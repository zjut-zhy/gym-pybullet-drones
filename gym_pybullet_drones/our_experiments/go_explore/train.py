"""Go-Explore Phase 1 -- deterministic exploration + per-cell trajectories.

Each cell independently stores its complete action sequence from
env.reset().  The best cell's sequence is used for Phase 2 demo.

State restoration uses **action replay** (reset + replay full_action_sequence)
for bit-identical determinism.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.go_explore.archive import Archive, Cell
from gym_pybullet_drones.our_experiments.go_explore.config import GoExploreConfig


def _make_env(cfg: GoExploreConfig) -> OurSingleRLAviary:
    return OurSingleRLAviary(
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        gui=False,
        record=False,
        arena_size_xy_m=cfg.arena_size,
        target_count=cfg.target_count,
        obstacle_count=cfg.obstacle_count,
        ctrl_freq=cfg.ctrl_freq,
        max_episode_len_sec=cfg.max_episode_len_sec,
        environment_seed=cfg.seed,
        enable_target_attraction=False,  # No shaping reward in Phase 1
    )


def _replay_to_cell(
    env: OurSingleRLAviary,
    cell: Cell,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Deterministic action replay: reset + replay full_action_sequence.

    Returns the observation after the last replayed action.
    """
    obs, _ = env.reset(seed=seed)
    if cell.full_action_sequence:
        for action in cell.full_action_sequence:
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                # Shouldn't happen if sequences are valid, but be safe
                break
    return obs


def _run_iteration(
    env: OurSingleRLAviary,
    archive: Archive,
    cfg: GoExploreConfig,
    rng: np.random.RandomState,
    iteration: int = 0,
) -> dict:
    act_dim = cfg.action_dim

    # -- select cell to explore from ---
    # Warmup: always start from reset to build diverse initial archive
    if iteration <= cfg.warmup_iterations:
        target_cell = None
    else:
        target_cell = archive.select() if len(archive) > 0 else None

    # -- return phase (action replay for determinism) ---
    source_cell = None
    if target_cell is not None and target_cell.full_action_sequence is not None:
        obs = _replay_to_cell(env, target_cell, seed=cfg.seed)
        source_cell = target_cell
    else:
        obs, info = env.reset(seed=cfg.seed)

    # -- explore phase ---
    all_obs: List[Dict[str, np.ndarray]] = []
    all_rewards: List[float] = []
    all_actions: List[np.ndarray] = []
    all_n_captured: List[int] = []

    action = None
    for step in range(cfg.explore_steps):
        # Hold each action for 30 steps (~1s at 30Hz ctrl_freq),
        # then sample a new direction.
        if step % 30 == 0:
            action = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        n_cap = int(info.get("target_capture_count", 0))

        if terminated or truncated:
            if n_cap >= cfg.target_count:
                all_obs.append({k: np.array(v, copy=True)
                                for k, v in obs.items()})
                all_rewards.append(float(reward))
                all_actions.append(np.array(action, copy=True))
                all_n_captured.append(n_cap)
            break

        all_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
        all_rewards.append(float(reward))
        all_actions.append(np.array(action, copy=True))
        all_n_captured.append(n_cap)

    # -- archive update with source_cell for sequence building ---
    new_cells = archive.update(
        all_obs, all_rewards, all_actions,
        trajectory_n_captured=all_n_captured,
        source_cell=source_cell,
    )

    return {
        "new_cells": len(new_cells),
        "total_reward": sum(all_rewards),
        "return_restored": source_cell is not None,
    }


def train(cfg: GoExploreConfig) -> None:
    rng = np.random.RandomState(cfg.seed)
    envs = [_make_env(cfg) for _ in range(cfg.n_envs)]

    max_steps = int(cfg.ctrl_freq * cfg.max_episode_len_sec)

    archive = Archive(
        cell_size=cfg.cell_size,
        arena_half=cfg.arena_size / 2.0,
        max_cells=cfg.max_cells,
        env_seed=cfg.seed,
        max_steps=max_steps,
        target_count=cfg.target_count,
    )
    archive.seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Go-Explore Phase 1 (action replay, per-cell full action sequences)")
    print(f"  n_envs={cfg.n_envs}  explore_steps={cfg.explore_steps}  "
          f"iters={cfg.total_iterations}  seed={cfg.seed}")
    print("=" * 70)

    t_start = time.time()
    total_new_cells = 0

    for iteration in range(1, cfg.total_iterations + 1):
        iter_new = 0
        iter_rewards = []

        for env in envs:
            result = _run_iteration(env, archive, cfg, rng, iteration)
            iter_new += result["new_cells"]
            iter_rewards.append(result["total_reward"])

        total_new_cells += iter_new

        if iteration % cfg.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            mean_rew = np.mean(iter_rewards)
            best = archive.get_best_cell()
            best_info = ""
            if best is not None:
                seq_len = (len(best.full_action_sequence)
                           if best.full_action_sequence else 0)
                best_info = (f"  best=({best.key[2]}cap, "
                             f"rew={best.cumulative_reward:+.1f}, "
                             f"seq={seq_len})")
            print(
                f"[iter {iteration:5d}]  "
                f"cells={len(archive):5d} (+{iter_new:3d})  "
                f"total_discovered={total_new_cells:6d}  "
                f"mean_rew={mean_rew:+8.2f}"
                f"{best_info}  elapsed={elapsed:.1f}s"
            )

        if iteration % cfg.save_interval == 0:
            archive.save(os.path.join(cfg.output_dir, "archive.json"))
            print(f"  -> saved archive ({len(archive)} cells)")

    archive.save(os.path.join(cfg.output_dir, "archive.json"))
    best = archive.get_best_cell()
    if best is not None:
        seq_len = (len(best.full_action_sequence)
                   if best.full_action_sequence else 0)
        print(f"\nBest cell: key={best.key}, captures={best.key[2]}, "
              f"reward={best.cumulative_reward:.2f}, "
              f"action_sequence={seq_len} steps")
    print(f"Exploration finished.  Total cells: {len(archive)}")

    for env in envs:
        env.close()


def _parse_args() -> GoExploreConfig:
    parser = argparse.ArgumentParser(description="Go-Explore Phase 1")
    cfg = GoExploreConfig()
    for f in fields(GoExploreConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default),
                            default=f.default)
    return GoExploreConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
