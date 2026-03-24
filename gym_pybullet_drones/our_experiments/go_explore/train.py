"""Original Go-Explore (Phase 1) -- snapshot-based return + random exploration.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.train \\
        --total_iterations 5000 --n_envs 4

Adapted for single-agent OurSingleRLAviary -- flat obs / action / reward.
Uses environment snapshots instead of action replay for deterministic state
restoration in dynamic environments.
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


# ---------------------------------------------------------------------------
#  Environment factory
# ---------------------------------------------------------------------------

def _make_env(cfg: GoExploreConfig, env_seed: int) -> OurSingleRLAviary:
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
        environment_seed=env_seed,
    )


# ---------------------------------------------------------------------------
#  One iteration: select -> restore snapshot -> explore
# ---------------------------------------------------------------------------

def _run_iteration(
    env: OurSingleRLAviary,
    archive: Archive,
    cfg: GoExploreConfig,
    rng: np.random.RandomState,
) -> dict:
    """Execute one Go-Explore iteration."""
    act_dim = cfg.action_dim

    # -- select ---
    target_cell: Optional[Cell] = archive.select() if len(archive) > 0 else None

    # -- return phase (snapshot restore OR fresh reset) ---
    all_obs: List[Dict[str, np.ndarray]] = []
    all_snapshots: List[dict] = []
    all_rewards: List[float] = []
    return_restored = False

    if target_cell is not None and target_cell.snapshot is not None:
        # Restore to the archived state WITHOUT calling reset() first,
        # because reset() rebuilds the PyBullet world and invalidates
        # all previously saved p.saveState() IDs.
        env.restore_snapshot(target_cell.snapshot)
        obs = env._computeObs()
        return_restored = True
    else:
        # No snapshot to restore -- do a fresh reset
        obs, info = env.reset()

    # -- explore phase (random actions) ---
    explore_count = 0
    all_n_captured: List[int] = []
    for _ in range(cfg.explore_steps):
        action = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        all_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
        all_snapshots.append(env.get_snapshot())
        all_rewards.append(float(reward))
        all_n_captured.append(int(info.get("target_capture_count", 0)))
        explore_count += 1

        if terminated or truncated:
            break

    # -- archive update ---
    new_cells = archive.update(all_obs, all_snapshots, all_rewards,
                               trajectory_n_captured=all_n_captured)

    return {
        "new_cells": len(new_cells),
        "total_reward": sum(all_rewards),
        "return_restored": return_restored,
        "explore_steps": explore_count,
    }


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def train(cfg: GoExploreConfig) -> None:
    """Run original Go-Explore (Phase 1 -- snapshot-based return)."""
    rng = np.random.RandomState(cfg.seed)

    # -- environments ---
    envs = [_make_env(cfg, env_seed=cfg.seed + i) for i in range(cfg.n_envs)]

    # -- archive ---
    archive = Archive(
        cell_size=cfg.cell_size,
        arena_half=cfg.arena_size / 2.0,
        max_cells=cfg.max_cells,
    )
    archive.seed(cfg.seed)

    # -- output dir ---
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # -- main loop ---
    print("=" * 70)
    print("Original Go-Explore (snapshot-based)  |  OurSingleRLAviary")
    print(f"  n_envs={cfg.n_envs}  "
          f"explore_steps={cfg.explore_steps}  iters={cfg.total_iterations}")
    print("=" * 70)

    t_start = time.time()
    total_new_cells = 0

    for iteration in range(1, cfg.total_iterations + 1):
        iter_new = 0
        iter_rewards = []

        for env in envs:
            result = _run_iteration(env, archive, cfg, rng)
            iter_new += result["new_cells"]
            iter_rewards.append(result["total_reward"])

        total_new_cells += iter_new

        if iteration % cfg.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            mean_rew = np.mean(iter_rewards)
            print(
                f"[iter {iteration:5d}]  "
                f"cells={len(archive):5d} (+{iter_new:3d})  "
                f"total_discovered={total_new_cells:6d}  "
                f"mean_rew={mean_rew:+8.2f}  "
                f"elapsed={elapsed:.1f}s"
            )

        if iteration % cfg.save_interval == 0:
            archive.save(os.path.join(cfg.output_dir, "archive.json"))
            print(f"  -> saved archive ({len(archive)} cells)")

    archive.save(os.path.join(cfg.output_dir, "archive.json"))
    print(f"\nExploration finished.  Total cells: {len(archive)}")

    for env in envs:
        env.close()


# --- CLI ---

def _parse_args() -> GoExploreConfig:
    parser = argparse.ArgumentParser(description="Original Go-Explore (snapshot-based)")
    cfg = GoExploreConfig()
    for f in fields(GoExploreConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return GoExploreConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
