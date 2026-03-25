"""Go-Explore Phase 1 -- deterministic exploration with tree archiving.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.train \\
        --total_iterations 5000 --n_envs 4

Phase 1 is FULLY DETERMINISTIC: all environments use the same seed and
all randomness comes from a fixed RNG.  The archive forms a tree where
each cell stores a parent pointer + action segment.  The best trajectory
can be reconstructed by tracing parent pointers back to the root.
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
#  Environment factory -- ALL envs use the SAME seed for determinism
# ---------------------------------------------------------------------------

def _make_env(cfg: GoExploreConfig) -> OurSingleRLAviary:
    """Create a deterministic environment with the config's seed."""
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
        environment_seed=cfg.seed,            # ALL envs share the same seed
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
    """Execute one Go-Explore iteration with tree-based archiving."""
    act_dim = cfg.action_dim

    # -- select a cell to explore from ---
    target_cell: Optional[Cell] = archive.select() if len(archive) > 0 else None

    # -- return phase: snapshot restore or fresh reset ---
    source_cell_key = None
    source_cum_reward = 0.0
    source_cost = 0
    return_restored = False

    if target_cell is not None and target_cell.snapshot is not None:
        env.restore_snapshot(target_cell.snapshot)
        obs = env._computeObs()
        source_cell_key = target_cell.key
        source_cum_reward = target_cell.cumulative_reward
        source_cost = target_cell.trajectory_cost
        return_restored = True
    else:
        obs, info = env.reset()

    # -- explore phase (random actions) ---
    all_obs: List[Dict[str, np.ndarray]] = []
    all_snapshots: List[dict] = []
    all_rewards: List[float] = []
    all_actions: List[np.ndarray] = []
    all_n_captured: List[int] = []
    explore_count = 0

    for _ in range(cfg.explore_steps):
        action = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        all_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
        all_snapshots.append(env.get_snapshot())
        all_rewards.append(float(reward))
        all_actions.append(np.array(action, copy=True))
        all_n_captured.append(int(info.get("target_capture_count", 0)))
        explore_count += 1

        if terminated or truncated:
            break

    # -- archive update with tree structure ---
    new_cells = archive.update(
        all_obs, all_snapshots, all_rewards, all_actions,
        trajectory_n_captured=all_n_captured,
        source_cell_key=source_cell_key,
        source_cum_reward=source_cum_reward,
        source_cost=source_cost,
    )

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
    """Run Go-Explore Phase 1 with deterministic exploration."""
    rng = np.random.RandomState(cfg.seed)

    # ALL envs share the same seed for determinism
    envs = [_make_env(cfg) for _ in range(cfg.n_envs)]

    archive = Archive(
        cell_size=cfg.cell_size,
        arena_half=cfg.arena_size / 2.0,
        max_cells=cfg.max_cells,
        env_seed=cfg.seed,
    )
    archive.seed(cfg.seed)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Go-Explore Phase 1 (deterministic, tree-archiving)")
    print(f"  n_envs={cfg.n_envs}  explore_steps={cfg.explore_steps}  "
          f"iters={cfg.total_iterations}  env_seed={cfg.seed}")
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
            best = archive.get_best_cell()
            best_info = ""
            if best is not None:
                best_info = (f"  best=({best.key[2]}cap, "
                             f"rew={best.cumulative_reward:+.1f}, "
                             f"cost={best.trajectory_cost})")
            print(
                f"[iter {iteration:5d}]  "
                f"cells={len(archive):5d} (+{iter_new:3d})  "
                f"total_discovered={total_new_cells:6d}  "
                f"mean_rew={mean_rew:+8.2f}"
                f"{best_info}  "
                f"elapsed={elapsed:.1f}s"
            )

        if iteration % cfg.save_interval == 0:
            archive.save(os.path.join(cfg.output_dir, "archive.json"))
            print(f"  -> saved archive ({len(archive)} cells)")

    archive.save(os.path.join(cfg.output_dir, "archive.json"))

    best = archive.get_best_cell()
    if best is not None:
        full_actions = archive.reconstruct_trajectory(best.key)
        print(f"\nBest cell: key={best.key}, captures={best.key[2]}, "
              f"reward={best.cumulative_reward:.2f}, "
              f"cost={best.trajectory_cost}")
        print(f"Reconstructed trajectory: {len(full_actions)} actions")
    print(f"Exploration finished.  Total cells: {len(archive)}")

    for env in envs:
        env.close()


# --- CLI ---

def _parse_args() -> GoExploreConfig:
    parser = argparse.ArgumentParser(
        description="Go-Explore Phase 1 (deterministic, tree-archiving)")
    cfg = GoExploreConfig()
    for f in fields(GoExploreConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default),
                            default=f.default)
    return GoExploreConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
