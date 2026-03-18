"""Original Go-Explore (Phase 1) — action-replay return + random exploration.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.train \\
        --num_drones 2 --total_iterations 5000 --n_envs 4

Adapted for PettingZoo ParallelEnv — each drone is a named agent with its
own observation, action, and reward.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from gym_pybullet_drones.envs.OurRLAviary_PettingZoo import OurRLAviaryPZ
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.go_explore.archive import Archive, Cell
from gym_pybullet_drones.our_experiments.go_explore.config import GoExploreConfig


# ─────────────────────────────────────────────────────────────────────────────
#  Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_env(cfg: GoExploreConfig, env_seed: int) -> OurRLAviaryPZ:
    return OurRLAviaryPZ(
        num_drones=cfg.num_drones,
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


# ─────────────────────────────────────────────────────────────────────────────
#  Random action helper
# ─────────────────────────────────────────────────────────────────────────────

def _random_actions(
    agents: List[str],
    action_dim: int,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """Generate random actions for all agents."""
    return {
        agent: rng.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
        for agent in agents
    }


# ─────────────────────────────────────────────────────────────────────────────
#  One iteration: select → return → explore
# ─────────────────────────────────────────────────────────────────────────────

def _run_iteration(
    env: OurRLAviaryPZ,
    archive: Archive,
    cfg: GoExploreConfig,
    rng: np.random.RandomState,
) -> dict:
    """Execute one Go-Explore iteration."""
    agents = env.possible_agents
    act_dim = cfg.action_dim

    # ── select ───────────────────────────────────────────────────
    target_cell: Optional[Cell] = archive.select() if len(archive) > 0 else None

    # ── reset ────────────────────────────────────────────────────
    observations, infos = env.reset()

    # ── return phase (action replay) ─────────────────────────────
    all_actions: List[Dict[str, np.ndarray]] = []
    all_obs: List[Dict[str, Dict[str, np.ndarray]]] = []
    all_rewards: List[Dict[str, float]] = []
    return_steps = 0

    if target_cell is not None and len(target_cell.action_sequence) > 0:
        for stored_actions in target_cell.action_sequence:
            # stored_actions is {agent_name: action_array}
            # ensure all current agents have an action
            actions = {}
            for agent in env.agents:
                if agent in stored_actions:
                    actions[agent] = np.asarray(stored_actions[agent], dtype=np.float32)
                else:
                    actions[agent] = np.zeros(act_dim, dtype=np.float32)

            observations, rewards, terminations, truncations, infos = env.step(actions)

            all_actions.append({a: np.array(v, copy=True) for a, v in actions.items()})
            all_obs.append({a: {k: np.array(v, copy=True) for k, v in obs.items()}
                           for a, obs in observations.items()})
            all_rewards.append(dict(rewards))
            return_steps += 1

            if any(terminations.values()) or any(truncations.values()):
                new_cells = archive.update(all_obs, all_actions, all_rewards)
                return {
                    "new_cells": len(new_cells),
                    "total_reward": sum(sum(r.values()) for r in all_rewards),
                    "return_steps": return_steps,
                    "explore_steps": 0,
                }

            # PettingZoo clears agents on done
            if not env.agents:
                new_cells = archive.update(all_obs, all_actions, all_rewards)
                return {
                    "new_cells": len(new_cells),
                    "total_reward": sum(sum(r.values()) for r in all_rewards),
                    "return_steps": return_steps,
                    "explore_steps": 0,
                }

    # ── explore phase (random actions) ───────────────────────────
    explore_count = 0
    for _ in range(cfg.explore_steps):
        if not env.agents:
            break
        actions = _random_actions(env.agents, act_dim, rng)
        observations, rewards, terminations, truncations, infos = env.step(actions)

        all_actions.append({a: np.array(v, copy=True) for a, v in actions.items()})
        all_obs.append({a: {k: np.array(v, copy=True) for k, v in obs.items()}
                       for a, obs in observations.items()})
        all_rewards.append(dict(rewards))
        explore_count += 1

        if any(terminations.values()) or any(truncations.values()):
            break
        if not env.agents:
            break

    # ── archive update ───────────────────────────────────────────
    new_cells = archive.update(all_obs, all_actions, all_rewards)

    return {
        "new_cells": len(new_cells),
        "total_reward": sum(sum(r.values()) for r in all_rewards),
        "return_steps": return_steps,
        "explore_steps": explore_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: GoExploreConfig) -> None:
    """Run original Go-Explore (Phase 1 — exploration via action replay)."""
    rng = np.random.RandomState(cfg.seed)

    # ── environments ─────────────────────────────────────────────
    envs = [_make_env(cfg, env_seed=cfg.seed + i) for i in range(cfg.n_envs)]

    # ── archive ──────────────────────────────────────────────────
    archive = Archive(
        cell_size=cfg.cell_size,
        arena_half=cfg.arena_size / 2.0,
        max_cells=cfg.max_cells,
    )
    archive.seed(cfg.seed)

    # ── output dir ───────────────────────────────────────────────
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── main loop ────────────────────────────────────────────────
    print("=" * 70)
    print("Original Go-Explore (action-replay)  |  OurRLAviaryPZ")
    print(f"  n_envs={cfg.n_envs}  n_drones={cfg.num_drones}  "
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
            print(f"  → saved archive ({len(archive)} cells)")

    archive.save(os.path.join(cfg.output_dir, "archive.json"))
    print(f"\n✓ Exploration finished.  Total cells: {len(archive)}")

    for env in envs:
        env.close()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> GoExploreConfig:
    parser = argparse.ArgumentParser(description="Original Go-Explore (action-replay)")
    cfg = GoExploreConfig()
    for f in fields(GoExploreConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return GoExploreConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
