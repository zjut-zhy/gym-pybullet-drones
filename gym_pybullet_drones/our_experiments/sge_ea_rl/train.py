"""SGE-EA (ablation) Phase 1 -- SAC-guided Go-Explore.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.sge_ea_rl.train \\
        --total_iterations 20000

Phase 1 is identical across all SEAM-RL variants. Reuses the same
SAC-guided Go-Explore logic.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.go_explore.archive import Archive, Cell
from gym_pybullet_drones.our_experiments.sge_ea_rl.config import SGEConfig

# Import train logic from sge_ea_mamba_rl (Phase 1 is identical)
from gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.train import (
    _SACExplorer, _make_env as _make_env_base, _replay_to_cell, _run_iteration,
)


def _make_env(cfg: SGEConfig) -> OurSingleRLAviary:
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
    )


def train(cfg: SGEConfig) -> None:
    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = _make_env(cfg)
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

    test_obs, _ = env.reset(seed=cfg.seed)
    obs_dim = sum(np.asarray(v).ravel().shape[0] for v in test_obs.values())
    print(f"Obs dim (flattened): {obs_dim}, action dim: {cfg.action_dim}")

    sac = _SACExplorer(
        obs_dim=obs_dim,
        action_dim=cfg.action_dim,
        lr=cfg.sac_lr,
        buffer_size=cfg.sac_buffer_size,
        batch_size=cfg.sac_batch_size,
        tau=cfg.sac_tau,
        gamma=cfg.sac_gamma,
    )

    print("=" * 70)
    print("SGE-EA (ablation) Phase 1 -- SAC-Guided Go-Explore")
    print(f"  iters={cfg.total_iterations}  explore_steps={cfg.explore_steps}  "
          f"seed={cfg.seed}")
    print("=" * 70)

    t_start = time.time()
    total_new_cells = 0
    total_sac_steps = 0

    for iteration in range(1, cfg.total_iterations + 1):
        result = _run_iteration(
            env, archive, sac, cfg, rng,
            iteration=iteration,
            total_sac_steps=total_sac_steps,
        )
        total_new_cells += result["new_cells"]
        total_sac_steps = result["total_sac_steps"]

        if iteration % cfg.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
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
                f"cells={len(archive):5d} (+{result['new_cells']:3d})  "
                f"total_discovered={total_new_cells:6d}  "
                f"rew={result['total_reward']:+8.2f}  "
                f"sac_steps={total_sac_steps:>8,}"
                f"{best_info}  elapsed={elapsed:.1f}s"
            )

        if iteration % cfg.save_interval == 0:
            archive.save(os.path.join(cfg.output_dir, "archive.json"))

    archive.save(os.path.join(cfg.output_dir, "archive.json"))
    best = archive.get_best_cell()
    if best is not None:
        seq_len = (len(best.full_action_sequence)
                   if best.full_action_sequence else 0)
        print(f"\nBest cell: key={best.key}, captures={best.key[2]}, "
              f"reward={best.cumulative_reward:.2f}, "
              f"action_sequence={seq_len} steps")

    sac_path = os.path.join(cfg.output_dir, "sac_explorer.pt")
    torch.save({
        "actor": sac.actor.state_dict(),
        "actor_mean": sac.actor_mean.state_dict(),
        "actor_log_std": sac.actor_log_std.state_dict(),
        "q1": sac.q1.state_dict(),
        "q2": sac.q2.state_dict(),
    }, sac_path)
    print(f"SAC explorer saved to {sac_path}")

    env.close()


def _parse_args() -> SGEConfig:
    parser = argparse.ArgumentParser(
        description="SGE-EA (ablation) Phase 1: SAC-Guided Go-Explore")
    cfg = SGEConfig()
    for f in fields(SGEConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default),
                            default=f.default)
    return SGEConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
