"""Hyperparameters for original (action-replay) Go-Explore."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GoExploreConfig:
    # ── environment ──────────────────────────────────────────────
    num_drones: int = 1
    arena_size: float = 10.0
    target_count: int = 18
    obstacle_count: int = 6
    ctrl_freq: int = 30
    max_episode_len_sec: float = 60.0

    # ── archive / cell ───────────────────────────────────────────
    cell_size: float = 0.5
    max_cells: int = 10_000

    # ── exploration ──────────────────────────────────────────────
    explore_steps: int = 300              # random-action steps after return
    total_iterations: int = 5000
    n_envs: int = 4                       # parallel environments

    # ── action ───────────────────────────────────────────────────
    action_dim: int = 2                   # VEL action type

    # ── logging / saving ─────────────────────────────────────────
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "results/go_explore"
    seed: int = 42
