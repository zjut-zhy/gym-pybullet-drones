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
    warmup_iterations: int = 100          # 前 N 轮从 reset 开始，不使用 cell 选择
    total_iterations: int = 5000
    n_envs: int = 1                       # 固定种子下多环境无意义
    early_stop_successes: int = 10        # 成功 cell 达到此数量后早停 (0=不早停)

    # ── action ───────────────────────────────────────────────────
    action_dim: int = 2                   # VEL action type

    # ── logging / saving ─────────────────────────────────────────
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "results/go_explore"
    seed: int = 42
