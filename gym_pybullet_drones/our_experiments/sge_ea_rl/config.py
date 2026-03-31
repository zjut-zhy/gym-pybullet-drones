"""Hyperparameters for SGE-EA (ablation) Phase 1 (SAC-guided Go-Explore)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SGEConfig:
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
    explore_steps: int = 300
    warmup_iterations: int = 100
    total_iterations: int = 20_000
    n_envs: int = 1

    # ── SAC explorer ─────────────────────────────────────────────
    sac_lr: float = 3e-4
    sac_buffer_size: int = 100_000
    sac_batch_size: int = 256
    sac_tau: float = 0.005
    sac_gamma: float = 0.99
    sac_train_freq: int = 1
    sac_gradient_steps: int = 1
    sac_learning_starts: int = 1000
    sac_ent_coef: str = "auto"
    sac_update_interval: int = 10

    # ── action ───────────────────────────────────────────────────
    action_dim: int = 2

    # ── logging / saving ─────────────────────────────────────────
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "results/sge_ea_rl_phase1"
    seed: int = 42
