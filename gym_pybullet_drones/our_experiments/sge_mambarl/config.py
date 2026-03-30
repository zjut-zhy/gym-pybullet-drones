"""Hyperparameters for SGE-MambaRL Phase 1 (SAC-guided Go-Explore)."""

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
    explore_steps: int = 300              # steps per Go-Explore iteration
    warmup_iterations: int = 100          # warm up from reset before using archive
    total_iterations: int = 20_000
    n_envs: int = 1                       # deterministic → single env

    # ── SAC explorer ─────────────────────────────────────────────
    sac_lr: float = 3e-4                  # SAC learning rate
    sac_buffer_size: int = 100_000        # SAC replay buffer capacity
    sac_batch_size: int = 256
    sac_tau: float = 0.005                # soft target update
    sac_gamma: float = 0.99
    sac_train_freq: int = 1               # gradient steps per env step
    sac_gradient_steps: int = 1
    sac_learning_starts: int = 1000       # random actions before SAC kicks in
    sac_ent_coef: str = "auto"            # automatic entropy tuning
    sac_update_interval: int = 10         # update SAC every N Go-Explore iterations

    # ── action ───────────────────────────────────────────────────
    action_dim: int = 2                   # VEL action type

    # ── logging / saving ─────────────────────────────────────────
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "results/sge_mambarl_phase1"
    seed: int = 42
