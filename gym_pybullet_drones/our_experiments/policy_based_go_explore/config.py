"""Centralised hyper-parameter configuration for Policy-Based Go-Explore."""

from dataclasses import dataclass, field


@dataclass
class GoExploreConfig:
    """All tuneable knobs live here so that CLI / sweep tools can override them."""

    # ── Environment ──────────────────────────────────────────────
    num_drones: int = 2
    arena_size: float = 10.0
    target_count: int = 18
    obstacle_count: int = 6
    ctrl_freq: int = 60
    max_episode_len_sec: float = 60.0

    # ── Archive ──────────────────────────────────────────────────
    cell_size: float = 0.5          # metres per grid cell
    max_cells: int = 10_000
    selection_method: str = "score_proportional"  # or "uniform"

    # ── Go-Explore phases ────────────────────────────────────────
    return_max_steps: int = 200
    explore_max_steps: int = 300
    n_envs: int = 4

    # ── Trajectory tracker ───────────────────────────────────────
    sub_goal_spacing: float = 1.0   # metres between sub-goals
    sub_goal_reach_thresh: float = 0.5  # metres
    sub_goal_reward: float = 1.0
    potential_reward_scale: float = 0.1

    # ── Network ──────────────────────────────────────────────────
    obs_embed_dim: int = 128
    goal_dim: int = 2
    goal_embed_dim: int = 16
    gru_hidden: int = 128
    num_gru_layers: int = 1
    action_dim: int = 2             # VEL action space (vx, vy)

    # ── PPO ──────────────────────────────────────────────────────
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 256

    # ── Training loop ────────────────────────────────────────────
    total_iterations: int = 5_000
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "results/go_explore"
    seed: int = 42
    device: str = "cpu"             # "cpu" or "cuda"
