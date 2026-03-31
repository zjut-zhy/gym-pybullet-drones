"""SGE-EA (ablation) Phase 2 -- EA RL + backward curriculum + SIL.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.sge_ea_rl.robustify \\
        --demo_path results/sge_ea_rl_phase1/demos_best.demo.pkl \\
        --total_timesteps 1000000

Ablation: Entity Attention only (no Mamba temporal modeling).
Uses EAActorCriticPolicy instead of EAMambaActorCriticPolicy.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.sge_ea_rl.ea_policy import (
    EAActorCriticPolicy,
)
from gym_pybullet_drones.our_experiments.sge_ea_rl.sil_buffer import SILBuffer

# Reuse backward curriculum wrapper and SIL callback from sge_ea_mamba_rl
from gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.robustify import (
    BackwardCurriculumWrapper,
    SILCallback,
)


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

@dataclass
class RobustifyConfig:
    # ── environment ──────────────────────────────────────────────────
    arena_size: float = 10.0
    target_count: int = 18
    obstacle_count: int = 6
    ctrl_freq: int = 30
    max_episode_len_sec: float = 60.0

    # ── PPO ──────────────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    n_steps: int = 2048
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    batch_size: int = 256

    # ── EA policy (no mamba) ─────────────────────────────────────────
    ea_d_model: int = 128
    ea_d_k: int = 32
    ea_d_attn_target: int = 64
    ea_d_attn_obstacle: int = 32
    ea_n_heads: int = 2

    # ── backward curriculum ──────────────────────────────────────────
    backward_step_size: int = 50
    success_threshold: float = 0.8
    eval_window: int = 20
    max_backward_iters: int = 30
    success_captures: int = 18
    curriculum_ratio: float = 0.5

    # ── SIL ──────────────────────────────────────────────────────────
    sil_coef: float = 0.1
    sil_batch_size: int = 128
    sil_capacity: int = 50_000
    sil_updates_per_rollout: int = 5
    sil_online_threshold: float = 0.0

    # ── evaluation ───────────────────────────────────────────────────
    eval_freq: int = 10_000
    n_eval_episodes: int = 3

    # ── demo / IO ────────────────────────────────────────────────────
    demo_path: str = "results/sge_ea_rl_phase1/demos_best.demo.pkl"
    output_dir: str = "results/sge_ea_rl"
    seed: int = 42
    device: str = "auto"


# ---------------------------------------------------------------------------
#  Env factory
# ---------------------------------------------------------------------------

def _make_env(cfg: RobustifyConfig, demo: dict):
    def _thunk():
        env = OurSingleRLAviary(
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
        env = BackwardCurriculumWrapper(
            env, demo,
            backward_step_size=cfg.backward_step_size,
            success_threshold=cfg.success_threshold,
            eval_window=cfg.eval_window,
            max_backward_iters=cfg.max_backward_iters,
            success_captures=cfg.success_captures,
            curriculum_ratio=cfg.curriculum_ratio,
        )
        return env
    return _thunk


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def train(cfg: RobustifyConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    with open(cfg.demo_path, "rb") as f:
        demo = pickle.load(f)
    demo_n_steps = demo["n_steps"]
    print(f"Loaded demo: {demo_n_steps} steps, "
          f"total_reward={demo['total_reward']:.2f}")

    train_env = DummyVecEnv([_make_env(cfg, demo)])
    eval_env = OurSingleRLAviary(
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        gui=False,
        record=False,
        arena_size_xy_m=cfg.arena_size,
        target_count=cfg.target_count,
        obstacle_count=cfg.obstacle_count,
        ctrl_freq=cfg.ctrl_freq,
        max_episode_len_sec=cfg.max_episode_len_sec,
    )

    backward_wrapper: BackwardCurriculumWrapper = train_env.envs[0]

    sil_buffer = SILBuffer(capacity=cfg.sil_capacity)
    sil_buffer.load_demo(
        demo["obs_list"], demo["action_list"], demo["returns"],
    )
    print(f"SIL buffer loaded with {len(sil_buffer)} demo transitions")

    tb_dir = os.path.join("runs", "sge_ea_rl")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(
        ea_d_model=cfg.ea_d_model,
        ea_d_k=cfg.ea_d_k,
        ea_d_attn_target=cfg.ea_d_attn_target,
        ea_d_attn_obstacle=cfg.ea_d_attn_obstacle,
        ea_n_heads=cfg.ea_n_heads,
    )

    model = PPO(
        EAActorCriticPolicy,
        train_env,
        verbose=1,
        device=cfg.device,
        tensorboard_log=tb_dir,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        n_epochs=cfg.n_epochs,
        clip_range=cfg.clip_range,
        max_grad_norm=cfg.max_grad_norm,
        vf_coef=cfg.vf_coef,
        seed=cfg.seed,
        policy_kwargs=policy_kwargs,
    )

    print("\n" + "=" * 70)
    print("EA Policy Architecture (ablation: no Mamba):")
    print(model.policy)
    print("=" * 70)

    sil_callback = SILCallback(
        sil_buffer=sil_buffer,
        backward_wrapper=backward_wrapper,
        sil_coef=cfg.sil_coef,
        sil_batch_size=cfg.sil_batch_size,
        sil_updates=cfg.sil_updates_per_rollout,
        sil_online_threshold=cfg.sil_online_threshold,
        gamma=cfg.gamma,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        verbose=1,
        best_model_save_path=cfg.output_dir,
        log_path=cfg.output_dir,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    print("=" * 70)
    print("SGE-EA (ablation) Phase 2 -- EA RL + SIL + Backward Curriculum")
    print(f"  EA: d_model={cfg.ea_d_model}, d_k={cfg.ea_d_k}, "
          f"n_heads={cfg.ea_n_heads}")
    print(f"  NOTE: NO Mamba temporal modeling (ablation baseline)")
    print(f"  Backward: demo_steps={demo_n_steps}, "
          f"step_size={cfg.backward_step_size}")
    print(f"  SIL: coef={cfg.sil_coef}, "
          f"updates/rollout={cfg.sil_updates_per_rollout}")
    print(f"  PPO: n_steps={cfg.n_steps}, "
          f"total_timesteps={cfg.total_timesteps}")
    print(f"  TensorBoard: {tb_dir}")
    print("=" * 70)

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[sil_callback, eval_callback],
        log_interval=1,
        progress_bar=False,
    )

    final_path = os.path.join(cfg.output_dir, "final_model")
    model.save(final_path)
    print(f"\nPhase 2 finished. Model saved to {final_path}.zip")

    if backward_wrapper.converged:
        print(f"  >> CONVERGED at start_idx=0, "
              f"success_rate={backward_wrapper.success_rate:.1%}")
    else:
        print(f"  Final backward start_idx={backward_wrapper.start_idx}, "
              f"success_rate={backward_wrapper.success_rate:.1%}")

    eval_env.close()
    train_env.close()


def _parse_args() -> RobustifyConfig:
    parser = argparse.ArgumentParser(
        description="SGE-EA (ablation) Phase 2 (EA RL)")
    cfg = RobustifyConfig()
    for f in fields(RobustifyConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default),
                            default=f.default)
    return RobustifyConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
