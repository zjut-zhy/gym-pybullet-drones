"""SEAM-RL Phase 2 -- EA-Mamba RL + backward curriculum + SIL.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.robustify \\
        --demo_path results/sge_ea_mamba_rl_phase1/demos_best.demo.pkl \\
        --total_timesteps 1000000

Phase 2 of SEAM-RL:
  1. Backward curriculum: start from near the end of demo, shift backward.
  2. SB3 PPO with EAMambaActorCriticPolicy (Entity Attention + Mamba).
  3. SIL: self-imitation learning from demo + high-return online trajectories.
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

from gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.ea_mamba_policy import (
    EAMambaActorCriticPolicy,
)
from gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.sil_buffer import SILBuffer


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

    # ── EA-Mamba policy ──────────────────────────────────────────────
    mamba_d_model: int = 128
    mamba_d_state: int = 64
    mamba_n_layers: int = 2
    mamba_headdim: int = 32
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
    demo_path: str = "results/sge_ea_mamba_rl_phase1/demos_best.demo.pkl"
    output_dir: str = "results/sge_ea_mamba_rl"
    seed: int = 42
    device: str = "auto"


# ---------------------------------------------------------------------------
#  Backward Curriculum Wrapper
# ---------------------------------------------------------------------------

class BackwardCurriculumWrapper(gym.Wrapper):
    """Backward algorithm reset logic wrapper."""

    def __init__(
        self,
        env: OurSingleRLAviary,
        demo: dict,
        backward_step_size: int = 50,
        success_threshold: float = 0.8,
        eval_window: int = 20,
        max_backward_iters: int = 30,
        success_captures: int = 18,
        curriculum_ratio: float = 0.5,
    ) -> None:
        super().__init__(env)
        self.demo = demo
        self.demo_n_steps = demo["n_steps"]
        self.backward_step_size = backward_step_size
        self.success_threshold = success_threshold
        self.eval_window = eval_window
        self.max_backward_iters = max_backward_iters
        self.success_captures = success_captures
        self.curriculum_ratio = curriculum_ratio

        self.start_idx = max(0, self.demo_n_steps - backward_step_size)
        self.episode_count = 0
        self.success_history: List[bool] = []
        self._level_iters = 0
        self._ep_captures = 0
        self._is_curriculum_ep = False

    def reset(self, *, seed=None, options=None):
        self.episode_count += 1
        self._ep_captures = 0

        self._is_curriculum_ep = np.random.random() < self.curriculum_ratio

        if self._is_curriculum_ep:
            demo_seed = self.demo.get("env_seed", None)
            obs, info = self.env.reset(seed=demo_seed, options=options)
            if self.start_idx > 0:
                idx = min(self.start_idx, len(self.demo["action_list"]))
                for action in self.demo["action_list"][:idx]:
                    obs, _, terminated, truncated, _ = self.env.step(action)
                    if terminated or truncated:
                        break
        else:
            obs, info = self.env.reset(options=options)
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self._ep_captures = int(info.get("target_capture_count", 0))

        if terminated or truncated:
            if self._is_curriculum_ep:
                self.success_history.append(
                    self._ep_captures >= self.success_captures
                )
            if len(self.success_history) > self.eval_window:
                self.success_history = self.success_history[-self.eval_window:]

        return obs, rew, terminated, truncated, info

    @property
    def success_rate(self) -> float:
        if not self.success_history:
            return 0.0
        return float(np.mean(self.success_history))

    def maybe_shift_backward(self) -> bool:
        self._level_iters += 1

        if len(self.success_history) < self.eval_window:
            return False

        if (self.success_rate >= self.success_threshold
                or self._level_iters >= self.max_backward_iters):
            old_idx = self.start_idx
            self.start_idx = max(0, self.start_idx - self.backward_step_size)
            if self.start_idx != old_idx:
                reason = (f"success_rate={self.success_rate:.1%}"
                          if self.success_rate >= self.success_threshold
                          else f"max_iters={self.max_backward_iters}")
                print(f"\n  >> BACKWARD SHIFT: start_idx {old_idx} -> "
                      f"{self.start_idx}  ({reason})")
                self._level_iters = 0
                self.success_history.clear()
                return True
        return False

    @property
    def converged(self) -> bool:
        return (self.start_idx == 0
                and self._level_iters >= self.eval_window
                and self.success_rate >= self.success_threshold)


# ---------------------------------------------------------------------------
#  SIL Callback
# ---------------------------------------------------------------------------

class SILCallback(BaseCallback):
    """SIL callback for SB3 PPO with EA-Mamba policy."""

    def __init__(
        self,
        sil_buffer: SILBuffer,
        backward_wrapper: BackwardCurriculumWrapper,
        sil_coef: float = 0.1,
        sil_batch_size: int = 128,
        sil_updates: int = 5,
        sil_online_threshold: float = 0.0,
        gamma: float = 0.99,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.sil_buffer = sil_buffer
        self.backward_wrapper = backward_wrapper
        self.sil_coef = sil_coef
        self.sil_batch_size = sil_batch_size
        self.sil_updates = sil_updates
        self.sil_online_threshold = sil_online_threshold
        self.gamma = gamma

        self._ep_obs: List[Dict[str, np.ndarray]] = []
        self._ep_actions: List[np.ndarray] = []
        self._ep_rewards: List[float] = []

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        obs = self.locals.get("new_obs")
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if obs is not None and actions is not None:
            self._ep_obs.append(
                {k: np.array(v[0], copy=True) for k, v in obs.items()}
            )
            self._ep_actions.append(np.array(actions[0], copy=True))
            self._ep_rewards.append(float(rewards[0]))

            if dones is not None and dones[0]:
                ep_total = sum(self._ep_rewards)
                if ep_total > self.sil_online_threshold:
                    self.sil_buffer.add_trajectory(
                        self._ep_obs, self._ep_actions,
                        self._ep_rewards, gamma=self.gamma,
                    )
                self._ep_obs = []
                self._ep_actions = []
                self._ep_rewards = []

                fe = self.model.policy.features_extractor
                if hasattr(fe, "reset_hidden"):
                    fe.reset_hidden(batch_size=1)

        return True

    def _on_rollout_end(self) -> None:
        sil_loss_avg = self._do_sil_updates()
        self.backward_wrapper.maybe_shift_backward()

        self.logger.record("sil/loss", sil_loss_avg)
        self.logger.record("sil/buffer_size", len(self.sil_buffer))
        self.logger.record("backward/start_idx",
                           self.backward_wrapper.start_idx)
        self.logger.record("backward/success_rate",
                           self.backward_wrapper.success_rate)
        self.logger.record("backward/episode_count",
                           self.backward_wrapper.episode_count)

    def _do_sil_updates(self) -> float:
        if self.sil_coef <= 0 or len(self.sil_buffer) == 0:
            return 0.0

        policy = self.model.policy
        optimizer = policy.optimizer
        device = policy.device
        total_loss = 0.0
        n_updates = 0

        for _ in range(self.sil_updates):
            sil_batch = self.sil_buffer.sample_batch(
                self.sil_batch_size, device)
            if sil_batch is None:
                continue

            obs_tensor = sil_batch["obs"]
            action_tensor = sil_batch["action"]
            demo_returns = sil_batch["return"]

            values, log_prob, entropy = policy.evaluate_actions(
                obs_tensor, action_tensor)

            values_squeezed = values.squeeze(-1)
            advantage = (demo_returns - values_squeezed).clamp(min=0.0)

            sil_policy_loss = -(advantage.detach() * log_prob.squeeze(-1)).mean()
            sil_value_loss = (
                advantage * (demo_returns - values_squeezed).pow(2)
            ).mean()

            sil_loss = self.sil_coef * (sil_policy_loss + 0.5 * sil_value_loss)

            if torch.isnan(sil_loss) or torch.isinf(sil_loss):
                continue

            optimizer.zero_grad()
            sil_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += sil_loss.item()
            n_updates += 1

        return total_loss / max(n_updates, 1)


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

    tb_dir = os.path.join("runs", "sge_ea_mamba_rl")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(
        mamba_d_model=cfg.mamba_d_model,
        mamba_d_state=cfg.mamba_d_state,
        mamba_n_layers=cfg.mamba_n_layers,
        mamba_headdim=cfg.mamba_headdim,
        ea_d_k=cfg.ea_d_k,
        ea_d_attn_target=cfg.ea_d_attn_target,
        ea_d_attn_obstacle=cfg.ea_d_attn_obstacle,
        ea_n_heads=cfg.ea_n_heads,
    )

    model = PPO(
        EAMambaActorCriticPolicy,
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
    print("EA-Mamba Policy Architecture:")
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
    print("SEAM-RL Phase 2 -- EA-Mamba RL + SIL + Backward Curriculum")
    print(f"  EA-Mamba: d_model={cfg.mamba_d_model}, d_state={cfg.mamba_d_state}, "
          f"n_layers={cfg.mamba_n_layers}, headdim={cfg.mamba_headdim}")
    print(f"  EA: d_k={cfg.ea_d_k}, d_attn_target={cfg.ea_d_attn_target}, "
          f"d_attn_obstacle={cfg.ea_d_attn_obstacle}, n_heads={cfg.ea_n_heads}")
    print(f"  Backward: demo_steps={demo_n_steps}, "
          f"step_size={cfg.backward_step_size}")
    print(f"  SIL: coef={cfg.sil_coef}, "
          f"updates/rollout={cfg.sil_updates_per_rollout}")
    print(f"  PPO: n_steps={cfg.n_steps}, "
          f"total_timesteps={cfg.total_timesteps}")
    print(f"  Eval: every {cfg.eval_freq} steps, {cfg.n_eval_episodes} eps")
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
        description="SEAM-RL Phase 2 (EA-Mamba RL)")
    cfg = RobustifyConfig()
    for f in fields(RobustifyConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default),
                            default=f.default)
    return RobustifyConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
