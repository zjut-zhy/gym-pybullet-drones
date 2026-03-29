"""Go-Explore Phase 2 -- Robustification via SB3 PPO + backward curriculum + SIL.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.robustify \\
        --demo_path results/go_explore/demos_best.demo.pkl \\
        --total_timesteps 1000000

Implements the full Phase 2 mechanism on top of Stable-Baselines3's PPO:
  1. Backward Algorithm: start training from near the end of the demo
     trajectory, gradually shifting the starting point backward toward t=0.
  2. SB3 PPO: online policy gradient (MultiInputPolicy, same hypers as
     sb3rl/train.py).
  3. SIL: self-imitation learning from demo + high-return online trajectories,
     applied as extra gradient steps after each PPO rollout.
  4. Alternating resets: backward waypoint (odd eps) / random seed (even eps).
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

from gym_pybullet_drones.our_experiments.go_explore.sil_buffer import SILBuffer


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

    # ── PPO (aligned with sb3rl/train.py) ────────────────────────────
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

    # ── backward algorithm ───────────────────────────────────────────
    backward_step_size: int = 50     # how many demo steps to shift backward
    success_threshold: float = 0.8   # success rate to trigger backward shift
    eval_window: int = 20            # number of recent episodes for success rate
    max_backward_iters: int = 30     # max PPO updates per backward level
    success_captures: int = 18       # target captures needed for "success"
    curriculum_ratio: float = 0.5    # fraction of episodes using backward waypoint reset

    # ── SIL ──────────────────────────────────────────────────────────
    sil_coef: float = 0.1
    sil_batch_size: int = 128
    sil_capacity: int = 50_000
    sil_updates_per_rollout: int = 5     # SIL gradient steps per PPO rollout
    sil_online_threshold: float = 0.0    # min ep reward to add to SIL buffer

    # ── evaluation (aligned with sb3rl/train.py) ─────────────────────
    eval_freq: int = 10_000
    n_eval_episodes: int = 3

    # ── demo / IO ────────────────────────────────────────────────────
    demo_path: str = "results/go_explore/demos_best.demo.pkl"
    output_dir: str = "results/go_explore_phase2"
    seed: int = 42
    device: str = "auto"


# ---------------------------------------------------------------------------
#  Backward Curriculum Wrapper
# ---------------------------------------------------------------------------

class BackwardCurriculumWrapper(gym.Wrapper):
    """Gymnasium wrapper implementing backward algorithm reset logic.

    Each episode randomly selects (with probability ``curriculum_ratio``):
      - Backward waypoint: replay demo actions to start_idx, then train
      - Normal reset: env.reset() with natural randomness (like train.py)

    Backward index shifts when success_rate >= threshold or after max_iters.
    """

    def __init__(
        self,
        env: OurSingleRLAviary,
        demo: dict,
        backward_step_size: int = 50,
        success_threshold: float = 0.8,
        eval_window: int = 20,
        max_backward_iters: int = 200,
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

        # Backward state
        self.start_idx = max(0, self.demo_n_steps - backward_step_size)
        self.episode_count = 0
        self.rollout_count = 0  # PPO update counter (set by callback)
        self.success_history: List[bool] = []
        self._level_iters = 0

        # Per-episode tracking
        self._ep_captures = 0
        self._is_curriculum_ep = False

    def reset(self, *, seed=None, options=None):
        self.episode_count += 1
        self._ep_captures = 0

        self._is_curriculum_ep = np.random.random() < self.curriculum_ratio

        if self._is_curriculum_ep:
            # Backward waypoint: replay demo actions to start_idx
            demo_seed = self.demo.get("env_seed", None)
            obs, info = self.env.reset(seed=demo_seed, options=options)
            if self.start_idx > 0:
                idx = min(self.start_idx, len(self.demo["action_list"]))
                for action in self.demo["action_list"][:idx]:
                    obs, _, terminated, truncated, _ = self.env.step(action)
                    if terminated or truncated:
                        break
        else:
            # Normal reset: rely on env's own randomness (same as train.py)
            obs, info = self.env.reset(options=options)
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self._ep_captures = int(info.get("target_capture_count", 0))

        if terminated or truncated:
            # Only track curriculum episodes for backward shift decisions;
            # random episodes are for generalization, not progress measurement.
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
        """Check and perform backward shift. Called by callback after PPO update."""
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
        """True if at start_idx=0 with sufficient success rate."""
        return (self.start_idx == 0
                and self._level_iters >= self.eval_window
                and self.success_rate >= self.success_threshold)


# ---------------------------------------------------------------------------
#  SIL Callback
# ---------------------------------------------------------------------------

class SILCallback(BaseCallback):
    """Self-Imitation Learning callback for SB3's PPO.

    After each PPO rollout:
      1. Do ``sil_updates`` gradient steps on the policy using the SIL buffer.
      2. Track online episodes and add high-return ones to the SIL buffer.
      3. Log backward curriculum + SIL stats to TensorBoard.
    """

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

        # Per-episode tracking for SIL online collection
        self._ep_obs: List[Dict[str, np.ndarray]] = []
        self._ep_actions: List[np.ndarray] = []
        self._ep_rewards: List[float] = []

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        """Track per-step data for SIL online collection."""
        # Collect transition data
        # For VecEnv, we get arrays of shape (n_envs, ...)
        # We only use n_envs=1
        obs = self.locals.get("new_obs")
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if obs is not None and actions is not None:
            # Store single-env data
            self._ep_obs.append(
                {k: np.array(v[0], copy=True) for k, v in obs.items()}
            )
            self._ep_actions.append(np.array(actions[0], copy=True))
            self._ep_rewards.append(float(rewards[0]))

            if dones is not None and dones[0]:
                # Episode finished — add to SIL if reward is high enough
                ep_total = sum(self._ep_rewards)
                if ep_total > self.sil_online_threshold:
                    self.sil_buffer.add_trajectory(
                        self._ep_obs, self._ep_actions,
                        self._ep_rewards, gamma=self.gamma,
                    )
                # Reset episode tracking
                self._ep_obs = []
                self._ep_actions = []
                self._ep_rewards = []

        return True

    def _on_rollout_end(self) -> None:
        """After rollout collection: do SIL updates + backward shift check."""
        # -- SIL gradient steps --
        sil_loss_avg = self._do_sil_updates()

        # -- backward shift check --
        shifted = self.backward_wrapper.maybe_shift_backward()

        # -- TensorBoard logging --
        self.logger.record("sil/loss", sil_loss_avg)
        self.logger.record("sil/buffer_size", len(self.sil_buffer))
        self.logger.record("backward/start_idx",
                           self.backward_wrapper.start_idx)
        self.logger.record("backward/success_rate",
                           self.backward_wrapper.success_rate)
        self.logger.record("backward/episode_count",
                           self.backward_wrapper.episode_count)

    def _do_sil_updates(self) -> float:
        """Perform SIL gradient steps on the SB3 policy."""
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

            # Evaluate actions using SB3's policy
            obs_tensor = sil_batch["obs"]
            action_tensor = sil_batch["action"]
            demo_returns = sil_batch["return"]

            # SB3's evaluate_actions returns (values, log_prob, entropy)
            values, log_prob, entropy = policy.evaluate_actions(
                obs_tensor, action_tensor)

            # SIL loss: imitate when demo return > current value
            values_squeezed = values.squeeze(-1)
            advantage = (demo_returns - values_squeezed).clamp(min=0.0)

            # Policy imitation (advantage-weighted)
            sil_policy_loss = -(advantage.detach() * log_prob.squeeze(-1)).mean()
            # Value regression toward demo returns (where demo is better)
            sil_value_loss = (advantage * (demo_returns - values_squeezed).pow(2)).mean()

            sil_loss = self.sil_coef * (sil_policy_loss + 0.5 * sil_value_loss)

            # NaN/Inf guard
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
    """Create OurSingleRLAviary wrapped with backward curriculum."""
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

    # -- load demo --
    with open(cfg.demo_path, "rb") as f:
        demo = pickle.load(f)
    demo_n_steps = demo["n_steps"]
    print(f"Loaded demo: {demo_n_steps} steps, "
          f"total_reward={demo['total_reward']:.2f}")

    # -- environments --
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

    # -- access backward wrapper --
    backward_wrapper: BackwardCurriculumWrapper = train_env.envs[0]

    # -- SIL buffer: load demo --
    sil_buffer = SILBuffer(capacity=cfg.sil_capacity)
    sil_buffer.load_demo(
        demo["obs_list"], demo["action_list"], demo["returns"],
    )
    print(f"SIL buffer loaded with {len(sil_buffer)} demo transitions")

    # -- TensorBoard (SB3 auto-creates PPO_1, PPO_2, … inside this dir) --
    tb_dir = os.path.join("runs", "go_explore_phase2")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # -- SB3 PPO model (same hypers as sb3rl/train.py) --
    model = PPO(
        "MultiInputPolicy",
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
    )

    # -- Callbacks --
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

    # -- Print config --
    print("=" * 70)
    print("Go-Explore Phase 2 -- Robustification (SB3 PPO + SIL)")
    print(f"  Backward algorithm: demo_steps={demo_n_steps}, "
          f"step_size={cfg.backward_step_size}")
    print(f"  SIL: coef={cfg.sil_coef}, updates/rollout={cfg.sil_updates_per_rollout}")
    print(f"  PPO: n_steps={cfg.n_steps}, total_timesteps={cfg.total_timesteps}")
    print(f"  Eval: every {cfg.eval_freq} steps, {cfg.n_eval_episodes} eps")
    print(f"  TensorBoard: {tb_dir}")
    print("=" * 70)

    # -- Train --
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[sil_callback, eval_callback],
        log_interval=1,
        progress_bar=False,
    )

    # -- Save final model --
    final_path = os.path.join(cfg.output_dir, "final_model")
    model.save(final_path)
    print(f"\nPhase 2 finished. Model saved to {final_path}.zip")

    # -- Check convergence --
    if backward_wrapper.converged:
        print(f"  >> CONVERGED at start_idx=0, "
              f"success_rate={backward_wrapper.success_rate:.1%}")
    else:
        print(f"  Final backward start_idx={backward_wrapper.start_idx}, "
              f"success_rate={backward_wrapper.success_rate:.1%}")

    eval_env.close()
    train_env.close()


def _parse_args() -> RobustifyConfig:
    parser = argparse.ArgumentParser(description="Go-Explore Phase 2 (SB3)")
    cfg = RobustifyConfig()
    for f in fields(RobustifyConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default),
                            default=f.default)
    return RobustifyConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
