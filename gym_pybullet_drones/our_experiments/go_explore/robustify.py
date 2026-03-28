"""Go-Explore Phase 2 -- Robustification via backward algorithm + PPO + SIL.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.robustify \\
        --demo_path results/go_explore/demos_best.demo.pkl \\
        --total_timesteps 1000000

Implements the full Phase 2 mechanism:
  1. Backward Algorithm: start training from near the end of the demo
     trajectory, gradually shifting the starting point backward toward t=0.
  2. PPO: online policy gradient with clipped surrogate objective.
  3. SIL: self-imitation learning from demo + high-return online trajectories.
  4. Alternating resets: backward waypoint (odd eps) / random seed (even eps).

Adapted for single-agent OurSingleRLAviary.  Training loop, eval, and
TensorBoard logging are aligned with sb3rl/train.py for fair comparison.
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.go_explore.networks import (
    ActorCritic, ObsEncoder,
)
from gym_pybullet_drones.our_experiments.go_explore.ppo import PPOTrainer
from gym_pybullet_drones.our_experiments.go_explore.rollout_buffer import RolloutBuffer
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
    action_dim: int = 2

    # ── PPO (aligned with sb3rl/train.py) ────────────────────────────
    total_timesteps: int = 1_000_000
    n_steps: int = 2048
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 5
    batch_size: int = 256

    # ── model ────────────────────────────────────────────────────────
    obs_embed_dim: int = 128
    gru_hidden: int = 128
    num_gru_layers: int = 1

    # ── backward algorithm ───────────────────────────────────────────
    backward_step_size: int = 50     # how many demo steps to shift backward
    success_threshold: float = 0.8   # success rate to trigger backward shift
    eval_window: int = 20            # number of recent episodes for success rate
    max_backward_iters: int = 200    # max PPO updates per backward level
    success_captures: int = 18       # target captures needed for "success"

    # ── SIL ──────────────────────────────────────────────────────────
    sil_coef: float = 0.1
    sil_batch_size: int = 128
    sil_capacity: int = 50_000
    sil_online_threshold: float = 0.0  # min ep reward to add to SIL buffer

    # ── evaluation (aligned with sb3rl/train.py) ─────────────────────
    eval_freq: int = 10_000          # evaluate every N env steps
    n_eval_episodes: int = 3

    # ── demo / IO ────────────────────────────────────────────────────
    demo_path: str = "results/go_explore/demos_best.demo.pkl"
    log_interval: int = 100          # print every N PPO updates
    output_dir: str = "results/go_explore_phase2"
    seed: int = 42
    device: str = "cpu"


# ---------------------------------------------------------------------------
#  Env factory
# ---------------------------------------------------------------------------

def _make_env(cfg: RobustifyConfig, env_seed: int) -> OurSingleRLAviary:
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
        environment_seed=env_seed,
    )


# ---------------------------------------------------------------------------
#  Obs helpers
# ---------------------------------------------------------------------------

def _obs_keys_shapes(env: OurSingleRLAviary) -> Dict[str, tuple]:
    return {k: box.shape for k, box in env.observation_space.spaces.items()}


def _obs_to_batch(
    obs: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Single obs dict → batch-of-1 tensors."""
    return {
        k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        for k, v in obs.items()
    }


def _obs_to_np_batch(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Single obs dict → batch-of-1 numpy arrays."""
    return {k: np.expand_dims(np.asarray(v, dtype=np.float32), 0) for k, v in obs.items()}


# ---------------------------------------------------------------------------
#  Backward starting state
# ---------------------------------------------------------------------------

def _restore_to_waypoint(
    env: OurSingleRLAviary,
    demo: dict,
    start_idx: int,
) -> Dict[str, np.ndarray]:
    """Restore env to demo waypoint ``start_idx`` via action replay."""
    seed = demo.get("env_seed", None)
    obs, _ = env.reset(seed=seed)
    if start_idx <= 0:
        return obs
    idx = min(start_idx, len(demo["action_list"]))
    for action in demo["action_list"][:idx]:
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return obs


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    model: ActorCritic,
    cfg: RobustifyConfig,
    device: torch.device,
    n_episodes: int = 3,
) -> dict:
    """Run evaluation episodes with random seeds, return stats."""
    model.eval()
    ep_rewards, ep_lengths, ep_captures = [], [], []

    eval_env = _make_env(cfg, env_seed=cfg.seed + 9999)

    for ep in range(n_episodes):
        ep_seed = np.random.randint(0, 2**31)
        obs, _ = eval_env.reset(seed=ep_seed)
        hidden = model.initial_hidden(1).to(device)
        total_rew = 0.0
        steps = 0
        captures = 0

        while True:
            obs_batch = _obs_to_batch(obs, device)
            with torch.no_grad():
                action_t, _, _, hidden = model.get_action(obs_batch, hidden)
            action = action_t.cpu().numpy()[0]
            obs, rew, terminated, truncated, info = eval_env.step(action)
            total_rew += float(rew)
            steps += 1
            captures = int(info.get("target_capture_count", 0))
            if terminated or truncated:
                break

        ep_rewards.append(total_rew)
        ep_lengths.append(steps)
        ep_captures.append(captures)

    eval_env.close()
    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "mean_length": float(np.mean(ep_lengths)),
        "mean_captures": float(np.mean(ep_captures)),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def train(cfg: RobustifyConfig) -> None:
    device = torch.device(cfg.device)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # -- load demo --
    with open(cfg.demo_path, "rb") as f:
        demo = pickle.load(f)
    demo_n_steps = demo["n_steps"]
    print(f"Loaded demo: {demo_n_steps} steps, "
          f"total_reward={demo['total_reward']:.2f}")

    # -- environment --
    env = _make_env(cfg, env_seed=cfg.seed)

    # -- model --
    obs_shapes = _obs_keys_shapes(env)
    encoder = ObsEncoder(
        self_state_dim=obs_shapes.get("self_state", (6,))[-1],
        target_state_dim=obs_shapes.get("target_state", (54,))[-1],
        obstacle_state_dim=obs_shapes.get("obstacle_state", (24,))[-1],
        embed_dim=cfg.obs_embed_dim,
    )
    model = ActorCritic(
        encoder, action_dim=cfg.action_dim,
        gru_hidden=cfg.gru_hidden, num_gru_layers=cfg.num_gru_layers,
    ).to(device)

    ppo = PPOTrainer(
        model, lr=cfg.lr, clip_eps=cfg.clip_eps, vf_coef=cfg.vf_coef,
        entropy_coef=cfg.entropy_coef, max_grad_norm=cfg.max_grad_norm,
        n_epochs=cfg.n_epochs, batch_size=cfg.batch_size,
        sil_coef=cfg.sil_coef, sil_batch_size=cfg.sil_batch_size,
        device=cfg.device,
    )

    buffer = RolloutBuffer(
        n_steps=cfg.n_steps, n_agents=1,
        obs_keys_shapes=obs_shapes, action_dim=cfg.action_dim,
        gru_hidden=cfg.gru_hidden, num_gru_layers=cfg.num_gru_layers,
    )

    # -- SIL buffer: load demo --
    sil_buffer = SILBuffer(capacity=cfg.sil_capacity)
    sil_buffer.load_demo(
        demo["obs_list"], demo["action_list"], demo["returns"],
    )
    print(f"SIL buffer loaded with {len(sil_buffer)} demo transitions")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # -- TensorBoard --
    tb_dir = os.path.join("runs", "go_explore_phase2")
    writer = SummaryWriter(log_dir=tb_dir)

    # ================================================================
    #  Backward Algorithm State
    # ================================================================
    start_idx = max(0, demo_n_steps - cfg.backward_step_size)
    success_history: List[bool] = []
    episode_count = 0  # total episodes completed (for alternating reset)

    print("=" * 70)
    print("Go-Explore Phase 2 -- Robustification")
    print(f"  Backward algorithm: demo_steps={demo_n_steps}, "
          f"step_size={cfg.backward_step_size}")
    print(f"  PPO + SIL (sil_coef={cfg.sil_coef})")
    print(f"  n_steps={cfg.n_steps}  total_timesteps={cfg.total_timesteps}")
    print(f"  Eval: every {cfg.eval_freq} steps, {cfg.n_eval_episodes} eps")
    print(f"  TensorBoard: {tb_dir}")
    print("=" * 70)

    global_step = 0
    t_start = time.time()
    level_iters = 0        # PPO updates at current backward level
    best_eval_reward = -float("inf")
    next_eval_step = cfg.eval_freq

    # -- initial reset (backward waypoint) --
    obs = _restore_to_waypoint(env, demo, start_idx)
    hidden = model.initial_hidden(1).to(device)

    # -- episode tracking --
    ep_reward = 0.0
    ep_length = 0
    ep_captures = 0
    ep_obs_buf: List[Dict[str, np.ndarray]] = []
    ep_act_buf: List[np.ndarray] = []
    ep_rew_buf: List[float] = []

    update_count = 0

    while global_step < cfg.total_timesteps:
        buffer.reset()
        model.eval()

        # -- collect n_steps transitions --
        for step in range(cfg.n_steps):
            obs_batch = _obs_to_batch(obs, device)
            action_t, lp_t, val_t, hidden = model.get_action(obs_batch, hidden)

            action_np = action_t.cpu().numpy()[0]
            lp_np = lp_t.cpu().numpy()
            val_np = val_t.cpu().numpy()

            new_obs, rew, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            ep_captures = int(info.get("target_capture_count", 0))

            clipped_rew = float(np.clip(rew, -10.0, 10.0))
            ep_reward += float(rew)
            ep_length += 1

            # Track for SIL
            ep_obs_buf.append({k: np.array(v, copy=True) for k, v in new_obs.items()})
            ep_act_buf.append(np.array(action_np, copy=True))
            ep_rew_buf.append(float(rew))

            # Store in buffer (batch dim = 1)
            buf_obs = _obs_to_np_batch(obs)
            buffer.add(
                obs=buf_obs,
                action=action_np[np.newaxis],
                log_prob=lp_np,
                reward=np.array([clipped_rew], dtype=np.float32),
                value=val_np,
                done=np.array([float(done)], dtype=np.float32),
                gru_hidden=hidden.detach().cpu().numpy(),
            )

            global_step += 1
            obs = new_obs

            if done:
                # -- TensorBoard episode logging --
                writer.add_scalar("rollout/ep_rew_mean", ep_reward, global_step)
                writer.add_scalar("rollout/ep_len_mean", ep_length, global_step)
                writer.add_scalar("rollout/ep_captures", ep_captures, global_step)

                # -- SIL: add high-return episodes --
                if sum(ep_rew_buf) > cfg.sil_online_threshold:
                    sil_buffer.add_trajectory(
                        ep_obs_buf, ep_act_buf, ep_rew_buf, gamma=cfg.gamma,
                    )

                # -- backward success tracking --
                success_history.append(ep_captures >= cfg.success_captures)
                if len(success_history) > cfg.eval_window:
                    success_history = success_history[-cfg.eval_window:]

                # -- alternating reset --
                episode_count += 1
                if episode_count % 2 == 0:
                    # Even episode: random seed reset (generalization)
                    rand_seed = np.random.randint(0, 2**31)
                    obs, _ = env.reset(seed=rand_seed)
                else:
                    # Odd episode: backward waypoint (curriculum)
                    obs = _restore_to_waypoint(env, demo, start_idx)

                hidden = model.initial_hidden(1).to(device)
                ep_reward = 0.0
                ep_length = 0
                ep_captures = 0
                ep_obs_buf = []
                ep_act_buf = []
                ep_rew_buf = []

        # -- compute GAE returns --
        with torch.no_grad():
            last_obs_batch = _obs_to_batch(obs, device)
            _, _, last_val, _ = model.get_action(last_obs_batch, hidden)
        buffer.compute_returns(last_val.cpu().numpy(), cfg.gamma, cfg.gae_lambda)

        # -- PPO + SIL update --
        model.train()
        loss_info = ppo.update(buffer, sil_buffer=sil_buffer)
        update_count += 1
        level_iters += 1

        # -- TensorBoard training logs --
        writer.add_scalar("train/policy_loss", loss_info["policy_loss"], global_step)
        writer.add_scalar("train/value_loss", loss_info["value_loss"], global_step)
        writer.add_scalar("train/sil_loss", loss_info["sil_loss"], global_step)
        writer.add_scalar("backward/start_idx", start_idx, global_step)
        success_rate = float(np.mean(success_history)) if success_history else 0.0
        writer.add_scalar("backward/success_rate", success_rate, global_step)

        # -- backward shift check --
        shifted = False
        if level_iters >= cfg.eval_window:
            if (success_rate >= cfg.success_threshold
                    or level_iters >= cfg.max_backward_iters):
                old_idx = start_idx
                start_idx = max(0, start_idx - cfg.backward_step_size)
                if start_idx != old_idx:
                    shifted = True
                    level_iters = 0
                    success_history.clear()
                    reason = (f"success_rate={success_rate:.1%}"
                              if success_rate >= cfg.success_threshold
                              else f"max_iters={cfg.max_backward_iters}")
                    print(f"\n  >> BACKWARD SHIFT: start_idx {old_idx} -> {start_idx}"
                          f"  ({reason})")

        # -- console logging --
        if update_count % cfg.log_interval == 0 or update_count == 1 or shifted:
            elapsed = time.time() - t_start
            print(
                f"[step {global_step:>8d}/{cfg.total_timesteps}]  "
                f"updates={update_count:4d}  "
                f"start={start_idx:4d}/{demo_n_steps}  "
                f"succ={success_rate:.0%}  "
                f"p={loss_info['policy_loss']:.4f}  "
                f"v={loss_info['value_loss']:.4f}  "
                f"sil={loss_info['sil_loss']:.4f}  "
                f"sil_buf={len(sil_buffer)}  t={elapsed:.0f}s"
            )

        # -- evaluation --
        if global_step >= next_eval_step:
            eval_stats = _evaluate(model, cfg, device, cfg.n_eval_episodes)
            writer.add_scalar("eval/mean_reward", eval_stats["mean_reward"], global_step)
            writer.add_scalar("eval/mean_length", eval_stats["mean_length"], global_step)
            writer.add_scalar("eval/mean_captures", eval_stats["mean_captures"], global_step)

            print(f"  [EVAL step={global_step}]  "
                  f"reward={eval_stats['mean_reward']:+.2f} ± {eval_stats['std_reward']:.2f}  "
                  f"captures={eval_stats['mean_captures']:.1f}")

            # Save best model
            if eval_stats["mean_reward"] > best_eval_reward:
                best_eval_reward = eval_stats["mean_reward"]
                torch.save(model.state_dict(),
                           os.path.join(cfg.output_dir, "best_model.pt"))
                print(f"  -> new best model (reward={best_eval_reward:+.2f})")

            next_eval_step += cfg.eval_freq

        # -- early stop: backward fully converged --
        if start_idx == 0 and level_iters >= cfg.eval_window:
            if success_rate >= cfg.success_threshold:
                print(f"\n  >> CONVERGED at start_idx=0, "
                      f"success_rate={success_rate:.1%}")
                break

    # -- final save --
    torch.save(model.state_dict(),
               os.path.join(cfg.output_dir, "model_final.pt"))
    writer.close()
    env.close()
    print(f"\nPhase 2 finished. total_steps={global_step}, "
          f"final start_idx={start_idx}")


def _parse_args() -> RobustifyConfig:
    parser = argparse.ArgumentParser(description="Go-Explore Phase 2")
    cfg = RobustifyConfig()
    for f in fields(RobustifyConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default),
                            default=f.default)
    return RobustifyConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
