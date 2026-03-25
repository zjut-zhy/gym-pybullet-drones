"""Go-Explore Phase 2 -- Robustification via backward algorithm + PPO + SIL.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.robustify \\
        --demo_path results/go_explore/best_demo.demo.pkl \\
        --total_iterations 3000 --n_envs 4

Implements the full Phase 2 mechanism from the original Go-Explore paper:
  1. Backward Algorithm: start training from near the end of the demo
     trajectory, gradually shifting the starting point backward toward t=0.
  2. PPO: online policy gradient with clipped surrogate objective.
  3. SIL: self-imitation learning from demo + high-return online trajectories.

Adapted for single-agent OurSingleRLAviary.
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
    n_envs: int = 4

    # ── PPO ──────────────────────────────────────────────────────────
    total_iterations: int = 3000
    policy_steps: int = 300
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 256

    # ── model ────────────────────────────────────────────────────────
    obs_embed_dim: int = 128
    gru_hidden: int = 128
    num_gru_layers: int = 1

    # ── backward algorithm ───────────────────────────────────────────
    backward_step_size: int = 50     # how many demo steps to shift backward
    success_threshold: float = 0.8   # success rate to trigger backward shift
    eval_window: int = 20            # number of recent episodes for success rate
    max_backward_iters: int = 500    # max iters per backward level
    success_captures: int = 18       # target captures needed for "success"

    # ── SIL ──────────────────────────────────────────────────────────
    sil_coef: float = 0.1
    sil_batch_size: int = 128
    sil_capacity: int = 50_000
    sil_online_threshold: float = 0.0  # min ep reward to add to SIL buffer

    # ── demo / IO ────────────────────────────────────────────────────
    demo_path: str = "results/go_explore/best_demo.demo.pkl"
    log_interval: int = 10
    save_interval: int = 100
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


def _stack_obs(
    obs_list: List[Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    keys = list(obs_list[0].keys())
    out = {}
    for key in keys:
        arrs = [np.asarray(obs[key], dtype=np.float32) for obs in obs_list]
        out[key] = torch.as_tensor(np.stack(arrs), dtype=torch.float32,
                                   device=device)
    return out


def _stack_obs_np(
    obs_list: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    keys = list(obs_list[0].keys())
    out = {}
    for key in keys:
        arrs = [np.asarray(obs[key], dtype=np.float32) for obs in obs_list]
        out[key] = np.stack(arrs)
    return out


# ---------------------------------------------------------------------------
#  Backward starting state
# ---------------------------------------------------------------------------

def _restore_to_waypoint(
    env: OurSingleRLAviary,
    demo: dict,
    start_idx: int,
) -> Dict[str, np.ndarray]:
    """Restore env to demo waypoint ``start_idx``.

    If start_idx == 0, does a fresh reset instead.
    """
    if start_idx <= 0:
        obs, _ = env.reset()
        return obs
    # Clamp to valid range
    idx = min(start_idx, len(demo["snapshot_list"]) - 1)
    env.restore_snapshot(demo["snapshot_list"][idx])
    obs = env._computeObs()
    return obs


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def train(cfg: RobustifyConfig) -> None:
    device = torch.device(cfg.device)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    n_envs = cfg.n_envs
    n_batch = n_envs  # single agent per env

    # -- load demo --
    with open(cfg.demo_path, "rb") as f:
        demo = pickle.load(f)
    demo_n_steps = demo["n_steps"]
    print(f"Loaded demo: {demo_n_steps} steps, "
          f"total_reward={demo['total_reward']:.2f}")

    # -- environments --
    envs = [_make_env(cfg, cfg.seed + i) for i in range(n_envs)]

    # -- model --
    obs_shapes = _obs_keys_shapes(envs[0])
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
        n_steps=cfg.policy_steps, n_agents=n_batch,
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

    # ================================================================
    #  Backward Algorithm
    # ================================================================
    # Start near the end of the demo trajectory and work backward
    start_idx = max(0, demo_n_steps - cfg.backward_step_size)
    success_history: List[bool] = []  # sliding window for backward scheduling

    print("=" * 70)
    print("Go-Explore Phase 2 -- Robustification")
    print(f"  Backward algorithm: demo_steps={demo_n_steps}, "
          f"step_size={cfg.backward_step_size}")
    print(f"  PPO + SIL (sil_coef={cfg.sil_coef})")
    print(f"  n_envs={n_envs}  policy_steps={cfg.policy_steps}")
    print("=" * 70)

    global_step = 0
    t_start = time.time()
    level_iters = 0

    for iteration in range(1, cfg.total_iterations + 1):
        buffer.reset()
        model.eval()

        # -- restore all envs to the current backward waypoint --
        obs_list: List[Dict[str, np.ndarray]] = []
        for env in envs:
            obs = _restore_to_waypoint(env, demo, start_idx)
            obs_list.append(obs)

        hidden = model.initial_hidden(n_batch).to(device)
        ep_rewards = [0.0] * n_envs
        ep_captures = [0] * n_envs      # track captures per env

        # -- collect per-env trajectories for potential SIL insertion --
        ep_obs_buf: List[List[Dict[str, np.ndarray]]] = [[] for _ in range(n_envs)]
        ep_act_buf: List[List[np.ndarray]] = [[] for _ in range(n_envs)]
        ep_rew_buf: List[List[float]] = [[] for _ in range(n_envs)]

        for step in range(cfg.policy_steps):
            obs_batch = _stack_obs(obs_list, device)

            action_t, lp_t, val_t, hidden = model.get_action(obs_batch, hidden)

            actions_np = action_t.cpu().numpy()
            lp_np = lp_t.cpu().numpy()
            val_np = val_t.cpu().numpy()

            new_obs_list = []
            rewards = np.zeros(n_batch, dtype=np.float32)
            dones = np.zeros(n_batch, dtype=np.float32)

            for e_idx, env in enumerate(envs):
                action = actions_np[e_idx]
                obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_captures[e_idx] = int(info.get("target_capture_count", 0))

                rewards[e_idx] = float(np.clip(rew, -10.0, 10.0))
                dones[e_idx] = float(done)
                ep_rewards[e_idx] += float(rew)

                # Track for SIL
                ep_obs_buf[e_idx].append(
                    {k: np.array(v, copy=True) for k, v in obs.items()})
                ep_act_buf[e_idx].append(np.array(action, copy=True))
                ep_rew_buf[e_idx].append(float(rew))

                new_obs_list.append(obs)

                if done:
                    # Add high-return episodes to SIL buffer
                    ep_total = sum(ep_rew_buf[e_idx])
                    if ep_total > cfg.sil_online_threshold:
                        sil_buffer.add_trajectory(
                            ep_obs_buf[e_idx], ep_act_buf[e_idx],
                            ep_rew_buf[e_idx], gamma=cfg.gamma,
                        )
                    # Reset trajectory tracking
                    ep_obs_buf[e_idx] = []
                    ep_act_buf[e_idx] = []
                    ep_rew_buf[e_idx] = []

                    # Restore to current waypoint again
                    rst_obs = _restore_to_waypoint(env, demo, start_idx)
                    new_obs_list[-1] = rst_obs
                    hidden[:, e_idx:e_idx + 1, :] = 0.0

            buf_obs = _stack_obs_np(obs_list)
            buffer.add(
                obs=buf_obs, action=actions_np, log_prob=lp_np,
                reward=rewards, value=val_np, done=dones,
                gru_hidden=hidden.detach().cpu().numpy(),
            )
            obs_list = new_obs_list
            global_step += n_batch

        with torch.no_grad():
            last_obs = _stack_obs(obs_list, device)
            _, _, last_val, _ = model.get_action(last_obs, hidden)
        buffer.compute_returns(last_val.cpu().numpy(), cfg.gamma, cfg.gae_lambda)

        model.train()
        loss_info = ppo.update(buffer, sil_buffer=sil_buffer)

        # -- track success for backward scheduling --
        mean_rew = np.mean(ep_rewards)
        # Success = all targets captured (per env)
        for e_idx in range(n_envs):
            success_history.append(ep_captures[e_idx] >= cfg.success_captures)
        if len(success_history) > cfg.eval_window * n_envs:
            success_history = success_history[-(cfg.eval_window * n_envs):]

        level_iters += 1

        # -- check backward shift condition --
        shifted = False
        if level_iters >= cfg.eval_window:
            success_rate = np.mean(success_history) if success_history else 0.0
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

        # -- logging --
        if iteration % cfg.log_interval == 0 or iteration == 1 or shifted:
            elapsed = time.time() - t_start
            success_rate = np.mean(success_history) if success_history else 0.0
            max_cap = max(ep_captures)
            print(
                f"[iter {iteration:5d}]  "
                f"start={start_idx:4d}/{demo_n_steps}  "
                f"succ={success_rate:.0%}  "
                f"cap={max_cap}/{cfg.success_captures}  "
                f"rew={mean_rew:+8.2f}  "
                f"p={loss_info['policy_loss']:.4f}  "
                f"v={loss_info['value_loss']:.4f}  "
                f"sil={loss_info['sil_loss']:.4f}  "
                f"sil_buf={len(sil_buffer)}  t={elapsed:.0f}s"
            )

        if iteration % cfg.save_interval == 0:
            ckpt = os.path.join(cfg.output_dir, f"model_iter{iteration}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")

        # Early stop: reached the beginning and converged
        if start_idx == 0 and level_iters >= cfg.eval_window:
            final_success_rate = np.mean(success_history) if success_history else 0.0
            if final_success_rate >= cfg.success_threshold:
                print(f"\n  >> CONVERGED at start_idx=0, "
                      f"success_rate={final_success_rate:.1%}")
                break

    torch.save(model.state_dict(),
               os.path.join(cfg.output_dir, "model_final.pt"))
    print(f"\nPhase 2 finished. Final start_idx={start_idx}")

    for env in envs:
        env.close()


def _parse_args() -> RobustifyConfig:
    parser = argparse.ArgumentParser(description="Go-Explore Phase 2")
    cfg = RobustifyConfig()
    for f in fields(RobustifyConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default),
                            default=f.default)
    return RobustifyConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
