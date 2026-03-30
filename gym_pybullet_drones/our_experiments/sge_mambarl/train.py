"""SGE-MambaRL Phase 1 -- SAC-guided Go-Explore.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.sge_mambarl.train \\
        --total_iterations 20000

Unlike the original Go-Explore (random actions), this version trains a
SAC policy online and uses it for the explore phase.  SAC's maximum-entropy
objective naturally encourages diverse exploration, producing higher-quality
trajectories for the archive.

The archive / cell / action-replay structure is identical to go_explore.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

# Reuse archive from go_explore (same data structure)
from gym_pybullet_drones.our_experiments.go_explore.archive import Archive, Cell
from gym_pybullet_drones.our_experiments.sge_mambarl.config import SGEConfig


# ---------------------------------------------------------------------------
#  Simple SAC explorer (lightweight, no SB3 dependency for Phase 1)
# ---------------------------------------------------------------------------

class _SACExplorer:
    """Lightweight SAC agent for Go-Explore exploration.

    Uses a small MLP policy to generate actions.  Trained online with a
    standard replay buffer.  The policy is used during the explore phase
    instead of random actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.distributions import Normal

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Actor (Gaussian policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        ).to(self.device)
        self.actor_mean = nn.Linear(256, action_dim).to(self.device)
        self.actor_log_std = nn.Linear(256, action_dim).to(self.device)

        # Twin Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)

        # Target Q-networks
        self.q1_target = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)
        self.q2_target = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Automatic entropy tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)

        # Optimizers
        actor_params = (list(self.actor.parameters())
                        + list(self.actor_mean.parameters())
                        + list(self.actor_log_std.parameters()))
        self.actor_opt = torch.optim.Adam(actor_params, lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer (simple)
        self._buf_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._buf_act = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self._buf_rew = np.zeros(buffer_size, dtype=np.float32)
        self._buf_next = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self._buf_done = np.zeros(buffer_size, dtype=np.float32)
        self._buf_size = buffer_size
        self._buf_ptr = 0
        self._buf_len = 0

        self._F = F
        self._Normal = Normal

    def _flatten_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten dict obs into a single vector."""
        parts = []
        for key in sorted(obs.keys()):
            parts.append(np.asarray(obs[key], dtype=np.float32).ravel())
        return np.concatenate(parts)

    def add_transition(
        self, obs: Dict[str, np.ndarray], action: np.ndarray,
        reward: float, next_obs: Dict[str, np.ndarray], done: bool,
    ) -> None:
        flat_obs = self._flatten_obs(obs)
        flat_next = self._flatten_obs(next_obs)
        idx = self._buf_ptr
        self._buf_obs[idx] = flat_obs
        self._buf_act[idx] = action
        self._buf_rew[idx] = reward
        self._buf_next[idx] = flat_next
        self._buf_done[idx] = float(done)
        self._buf_ptr = (self._buf_ptr + 1) % self._buf_size
        self._buf_len = min(self._buf_len + 1, self._buf_size)

    @torch.no_grad()
    def select_action(
        self, obs: Dict[str, np.ndarray], deterministic: bool = False,
    ) -> np.ndarray:
        flat = self._flatten_obs(obs)
        x = torch.as_tensor(flat, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        h = self.actor(x)
        mean = self.actor_mean(h)
        if deterministic:
            return torch.tanh(mean).squeeze(0).cpu().numpy()
        log_std = self.actor_log_std(h).clamp(-5.0, 2.0)
        std = log_std.exp()
        dist = self._Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        return action.squeeze(0).cpu().numpy()

    def train_step(self) -> Optional[dict]:
        """One SAC gradient step. Returns loss dict or None if not enough data."""
        if self._buf_len < self.batch_size:
            return None

        F = self._F
        Normal = self._Normal

        # Sample batch
        indices = np.random.choice(self._buf_len, self.batch_size, replace=False)
        obs = torch.as_tensor(self._buf_obs[indices], device=self.device)
        acts = torch.as_tensor(self._buf_act[indices], device=self.device)
        rews = torch.as_tensor(self._buf_rew[indices], device=self.device)
        next_obs = torch.as_tensor(self._buf_next[indices], device=self.device)
        dones = torch.as_tensor(self._buf_done[indices], device=self.device)

        alpha = self.log_alpha.exp().detach()

        # -- Q update --
        with torch.no_grad():
            h_next = self.actor(next_obs)
            mean_next = self.actor_mean(h_next)
            log_std_next = self.actor_log_std(h_next).clamp(-5.0, 2.0)
            dist_next = Normal(mean_next, log_std_next.exp())
            z_next = dist_next.rsample()
            a_next = torch.tanh(z_next)
            log_prob_next = (dist_next.log_prob(z_next)
                             - torch.log(1 - a_next.pow(2) + 1e-6))
            log_prob_next = log_prob_next.sum(dim=-1, keepdim=True)

            sa_next = torch.cat([next_obs, a_next], dim=-1)
            q1_targ = self.q1_target(sa_next)
            q2_targ = self.q2_target(sa_next)
            q_targ = torch.min(q1_targ, q2_targ) - alpha * log_prob_next
            td_target = rews.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * q_targ

        sa = torch.cat([obs, acts], dim=-1)
        q1_pred = self.q1(sa)
        q2_pred = self.q2(sa)
        q_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # -- Actor update --
        h = self.actor(obs)
        mean = self.actor_mean(h)
        log_std = self.actor_log_std(h).clamp(-5.0, 2.0)
        dist = Normal(mean, log_std.exp())
        z = dist.rsample()
        a = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        sa_new = torch.cat([obs, a], dim=-1)
        q1_new = self.q1(sa_new)
        q2_new = self.q2(sa_new)
        q_min = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -- Alpha update --
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # -- Soft target update --
        for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_targ.data.mul_(1 - self.tau).add_(p.data * self.tau)
        for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_targ.data.mul_(1 - self.tau).add_(p.data * self.tau)

        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
        }


# ---------------------------------------------------------------------------
#  Environment factory
# ---------------------------------------------------------------------------

def _make_env(cfg: SGEConfig) -> OurSingleRLAviary:
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
        environment_seed=cfg.seed,
    )


def _replay_to_cell(
    env: OurSingleRLAviary,
    cell: Cell,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Deterministic action replay to reach a cell."""
    obs, _ = env.reset(seed=seed)
    if cell.full_action_sequence:
        for action in cell.full_action_sequence:
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return obs


# ---------------------------------------------------------------------------
#  One iteration: select → restore → SAC explore
# ---------------------------------------------------------------------------

def _run_iteration(
    env: OurSingleRLAviary,
    archive: Archive,
    sac: _SACExplorer,
    cfg: SGEConfig,
    rng: np.random.RandomState,
    iteration: int = 0,
    total_sac_steps: int = 0,
) -> dict:
    act_dim = cfg.action_dim

    # -- select cell to explore from --
    if iteration <= cfg.warmup_iterations:
        target_cell = None
    else:
        target_cell = archive.select() if len(archive) > 0 else None

    # -- return phase (action replay) --
    source_cell = None
    if target_cell is not None and target_cell.full_action_sequence is not None:
        obs = _replay_to_cell(env, target_cell, seed=cfg.seed)
        source_cell = target_cell
    else:
        obs, info = env.reset(seed=cfg.seed)

    # -- explore phase: SAC policy instead of random --
    all_obs: List[Dict[str, np.ndarray]] = []
    all_rewards: List[float] = []
    all_actions: List[np.ndarray] = []
    all_n_captured: List[int] = []
    sac_train_count = 0
    prev_obs = obs

    for step in range(cfg.explore_steps):
        # Use SAC policy (or random during warmup)
        if total_sac_steps < cfg.sac_learning_starts:
            action = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        else:
            action = sac.select_action(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action)
        n_cap = int(info.get("target_capture_count", 0))

        # Add to SAC replay buffer
        sac.add_transition(prev_obs, action, reward, obs,
                           terminated or truncated)
        total_sac_steps += 1

        # SAC gradient step
        if total_sac_steps >= cfg.sac_learning_starts:
            for _ in range(cfg.sac_gradient_steps):
                sac.train_step()
                sac_train_count += 1

        if terminated or truncated:
            if n_cap >= cfg.target_count:
                all_obs.append({k: np.array(v, copy=True)
                                for k, v in obs.items()})
                all_rewards.append(float(reward))
                all_actions.append(np.array(action, copy=True))
                all_n_captured.append(n_cap)
            break

        all_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
        all_rewards.append(float(reward))
        all_actions.append(np.array(action, copy=True))
        all_n_captured.append(n_cap)
        prev_obs = obs

    # -- archive update --
    new_cells = archive.update(
        all_obs, all_rewards, all_actions,
        trajectory_n_captured=all_n_captured,
        source_cell=source_cell,
    )

    return {
        "new_cells": len(new_cells),
        "total_reward": sum(all_rewards),
        "return_restored": source_cell is not None,
        "sac_train_count": sac_train_count,
        "total_sac_steps": total_sac_steps,
    }


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def train(cfg: SGEConfig) -> None:
    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = _make_env(cfg)

    max_steps = int(cfg.ctrl_freq * cfg.max_episode_len_sec)

    archive = Archive(
        cell_size=cfg.cell_size,
        arena_half=cfg.arena_size / 2.0,
        max_cells=cfg.max_cells,
        env_seed=cfg.seed,
        max_steps=max_steps,
        target_count=cfg.target_count,
    )
    archive.seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # -- Compute obs dimension for SAC --
    test_obs, _ = env.reset(seed=cfg.seed)
    obs_dim = sum(np.asarray(v).ravel().shape[0] for v in test_obs.values())
    print(f"Obs dim (flattened): {obs_dim}, action dim: {cfg.action_dim}")

    # -- Create SAC explorer --
    sac = _SACExplorer(
        obs_dim=obs_dim,
        action_dim=cfg.action_dim,
        lr=cfg.sac_lr,
        buffer_size=cfg.sac_buffer_size,
        batch_size=cfg.sac_batch_size,
        tau=cfg.sac_tau,
        gamma=cfg.sac_gamma,
    )

    print("=" * 70)
    print("SGE Phase 1 -- SAC-Guided Go-Explore")
    print(f"  iters={cfg.total_iterations}  explore_steps={cfg.explore_steps}  "
          f"seed={cfg.seed}")
    print(f"  SAC: lr={cfg.sac_lr}, buffer={cfg.sac_buffer_size}, "
          f"learning_starts={cfg.sac_learning_starts}")
    print("=" * 70)

    t_start = time.time()
    total_new_cells = 0
    total_sac_steps = 0

    for iteration in range(1, cfg.total_iterations + 1):
        result = _run_iteration(
            env, archive, sac, cfg, rng,
            iteration=iteration,
            total_sac_steps=total_sac_steps,
        )
        total_new_cells += result["new_cells"]
        total_sac_steps = result["total_sac_steps"]

        if iteration % cfg.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            best = archive.get_best_cell()
            best_info = ""
            if best is not None:
                seq_len = (len(best.full_action_sequence)
                           if best.full_action_sequence else 0)
                best_info = (f"  best=({best.key[2]}cap, "
                             f"rew={best.cumulative_reward:+.1f}, "
                             f"seq={seq_len})")
            print(
                f"[iter {iteration:5d}]  "
                f"cells={len(archive):5d} (+{result['new_cells']:3d})  "
                f"total_discovered={total_new_cells:6d}  "
                f"rew={result['total_reward']:+8.2f}  "
                f"sac_steps={total_sac_steps:>8,}"
                f"{best_info}  elapsed={elapsed:.1f}s"
            )

        if iteration % cfg.save_interval == 0:
            archive.save(os.path.join(cfg.output_dir, "archive.json"))
            print(f"  -> saved archive ({len(archive)} cells)")

    # -- Save final --
    archive.save(os.path.join(cfg.output_dir, "archive.json"))
    best = archive.get_best_cell()
    if best is not None:
        seq_len = (len(best.full_action_sequence)
                   if best.full_action_sequence else 0)
        print(f"\nBest cell: key={best.key}, captures={best.key[2]}, "
              f"reward={best.cumulative_reward:.2f}, "
              f"action_sequence={seq_len} steps")
    print(f"SGE exploration finished. Total cells: {len(archive)}")

    # Save SAC model
    sac_path = os.path.join(cfg.output_dir, "sac_explorer.pt")
    torch.save({
        "actor": sac.actor.state_dict(),
        "actor_mean": sac.actor_mean.state_dict(),
        "actor_log_std": sac.actor_log_std.state_dict(),
        "q1": sac.q1.state_dict(),
        "q2": sac.q2.state_dict(),
    }, sac_path)
    print(f"SAC explorer saved to {sac_path}")

    env.close()


# --- CLI ---

def _parse_args() -> SGEConfig:
    parser = argparse.ArgumentParser(
        description="SGE Phase 1: SAC-Guided Go-Explore")
    cfg = SGEConfig()
    for f in fields(SGEConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default),
                            default=f.default)
    return SGEConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
