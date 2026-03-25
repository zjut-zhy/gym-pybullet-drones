"""Go-Explore Phase 2 -- Robustification via snapshot-based curriculum.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.robustify \\
        --archive_path results/go_explore/archive.json \\
        --total_iterations 3000 --n_envs 4

Adapted for single-agent OurSingleRLAviary -- flat obs / action / reward.
Uses environment snapshots (instead of action replay) to set starting states.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.go_explore.archive import Archive, Cell
from gym_pybullet_drones.our_experiments.go_explore.networks import ActorCritic, ObsEncoder
from gym_pybullet_drones.our_experiments.go_explore.ppo import PPOTrainer
from gym_pybullet_drones.our_experiments.go_explore.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

@dataclass
class RobustifyConfig:
    arena_size: float = 10.0
    target_count: int = 18
    obstacle_count: int = 6
    ctrl_freq: int = 30
    max_episode_len_sec: float = 60.0
    action_dim: int = 2
    n_envs: int = 4
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
    obs_embed_dim: int = 128
    gru_hidden: int = 128
    num_gru_layers: int = 1
    curriculum_interval: int = 50
    archive_path: str = "results/go_explore/archive.json"
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
#  Obs helpers -- stack flat obs across envs
# ---------------------------------------------------------------------------

def _obs_keys_shapes(env: OurSingleRLAviary) -> Dict[str, tuple]:
    """Feature shapes from single-agent obs space."""
    return {k: box.shape for k, box in env.observation_space.spaces.items()}


def _stack_obs(
    obs_list: List[Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Stack per-env obs into (n_envs, feat) tensors."""
    keys = list(obs_list[0].keys())
    out = {}
    for key in keys:
        arrs = [np.asarray(obs[key], dtype=np.float32) for obs in obs_list]
        out[key] = torch.as_tensor(np.stack(arrs), dtype=torch.float32, device=device)
    return out


def _stack_obs_np(
    obs_list: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Same as above, numpy."""
    keys = list(obs_list[0].keys())
    out = {}
    for key in keys:
        arrs = [np.asarray(obs[key], dtype=np.float32) for obs in obs_list]
        out[key] = np.stack(arrs)
    return out


# ---------------------------------------------------------------------------
#  Snapshot-based starting state
# ---------------------------------------------------------------------------

def _restore_from_archive(
    env: OurSingleRLAviary,
    archive: Archive,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """Restore a snapshot from the archive, or fresh-reset if unavailable.

    Returns the observation after restoration.
    """
    cell = archive.select()
    if cell is not None and cell.snapshot is not None:
        # Restore WITHOUT calling reset() first -- reset() rebuilds
        # the PyBullet world and invalidates all saved state IDs.
        env.restore_snapshot(cell.snapshot)
        obs = env._computeObs()
    else:
        obs, _ = env.reset()
    return obs


def _best_cell(archive: Archive) -> Optional[Cell]:
    """Return the cell with the highest cumulative reward."""
    best = None
    for cell in archive.cells.values():
        if best is None or cell.cumulative_reward > best.cumulative_reward:
            best = cell
    return best


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def train(cfg: RobustifyConfig) -> None:
    device = torch.device(cfg.device)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    n_envs = cfg.n_envs
    n_batch = n_envs  # single agent per env

    # -- load archive ---
    archive = Archive(arena_half=cfg.arena_size / 2.0)
    archive.load(cfg.archive_path)
    archive.seed(cfg.seed)
    best = _best_cell(archive)
    has_snapshots = any(c.snapshot is not None for c in archive.cells.values())
    print(f"Loaded archive: {len(archive)} cells, "
          f"best reward: {best.cumulative_reward:.2f}" if best else "no cells",
          f"  snapshots available: {has_snapshots}")

    if len(archive) == 0:
        print("ERROR: empty archive. Run Phase 1 first.")
        return

    # -- environments ---
    envs = [_make_env(cfg, cfg.seed + i) for i in range(n_envs)]

    # -- model ---
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
        n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, device=cfg.device,
    )

    buffer = RolloutBuffer(
        n_steps=cfg.policy_steps, n_agents=n_batch,
        obs_keys_shapes=obs_shapes, action_dim=cfg.action_dim,
        gru_hidden=cfg.gru_hidden, num_gru_layers=cfg.num_gru_layers,
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(cfg.seed)

    print("=" * 70)
    print("Go-Explore Phase 2 -- Robustification (snapshot-based curriculum)")
    print(f"  archive_cells={len(archive)}  "
          f"n_envs={n_envs}  policy_steps={cfg.policy_steps}")
    print("=" * 70)

    global_step = 0
    t_start = time.time()

    for iteration in range(1, cfg.total_iterations + 1):
        buffer.reset()
        model.eval()

        # -- restore from archive snapshot for all envs ---
        obs_list: List[Dict[str, np.ndarray]] = []
        for env in envs:
            obs = _restore_from_archive(env, archive, rng)
            obs_list.append(obs)

        hidden = model.initial_hidden(n_batch).to(device)
        ep_rewards = [0.0] * n_envs

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
                obs, rew, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rewards[e_idx] = float(np.clip(rew, -10.0, 10.0))
                dones[e_idx] = float(done)
                ep_rewards[e_idx] += float(rew)

                new_obs_list.append(obs)

                if done:
                    rst_obs = _restore_from_archive(env, archive, rng)
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
        loss_info = ppo.update(buffer)

        if iteration % cfg.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            mean_rew = np.mean(ep_rewards)
            print(
                f"[iter {iteration:5d}]  "
                f"step={global_step:>9,}  mean_rew={mean_rew:+8.2f}  "
                f"p_loss={loss_info['policy_loss']:.4f}  "
                f"v_loss={loss_info['value_loss']:.4f}  "
                f"ent={loss_info['entropy']:.4f}  t={elapsed:.0f}s"
            )

        if iteration % cfg.save_interval == 0:
            ckpt = os.path.join(cfg.output_dir, f"model_iter{iteration}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> saved {ckpt}")

    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "model_final.pt"))
    print(f"\nPhase 2 finished.")

    for env in envs:
        env.close()


def _parse_args() -> RobustifyConfig:
    parser = argparse.ArgumentParser(description="Go-Explore Phase 2")
    cfg = RobustifyConfig()
    for f in fields(RobustifyConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return RobustifyConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
