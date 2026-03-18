"""Go-Explore Phase 2 — Robustification via backward curriculum.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.robustify \\
        --archive_path results/go_explore/archive.json \\
        --total_iterations 3000 --n_envs 4

Adapted for PettingZoo ParallelEnv — per-agent observations and actions.
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

from gym_pybullet_drones.envs.OurRLAviary_PettingZoo import OurRLAviaryPZ
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.go_explore.archive import Archive
from gym_pybullet_drones.our_experiments.go_explore.networks import ActorCritic, ObsEncoder
from gym_pybullet_drones.our_experiments.go_explore.ppo import PPOTrainer
from gym_pybullet_drones.our_experiments.go_explore.rollout_buffer import RolloutBuffer


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RobustifyConfig:
    num_drones: int = 2
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
    curriculum_step: int = 5
    curriculum_interval: int = 50
    archive_path: str = "results/go_explore/archive.json"
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "results/go_explore_phase2"
    seed: int = 42
    device: str = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
#  Env factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_env(cfg: RobustifyConfig, env_seed: int) -> OurRLAviaryPZ:
    return OurRLAviaryPZ(
        num_drones=cfg.num_drones,
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


# ─────────────────────────────────────────────────────────────────────────────
#  Obs helpers — PettingZoo per-agent → batch tensor
# ─────────────────────────────────────────────────────────────────────────────

def _obs_keys_shapes(env: OurRLAviaryPZ) -> Dict[str, tuple]:
    """Per-agent feature shapes from PettingZoo obs space."""
    agent = env.possible_agents[0]
    space = env.observation_space(agent)
    return {k: box.shape for k, box in space.spaces.items()}


def _stack_agent_obs(
    pz_obs_list: List[Dict[str, Dict[str, np.ndarray]]],
    agents: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Stack per-env per-agent obs into (n_envs * n_agents, feat) tensors."""
    keys = list(pz_obs_list[0][agents[0]].keys())
    out = {}
    for key in keys:
        arrs = []
        for pz_obs in pz_obs_list:
            for agent in agents:
                arrs.append(np.asarray(pz_obs[agent][key], dtype=np.float32))
        out[key] = torch.as_tensor(np.stack(arrs), dtype=torch.float32, device=device)
    return out


def _stack_agent_obs_np(
    pz_obs_list: List[Dict[str, Dict[str, np.ndarray]]],
    agents: List[str],
) -> Dict[str, np.ndarray]:
    """Same as above, numpy."""
    keys = list(pz_obs_list[0][agents[0]].keys())
    out = {}
    for key in keys:
        arrs = []
        for pz_obs in pz_obs_list:
            for agent in agents:
                arrs.append(np.asarray(pz_obs[agent][key], dtype=np.float32))
        out[key] = np.stack(arrs)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Replay prefix
# ─────────────────────────────────────────────────────────────────────────────

def _replay_prefix(
    env: OurRLAviaryPZ,
    demo_actions: list,
    replay_len: int,
) -> tuple:
    """Reset env and replay ``replay_len`` demo actions.

    Returns (pz_observations, episode_ended).
    """
    observations, _ = env.reset()
    for i in range(min(replay_len, len(demo_actions))):
        stored = demo_actions[i]  # {agent: action}
        actions = {}
        for agent in env.agents:
            if agent in stored:
                actions[agent] = np.asarray(stored[agent], dtype=np.float32)
            else:
                actions[agent] = np.zeros(env.action_space(agent).shape, dtype=np.float32)

        observations, _, terminations, truncations, _ = env.step(actions)
        if any(terminations.values()) or any(truncations.values()) or not env.agents:
            observations, _ = env.reset()
            return observations, True
    return observations, False


def _best_trajectory(archive: Archive) -> list:
    best = None
    for cell in archive.cells.values():
        if best is None or cell.cumulative_reward > best.cumulative_reward:
            best = cell
    return best.action_sequence if best and best.action_sequence else []


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: RobustifyConfig) -> None:
    device = torch.device(cfg.device)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    num_drones = cfg.num_drones
    n_envs = cfg.n_envs
    n_batch = n_envs * num_drones

    # ── load archive ─────────────────────────────────────────────
    archive = Archive(arena_half=cfg.arena_size / 2.0)
    archive.load(cfg.archive_path)
    demo_actions = _best_trajectory(archive)
    demo_len = len(demo_actions)
    print(f"Loaded archive: {len(archive)} cells, best traj len: {demo_len}")

    if demo_len == 0:
        print("ERROR: no trajectory in archive. Run Phase 1 first.")
        return

    # ── environments ─────────────────────────────────────────────
    envs = [_make_env(cfg, cfg.seed + i) for i in range(n_envs)]
    agents = envs[0].possible_agents

    # ── model ────────────────────────────────────────────────────
    obs_shapes = _obs_keys_shapes(envs[0])
    encoder = ObsEncoder(
        self_state_dim=obs_shapes.get("self_state", (6,))[-1],
        action_history_dim=obs_shapes.get("action_history", (60,))[-1],
        teammate_state_dim=obs_shapes.get("teammate_state", (48,))[-1],
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
    replay_len = max(0, demo_len - cfg.curriculum_step)

    print("=" * 70)
    print("Go-Explore Phase 2 — Robustification (backward curriculum)")
    print(f"  demo_len={demo_len}  initial_replay={replay_len}  "
          f"n_envs={n_envs}  policy_steps={cfg.policy_steps}")
    print("=" * 70)

    global_step = 0
    t_start = time.time()

    for iteration in range(1, cfg.total_iterations + 1):
        buffer.reset()
        model.eval()

        if iteration > 1 and (iteration - 1) % cfg.curriculum_interval == 0:
            replay_len = max(0, replay_len - cfg.curriculum_step)

        # ── replay prefix for all envs ───────────────────────
        pz_obs_list: List[Dict[str, Dict[str, np.ndarray]]] = []
        for env in envs:
            obs, _ = _replay_prefix(env, demo_actions, replay_len)
            pz_obs_list.append(obs)

        hidden = model.initial_hidden(n_batch).to(device)
        ep_rewards = [0.0] * n_envs

        for step in range(cfg.policy_steps):
            obs_batch = _stack_agent_obs(pz_obs_list, agents, device)

            action_t, lp_t, val_t, hidden = model.get_action(obs_batch, hidden)

            actions_np = action_t.cpu().numpy()
            lp_np = lp_t.cpu().numpy()
            val_np = val_t.cpu().numpy()

            new_pz_obs_list = []
            rewards = np.zeros(n_batch, dtype=np.float32)
            dones = np.zeros(n_batch, dtype=np.float32)

            for e_idx, env in enumerate(envs):
                # split batch actions into per-agent dict
                pz_actions = {}
                for a_idx, agent in enumerate(agents):
                    flat_idx = e_idx * num_drones + a_idx
                    pz_actions[agent] = actions_np[flat_idx]

                new_obs, pz_rew, pz_term, pz_trunc, _ = env.step(pz_actions)
                done = any(pz_term.values()) or any(pz_trunc.values())

                for a_idx, agent in enumerate(agents):
                    flat_idx = e_idx * num_drones + a_idx
                    rewards[flat_idx] = pz_rew.get(agent, 0.0)
                    dones[flat_idx] = float(done)
                    ep_rewards[e_idx] += pz_rew.get(agent, 0.0)

                new_pz_obs_list.append(new_obs)

                if done or not env.agents:
                    rst_obs, _ = _replay_prefix(env, demo_actions, replay_len)
                    new_pz_obs_list[-1] = rst_obs
                    s = e_idx * num_drones
                    hidden[:, s:s + num_drones, :] = 0.0

            buf_obs = _stack_agent_obs_np(pz_obs_list, agents)
            buffer.add(
                obs=buf_obs, action=actions_np, log_prob=lp_np,
                reward=rewards, value=val_np, done=dones,
                gru_hidden=hidden.detach().cpu().numpy(),
            )
            pz_obs_list = new_pz_obs_list
            global_step += n_batch

        with torch.no_grad():
            last_obs = _stack_agent_obs(pz_obs_list, agents, device)
            _, _, last_val, _ = model.get_action(last_obs, hidden)
        buffer.compute_returns(last_val.cpu().numpy(), cfg.gamma, cfg.gae_lambda)

        model.train()
        loss_info = ppo.update(buffer)

        if iteration % cfg.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            mean_rew = np.mean(ep_rewards)
            print(
                f"[iter {iteration:5d}]  replay={replay_len:4d}/{demo_len}  "
                f"step={global_step:>9,}  mean_rew={mean_rew:+8.2f}  "
                f"p_loss={loss_info['policy_loss']:.4f}  "
                f"v_loss={loss_info['value_loss']:.4f}  "
                f"ent={loss_info['entropy']:.4f}  t={elapsed:.0f}s"
            )

        if iteration % cfg.save_interval == 0:
            ckpt = os.path.join(cfg.output_dir, f"model_iter{iteration}.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved {ckpt}")

    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "model_final.pt"))
    print(f"\n✓ Phase 2 finished.  replay_len={replay_len}")

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
