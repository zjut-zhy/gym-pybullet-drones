"""Policy-Based Go-Explore training loop — PettingZoo version.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.policy_based_go_explore.train \\
        --num_drones 2 --total_iterations 500 --n_envs 4

With PettingZoo, each drone is already a named agent with per-agent obs
``{key: (feat,)}``. We stack per-agent observations across all envs into a
flat batch ``(n_envs * n_drones, feat)`` for the shared policy network.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from gym_pybullet_drones.envs.OurRLAviary_PettingZoo import OurRLAviaryPZ
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.policy_based_go_explore.archive import Archive
from gym_pybullet_drones.our_experiments.policy_based_go_explore.config import GoExploreConfig
from gym_pybullet_drones.our_experiments.policy_based_go_explore.goal_conditioned_env import GoExploreEnvWrapper
from gym_pybullet_drones.our_experiments.policy_based_go_explore.networks import (
    GoalConditionedActorCritic,
    ObsEncoder,
)
from gym_pybullet_drones.our_experiments.policy_based_go_explore.ppo import PPOTrainer
from gym_pybullet_drones.our_experiments.policy_based_go_explore.rollout_buffer import GoExploreRolloutBuffer


# ─────────────────────────────────────────────────────────────────────────────
#  Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_env(cfg: GoExploreConfig, env_seed: int) -> GoExploreEnvWrapper:
    base_env = OurRLAviaryPZ(
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
    return GoExploreEnvWrapper(
        env=base_env,
        return_max_steps=cfg.return_max_steps,
        explore_max_steps=cfg.explore_max_steps,
        tracker_kwargs=dict(
            sub_goal_spacing=cfg.sub_goal_spacing,
            reach_threshold=cfg.sub_goal_reach_thresh,
            sub_goal_reward=cfg.sub_goal_reward,
            potential_scale=cfg.potential_reward_scale,
            arena_half=cfg.arena_size / 2.0,
        ),
        arena_half=cfg.arena_size / 2.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Obs helpers — PettingZoo per-agent → batch tensors
# ─────────────────────────────────────────────────────────────────────────────

def _obs_keys_shapes(env: GoExploreEnvWrapper) -> Dict[str, tuple]:
    """Per-agent feature shapes (PettingZoo obs space)."""
    agent = env.possible_agents[0]
    space = env.observation_space(agent)
    return {k: box.shape for k, box in space.spaces.items()}


def _stack_obs(
    pz_obs_list: List[Dict[str, Dict[str, np.ndarray]]],
    agents: List[str],
    device: torch.device,
    exclude: tuple = (),
) -> Dict[str, torch.Tensor]:
    """Stack per-env per-agent obs into (n_envs*n_agents, feat) tensors."""
    first_agent_obs = pz_obs_list[0][agents[0]]
    keys = [k for k in first_agent_obs if k not in exclude]
    out = {}
    for key in keys:
        arrs = []
        for pz_obs in pz_obs_list:
            for agent in agents:
                arrs.append(np.asarray(pz_obs[agent][key], dtype=np.float32))
        out[key] = torch.as_tensor(np.stack(arrs), dtype=torch.float32, device=device)
    return out


def _stack_obs_np(
    pz_obs_list: List[Dict[str, Dict[str, np.ndarray]]],
    agents: List[str],
    exclude: tuple = (),
) -> Dict[str, np.ndarray]:
    first_agent_obs = pz_obs_list[0][agents[0]]
    keys = [k for k in first_agent_obs if k not in exclude]
    out = {}
    for key in keys:
        arrs = []
        for pz_obs in pz_obs_list:
            for agent in agents:
                arrs.append(np.asarray(pz_obs[agent][key], dtype=np.float32))
        out[key] = np.stack(arrs)
    return out


def _stack_field(pz_obs_list, agents, key, device):
    arrs = []
    for pz_obs in pz_obs_list:
        for agent in agents:
            arrs.append(np.asarray(pz_obs[agent][key], dtype=np.float32))
    return torch.as_tensor(np.stack(arrs), dtype=torch.float32, device=device)


def _stack_field_np(pz_obs_list, agents, key):
    arrs = []
    for pz_obs in pz_obs_list:
        for agent in agents:
            arrs.append(np.asarray(pz_obs[agent][key], dtype=np.float32))
    return np.stack(arrs)


# ─────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: GoExploreConfig) -> None:
    device = torch.device(cfg.device)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    num_drones = cfg.num_drones
    n_envs = cfg.n_envs
    n_batch = n_envs * num_drones

    # ── environments ─────────────────────────────────────────────
    envs = [_make_env(cfg, env_seed=cfg.seed + i) for i in range(n_envs)]
    agents = envs[0].possible_agents  # ["drone_0", "drone_1", ...]

    # ── archive ──────────────────────────────────────────────────
    archive = Archive(
        cell_size=cfg.cell_size,
        arena_half=cfg.arena_size / 2.0,
        max_cells=cfg.max_cells,
    )
    archive.seed(cfg.seed)
    for env in envs:
        env.set_archive(archive)

    # ── model ────────────────────────────────────────────────────
    raw_obs_shapes = _obs_keys_shapes(envs[0])
    # obs shapes for the network (exclude goal/phase which are separate)
    net_obs_shapes = {k: v for k, v in raw_obs_shapes.items()
                      if k not in ("goal", "phase")}
    self_state_dim = net_obs_shapes.get("self_state", (6,))[-1]
    action_history_dim = net_obs_shapes.get("action_history", (60,))[-1]
    teammate_state_dim = net_obs_shapes.get("teammate_state", (48,))[-1]
    target_state_dim = net_obs_shapes.get("target_state", (54,))[-1]
    obstacle_state_dim = net_obs_shapes.get("obstacle_state", (24,))[-1]

    obs_encoder = ObsEncoder(
        self_state_dim=self_state_dim,
        action_history_dim=action_history_dim,
        teammate_state_dim=teammate_state_dim,
        target_state_dim=target_state_dim,
        obstacle_state_dim=obstacle_state_dim,
        embed_dim=cfg.obs_embed_dim,
    )
    model = GoalConditionedActorCritic(
        obs_encoder=obs_encoder,
        goal_dim=cfg.goal_dim,
        action_dim=cfg.action_dim,
        gru_hidden=cfg.gru_hidden,
        num_gru_layers=cfg.num_gru_layers,
        goal_embed_dim=cfg.goal_embed_dim,
    ).to(device)

    ppo = PPOTrainer(model, cfg)
    n_steps = cfg.return_max_steps + cfg.explore_max_steps

    buffer_obs_shapes = {k: v for k, v in net_obs_shapes.items()}
    buffer = GoExploreRolloutBuffer(
        n_steps=n_steps,
        n_agents=n_batch,
        obs_keys_shapes=buffer_obs_shapes,
        goal_dim=cfg.goal_dim,
        action_dim=cfg.action_dim,
        gru_hidden=cfg.gru_hidden,
        num_gru_layers=cfg.num_gru_layers,
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Policy-Based Go-Explore  |  OurRLAviaryPZ")
    print(f"  n_envs={n_envs}  n_drones={num_drones}  batch={n_batch}  "
          f"n_steps={n_steps}  iters={cfg.total_iterations}")
    print("=" * 70)

    global_step = 0
    t_start = time.time()

    for iteration in range(1, cfg.total_iterations + 1):
        buffer.reset()
        model.eval()

        # reset all envs — each returns {agent: obs_dict}
        pz_obs_list: List[Dict[str, Dict[str, np.ndarray]]] = []
        for env in envs:
            obs, _ = env.reset()
            pz_obs_list.append(obs)

        hidden = model.initial_hidden(batch_size=n_batch).to(device)

        traj_obs: List[List[Dict[str, Dict[str, np.ndarray]]]] = [[] for _ in range(n_envs)]
        traj_rewards: List[List[Dict[str, float]]] = [[] for _ in range(n_envs)]

        for step in range(n_steps):
            # stack per-agent obs across envs → (n_batch, feat)
            obs_batch = _stack_obs(pz_obs_list, agents, device, exclude=("goal", "phase"))
            goal_batch = _stack_field(pz_obs_list, agents, "goal", device)
            phase_batch = _stack_field(pz_obs_list, agents, "phase", device)

            action_t, log_prob_t, value_t, hidden = model.get_action(
                obs_batch, goal_batch, phase_batch, hidden,
            )

            actions_np = action_t.cpu().numpy()
            log_probs_np = log_prob_t.cpu().numpy()
            values_np = value_t.cpu().numpy()

            new_pz_obs_list = []
            rewards_flat = np.zeros(n_batch, dtype=np.float32)
            dones_flat = np.zeros(n_batch, dtype=np.float32)
            valids_flat = np.ones(n_batch, dtype=np.float32)

            for e_idx, env in enumerate(envs):
                # convert batch actions → per-agent dict
                pz_actions = {}
                for a_idx, agent in enumerate(agents):
                    flat_idx = e_idx * num_drones + a_idx
                    pz_actions[agent] = actions_np[flat_idx]

                new_obs, pz_rew, pz_term, pz_trunc, pz_info = env.step(pz_actions)
                done = any(pz_term.values()) or any(pz_trunc.values())

                for a_idx, agent in enumerate(agents):
                    flat_idx = e_idx * num_drones + a_idx
                    rewards_flat[flat_idx] = pz_rew.get(agent, 0.0)
                    dones_flat[flat_idx] = float(done)
                    agent_info = pz_info.get(agent, {})
                    valids_flat[flat_idx] = float(agent_info.get("valid_mask", True))

                new_pz_obs_list.append(new_obs)

                # record for archive (per-agent obs without goal/phase)
                step_obs = {}
                for agent in new_obs:
                    step_obs[agent] = {k: np.array(v, copy=True)
                                       for k, v in new_obs[agent].items()
                                       if k not in ("goal", "phase")}
                traj_obs[e_idx].append(step_obs)
                traj_rewards[e_idx].append(dict(pz_rew))

                if done or not env.agents:
                    reset_obs, _ = env.reset()
                    new_pz_obs_list[-1] = reset_obs
                    s = e_idx * num_drones
                    hidden[:, s:s + num_drones, :] = 0.0

            # store in buffer
            buf_obs = _stack_obs_np(pz_obs_list, agents, exclude=("goal", "phase"))
            buf_goal = _stack_field_np(pz_obs_list, agents, "goal")
            buf_phase = _stack_field_np(pz_obs_list, agents, "phase")

            buffer.add(
                obs=buf_obs, goal=buf_goal, phase=buf_phase,
                action=actions_np, log_prob=log_probs_np,
                reward=rewards_flat, value=values_np,
                done=dones_flat, valid_mask=valids_flat,
                gru_hidden=hidden.detach().cpu().numpy(),
            )

            pz_obs_list = new_pz_obs_list
            global_step += n_batch

        # bootstrap
        with torch.no_grad():
            last_obs = _stack_obs(pz_obs_list, agents, device, exclude=("goal", "phase"))
            last_g = _stack_field(pz_obs_list, agents, "goal", device)
            last_p = _stack_field(pz_obs_list, agents, "phase", device)
            _, _, last_val, _ = model.get_action(last_obs, last_g, last_p, hidden)
        buffer.compute_returns(last_val.cpu().numpy(), cfg.gamma, cfg.gae_lambda)

        model.train()
        loss_info = ppo.update(buffer)

        # archive update
        new_cells_total = 0
        for e_idx in range(n_envs):
            new_cells_total += len(archive.update(traj_obs[e_idx], traj_rewards[e_idx]))

        if iteration % cfg.log_interval == 0 or iteration == 1:
            elapsed = time.time() - t_start
            fps = global_step / max(elapsed, 1e-6)
            total_rew = sum(
                sum(sum(r.values()) for r in tr) for tr in traj_rewards
            )
            mean_rew = total_rew / n_envs
            print(
                f"[iter {iteration:5d}]  step={global_step:>9,}  "
                f"cells={len(archive):5d} (+{new_cells_total:3d})  "
                f"mean_rew={mean_rew:+8.2f}  "
                f"p_loss={loss_info['policy_loss']:.4f}  "
                f"v_loss={loss_info['value_loss']:.4f}  "
                f"ent={loss_info['entropy']:.4f}  FPS={fps:.0f}"
            )

        if iteration % cfg.save_interval == 0:
            ckpt = os.path.join(cfg.output_dir, f"model_iter{iteration}.pt")
            torch.save(model.state_dict(), ckpt)
            archive.save(os.path.join(cfg.output_dir, "archive.json"))
            print(f"  → saved {ckpt}")

    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "model_final.pt"))
    archive.save(os.path.join(cfg.output_dir, "archive.json"))
    print(f"\n✓ Training finished. Cells: {len(archive)}")

    for env in envs:
        env.close()


def _parse_args() -> GoExploreConfig:
    parser = argparse.ArgumentParser(description="Policy-Based Go-Explore (PettingZoo)")
    cfg = GoExploreConfig()
    for f in fields(GoExploreConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return GoExploreConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(_parse_args())
