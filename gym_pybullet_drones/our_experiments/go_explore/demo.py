"""Go-Explore -- Demo: load trained Phase 2 model and run GUI episodes.

Supports both SB3 PPO models (.zip) and legacy custom models (.pt).

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.demo \\
        --model_path results/go_explore_phase2/best_model.zip \\
        --n_episodes 3

    # Legacy custom model:
    python -m gym_pybullet_drones.our_experiments.go_explore.demo \\
        --model_path results/go_explore_phase2/model_final.pt \\
        --n_episodes 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync


def _run_sb3(model_path: str, env: OurSingleRLAviary, n_episodes: int,
             target_count: int) -> None:
    """Run demo with SB3 PPO model."""
    model = PPO.load(model_path)
    print(f"[INFO] Loaded SB3 PPO model from {model_path}")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=np.random.randint(0, 2**31))
        ep_reward = 0.0
        step_count = 0
        done = False
        start = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(rew)
            step_count += 1
            sync(step_count, start, 1.0 / env.CTRL_FREQ)

        captures = int(info.get("target_capture_count", 0))
        print(f"[Episode {ep+1}/{n_episodes}]  "
              f"reward={ep_reward:+.2f}  steps={step_count}  "
              f"captures={captures}/{target_count}")


def _run_legacy(model_path: str, env: OurSingleRLAviary, n_episodes: int,
                target_count: int) -> None:
    """Run demo with legacy custom ActorCritic model (.pt)."""
    from gym_pybullet_drones.our_experiments.go_explore.networks import (
        ActorCritic, ObsEncoder,
    )

    device = torch.device("cpu")
    obs_shapes = {k: box.shape for k, box in env.observation_space.spaces.items()}
    encoder = ObsEncoder(
        self_state_dim=obs_shapes.get("self_state", (6,))[-1],
        target_state_dim=obs_shapes.get("target_state", (54,))[-1],
        obstacle_state_dim=obs_shapes.get("obstacle_state", (24,))[-1],
        embed_dim=128,
    )
    model = ActorCritic(encoder, action_dim=2, gru_hidden=128, num_gru_layers=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded legacy model from {model_path}")

    for ep in range(n_episodes):
        obs, info = env.reset(seed=np.random.randint(0, 2**31))
        hidden = model.initial_hidden(1).to(device)
        ep_reward = 0.0
        step_count = 0
        done = False
        start = time.time()

        while not done:
            obs_t = {k: torch.as_tensor(np.array(v, dtype=np.float32),
                                         device=device).unsqueeze(0)
                     for k, v in obs.items()}
            action, _, _, hidden = model.get_action(obs_t, hidden)
            action_np = action.squeeze(0).cpu().numpy()
            obs, rew, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            ep_reward += float(rew)
            step_count += 1
            sync(step_count, start, 1.0 / env.CTRL_FREQ)

        captures = int(info.get("target_capture_count", 0))
        print(f"[Episode {ep+1}/{n_episodes}]  "
              f"reward={ep_reward:+.2f}  steps={step_count}  "
              f"captures={captures}/{target_count}")


def run(
    model_path: str,
    n_episodes: int = 3,
    arena_size: float = 10.0,
    target_count: int = 18,
    obstacle_count: int = 6,
) -> None:
    env = OurSingleRLAviary(
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        gui=True,
        record=False,
        arena_size_xy_m=arena_size,
        target_count=target_count,
        obstacle_count=obstacle_count,
    )

    if model_path.endswith(".zip"):
        _run_sb3(model_path, env, n_episodes, target_count)
    elif model_path.endswith(".pt"):
        _run_legacy(model_path, env, n_episodes, target_count)
    else:
        # Try SB3 first, fall back to legacy
        try:
            _run_sb3(model_path, env, n_episodes, target_count)
        except Exception:
            _run_legacy(model_path + ".pt", env, n_episodes, target_count)

    env.close()
    print("\n[INFO] Demo finished.")


def main():
    parser = argparse.ArgumentParser(description="Go-Explore Demo")
    parser.add_argument("--model_path", type=str,
                        default="results/go_explore_phase2/best_model.zip")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--arena_size", type=float, default=10.0)
    parser.add_argument("--target_count", type=int, default=18)
    parser.add_argument("--obstacle_count", type=int, default=6)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
