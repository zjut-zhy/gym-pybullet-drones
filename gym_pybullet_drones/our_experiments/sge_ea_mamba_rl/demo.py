"""SEAM-RL -- GUI demo: load trained EA-Mamba policy and visualise.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.demo \\
        --model_path results/sge_ea_mamba_rl/best_model.zip --n_episodes 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.ea_mamba_policy import (
    EAMambaActorCriticPolicy,
)


def run_demo(
    model_path: str,
    n_episodes: int = 3,
    playback_speed: float = 1.0,
) -> None:
    print(f"Loading model from {model_path} ...")
    custom_objects = {
        "policy_class": EAMambaActorCriticPolicy,
    }
    model = PPO.load(model_path, custom_objects=custom_objects)
    print("Model loaded successfully.")

    env = OurSingleRLAviary(
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        gui=True,
        record=False,
        visualize_coverage=True,
    )

    print(f"\nRunning {n_episodes} demo episodes ...")
    print("=" * 70)

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0
        start_time = time.time()

        fe = model.policy.features_extractor
        if hasattr(fe, "reset_hidden"):
            fe.reset_hidden(batch_size=1)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += float(reward)
            step_count += 1

            captures = info.get("target_capture_count", 0)
            coverage = info.get("coverage_ratio", 0.0)

            if step_count % 30 == 0:
                print(
                    f"  [Ep {ep}] Step {step_count:4d}  "
                    f"reward={ep_reward:+8.2f}  "
                    f"captures={captures}/{env.TARGET_COUNT}  "
                    f"coverage={coverage:.1%}"
                )

            env.render()
            sync(step_count, start_time,
                 env.CTRL_TIMESTEP / playback_speed)

        end_reason = "terminated" if terminated else "truncated"
        captures = info.get("target_capture_count", 0)
        coverage = info.get("coverage_ratio", 0.0)
        print(
            f"\n  Episode {ep} done ({end_reason}): "
            f"{step_count} steps, reward={ep_reward:+.2f}, "
            f"captures={captures}/{env.TARGET_COUNT}, "
            f"coverage={coverage:.1%}\n"
        )

    env.close()
    print("Demo finished.")


def main():
    parser = argparse.ArgumentParser(
        description="SEAM-RL demo: visualise trained EA-Mamba policy")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SB3 model (.zip)")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--playback_speed", type=float, default=1.0)
    args = parser.parse_args()
    run_demo(**vars(args))


if __name__ == "__main__":
    main()
