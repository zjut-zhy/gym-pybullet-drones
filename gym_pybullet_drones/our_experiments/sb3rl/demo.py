"""Load a trained best_model and run a visual demo in the GUI.

Supported algorithms: PPO, SAC, TD3, DDPG (all from Stable-Baselines3).

Usage:
    # Specify the folder that contains best_model.zip (auto-detect algorithm from folder name):
    python -m gym_pybullet_drones.our_experiments.sb3rl.demo --model_path results/ppo-03.23.2026_19.34.20

    # Explicitly specify algorithm and model file:
    python -m gym_pybullet_drones.our_experiments.sb3rl.demo --algo ppo --model_path results/ppo-03.23.2026_19.34.20/best_model.zip

    # Run multiple episodes:
    python -m gym_pybullet_drones.our_experiments.sb3rl.demo --model_path results/sac-03.23.2026_20.35.11 --n_episodes 5
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO, SAC, TD3, DDPG

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import str2bool, sync

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("vel")
DEFAULT_VISUALIZE_COVERAGE = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_N_EPISODES = 5

ALGO_MAP = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


def _guess_algo_from_path(path: str) -> str | None:
    """Try to infer the algorithm name from the model path or its parent folder."""
    basename = os.path.basename(os.path.normpath(path)).lower()
    parent = os.path.basename(os.path.dirname(os.path.normpath(path))).lower()
    for name in ALGO_MAP:
        if basename.startswith(name) or parent.startswith(name):
            return name
    return None


def _resolve_model_path(model_path: str) -> str:
    """If model_path is a directory, look for best_model.zip (then final_model.zip) inside it."""
    if os.path.isdir(model_path):
        best = os.path.join(model_path, "best_model.zip")
        final = os.path.join(model_path, "final_model.zip")
        if os.path.isfile(best):
            return best
        if os.path.isfile(final):
            print(f"[WARN] best_model.zip not found, falling back to final_model.zip")
            return final
        raise FileNotFoundError(
            f"No best_model.zip or final_model.zip found in {model_path}"
        )
    if os.path.isfile(model_path):
        return model_path
    raise FileNotFoundError(f"Model path does not exist: {model_path}")


# ── Main ─────────────────────────────────────────────────────────────
def run(
    algo: str | None = None,
    model_path: str = "",
    record_video: bool = DEFAULT_RECORD_VIDEO,
    visualize_coverage: bool = DEFAULT_VISUALIZE_COVERAGE,
    n_episodes: int = DEFAULT_N_EPISODES,
):
    # ── Resolve model path ────────────────────────────────────────
    model_file = _resolve_model_path(model_path)
    print(f"[INFO] Model file : {model_file}")

    # ── Determine algorithm ───────────────────────────────────────
    if algo is None:
        algo = _guess_algo_from_path(model_file)
    if algo is None:
        algo = _guess_algo_from_path(model_path)
    if algo is None:
        raise ValueError(
            "Cannot auto-detect algorithm from path. "
            "Please specify --algo explicitly (ppo / sac / td3 / ddpg)."
        )
    algo = algo.lower()
    if algo not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose from {list(ALGO_MAP)}")

    AlgoCls = ALGO_MAP[algo]
    print(f"[INFO] Algorithm  : {algo.upper()}")

    # ── Load model ────────────────────────────────────────────────
    model = AlgoCls.load(model_file)
    print(f"[INFO] Model loaded successfully.")

    # ── GUI demo ──────────────────────────────────────────────────
    print(f"\n[INFO] Launching GUI demo ({n_episodes} episode(s)) ...")
    test_env = OurSingleRLAviary(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        gui=True,
        record=record_video,
        visualize_coverage=visualize_coverage,
    )

    for ep in range(n_episodes):
        ep_seed = np.random.randint(0, 2**31)
        obs, info = test_env.reset(seed=ep_seed)
        start = time.time()
        ep_reward = 0.0
        step_count = 0
        done = False

        print(f"\n{'='*60}")
        print(f"  Episode {ep + 1} / {n_episodes}")
        print(f"{'='*60}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            ep_reward += reward
            step_count += 1
            dr = info.get("drone_reward", reward)
            captures = info.get("target_capture_count", 0)
            coverage = info.get("coverage_ratio", 0.0)

            if step_count % 30 == 0 or done:
                print(
                    f"  Step {step_count:04d}  reward={dr:+.4f}  "
                    f"coverage={coverage:.3f}  captures={captures}"
                )

            # test_env.render()
            sync(step_count - 1, start, test_env.CTRL_TIMESTEP)

        ep_captures = info.get("target_capture_count", 0)
        ep_coverage = info.get("coverage_ratio", 0.0)
        print(f"\n  Episode {ep + 1} finished: "
              f"total_reward={ep_reward:+.4f}  "
              f"steps={step_count}  "
              f"captures={ep_captures}  "
              f"coverage={ep_coverage:.3f}")

    test_env.close()
    print(f"\n[INFO] Demo complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Load a trained best_model and run a visual GUI demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m gym_pybullet_drones.our_experiments.sb3rl.demo --model_path results/ppo-03.23.2026_19.34.20
  python -m gym_pybullet_drones.our_experiments.sb3rl.demo --algo sac --model_path results/sac-03.23.2026_20.35.11/best_model.zip
  python -m gym_pybullet_drones.our_experiments.sb3rl.demo --model_path results/td3-03.24.2026_03.38.27 --n_episodes 5
""",
    )
    p.add_argument(
        "--model_path", type=str, required=True,
        help="Path to best_model.zip or the folder containing it",
    )
    p.add_argument(
        "--algo", type=str, default=None, choices=list(ALGO_MAP),
        help="RL algorithm (auto-detected from folder name if omitted)",
    )
    p.add_argument(
        "--n_episodes", type=int, default=DEFAULT_N_EPISODES,
        help=f"Number of GUI demo episodes (default: {DEFAULT_N_EPISODES})",
    )
    p.add_argument(
        "--record_video", type=str2bool, default=DEFAULT_RECORD_VIDEO,
        help="Record video during GUI demo",
    )
    p.add_argument(
        "--visualize_coverage", type=str2bool, default=DEFAULT_VISUALIZE_COVERAGE,
        help="Draw observation / threat / coverage rings in GUI",
    )
    args = p.parse_args()
    run(**vars(args))
