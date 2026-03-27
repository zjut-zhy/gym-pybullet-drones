"""Test environment determinism: same env, repeated reset(seed=0), no GUI.

Mirrors demo.py but without GUI — creates ONE environment, then calls
reset(seed=0) for each episode, recording per-step data and comparing
across episodes to check if results diverge.

Usage:
    python -m gym_pybullet_drones.our_experiments.sb3rl.test_determinism --model_path results/ppo-03.23.2026_19.34.20
"""

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO, SAC, TD3, DDPG

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

ALGO_MAP = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}


def _guess_algo(path: str):
    base = os.path.basename(os.path.normpath(path)).lower()
    parent = os.path.basename(os.path.dirname(os.path.normpath(path))).lower()
    for name in ALGO_MAP:
        if base.startswith(name) or parent.startswith(name):
            return name
    return None


def _resolve_model(path: str) -> str:
    if os.path.isdir(path):
        for name in ("best_model.zip", "final_model.zip"):
            f = os.path.join(path, name)
            if os.path.isfile(f):
                return f
        raise FileNotFoundError(f"No model zip found in {path}")
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(path)


def compare(r1, r2, idx1, idx2):
    """Compare two episode recordings and report divergence."""
    min_len = min(len(r1), len(r2))
    print(f"\n{'='*70}")
    print(f"  Episode {idx1} ({len(r1)} steps) vs Episode {idx2} ({len(r2)} steps)")
    print(f"{'='*70}")

    if len(r1) != len(r2):
        print(f"  [DIFF] Episode lengths differ: {len(r1)} vs {len(r2)}")

    first_diff_step = None
    max_action_diff = 0.0
    max_obs_diff = 0.0
    max_reward_diff = 0.0

    for i in range(min_len):
        a_diff = float(np.max(np.abs(r1[i]["action"] - r2[i]["action"])))
        o_diff = float(np.max(np.abs(r1[i]["obs_self"] - r2[i]["obs_self"])))
        rw_diff = abs(r1[i]["reward"] - r2[i]["reward"])

        max_action_diff = max(max_action_diff, a_diff)
        max_obs_diff = max(max_obs_diff, o_diff)
        max_reward_diff = max(max_reward_diff, rw_diff)

        if first_diff_step is None and (a_diff > 1e-9 or o_diff > 1e-9 or rw_diff > 1e-9):
            first_diff_step = i

    print(f"\n  Max action  diff: {max_action_diff:.2e}")
    print(f"  Max obs     diff: {max_obs_diff:.2e}")
    print(f"  Max reward  diff: {max_reward_diff:.2e}")

    r1_total = sum(r["reward"] for r in r1)
    r2_total = sum(r["reward"] for r in r2)
    r1_cap = r1[-1]["captures"] if r1 else 0
    r2_cap = r2[-1]["captures"] if r2 else 0
    print(f"\n  Ep{idx1} total reward: {r1_total:+.4f}  captures: {r1_cap}")
    print(f"  Ep{idx2} total reward: {r2_total:+.4f}  captures: {r2_cap}")

    if first_diff_step is None and len(r1) == len(r2):
        print(f"\n  [OK] IDENTICAL")
    else:
        print(f"\n  [!!] DIVERGED at step {first_diff_step}")
        if first_diff_step is not None and first_diff_step < min_len:
            s = first_diff_step
            print(f"     step {s}: action diff={np.max(np.abs(r1[s]['action']-r2[s]['action'])):.2e}, "
                  f"obs diff={np.max(np.abs(r1[s]['obs_self']-r2[s]['obs_self'])):.2e}, "
                  f"reward diff={abs(r1[s]['reward']-r2[s]['reward']):.2e}")

    print(f"{'='*70}")
    return first_diff_step is None and len(r1) == len(r2)


def main():
    p = argparse.ArgumentParser(description="Test env determinism (no GUI, repeated resets).")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--algo", type=str, default=None, choices=list(ALGO_MAP))
    p.add_argument("--seed", type=int, default=0, help="Reset seed (default: 0)")
    p.add_argument("--n_episodes", type=int, default=10, help="Number of episodes (default: 10)")
    args = p.parse_args()

    model_file = _resolve_model(args.model_path)
    algo = args.algo or _guess_algo(model_file) or _guess_algo(args.model_path)
    if algo is None:
        print("[ERROR] Cannot detect algorithm. Use --algo.")
        sys.exit(1)

    print(f"[INFO] Model      : {model_file}")
    print(f"[INFO] Algo       : {algo.upper()}")
    print(f"[INFO] Seed       : {args.seed}")
    print(f"[INFO] Episodes   : {args.n_episodes}")
    print(f"[INFO] GUI        : False")

    model = ALGO_MAP[algo].load(model_file)

    # ── Create ONE environment (no GUI), just like demo.py but gui=False ──
    test_env = OurSingleRLAviary(
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        gui=False,
        record=False,
        visualize_coverage=False,
    )

    all_records = []

    for ep in range(args.n_episodes):
        obs, info = test_env.reset(seed=args.seed)
        records = []
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            records.append({
                "step": step,
                "action": action.copy(),
                "reward": float(reward),
                "obs_self": obs["self_state"].copy(),
                "captures": info.get("target_capture_count", 0),
                "coverage": info.get("coverage_ratio", 0.0),
            })
            step += 1

        total_r = sum(r["reward"] for r in records)
        caps = records[-1]["captures"] if records else 0
        print(f"  Episode {ep+1:2d}: {len(records):4d} steps, "
              f"reward={total_r:+.4f}, captures={caps}")
        all_records.append(records)

    # ── Compare all episodes against episode 1 ──
    all_identical = True
    for i in range(1, len(all_records)):
        if not compare(all_records[0], all_records[i], 1, i + 1):
            all_identical = False

    print()
    if all_identical:
        print("[PASS] All episodes are bit-identical in DIRECT (no GUI) mode.")
        print("   If demo.py (GUI mode) gives different results, the cause is PyBullet's GUI renderer.")
    else:
        print("[FAIL] Episodes diverged even without GUI.")
        print("   Non-determinism is in the physics engine or env logic itself, not the GUI.")

    test_env.close()


if __name__ == "__main__":
    main()
