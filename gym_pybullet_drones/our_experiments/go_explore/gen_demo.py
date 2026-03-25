"""Go-Explore Bridge Layer -- tree-trace + deterministic replay demo.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo \\
        --archive_path results/go_explore/archive.json

1. Load the Phase 1 archive.
2. Find the best cell (most captures, then highest reward).
3. Trace the parent-pointer tree back to root → reconstruct the full
   action sequence.
4. Replay that action sequence in a *deterministic* copy of the Phase 1
   environment (same seed) to collect ``(obs, action, reward, snapshot)``
   at every step.
5. Save the result as ``.demo.pkl`` for Phase 2.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.go_explore.archive import Archive
from gym_pybullet_drones.our_experiments.go_explore.config import GoExploreConfig


def _make_env(cfg: GoExploreConfig, env_seed: int) -> OurSingleRLAviary:
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


def gen_demo(
    archive_path: str,
    output_path: str,
    cfg: GoExploreConfig,
) -> None:
    # -- load archive --
    archive = Archive()
    archive.load(archive_path)
    print(f"Loaded archive: {len(archive)} cells, env_seed={archive.env_seed}")

    # -- find best cell (most captures → highest reward) --
    best = archive.get_best_cell()
    if best is None:
        print("ERROR: empty archive.")
        return
    print(f"Best cell: key={best.key}, captures={best.key[2]}, "
          f"reward={best.cumulative_reward:.2f}, "
          f"cost={best.trajectory_cost}")

    # -- tree-trace: reconstruct full action sequence --
    full_actions = archive.reconstruct_trajectory(best.key)
    print(f"Reconstructed trajectory: {len(full_actions)} actions "
          f"(should match cost={best.trajectory_cost})")

    if len(full_actions) == 0:
        print("ERROR: empty trajectory (root cell with no actions).")
        return

    # -- deterministic replay --
    print(f"Replaying in deterministic env (seed={archive.env_seed}) ...")
    env = _make_env(cfg, env_seed=archive.env_seed)
    obs, _ = env.reset()

    demo_obs = []
    demo_actions = []
    demo_rewards = []
    demo_snapshots = []
    demo_n_captured = []

    for i, action in enumerate(full_actions):
        obs, rew, terminated, truncated, info = env.step(action)

        demo_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
        demo_actions.append(np.array(action, copy=True))
        demo_rewards.append(float(rew))
        demo_snapshots.append(env.get_snapshot())
        demo_n_captured.append(int(info.get("target_capture_count", 0)))

        if terminated or truncated:
            print(f"  Episode ended at step {i+1} "
                  f"({'terminated' if terminated else 'truncated'})")
            break

    env.close()

    # -- compute MC returns --
    gamma = 0.99
    rewards_arr = np.array(demo_rewards, dtype=np.float64)
    returns = np.zeros_like(rewards_arr)
    running = 0.0
    for t in reversed(range(len(rewards_arr))):
        running = rewards_arr[t] + gamma * running
        returns[t] = running

    total_reward = sum(demo_rewards)

    # -- save demo --
    demo = {
        "obs_list": demo_obs,
        "action_list": demo_actions,
        "reward_list": demo_rewards,
        "snapshot_list": demo_snapshots,
        "n_captured_list": demo_n_captured,
        "returns": returns.tolist(),
        "total_reward": total_reward,
        "n_steps": len(demo_obs),
        "env_seed": archive.env_seed,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(demo, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nDemo saved to {output_path}")
    print(f"  Steps       : {demo['n_steps']}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Return(t=0) : {returns[0]:.2f}")
    print(f"  Final caps  : {demo_n_captured[-1] if demo_n_captured else 0}")


def main():
    parser = argparse.ArgumentParser(
        description="Go-Explore: extract demo via tree-trace + replay")
    parser.add_argument("--archive_path", type=str,
                        default="results/go_explore/archive.json")
    parser.add_argument("--output_path", type=str,
                        default="results/go_explore/best_demo.demo.pkl")
    # Environment params (must match Phase 1)
    parser.add_argument("--arena_size", type=float, default=10.0)
    parser.add_argument("--target_count", type=int, default=18)
    parser.add_argument("--obstacle_count", type=int, default=6)
    parser.add_argument("--ctrl_freq", type=int, default=30)
    parser.add_argument("--max_episode_len_sec", type=float, default=60.0)
    args = parser.parse_args()

    cfg = GoExploreConfig(
        arena_size=args.arena_size,
        target_count=args.target_count,
        obstacle_count=args.obstacle_count,
        ctrl_freq=args.ctrl_freq,
        max_episode_len_sec=args.max_episode_len_sec,
    )
    gen_demo(args.archive_path, args.output_path, cfg)


if __name__ == "__main__":
    main()
