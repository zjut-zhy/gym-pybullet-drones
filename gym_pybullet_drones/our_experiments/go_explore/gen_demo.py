"""Go-Explore Bridge -- replay successful cells' action sequences to gen demos.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo \\
        --archive_path results/go_explore/archive.json

Loads ALL successful cells (n_captured >= target_count) from the archive,
replays each in a deterministic environment to collect per-step data, and
saves them as .demo.pkl files for Phase 2.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

from gym_pybullet_drones.our_experiments.go_explore.archive import Archive, Cell
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


def _replay_cell(env: OurSingleRLAviary, cell: Cell, env_seed: int) -> dict:
    """Replay one cell's action sequence and collect per-step data."""
    actions = cell.full_action_sequence
    obs, _ = env.reset(seed=env_seed)

    demo_obs, demo_actions, demo_rewards = [], [], []
    demo_n_captured = []

    for i, action in enumerate(actions):
        obs, rew, terminated, truncated, info = env.step(action)

        demo_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
        demo_actions.append(np.array(action, copy=True))
        demo_rewards.append(float(rew))
        demo_n_captured.append(int(info.get("target_capture_count", 0)))

        if terminated or truncated:
            if i + 1 < len(actions):
                print(f"  WARNING: replay terminated at step {i+1}/{len(actions)}")
            break

    # -- MC returns --
    gamma = 0.99
    rewards_arr = np.array(demo_rewards, dtype=np.float64)
    returns = np.zeros_like(rewards_arr)
    running = 0.0
    for t in reversed(range(len(rewards_arr))):
        running = rewards_arr[t] + gamma * running
        returns[t] = running

    return {
        "obs_list": demo_obs,
        "action_list": demo_actions,
        "reward_list": demo_rewards,
        "n_captured_list": demo_n_captured,
        "returns": returns.tolist(),
        "total_reward": sum(demo_rewards),
        "n_steps": len(demo_obs),
        "max_captured": max(demo_n_captured) if demo_n_captured else 0,
        "env_seed": env_seed,
        "cell_key": list(cell.key),
    }


def gen_demo(archive_path: str, output_path: str, cfg: GoExploreConfig) -> None:
    archive = Archive()
    archive.load(archive_path)
    print(f"Loaded archive: {len(archive)} cells, env_seed={archive.env_seed}")

    # Get all successful cells
    successful = archive.get_successful_cells(cfg.target_count)
    if not successful:
        # Fallback: use best cell even if not fully successful
        best = archive.get_best_cell()
        if best is None:
            print("ERROR: empty archive.")
            return
        successful = [best]
        print(f"No fully successful cells found. Using best cell: "
              f"key={best.key}, captures={best.key[2]}")
    else:
        print(f"Found {len(successful)} successful cells "
              f"(n_captured >= {cfg.target_count})")

    env = _make_env(cfg, env_seed=archive.env_seed)
    all_demos = []

    for idx, cell in enumerate(successful):
        actions = cell.full_action_sequence
        if actions is None or len(actions) == 0:
            print(f"  [{idx}] Skipping cell {cell.key}: no action sequence")
            continue

        print(f"  [{idx}] Replaying cell key={cell.key}, "
              f"captures={cell.key[2]}, seq={len(actions)} actions ...")
        demo = _replay_cell(env, cell, archive.env_seed)
        all_demos.append(demo)

        print(f"       -> steps={demo['n_steps']}, "
              f"captures={demo['max_captured']}, "
              f"reward={demo['total_reward']:.2f}, "
              f"return(t=0)={demo['returns'][0]:.2f}")

    env.close()

    if not all_demos:
        print("ERROR: no valid demos generated.")
        return

    # Save individual demos + combined multi-demo file
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined file with all demos
    combined = {
        "demos": all_demos,
        "n_demos": len(all_demos),
        "env_seed": archive.env_seed,
        "target_count": cfg.target_count,
    }
    with open(output_path, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also save the best single demo for backward compatibility
    best_demo_path = str(Path(output_path).with_suffix("")) + "_best.demo.pkl"
    with open(best_demo_path, "wb") as f:
        pickle.dump(all_demos[0], f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n{'='*50}")
    print(f"Saved {len(all_demos)} demos to {output_path}")
    print(f"Best single demo saved to {best_demo_path}")
    print(f"  Best: steps={all_demos[0]['n_steps']}, "
          f"captures={all_demos[0]['max_captured']}, "
          f"reward={all_demos[0]['total_reward']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Go-Explore: gen demos from successful cells")
    parser.add_argument("--archive_path", type=str,
                        default="results/go_explore/archive.json")
    parser.add_argument("--output_path", type=str,
                        default="results/go_explore/demos.pkl")
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

