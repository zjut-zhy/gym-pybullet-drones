"""Replay successful cells from a saved Go-Explore archive in GUI.

Loads the archive, finds cells with n_captured >= target_count,
and replays their action sequences in a PyBullet GUI environment.
Uses sync() for frame-accurate wall-clock playback.

Usage
-----
    python -m gym_pybullet_drones.our_experiments.go_explore.demo_archive \
        --archive_path results/go_explore/archive.json \
        --n_demos 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.our_experiments.go_explore.archive import Archive


def run(
    archive_path: str = "results/go_explore/archive.json",
    n_demos: int = 3,
    arena_size: float = 10.0,
    target_count: int = 18,
    obstacle_count: int = 6,
    seed: int = 42,
    playback_speed: float = 1.0,
) -> None:
    # ── load archive ──
    archive = Archive()
    archive.load(archive_path)
    print(f"[INFO] Loaded archive: {len(archive)} cells")

    # ── find successful cells ──
    successful = archive.get_successful_cells(target_count)
    if not successful:
        print(f"[WARN] No cells with {target_count} captures found.")
        print("       Showing best available cells instead.")
        # Fallback: show cells sorted by (n_captured desc, reward desc)
        successful = sorted(
            archive.cells.values(),
            key=lambda c: (c.key[2], c.cumulative_reward),
            reverse=True,
        )

    n_demos = min(n_demos, len(successful))
    print(f"[INFO] Replaying top {n_demos} cell(s)  "
          f"(best captures={successful[0].key[2]})\n")

    # ── create GUI environment ──
    env = OurSingleRLAviary(
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        gui=True,
        record=False,
        arena_size_xy_m=arena_size,
        target_count=target_count,
        obstacle_count=obstacle_count,
        environment_seed=seed,
    )

    ctrl_timestep = 1.0 / env.CTRL_FREQ / playback_speed

    for i, cell in enumerate(successful[:n_demos]):
        seq = cell.full_action_sequence
        if not seq:
            print(f"[Cell {i+1}] No action sequence stored, skipping.")
            continue

        print(f"[Cell {i+1}/{n_demos}]  key={cell.key}  "
              f"captures={cell.key[2]}  reward={cell.cumulative_reward:+.1f}  "
              f"steps={len(seq)}")

        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        start = time.time()

        for step_idx, action in enumerate(seq):
            obs, rew, terminated, truncated, info = env.step(action)
            ep_reward += float(rew)
            sync(step_idx, start, ctrl_timestep)

            if terminated or truncated:
                break

        captures = int(info.get("target_capture_count", 0))
        print(f"         -> replayed {step_idx+1}/{len(seq)} steps  "
              f"reward={ep_reward:+.2f}  captures={captures}/{target_count}")

        # Pause between episodes
        if i < n_demos - 1:
            print("         (pausing 2s before next replay...)")
            time.sleep(2.0)

    env.close()
    print("\n[INFO] Archive demo finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Replay successful Go-Explore archive cells in GUI")
    parser.add_argument("--archive_path", type=str,
                        default="results/go_explore/archive.json")
    parser.add_argument("--n_demos", type=int, default=3,
                        help="Number of successful cells to replay")
    parser.add_argument("--arena_size", type=float, default=10.0)
    parser.add_argument("--target_count", type=int, default=18)
    parser.add_argument("--obstacle_count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--playback_speed", type=float, default=1.0,
                        help="Playback speed multiplier (e.g. 2.0 = 2x faster)")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
