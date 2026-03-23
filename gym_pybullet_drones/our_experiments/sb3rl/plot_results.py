"""Plot learning curves for the latest 4 SB3 training runs.

Reads `evaluations.npz` from the 4 most recent result folders in `results/`
and draws a comparison chart.

Usage:
    python -m gym_pybullet_drones.our_experiments.sb3rl.plot_results
    python -m gym_pybullet_drones.our_experiments.sb3rl.plot_results --results_dir results
    python -m gym_pybullet_drones.our_experiments.sb3rl.plot_results --n 2   # only latest 2
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Default paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # gym-pybullet-drones/
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def _parse_timestamp(folder_name: str) -> datetime | None:
    """Extract datetime from folder names like 'ppo-03.22.2026_13.25.20'."""
    m = re.search(r"(\d{2}\.\d{2}\.\d{4}_\d{2}\.\d{2}\.\d{2})$", folder_name)
    if m:
        return datetime.strptime(m.group(1), "%m.%d.%Y_%H.%M.%S")
    return None


def _extract_algo(folder_name: str) -> str:
    """Extract algorithm name from folder name, e.g. 'ppo-03...' -> 'PPO'."""
    return folder_name.split("-")[0].upper()


def load_latest_results(results_dir: Path, n: int = 4):
    """Return list of dicts with keys: algo, timesteps, mean, std, folder."""
    candidates = []
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        ts = _parse_timestamp(d.name)
        if ts is not None and (d / "evaluations.npz").exists():
            candidates.append((ts, d))

    # Sort by timestamp descending, take latest n
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:n]
    selected.reverse()  # chronological order (earliest first)

    runs = []
    for ts, folder in selected:
        data = np.load(folder / "evaluations.npz")
        timesteps = data["timesteps"]
        results = data["results"]  # shape: (n_evals, n_episodes)
        runs.append(
            dict(
                algo=_extract_algo(folder.name),
                timesteps=timesteps,
                mean=results.mean(axis=1),
                std=results.std(axis=1),
                folder=folder.name,
                time=ts,
            )
        )
    return runs


# ── Plotting ─────────────────────────────────────────────────────────
COLORS = {
    "PPO": "#2196F3",
    "SAC": "#FF5722",
    "TD3": "#4CAF50",
    "DDPG": "#9C27B0",
}


def plot_comparison(runs, save_path: Path | None = None):
    """Draw a comparison learning-curve chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for run in runs:
        color = COLORS.get(run["algo"], None)
        label = f"{run['algo']}  ({run['folder']})"
        ax.plot(
            run["timesteps"],
            run["mean"],
            label=label,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax.fill_between(
            run["timesteps"],
            run["mean"] - run["std"],
            run["mean"] + run["std"],
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Training Steps", fontsize=13)
    ax.set_ylabel("Mean Episode Reward", fontsize=13)
    ax.set_title("SB3 Algorithm Comparison — OurSingleRLAviary", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")

    plt.show()


def plot_individual(runs, save_dir: Path | None = None):
    """Draw one subplot per algorithm."""
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        color = COLORS.get(run["algo"], "#333")
        ax.plot(run["timesteps"], run["mean"], color=color, linewidth=2, marker="o", markersize=3)
        ax.fill_between(
            run["timesteps"],
            run["mean"] - run["std"],
            run["mean"] + run["std"],
            alpha=0.15,
            color=color,
        )
        ax.set_title(run["algo"], fontsize=14, fontweight="bold")
        ax.set_xlabel("Training Steps")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Episode Reward", fontsize=13)
    fig.suptitle("SB3 Individual Learning Curves", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_dir:
        path = save_dir / "individual_curves.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Figure saved to {path}")

    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot latest SB3 training results")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Path to results directory")
    parser.add_argument("--n", type=int, default=4, help="Number of latest runs to plot")
    parser.add_argument("--save", action="store_true", help="Save figures to results directory")
    parser.add_argument("--individual", action="store_true", help="Also plot individual subplots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return

    runs = load_latest_results(results_dir, n=args.n)
    if not runs:
        print("[ERROR] No valid result folders found.")
        return

    print(f"[INFO] Found {len(runs)} latest runs:")
    for r in runs:
        print(f"  • {r['algo']:5s}  {r['folder']}  ({len(r['timesteps'])} eval points)")

    save_path = results_dir / "comparison_curve.png" if args.save else None
    plot_comparison(runs, save_path=save_path)

    if args.individual:
        plot_individual(runs, save_dir=results_dir if args.save else None)


if __name__ == "__main__":
    main()
