"""Archive for original Go-Explore -- stores environment snapshots.

Adapted for single-agent OurSingleRLAviary: observations are flat dicts
``{key: (feat,)}`` and actions are plain numpy arrays.

Instead of storing action sequences for replay (which breaks in dynamic
environments), each cell now stores a full environment *snapshot* that can
be restored with ``env.restore_snapshot(cell.snapshot)``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Cell:
    """One entry in the archive."""

    key: Tuple[int, int, int]                    # (gx, gy, n_captured)
    center_xy: np.ndarray                        # (2,) world-frame metres
    snapshot: Optional[dict] = None               # env.get_snapshot() result
    trajectory_cost: int = 0                      # steps to reach
    cumulative_reward: float = 0.0
    visit_count: int = 0
    score: float = 1.0

    def update_score(self) -> None:
        novelty = 1.0 / (1.0 + self.visit_count)
        reward_bonus = max(0.0, self.cumulative_reward)
        self.score = novelty + 0.1 * reward_bonus


class Archive:
    """Cell archive with snapshot-based return support.

    Parameters
    ----------
    cell_size : float
        Grid cell side-length in metres.
    arena_half : float
        Half of the arena side-length; used for coordinate un-normalisation.
    max_cells : int
        Maximum number of stored cells.
    """

    def __init__(
        self,
        cell_size: float = 0.5,
        arena_half: float = 5.0,
        max_cells: int = 10_000,
    ) -> None:
        self.cell_size = cell_size
        self.arena_half = arena_half
        self.max_cells = max_cells
        self.cells: Dict[Tuple[int, int, int], Cell] = {}
        self._rng = np.random.RandomState(0)

    def seed(self, s: int) -> None:
        self._rng = np.random.RandomState(s)

    # -- grid helpers --

    def _xy_to_key(self, norm_xy: np.ndarray) -> Tuple[int, int]:
        world_xy = norm_xy * self.arena_half
        gx = int(math.floor(world_xy[0] / self.cell_size))
        gy = int(math.floor(world_xy[1] / self.cell_size))
        return (gx, gy)

    def obs_to_cell_key(self, obs: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
        """Return cell key for a single-agent observation dict.

        Key = (grid_x, grid_y, n_captured_targets).
        The captured count is extracted from target_state: every 3rd element
        starting at index 2 is a captured flag (0 or 1).
        """
        self_state = np.asarray(obs["self_state"])
        gx, gy = self._xy_to_key(self_state[:2])
        target_state = np.asarray(obs["target_state"])
        n_captured = int(target_state[2::3].sum())
        return (gx, gy, n_captured)

    def _cell_center(self, key: Tuple[int, int, int]) -> np.ndarray:
        """Return world-frame XY center (ignoring the capture-count dimension)."""
        return np.array(
            [(key[0] + 0.5) * self.cell_size, (key[1] + 0.5) * self.cell_size],
            dtype=np.float32,
        )

    # -- update --

    def update(
        self,
        trajectory_obs: List[Dict[str, np.ndarray]],
        trajectory_snapshots: List[dict],
        trajectory_rewards: List[float],
    ) -> List[Cell]:
        """Process one trajectory and upsert cells.

        Parameters
        ----------
        trajectory_obs : list of obs_dict
            Single-agent obs at each step.
        trajectory_snapshots : list of dict
            ``env.get_snapshot()`` result at each step.
        trajectory_rewards : list of float
            Scalar reward at each step.

        Returns list of newly created cells.
        """
        new_cells: List[Cell] = []
        cum_reward = 0.0

        for step_idx, (obs, snapshot, reward) in enumerate(
            zip(trajectory_obs, trajectory_snapshots, trajectory_rewards)
        ):
            cum_reward += reward
            key = self.obs_to_cell_key(obs)

            if key not in self.cells:
                if len(self.cells) >= self.max_cells:
                    continue
                cell = Cell(
                    key=key,
                    center_xy=self._cell_center(key),
                    snapshot=copy.deepcopy(snapshot),
                    trajectory_cost=step_idx,
                    cumulative_reward=cum_reward,
                    visit_count=1,
                )
                cell.update_score()
                self.cells[key] = cell
                new_cells.append(cell)
            else:
                existing = self.cells[key]
                existing.visit_count += 1
                if step_idx < existing.trajectory_cost or cum_reward > existing.cumulative_reward:
                    existing.trajectory_cost = step_idx
                    existing.cumulative_reward = cum_reward
                    existing.snapshot = copy.deepcopy(snapshot)
                existing.update_score()
        return new_cells

    # -- selection --

    def select(self) -> Optional[Cell]:
        """Score-proportional sampling."""
        if not self.cells:
            return None
        keys = list(self.cells.keys())
        scores = np.array([self.cells[k].score for k in keys], dtype=np.float64)
        total = scores.sum()
        probs = scores / total if total > 0 else np.ones(len(keys)) / len(keys)
        idx = int(self._rng.choice(len(keys), p=probs))
        return self.cells[keys[idx]]

    # -- persistence --

    def save(self, path: str) -> None:
        """Save archive: JSON metadata + pickle for snapshots."""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        snapshot_dir = base.parent / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        meta = []
        for i, cell in enumerate(self.cells.values()):
            snap_file = str(snapshot_dir / f"cell_{i}.pkl")
            if cell.snapshot is not None:
                with open(snap_file, "wb") as f:
                    pickle.dump(cell.snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            meta.append({
                "key": list(cell.key),
                "center_xy": cell.center_xy.tolist(),
                "trajectory_cost": cell.trajectory_cost,
                "cumulative_reward": float(cell.cumulative_reward),
                "visit_count": cell.visit_count,
                "score": float(cell.score),
                "snapshot_file": snap_file if cell.snapshot is not None else None,
            })
        with open(str(base), "w") as f:
            json.dump({"cell_size": self.cell_size, "cells": meta}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            raw = json.load(f)
        self.cell_size = raw.get("cell_size", self.cell_size)
        for entry in raw["cells"]:
            key = tuple(entry["key"])
            snapshot = None
            snap_file = entry.get("snapshot_file")
            if snap_file and os.path.exists(snap_file):
                with open(snap_file, "rb") as f:
                    snapshot = pickle.load(f)
            cell = Cell(
                key=key,
                center_xy=np.array(entry["center_xy"], dtype=np.float32),
                snapshot=snapshot,
                trajectory_cost=entry["trajectory_cost"],
                cumulative_reward=entry["cumulative_reward"],
                visit_count=entry["visit_count"],
                score=entry["score"],
            )
            self.cells[key] = cell

    def __len__(self) -> int:
        return len(self.cells)
