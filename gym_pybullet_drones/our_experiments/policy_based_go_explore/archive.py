"""Archive of discovered cells for Policy-Based Go-Explore.

Adapted for single-agent OurSingleRLAviary: observations are flat dicts
``{key: (feat,)}`` and rewards are scalar floats.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Cell:
    """One entry in the archive."""

    key: Tuple[int, int, int]                    # (gx, gy, n_captured)
    center_xy: np.ndarray                     # (2,) world-frame metres
    obs_snapshot: Optional[Dict[str, Any]]     # single-agent obs at arrival
    trajectory_cost: int = 0                   # steps needed to reach
    cumulative_reward: float = 0.0
    visit_count: int = 0
    score: float = 1.0

    def update_score(self) -> None:
        novelty = 1.0 / (1.0 + self.visit_count)
        reward_bonus = max(0.0, self.cumulative_reward)
        self.score = novelty + 0.1 * reward_bonus


class Archive:
    """Shared knowledge-base of discovered cells.

    Parameters
    ----------
    cell_size : float
        Side-length of each squared grid cell in metres.
    arena_half : float
        Half of the arena side-length; used for goal normalisation.
    max_cells : int
        Hard cap on the number of cells stored.
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

    def _xy_to_key(self, norm_xy: np.ndarray) -> Tuple[int, int]:
        world_xy = norm_xy * self.arena_half
        gx = int(math.floor(world_xy[0] / self.cell_size))
        gy = int(math.floor(world_xy[1] / self.cell_size))
        return (gx, gy)

    def obs_to_cell_key(self, obs: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
        """Cell key from a single-agent obs dict.

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

    def get_goal(self, cell: Cell) -> np.ndarray:
        """Return normalised 2-D goal for the network (in [-1, 1])."""
        return np.clip(cell.center_xy / self.arena_half, -1.0, 1.0).astype(np.float32)

    # -- core API --

    def update(
        self,
        trajectory_obs: List[Dict[str, np.ndarray]],
        trajectory_rewards: List[float],
    ) -> List[Cell]:
        """Insert / update cells from one trajectory.

        Parameters
        ----------
        trajectory_obs : list of obs_dict
            Single-agent obs at each step.
        trajectory_rewards : list of float
            Scalar reward per step.

        Returns the list of *newly created* cells.
        """
        new_cells: List[Cell] = []
        cum_reward = 0.0

        for step_idx, (obs, reward) in enumerate(
            zip(trajectory_obs, trajectory_rewards)
        ):
            cum_reward += reward
            key = self.obs_to_cell_key(obs)

            if key not in self.cells:
                if len(self.cells) >= self.max_cells:
                    continue
                cell = Cell(
                    key=key,
                    center_xy=self._cell_center(key),
                    obs_snapshot={k: np.array(v, copy=True) for k, v in obs.items()},
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
                    existing.obs_snapshot = {k: np.array(v, copy=True) for k, v in obs.items()}
                existing.update_score()
        return new_cells

    def select(self) -> Optional[Cell]:
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
        data = []
        for cell in self.cells.values():
            data.append({
                "key": list(cell.key),
                "center_xy": cell.center_xy.tolist(),
                "trajectory_cost": cell.trajectory_cost,
                "cumulative_reward": float(cell.cumulative_reward),
                "visit_count": cell.visit_count,
                "score": float(cell.score),
            })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"cell_size": self.cell_size, "cells": data}, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            raw = json.load(f)
        self.cell_size = raw.get("cell_size", self.cell_size)
        for entry in raw["cells"]:
            key = tuple(entry["key"])
            cell = Cell(
                key=key,
                center_xy=np.array(entry["center_xy"], dtype=np.float32),
                obs_snapshot=None,
                trajectory_cost=entry["trajectory_cost"],
                cumulative_reward=entry["cumulative_reward"],
                visit_count=entry["visit_count"],
                score=entry["score"],
            )
            self.cells[key] = cell

    def __len__(self) -> int:
        return len(self.cells)
