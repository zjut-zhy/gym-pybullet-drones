"""Archive for Go-Explore Phase 1 -- tree-structured cell database.

Each cell stores a **parent pointer** and the **action segment** from
parent to self.  To reconstruct the full trajectory to any cell, trace
the parent chain back to the root and concatenate the action segments
in reverse order -- the "linked-list tree tracing" approach.

The archive also stores the environment seed so that ``gen_demo`` can
replay the reconstructed action sequence in an identical deterministic
environment.
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


# -----------------------------------------------------------------------
#  Cell dataclass
# -----------------------------------------------------------------------

@dataclass
class Cell:
    """One node in the exploration tree.

    Attributes
    ----------
    key : (gx, gy, n_captured)
        Unique cell identifier.
    center_xy : np.ndarray
        Cell centre in world coordinates.
    snapshot : dict or None
        ``env.get_snapshot()`` for restoring during Phase 1 exploration.
    trajectory_cost : int
        Total steps from root to reach this cell (accumulated across the
        parent chain).
    cumulative_reward : float
        Total reward from root to this cell.
    visit_count, score : int, float
        Cell visitation statistics for exploration sampling.
    parent_key : tuple or None
        Key of the cell we explored *from* to discover this cell.
        ``None`` means this cell was reached from a fresh ``env.reset()``.
    actions_from_parent : list[np.ndarray] or None
        The action sequence executed from the parent cell's snapshot
        (or from reset) that led to this cell.  Together with the parent
        chain, these segments concatenate into the full trajectory.
    """

    key: Tuple[int, int, int]
    center_xy: np.ndarray
    snapshot: Optional[dict] = None
    trajectory_cost: int = 0
    cumulative_reward: float = 0.0
    visit_count: int = 0
    score: float = 1.0

    # ── tree structure ──
    parent_key: Optional[Tuple[int, int, int]] = None
    actions_from_parent: Optional[List[np.ndarray]] = None

    def update_score(self) -> None:
        novelty = 1.0 / (1.0 + self.visit_count)
        reward_bonus = max(0.0, self.cumulative_reward)
        self.score = novelty + 0.1 * reward_bonus


# -----------------------------------------------------------------------
#  Archive
# -----------------------------------------------------------------------

class Archive:
    """Cell archive with snapshot-based return + tree-structured
    trajectory reconstruction.

    Parameters
    ----------
    cell_size : float
        Grid cell side-length in metres.
    arena_half : float
        Half-arena side-length; used for coordinate un-normalisation.
    max_cells : int
        Hard cap on the number of stored cells.
    env_seed : int
        Environment seed used in Phase 1 (stored for gen_demo replay).
    """

    def __init__(
        self,
        cell_size: float = 0.5,
        arena_half: float = 5.0,
        max_cells: int = 10_000,
        env_seed: int = 42,
    ) -> None:
        self.cell_size = cell_size
        self.arena_half = arena_half
        self.max_cells = max_cells
        self.env_seed = env_seed
        self.cells: Dict[Tuple[int, int, int], Cell] = {}
        self._rng = np.random.RandomState(0)

    def seed(self, s: int) -> None:
        self._rng = np.random.RandomState(s)

    # ── grid helpers ──

    def _xy_to_key(self, norm_xy: np.ndarray) -> Tuple[int, int]:
        world_xy = norm_xy * self.arena_half
        gx = int(math.floor(world_xy[0] / self.cell_size))
        gy = int(math.floor(world_xy[1] / self.cell_size))
        return (gx, gy)

    def obs_to_cell_key(
        self, obs: Dict[str, np.ndarray], n_captured: int = 0,
    ) -> Tuple[int, int, int]:
        self_state = np.asarray(obs["self_state"])
        gx, gy = self._xy_to_key(self_state[:2])
        return (gx, gy, int(n_captured))

    def _cell_center(self, key: Tuple[int, int, int]) -> np.ndarray:
        return np.array(
            [(key[0] + 0.5) * self.cell_size,
             (key[1] + 0.5) * self.cell_size],
            dtype=np.float32,
        )

    # ── update (one exploration trajectory) ──

    def update(
        self,
        trajectory_obs: List[Dict[str, np.ndarray]],
        trajectory_snapshots: List[dict],
        trajectory_rewards: List[float],
        trajectory_actions: List[np.ndarray],
        trajectory_n_captured: Optional[List[int]] = None,
        source_cell_key: Optional[Tuple[int, int, int]] = None,
        source_cum_reward: float = 0.0,
        source_cost: int = 0,
    ) -> List[Cell]:
        """Process one Go-Explore trajectory and upsert cells.

        Parameters
        ----------
        trajectory_obs, trajectory_snapshots, trajectory_rewards,
        trajectory_actions : per-step data from the exploration.
        trajectory_n_captured : per-step target capture count.
        source_cell_key : key of the cell we explored *from*.
            ``None`` means we started from ``env.reset()``.
        source_cum_reward : cumulative reward of the source cell
            (so we can compute the total from root for each new cell).
        source_cost : trajectory_cost of the source cell.
        """
        new_cells: List[Cell] = []
        cum_reward = source_cum_reward
        if trajectory_n_captured is None:
            trajectory_n_captured = [0] * len(trajectory_obs)

        for step_idx, (obs, snapshot, reward, action, n_cap) in enumerate(
            zip(trajectory_obs, trajectory_snapshots, trajectory_rewards,
                trajectory_actions, trajectory_n_captured)
        ):
            cum_reward += reward
            total_cost = source_cost + step_idx + 1
            key = self.obs_to_cell_key(obs, n_captured=n_cap)

            # Actions from the source cell to this step (inclusive)
            seg_actions = [np.array(a, copy=True)
                           for a in trajectory_actions[: step_idx + 1]]

            if key not in self.cells:
                if len(self.cells) >= self.max_cells:
                    continue
                cell = Cell(
                    key=key,
                    center_xy=self._cell_center(key),
                    snapshot=copy.deepcopy(snapshot),
                    trajectory_cost=total_cost,
                    cumulative_reward=cum_reward,
                    visit_count=1,
                    parent_key=source_cell_key,
                    actions_from_parent=seg_actions,
                )
                cell.update_score()
                self.cells[key] = cell
                new_cells.append(cell)
            else:
                existing = self.cells[key]
                existing.visit_count += 1
                # Pareto dominance: update only if strictly better
                if (total_cost <= existing.trajectory_cost
                        and cum_reward >= existing.cumulative_reward
                        and (total_cost < existing.trajectory_cost
                             or cum_reward > existing.cumulative_reward)):
                    existing.trajectory_cost = total_cost
                    existing.cumulative_reward = cum_reward
                    existing.snapshot = copy.deepcopy(snapshot)
                    existing.parent_key = source_cell_key
                    existing.actions_from_parent = seg_actions
                existing.update_score()

        return new_cells

    # ── tree reconstruction ──

    def reconstruct_trajectory(
        self, target_key: Tuple[int, int, int],
    ) -> List[np.ndarray]:
        """Trace parent pointers from *target_key* back to root.

        Returns the full action sequence from ``env.reset()`` to the target
        cell.  The segments are collected leaf-to-root and then reversed.
        """
        chain: List[List[np.ndarray]] = []
        current = target_key

        while current is not None:
            cell = self.cells[current]
            if cell.actions_from_parent:
                chain.append(cell.actions_from_parent)
            current = cell.parent_key

        chain.reverse()
        # Flatten into one continuous action list
        return [a for segment in chain for a in segment]

    # ── best cell ──

    def get_best_cell(self) -> Optional[Cell]:
        """Return the cell with the most target captures (then highest
        cumulative reward as tie-breaker)."""
        if not self.cells:
            return None
        return max(self.cells.values(),
                   key=lambda c: (c.key[2], c.cumulative_reward))

    # ── selection ──

    def select(self) -> Optional[Cell]:
        """Score-proportional sampling."""
        if not self.cells:
            return None
        keys = list(self.cells.keys())
        scores = np.array([self.cells[k].score for k in keys],
                          dtype=np.float64)
        total = scores.sum()
        probs = scores / total if total > 0 else np.ones(len(keys)) / len(keys)
        idx = int(self._rng.choice(len(keys), p=probs))
        return self.cells[keys[idx]]

    # ── persistence ──

    def save(self, path: str) -> None:
        """Save archive: JSON metadata + per-cell pickle files."""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        cell_data_dir = base.parent / "cell_data"
        cell_data_dir.mkdir(parents=True, exist_ok=True)

        meta = []
        for i, cell in enumerate(self.cells.values()):
            # Pack snapshot + actions_from_parent into one pickle
            cell_file = str(cell_data_dir / f"cell_{i}.pkl")
            cell_payload = {
                "snapshot": cell.snapshot,
                "actions_from_parent": cell.actions_from_parent,
            }
            with open(cell_file, "wb") as f:
                pickle.dump(cell_payload, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            meta.append({
                "key": list(cell.key),
                "center_xy": cell.center_xy.tolist(),
                "trajectory_cost": cell.trajectory_cost,
                "cumulative_reward": float(cell.cumulative_reward),
                "visit_count": cell.visit_count,
                "score": float(cell.score),
                "parent_key": list(cell.parent_key) if cell.parent_key else None,
                "cell_file": cell_file,
            })

        with open(str(base), "w") as f:
            json.dump({
                "cell_size": self.cell_size,
                "env_seed": self.env_seed,
                "cells": meta,
            }, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            raw = json.load(f)
        self.cell_size = raw.get("cell_size", self.cell_size)
        self.env_seed = raw.get("env_seed", self.env_seed)

        for entry in raw["cells"]:
            key = tuple(entry["key"])
            snapshot = None
            actions_from_parent = None

            cell_file = entry.get("cell_file")
            if cell_file and os.path.exists(cell_file):
                with open(cell_file, "rb") as f:
                    payload = pickle.load(f)
                snapshot = payload.get("snapshot")
                actions_from_parent = payload.get("actions_from_parent")

            parent_key = (tuple(entry["parent_key"])
                          if entry.get("parent_key") else None)

            cell = Cell(
                key=key,
                center_xy=np.array(entry["center_xy"], dtype=np.float32),
                snapshot=snapshot,
                trajectory_cost=entry["trajectory_cost"],
                cumulative_reward=entry["cumulative_reward"],
                visit_count=entry["visit_count"],
                score=entry["score"],
                parent_key=parent_key,
                actions_from_parent=actions_from_parent,
            )
            self.cells[key] = cell

    def __len__(self) -> int:
        return len(self.cells)
