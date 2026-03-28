"""Archive for Go-Explore Phase 1.

Each cell stores a **full_action_sequence**: the complete action list from
``env.reset()`` to this cell, recorded at creation time.  Independent of
other cells.

When cell B is discovered from cell A at exploration step k:
    B.full_action_sequence = A.full_action_sequence + actions[0:k+1]

This is an independent copy.  Cells are "frozen" after creation (action
sequence is never updated), so each cell's sequence is always valid for
deterministic replay.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Cell:
    """One entry in the archive."""

    key: Tuple[int, int, int]                    # (gx, gy, n_captured)
    center_xy: np.ndarray
    trajectory_cost: int = 0                      # total steps from reset
    cumulative_reward: float = 0.0
    visit_count: int = 0
    score: float = 1.0

    # Complete action sequence from env.reset() to this cell.
    # Each cell owns an independent copy; not affected by other cells.
    full_action_sequence: Optional[List[np.ndarray]] = None

    def update_score(self, max_steps: int = 1800,
                     target_count: int = 18) -> None:
        novelty = 1.0 / (1.0 + self.visit_count)
        n_captured = self.key[2]
        # Capture efficiency: normalised so perfect efficiency ≈ 1.0.
        #   raw  = n_captured / trajectory_cost   (captures per step)
        #   coef = max_steps / target_count        (scaling factor)
        # A cell that captures all targets at uniform rate gets ~1.0.
        coef = max_steps / max(1, target_count)
        efficiency = (n_captured / max(1, self.trajectory_cost)) * coef
        self.score = novelty + efficiency


class Archive:
    """Cell archive with per-cell full action sequences.

    Parameters
    ----------
    cell_size, arena_half, max_cells, env_seed, max_steps : see Phase 1 config.
    """

    def __init__(
        self,
        cell_size: float = 0.5,
        arena_half: float = 5.0,
        max_cells: int = 10_000,
        env_seed: int = 42,
        max_steps: int = 1800,
        target_count: int = 18,
    ) -> None:
        self.cell_size = cell_size
        self.arena_half = arena_half
        self.max_cells = max_cells
        self.env_seed = env_seed
        self.max_steps = max_steps
        self.target_count = target_count
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

    # ── update ──

    def update(
        self,
        trajectory_obs: List[Dict[str, np.ndarray]],
        trajectory_rewards: List[float],
        trajectory_actions: List[np.ndarray],
        trajectory_n_captured: Optional[List[int]] = None,
        source_cell: Optional["Cell"] = None,
    ) -> List[Cell]:
        """Process one exploration trajectory and upsert cells.

        Parameters
        ----------
        trajectory_* : per-step data from the exploration.
        source_cell : the Cell we explored from (None = fresh reset).
            Used to build each new cell's full_action_sequence.
        """
        new_cells: List[Cell] = []
        if trajectory_n_captured is None:
            trajectory_n_captured = [0] * len(trajectory_obs)

        # Prefix: the complete action history up to the source cell
        prefix_actions: List[np.ndarray] = []
        source_cum_reward = 0.0
        source_cost = 0
        if source_cell is not None:
            if source_cell.full_action_sequence is not None:
                prefix_actions = source_cell.full_action_sequence
            source_cum_reward = source_cell.cumulative_reward
            source_cost = source_cell.trajectory_cost

        cum_reward = source_cum_reward

        for step_idx, (obs, reward, action, n_cap) in enumerate(
            zip(trajectory_obs, trajectory_rewards,
                trajectory_actions, trajectory_n_captured)
        ):
            cum_reward += reward
            total_cost = source_cost + step_idx + 1
            key = self.obs_to_cell_key(obs, n_captured=n_cap)

            if key not in self.cells:
                if len(self.cells) >= self.max_cells:
                    continue
                # Lazy build: only construct full_seq when actually needed
                full_seq = prefix_actions + trajectory_actions[: step_idx + 1]
                cell = Cell(
                    key=key,
                    center_xy=self._cell_center(key),
                    trajectory_cost=total_cost,
                    cumulative_reward=cum_reward,
                    visit_count=1,
                    full_action_sequence=full_seq,
                )
                cell.update_score(self.max_steps, self.target_count)
                self.cells[key] = cell
                new_cells.append(cell)
            else:
                existing = self.cells[key]
                existing.visit_count += 1
                # Pareto update: replace if new path is not worse on
                # both (cost, reward) and strictly better on at least one.
                # With target_attraction disabled in Phase 1, cumulative_reward
                # only reflects capture bonuses + safety penalties.
                cost_ok = total_cost <= existing.trajectory_cost
                reward_ok = cum_reward >= existing.cumulative_reward
                strictly_better = (total_cost < existing.trajectory_cost
                                   or cum_reward > existing.cumulative_reward)
                if cost_ok and reward_ok and strictly_better:
                    full_seq = prefix_actions + trajectory_actions[: step_idx + 1]
                    existing.trajectory_cost = total_cost
                    existing.cumulative_reward = cum_reward
                    existing.full_action_sequence = full_seq
                existing.update_score(self.max_steps, self.target_count)

        return new_cells

    # ── best cell ──

    def get_best_cell(self) -> Optional[Cell]:
        """Return cell with most captures (then highest reward)."""
        if not self.cells:
            return None
        return max(self.cells.values(),
                   key=lambda c: (c.key[2], c.cumulative_reward))

    def get_successful_cells(self, target_count: int) -> List[Cell]:
        """Return all cells with n_captured >= target_count, sorted by reward."""
        successful = [
            c for c in self.cells.values() if c.key[2] >= target_count
        ]
        successful.sort(key=lambda c: c.cumulative_reward, reverse=True)
        return successful

    # ── selection ──

    def select(self) -> Optional[Cell]:
        if not self.cells:
            return None
        # Exclude done cells (all targets captured) — replaying them would
        # immediately terminate the env, wasting the iteration.
        keys = [k for k in self.cells if k[2] < self.target_count]
        if not keys:
            return None
        scores = np.array([self.cells[k].score for k in keys],
                          dtype=np.float64)
        total = scores.sum()
        probs = scores / total if total > 0 else np.ones(len(keys)) / len(keys)
        idx = int(self._rng.choice(len(keys), p=probs))
        return self.cells[keys[idx]]

    # ── persistence ──

    def save(self, path: str) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        action_seq_dir = base.parent / "action_sequences"
        action_seq_dir.mkdir(parents=True, exist_ok=True)

        meta = []
        for i, cell in enumerate(self.cells.values()):
            # Save this cell's full action sequence
            seq_file = str(action_seq_dir / f"cell_{i}_actions.pkl")
            if cell.full_action_sequence is not None:
                with open(seq_file, "wb") as f:
                    pickle.dump(cell.full_action_sequence, f,
                                protocol=pickle.HIGHEST_PROTOCOL)

            meta.append({
                "key": list(cell.key),
                "center_xy": cell.center_xy.tolist(),
                "trajectory_cost": cell.trajectory_cost,
                "cumulative_reward": float(cell.cumulative_reward),
                "visit_count": cell.visit_count,
                "score": float(cell.score),
                "action_seq_file": seq_file if cell.full_action_sequence is not None else None,
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

            # Load this cell's full action sequence
            full_action_sequence = None
            seq_file = entry.get("action_seq_file")
            if seq_file and os.path.exists(seq_file):
                with open(seq_file, "rb") as f:
                    full_action_sequence = pickle.load(f)

            cell = Cell(
                key=key,
                center_xy=np.array(entry["center_xy"], dtype=np.float32),
                trajectory_cost=entry["trajectory_cost"],
                cumulative_reward=entry["cumulative_reward"],
                visit_count=entry["visit_count"],
                score=entry["score"],
                full_action_sequence=full_action_sequence,
            )
            self.cells[key] = cell

    def __len__(self) -> int:
        return len(self.cells)
