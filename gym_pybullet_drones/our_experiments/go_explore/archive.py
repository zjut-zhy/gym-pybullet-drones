"""Archive for original Go-Explore — stores action sequences for replay.

Unlike the policy-based version, cells here store the exact action trajectory
from env reset (start of episode) to the cell, enabling deterministic replay
for the "return" phase.

Adapted for PettingZoo per-agent observations: ``obs_to_cell_key()`` takes a
**single-agent** obs dict ``{key: (feat,)}``.
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

    key: Tuple[int, int]
    center_xy: np.ndarray                        # (2,) world-frame metres
    action_sequence: List[np.ndarray]             # stored action trajectory
    trajectory_cost: int = 0                      # steps to reach
    cumulative_reward: float = 0.0
    visit_count: int = 0
    score: float = 1.0

    def update_score(self) -> None:
        novelty = 1.0 / (1.0 + self.visit_count)
        reward_bonus = max(0.0, self.cumulative_reward)
        self.score = novelty + 0.1 * reward_bonus


class Archive:
    """Cell archive with action-replay support.

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
        self.cells: Dict[Tuple[int, int], Cell] = {}
        self._rng = np.random.RandomState(0)

    def seed(self, s: int) -> None:
        self._rng = np.random.RandomState(s)

    # ── grid helpers ─────────────────────────────────────────────

    def _xy_to_key(self, norm_xy: np.ndarray) -> Tuple[int, int]:
        world_xy = norm_xy * self.arena_half
        gx = int(math.floor(world_xy[0] / self.cell_size))
        gy = int(math.floor(world_xy[1] / self.cell_size))
        return (gx, gy)

    def obs_to_cell_key(self, agent_obs: Dict[str, np.ndarray]) -> Tuple[int, int]:
        """Return cell key for a single-agent PettingZoo observation.

        ``agent_obs["self_state"]`` has shape ``(feat_dim,)`` — the first two
        elements are normalised XY.
        """
        self_state = np.asarray(agent_obs["self_state"])
        return self._xy_to_key(self_state[:2])

    def _cell_center(self, key: Tuple[int, int]) -> np.ndarray:
        return np.array(
            [(key[0] + 0.5) * self.cell_size, (key[1] + 0.5) * self.cell_size],
            dtype=np.float32,
        )

    # ── update ───────────────────────────────────────────────────

    def update(
        self,
        trajectory_agent_obs: List[Dict[str, Dict[str, np.ndarray]]],
        trajectory_actions: List[Dict[str, np.ndarray]],
        trajectory_rewards: List[Dict[str, float]],
    ) -> List[Cell]:
        """Process one trajectory and upsert cells.

        Parameters
        ----------
        trajectory_agent_obs : list of {agent: obs_dict}
            PettingZoo per-step, per-agent observations.
        trajectory_actions : list of {agent: action}
            Actions taken at each step *from reset*.
        trajectory_rewards : list of {agent: float}
            Per-agent reward at each step.

        Returns list of newly created cells.
        """
        new_cells: List[Cell] = []
        # track cumulative reward per agent
        agents = list(trajectory_agent_obs[0].keys()) if trajectory_agent_obs else []
        cum_rewards = {a: 0.0 for a in agents}

        for step_idx, (obs_dict, rew_dict) in enumerate(
            zip(trajectory_agent_obs, trajectory_rewards)
        ):
            for agent in obs_dict:
                cum_rewards[agent] = cum_rewards.get(agent, 0.0) + rew_dict.get(agent, 0.0)
                key = self.obs_to_cell_key(obs_dict[agent])
                cum_r = cum_rewards[agent]

                if key not in self.cells:
                    if len(self.cells) >= self.max_cells:
                        continue
                    cell = Cell(
                        key=key,
                        center_xy=self._cell_center(key),
                        action_sequence=[
                            {a: np.array(act[a], copy=True) for a in act}
                            for act in trajectory_actions[: step_idx + 1]
                        ],
                        trajectory_cost=step_idx,
                        cumulative_reward=cum_r,
                        visit_count=1,
                    )
                    cell.update_score()
                    self.cells[key] = cell
                    new_cells.append(cell)
                else:
                    existing = self.cells[key]
                    existing.visit_count += 1
                    if step_idx < existing.trajectory_cost or cum_r > existing.cumulative_reward:
                        existing.trajectory_cost = step_idx
                        existing.cumulative_reward = cum_r
                        existing.action_sequence = [
                            {a: np.array(act[a], copy=True) for a in act}
                            for act in trajectory_actions[: step_idx + 1]
                        ]
                    existing.update_score()
        return new_cells

    # ── selection ────────────────────────────────────────────────

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

    # ── persistence ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save archive to JSON (action sequences stored as lists)."""
        data = []
        for cell in self.cells.values():
            # serialise action sequence: list of {agent: list}
            seq = []
            for act_dict in cell.action_sequence:
                seq.append({a: v.tolist() if hasattr(v, 'tolist') else v
                            for a, v in act_dict.items()})
            data.append({
                "key": list(cell.key),
                "center_xy": cell.center_xy.tolist(),
                "trajectory_cost": cell.trajectory_cost,
                "cumulative_reward": float(cell.cumulative_reward),
                "visit_count": cell.visit_count,
                "score": float(cell.score),
                "action_sequence": seq,
            })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"cell_size": self.cell_size, "cells": data}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            raw = json.load(f)
        self.cell_size = raw.get("cell_size", self.cell_size)
        for entry in raw["cells"]:
            key = tuple(entry["key"])
            seq = []
            for act_dict in entry.get("action_sequence", []):
                seq.append({a: np.array(v, dtype=np.float32) for a, v in act_dict.items()})
            cell = Cell(
                key=key,
                center_xy=np.array(entry["center_xy"], dtype=np.float32),
                action_sequence=seq,
                trajectory_cost=entry["trajectory_cost"],
                cumulative_reward=entry["cumulative_reward"],
                visit_count=entry["visit_count"],
                score=entry["score"],
            )
            self.cells[key] = cell

    def __len__(self) -> int:
        return len(self.cells)
