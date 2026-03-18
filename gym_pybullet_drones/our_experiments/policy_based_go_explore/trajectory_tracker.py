"""Sub-goal trajectory tracker for the *return* phase of Go-Explore.

Decomposes the straight-line path from the drone's current position to a
target cell centre into equally-spaced sub-goals and provides shaped reward
as sub-goals are reached.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class TrajectoryTracker:
    """Break a long return path into manageable sub-goals.

    Parameters
    ----------
    sub_goal_spacing : float
        Distance in metres between consecutive sub-goals.
    reach_threshold : float
        Distance in metres below which a sub-goal counts as reached.
    sub_goal_reward : float
        Bonus reward granted when a sub-goal is reached.
    potential_scale : float
        Scale for the continuous potential-based shaping reward.
    arena_half : float
        Half of the arena side-length; used for un-normalising positions.
    """

    def __init__(
        self,
        sub_goal_spacing: float = 1.0,
        reach_threshold: float = 0.5,
        sub_goal_reward: float = 1.0,
        potential_scale: float = 0.1,
        arena_half: float = 5.0,
    ) -> None:
        self.sub_goal_spacing = sub_goal_spacing
        self.reach_threshold = reach_threshold
        self.sub_goal_reward = sub_goal_reward
        self.potential_scale = potential_scale
        self.arena_half = arena_half

        self._sub_goals: List[np.ndarray] = []
        self._current_idx: int = 0
        self._prev_dist: Optional[float] = None
        self._active: bool = False

    # ── public API ───────────────────────────────────────────────

    def set_goal(
        self,
        current_xy_world: np.ndarray,
        target_xy_world: np.ndarray,
    ) -> None:
        """Compute sub-goal sequence from *current* to *target* in world frame.

        Both inputs should be in metres (world frame), **not** normalised.
        """
        diff = target_xy_world - current_xy_world
        total_dist = float(np.linalg.norm(diff))

        self._sub_goals = []
        if total_dist < 1e-4:
            self._sub_goals.append(target_xy_world.copy())
        else:
            n_segments = max(1, int(np.ceil(total_dist / self.sub_goal_spacing)))
            for i in range(1, n_segments + 1):
                t = i / n_segments
                sg = current_xy_world + t * diff
                self._sub_goals.append(sg.astype(np.float32))

        self._current_idx = 0
        self._prev_dist = None
        self._active = True

    def reset(self) -> None:
        """Deactivate the tracker (used when switching to explore)."""
        self._sub_goals = []
        self._current_idx = 0
        self._prev_dist = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def get_current_subgoal(self) -> Optional[np.ndarray]:
        """Return the current sub-goal in world-frame metres, or None."""
        if not self._active or self._current_idx >= len(self._sub_goals):
            return None
        return self._sub_goals[self._current_idx]

    def get_current_subgoal_normalised(self) -> np.ndarray:
        """Return the current sub-goal normalised to [-1, 1] for the network."""
        sg = self.get_current_subgoal()
        if sg is None:
            return np.zeros(2, dtype=np.float32)
        return np.clip(sg / self.arena_half, -1.0, 1.0).astype(np.float32)

    def advance(
        self,
        current_xy_world: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        """Step the tracker given the drone's current world-frame XY.

        Returns
        -------
        subgoal : ndarray (2,)
            Current sub-goal (world-frame).
        shaped_reward : float
            Shaped reward for this step.
        reached_final : bool
            True when the final sub-goal has been reached.
        """
        if not self._active or self._current_idx >= len(self._sub_goals):
            return np.zeros(2, dtype=np.float32), 0.0, True

        sg = self._sub_goals[self._current_idx]
        dist = float(np.linalg.norm(current_xy_world - sg))

        # potential-based shaping
        shaped_reward = 0.0
        if self._prev_dist is not None:
            shaped_reward = self.potential_scale * (self._prev_dist - dist)
        self._prev_dist = dist

        # check if current sub-goal is reached
        reached_final = False
        if dist < self.reach_threshold:
            shaped_reward += self.sub_goal_reward
            self._current_idx += 1
            self._prev_dist = None
            if self._current_idx >= len(self._sub_goals):
                reached_final = True
                self._active = False

        return sg, shaped_reward, reached_final
