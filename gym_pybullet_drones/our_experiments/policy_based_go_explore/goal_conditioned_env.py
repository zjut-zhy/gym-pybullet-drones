"""Wrapper that adds Go-Explore return/explore phases on top of OurSingleRLAviary.

Injects:
* goal vector (normalised 2-D XY of the target cell centre)
* phase indicator (0 = return, 1 = explore)
* shaped reward and valid-mask metadata via *info*

Wraps the standard Gymnasium single-agent OurSingleRLAviary.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gym_pybullet_drones.our_experiments.policy_based_go_explore.archive import Archive, Cell
from gym_pybullet_drones.our_experiments.policy_based_go_explore.trajectory_tracker import TrajectoryTracker


class Phase(IntEnum):
    RETURN = 0
    EXPLORE = 1


class GoExploreEnvWrapper:
    """Wraps *OurSingleRLAviary* with Go-Explore return -> explore phase logic."""

    def __init__(
        self,
        env,
        return_max_steps: int = 200,
        explore_max_steps: int = 300,
        tracker_kwargs: Optional[dict] = None,
        arena_half: float = 5.0,
    ) -> None:
        self.env = env
        self.return_max_steps = return_max_steps
        self.explore_max_steps = explore_max_steps
        self.arena_half = arena_half

        tk = tracker_kwargs or {}
        tk.setdefault("arena_half", arena_half)
        self.tracker = TrajectoryTracker(**tk)

        self._archive: Optional[Archive] = None
        self._phase: Phase = Phase.EXPLORE
        self._phase_step: int = 0
        self._target_cell: Optional[Cell] = None

    # -- delegates --

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def close(self):
        self.env.close()

    # -- archive injection --

    def set_archive(self, archive: Archive) -> None:
        self._archive = archive

    # -- reset / step --

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._begin_return_phase(obs)
        obs = self._augment_obs(obs)
        info["valid_mask"] = True
        info["phase"] = int(self._phase)
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._phase_step += 1

        if self._phase == Phase.RETURN:
            xy_world = self._extract_world_xy(obs)
            _sg, shaped_reward, reached_final = self.tracker.advance(xy_world)
            reward = shaped_reward
            if reached_final or self._phase_step >= self.return_max_steps:
                self._begin_explore_phase()
        else:
            if self._phase_step >= self.explore_max_steps:
                truncated = True

        obs = self._augment_obs(obs)
        info["valid_mask"] = True
        info["phase"] = int(self._phase)
        return obs, reward, terminated, truncated, info

    # -- phase transitions --

    def _begin_return_phase(self, obs):
        self._phase_step = 0
        if self._archive is not None and len(self._archive) > 0:
            self._target_cell = self._archive.select()
        else:
            self._target_cell = None

        if self._target_cell is not None:
            self._phase = Phase.RETURN
            xy = self._extract_world_xy(obs)
            self.tracker.set_goal(xy, self._target_cell.center_xy)
        else:
            self._phase = Phase.EXPLORE
            self.tracker.reset()

    def _begin_explore_phase(self):
        self._phase = Phase.EXPLORE
        self._phase_step = 0
        self.tracker.reset()

    # -- observation helpers --

    def _augment_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Append goal and phase to the obs dict."""
        if self._phase == Phase.RETURN and self.tracker.active:
            goal_2d = self.tracker.get_current_subgoal_normalised()
        elif self._target_cell is not None and self._phase == Phase.RETURN:
            goal_2d = self._archive.get_goal(self._target_cell) if self._archive else np.zeros(2, dtype=np.float32)
        else:
            goal_2d = np.zeros(2, dtype=np.float32)

        aug = dict(obs)
        aug["goal"] = goal_2d.copy().astype(np.float32)
        aug["phase"] = np.array([float(self._phase)], dtype=np.float32)
        return aug

    def _extract_world_xy(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get world-frame XY from self_state."""
        self_state = np.asarray(obs["self_state"])
        norm_xy = self_state[:2]
        return (norm_xy * self.arena_half).astype(np.float32)
