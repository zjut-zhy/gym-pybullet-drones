"""Wrapper that adds Go-Explore return/explore phases on top of OurRLAviaryPZ.

Wraps the PettingZoo ``OurRLAviaryPZ`` and injects:
* goal vector (normalised 2-D XY of the target cell centre)
* phase indicator (0 = return, 1 = explore)
* shaped reward and valid-mask metadata via *info*

Since PZ gives per-agent observations {key: (feat,)}, goal and phase are
appended as additional keys to each agent's obs dict.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gym_pybullet_drones.our_experiments.policy_based_go_explore.archive import Archive, Cell
from gym_pybullet_drones.our_experiments.policy_based_go_explore.trajectory_tracker import TrajectoryTracker


class Phase(IntEnum):
    RETURN = 0
    EXPLORE = 1


class GoExploreEnvWrapper:
    """Wraps *OurRLAviaryPZ* with Go-Explore return → explore phase logic.

    This is NOT a gym.Wrapper because the underlying env is a PettingZoo
    ParallelEnv. We implement a thin passthrough that adds goal/phase to
    each agent's observation.
    """

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

    # ── delegates ────────────────────────────────────────────────

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def agents(self):
        return self.env.agents

    @property
    def num_drones(self):
        return self.env.num_drones

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def close(self):
        self.env.close()

    # ── archive injection ────────────────────────────────────────

    def set_archive(self, archive: Archive) -> None:
        self._archive = archive

    # ── reset / step ─────────────────────────────────────────────

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self._begin_return_phase(observations)
        observations = self._augment_obs(observations)
        for agent in infos:
            infos[agent]["valid_mask"] = True
            infos[agent]["phase"] = int(self._phase)
        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._phase_step += 1

        if self._phase == Phase.RETURN:
            # use drone_0's position for sub-goal tracking
            drone0 = self.possible_agents[0]
            if drone0 in observations:
                xy_world = self._extract_world_xy(observations[drone0])
                _sg, shaped_reward, reached_final = self.tracker.advance(xy_world)
                # override rewards with shaped reward during return
                for agent in rewards:
                    rewards[agent] = shaped_reward
                if reached_final or self._phase_step >= self.return_max_steps:
                    self._begin_explore_phase()
        else:
            if self._phase_step >= self.explore_max_steps:
                for agent in truncations:
                    truncations[agent] = True

        observations = self._augment_obs(observations)
        for agent in infos:
            infos[agent]["valid_mask"] = True
            infos[agent]["phase"] = int(self._phase)
        return observations, rewards, terminations, truncations, infos

    # ── phase transitions ────────────────────────────────────────

    def _begin_return_phase(self, observations):
        self._phase_step = 0
        if self._archive is not None and len(self._archive) > 0:
            self._target_cell = self._archive.select()
        else:
            self._target_cell = None

        if self._target_cell is not None:
            self._phase = Phase.RETURN
            drone0 = self.possible_agents[0]
            if drone0 in observations:
                xy = self._extract_world_xy(observations[drone0])
                self.tracker.set_goal(xy, self._target_cell.center_xy)
        else:
            self._phase = Phase.EXPLORE
            self.tracker.reset()

    def _begin_explore_phase(self):
        self._phase = Phase.EXPLORE
        self._phase_step = 0
        self.tracker.reset()

    # ── observation helpers ──────────────────────────────────────

    def _augment_obs(self, observations):
        """Append goal and phase to each agent's obs dict."""
        if self._phase == Phase.RETURN and self.tracker.active:
            goal_2d = self.tracker.get_current_subgoal_normalised()
        elif self._target_cell is not None and self._phase == Phase.RETURN:
            goal_2d = self._archive.get_goal(self._target_cell) if self._archive else np.zeros(2, dtype=np.float32)
        else:
            goal_2d = np.zeros(2, dtype=np.float32)

        augmented = {}
        for agent, obs in observations.items():
            aug = dict(obs)
            aug["goal"] = goal_2d.copy().astype(np.float32)
            aug["phase"] = np.array([float(self._phase)], dtype=np.float32)
            augmented[agent] = aug
        return augmented

    def _extract_world_xy(self, agent_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get world-frame XY from a single-agent self_state."""
        self_state = np.asarray(agent_obs["self_state"])
        norm_xy = self_state[:2]
        return (norm_xy * self.arena_half).astype(np.float32)
