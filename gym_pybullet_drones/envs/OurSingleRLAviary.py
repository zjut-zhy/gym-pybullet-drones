"""Single-agent variant of OurRLAviary.

Inherits all environment logic (arena, targets, obstacles, physics) from
:class:`OurRLAviary` and adapts the Gymnasium interface for a **single UAV**:

* Observation / action spaces have no leading ``NUM_DRONES`` dimension.
* ``teammate_state`` is removed from observations.
* UAV-pair collision penalties and termination are removed (only one drone).
* ``step()`` / ``reset()`` return plain arrays instead of batched arrays.
"""

import math
from typing import Optional

import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.OurRLAviary import OurRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


class OurSingleRLAviary(OurRLAviary):
    """Single-UAV target-coverage environment.

    Exactly the same arena, targets, obstacles, and dynamics as
    :class:`OurRLAviary`, but the Gymnasium spaces are shaped for standard
    single-agent RL algorithms (PPO, SAC, …) that expect flat tensors.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.VEL,
        arena_size_xy_m: float = 10.0,
        wall_height: float = 2.0,
        wall_thickness: float = 0.05,
        coverage_radius_m: float = 0.75,
        observation_radius_m: float = 3.0,
        target_count: int = 18,
        target_radius_m: float = 0.2,
        obstacle_count: int = 6,
        environment_seed: int = 0,
        randomize_obstacles_on_reset: bool = True,
        obstacle_keepout_center_m: float = 2.5,
        min_separation_m: float = 0.6,
        threat_zone_width_m: float = 1.5,
        max_episode_len_sec: float = 60.0,
        coverage_completion_ratio: float = 1.0,
        visualize_coverage: bool = False,
    ):
        if num_drones != 1:
            raise ValueError(
                f"OurSingleRLAviary only supports num_drones=1, got {num_drones}"
            )
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
            arena_size_xy_m=arena_size_xy_m,
            wall_height=wall_height,
            wall_thickness=wall_thickness,
            coverage_radius_m=coverage_radius_m,
            observation_radius_m=observation_radius_m,
            target_count=target_count,
            target_radius_m=target_radius_m,
            obstacle_count=obstacle_count,
            environment_seed=environment_seed,
            randomize_obstacles_on_reset=randomize_obstacles_on_reset,
            obstacle_keepout_center_m=obstacle_keepout_center_m,
            min_separation_m=min_separation_m,
            threat_zone_width_m=threat_zone_width_m,
            max_episode_len_sec=max_episode_len_sec,
            coverage_completion_ratio=coverage_completion_ratio,
            visualize_coverage=visualize_coverage,
        )

    # ------------------------------------------------------------------
    # Observation space  (no teammate, no NUM_DRONES axis)
    # ------------------------------------------------------------------

    def _observationSpace(self):
        self_low = np.full(self.SELF_STATE_DIM, -1.0, dtype=np.float32)
        self_high = np.full(self.SELF_STATE_DIM, 1.0, dtype=np.float32)

        target_low = np.full(self.TARGET_STATE_DIM, -1.0, dtype=np.float32)
        target_high = np.full(self.TARGET_STATE_DIM, 1.0, dtype=np.float32)

        obstacle_slot_low = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        obstacle_slot_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        obstacle_low = np.tile(obstacle_slot_low, self.OBSTACLE_SLOTS).astype(np.float32)
        obstacle_high = np.tile(obstacle_slot_high, self.OBSTACLE_SLOTS).astype(np.float32)

        return spaces.Dict(
            {
                "self_state": spaces.Box(low=self_low, high=self_high, dtype=np.float32),
                "target_state": spaces.Box(low=target_low, high=target_high, dtype=np.float32),
                "obstacle_state": spaces.Box(low=obstacle_low, high=obstacle_high, dtype=np.float32),
            }
        )

    # ------------------------------------------------------------------
    # Compute observation  (drone 0 only, no teammate)
    # ------------------------------------------------------------------

    def _computeObs(self):
        self._process_target_captures()
        self._update_coverage_visuals()

        return {
            "self_state": self._build_self_state(0),
            "target_state": self._build_target_state(0).reshape(-1),
            "obstacle_state": self._build_obstacle_state(0).reshape(-1),
        }

    # ------------------------------------------------------------------
    # Action space  (flat, no NUM_DRONES axis)
    # ------------------------------------------------------------------

    def _actionSpace(self):
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4 if self.ACT_TYPE == ActionType.RPM else self.ACTION_DIM
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            raise ValueError("Unsupported action type for OurSingleRLAviary")

        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((1, size)))

        return spaces.Box(low=-np.ones(size, dtype=np.float32),
                          high=np.ones(size, dtype=np.float32),
                          dtype=np.float32)


    # ------------------------------------------------------------------
    # Info  (simplified, no adjacency / multi-drone fields)
    # ------------------------------------------------------------------

    def _computeInfo(self):
        coverage_ratio = min(1.0, float(self._episode_target_captures) / float(max(1, self.TARGET_COUNT)))
        return {
            "coverage_ratio": coverage_ratio,
            "target_capture_count": int(self._episode_target_captures),
            "coverage_target_count": int(self.COVERAGE_TARGET_COUNT),
            "target_count": int(self.TARGET_COUNT),
            "obstacle_count": int(self.OBSTACLE_COUNT),
            "team_new_cells": int(self._last_target_captures),
            "obstacle_collision_count": int(self._last_obstacle_collision_count),
            "out_of_bounds_count": int(self._last_out_of_bounds_count),
            "target_positions": self._target_positions.copy(),
            "obstacle_positions": self._obstacle_positions_xy.copy(),
            "drone_reward": float(self._last_drone_rewards[0]) if self._last_drone_rewards is not None else 0.0,
        }
