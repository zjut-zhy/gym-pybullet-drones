import copy
import math
from collections import deque
from importlib import resources
from typing import Optional

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


class OurRLAviary(BaseRLAviary):
    """Multi-UAV target-coverage environment with entity-centric observations.

    Each UAV observes only entities within a fixed local radius and all entity
    features preserve their global IDs in the observation tensors. Targets and
    obstacles move continuously with random directions and random speeds. Once a
    target is covered by any UAV, it is immediately respawned elsewhere to keep a
    constant environment complexity.
    """

    MAX_NUM_DRONES = 4
    MAX_TARGET_COUNT = 18
    MAX_OBSTACLE_COUNT = 6
    MIN_OBSTACLE_SIZE_XY_M = 0.5
    MAX_OBSTACLE_SIZE_XY_M = 1.5
    COVERAGE_RING_ALPHA = 0.55
    COVERAGE_Z_OFFSETS_M = (0.0, 0.01, 0.02)

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 4,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 60,
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
        visualize_coverage: bool = True,
    ):
        if obs != ObservationType.KIN:
            raise ValueError("OurRLAviary currently supports ObservationType.KIN only")
        if act not in [ActionType.VEL, ActionType.RPM, ActionType.PID, ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            raise ValueError("Unsupported action type for OurRLAviary")
        if num_drones < 1:
            raise ValueError("num_drones must be >= 1")
        if num_drones > self.MAX_NUM_DRONES:
            raise ValueError(f"num_drones must be <= MAX_NUM_DRONES ({self.MAX_NUM_DRONES})")
        if target_count < 1:
            raise ValueError("target_count must be >= 1")
        if target_count > self.MAX_TARGET_COUNT:
            raise ValueError(f"target_count must be <= MAX_TARGET_COUNT ({self.MAX_TARGET_COUNT})")
        if obstacle_count < 0:
            raise ValueError("obstacle_count must be >= 0")
        if obstacle_count > self.MAX_OBSTACLE_COUNT:
            raise ValueError(f"obstacle_count must be <= MAX_OBSTACLE_COUNT ({self.MAX_OBSTACLE_COUNT})")

        self.ARENA_SIZE_XY_M = float(arena_size_xy_m)
        self.WALL_HEIGHT = float(wall_height)
        self.WALL_THICKNESS = float(wall_thickness)
        if initial_xyzs is None:
            initial_xyzs = self._default_initial_xyzs(num_drones)
        else:
            initial_xyzs = np.asarray(initial_xyzs, dtype=np.float32)

        self.COVERAGE_RADIUS_M = float(coverage_radius_m)
        self.OBSERVATION_RADIUS_M = float(observation_radius_m)
        self.TARGET_COUNT = int(target_count)
        self.TARGET_RADIUS_M = float(target_radius_m)
        self.OBSTACLE_COUNT = int(obstacle_count)
        default_target_height = float(np.mean(initial_xyzs[:, 2]))
        default_obstacle_height = float(np.mean(initial_xyzs[:, 2]))
        self.OBSTACLE_HEIGHT_M = default_obstacle_height
        self.TARGET_HEIGHT_M = default_target_height
        self.OBSTACLE_Z_M = self.OBSTACLE_HEIGHT_M / 2.0
        self.TARGET_Z_M = self.TARGET_HEIGHT_M / 2.0
        self.ENVIRONMENT_SEED = int(environment_seed)
        self.RANDOMIZE_OBSTACLES_ON_RESET = bool(randomize_obstacles_on_reset)
        self.OBSTACLE_KEEPOUT_CENTER_M = float(obstacle_keepout_center_m)
        self.MIN_SEPARATION_M = float(min_separation_m)
        self.THREAT_ZONE_WIDTH_M = float(threat_zone_width_m)
        self.EPISODE_LEN_SEC = float(max_episode_len_sec)
        self.COVERAGE_COMPLETION_RATIO = float(np.clip(coverage_completion_ratio, 0.0, 1.0))
        self.VIS_COVERAGE = bool(visualize_coverage)
        self.VEL_Z_COMPONENT = 0.0
        self.VEL_SPEED_SCALE = 1.0

        if act == ActionType.VEL:
            self.ACTION_DIM = 2
        elif act == ActionType.RPM:
            self.ACTION_DIM = 4
        elif act == ActionType.PID:
            self.ACTION_DIM = 3
        else:
            self.ACTION_DIM = 1

        if self.ARENA_SIZE_XY_M <= 0:
            raise ValueError("arena_size_xy_m must be > 0")
        if self.WALL_HEIGHT <= 0:
            raise ValueError("wall_height must be > 0")
        if self.WALL_THICKNESS <= 0:
            raise ValueError("wall_thickness must be > 0")
        if self.COVERAGE_RADIUS_M <= 0:
            raise ValueError("coverage_radius_m must be > 0")
        if self.OBSERVATION_RADIUS_M <= 0:
            raise ValueError("observation_radius_m must be > 0")
        if self.TARGET_RADIUS_M <= 0:
            raise ValueError("target_radius_m must be > 0")
        if self.OBSTACLE_HEIGHT_M <= 0:
            raise ValueError("derived obstacle height must be > 0; check initial_xyzs z values")
        if self.MIN_SEPARATION_M <= 0:
            raise ValueError("min_separation_m must be > 0")
        if self.THREAT_ZONE_WIDTH_M < 0:
            raise ValueError("threat_zone_width_m must be >= 0")
        if self.EPISODE_LEN_SEC <= 0:
            raise ValueError("max_episode_len_sec must be > 0")

        half = self.ARENA_SIZE_XY_M / 2.0
        self.GRID_BOUNDS = ((-half, half), (-half, half))
        self.UAV_RADIUS_M = float(self.COLLISION_R) if hasattr(self, "COLLISION_R") else 0.08
        self.DRONE_SLOTS = self.MAX_NUM_DRONES
        self.TEAMMATE_SLOTS = self.MAX_NUM_DRONES
        self.TARGET_SLOTS = self.MAX_TARGET_COUNT
        self.OBSTACLE_SLOTS = self.MAX_OBSTACLE_COUNT
        self.SELF_STATE_DIM = 6
        self.TEAMMATE_FEATURE_DIM = 6
        self.TARGET_FEATURE_DIM = 3
        self.OBSTACLE_FEATURE_DIM = 4
        self.TEAMMATE_STATE_DIM = self.TEAMMATE_SLOTS * self.TEAMMATE_FEATURE_DIM
        self.TARGET_STATE_DIM = self.TARGET_SLOTS * self.TARGET_FEATURE_DIM
        self.OBSTACLE_STATE_DIM = self.OBSTACLE_SLOTS * self.OBSTACLE_FEATURE_DIM
        self.COVERAGE_TARGET_COUNT = max(1, int(math.ceil(self.TARGET_COUNT * self.COVERAGE_COMPLETION_RATIO)))
        self.ENTITY_DIRECTION_HOLD_STEPS = int(ctrl_freq)  # 相当于约 1s 保持一个方向

        self._rng = np.random.RandomState(self.ENVIRONMENT_SEED)
        self._obstacle_positions_xy = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_velocities_xy = np.zeros((0, 2), dtype=np.float32)
        self._obstacle_direction_steps = np.zeros((0,), dtype=np.int32)
        self._obstacle_radii = np.zeros((0,), dtype=np.float32)
        self._obstacle_body_ids = []
        self._target_positions = np.zeros((0, 3), dtype=np.float32)
        self._target_velocities_xy = np.zeros((0, 2), dtype=np.float32)
        self._target_direction_steps = np.zeros((0,), dtype=np.int32)
        self._target_body_ids = []
        self._episode_counter = 0
        self._last_target_captures = 0
        self._last_drone_captures = None
        self._last_drone_rewards = None
        self._episode_target_captures = 0
        self._targets_spawned = self.TARGET_COUNT
        self._last_uav_collision_count = 0
        self._last_obstacle_collision_count = 0
        self._last_out_of_bounds_count = 0
        self._coverage_body_ids = []
        self._coverage_visual_shape_ids = []

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
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
        )
        self.SPEED_LIMIT = 1

        self.UAV_RADIUS_M = float(self.COLLISION_R)

    ################################################################################

    def _default_initial_xyzs(self, num_drones: int) -> np.ndarray:
        radius = 1.25
        z0 = self.WALL_HEIGHT * 0.5
        xyzs = np.zeros((num_drones, 3), dtype=np.float32)
        for i in range(num_drones):
            angle = (2.0 * np.pi * i) / float(num_drones) 
            xyzs[i, 0] = radius * np.cos(angle)
            xyzs[i, 1] = radius * np.sin(angle)
            xyzs[i, 2] = z0
        return xyzs

    ################################################################################

    def reset(self, seed: int = None, options: dict = None):
        current_seed = self.ENVIRONMENT_SEED
        if self.RANDOMIZE_OBSTACLES_ON_RESET:
            if seed is not None:
                current_seed = int(seed)
            else:
                current_seed = int(self.ENVIRONMENT_SEED + self._episode_counter)

        self._rng = np.random.RandomState(current_seed)
        self._last_target_captures = 0
        self._last_drone_captures = np.zeros(self.NUM_DRONES, dtype=np.int32)
        self._last_drone_rewards = np.zeros(self.NUM_DRONES, dtype=np.float32)
        self._episode_target_captures = 0
        self._targets_spawned = self.TARGET_COUNT
        self._last_uav_collision_count = 0
        self._last_obstacle_collision_count = 0
        self._last_out_of_bounds_count = 0
        self._clear_coverage_visuals()
        obs, info = super().reset(seed=seed, options=options)
        self._episode_counter += 1
        return obs, info

    ################################################################################

    def _housekeeping(self):
        super()._housekeeping()
        self._coverage_body_ids = []
        self._coverage_visual_shape_ids = []

    ################################################################################

    def _clear_coverage_visuals(self):
        if not hasattr(self, "_coverage_body_ids"):
            return
        if self.GUI:
            for body_ids in self._coverage_body_ids:
                for body_id in body_ids:
                    try:
                        p.removeBody(body_id, physicsClientId=self.CLIENT)
                    except Exception:
                        pass
        self._coverage_body_ids = []
        self._coverage_visual_shape_ids = []

    def _create_coverage_visuals(self):
        if not (self.GUI and self.VIS_COVERAGE):
            return
        if len(self._coverage_body_ids) == self.NUM_DRONES:
            return

        self._clear_coverage_visuals()
        ring_mesh_path = str(resources.files("gym_pybullet_drones").joinpath("assets", "coverage_ring.obj"))
        threat_radius = self.UAV_RADIUS_M + self.THREAT_ZONE_WIDTH_M
        circle_specs = [
            (self.OBSERVATION_RADIUS_M, [1.0, 1.0, 0.0, self.COVERAGE_RING_ALPHA]),
            (threat_radius, [1.0, 0.0, 0.0, self.COVERAGE_RING_ALPHA]),
            (self.COVERAGE_RADIUS_M, [0.0, 1.0, 0.0, self.COVERAGE_RING_ALPHA]),
        ]

        for radius, rgba in circle_specs:
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=ring_mesh_path,
                meshScale=[float(radius), float(radius), 1.0],
                rgbaColor=rgba,
                specularColor=[0.0, 0.0, 0.0],
                physicsClientId=self.CLIENT,
            )
            self._coverage_visual_shape_ids.append(visual_shape_id)

        for drone_idx in range(self.NUM_DRONES):
            if hasattr(self, "pos") and self.pos.shape[0] == self.NUM_DRONES:
                center_xy = self.pos[drone_idx, :2]
            else:
                center_xy = self.INIT_XYZS[drone_idx, :2]
            z_base = float(self.INIT_XYZS[drone_idx, 2])
            drone_body_ids = []
            for visual_shape_id, z_offset in zip(self._coverage_visual_shape_ids, self.COVERAGE_Z_OFFSETS_M):
                body_id = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=[float(center_xy[0]), float(center_xy[1]), z_base + float(z_offset)],
                    baseOrientation=[0, 0, 0, 1],
                    physicsClientId=self.CLIENT,
                )
                drone_body_ids.append(body_id)
            self._coverage_body_ids.append(drone_body_ids)

    def _update_coverage_visuals(self):
        if not (self.GUI and self.VIS_COVERAGE):
            return
        if len(self._coverage_body_ids) != self.NUM_DRONES:
            self._create_coverage_visuals()
        if len(self._coverage_body_ids) != self.NUM_DRONES:
            return

        for drone_idx, body_ids in enumerate(self._coverage_body_ids):
            center_xy = self.pos[drone_idx, :2]
            z_base = float(self.INIT_XYZS[drone_idx, 2])
            for body_id, z_offset in zip(body_ids, self.COVERAGE_Z_OFFSETS_M):
                p.resetBasePositionAndOrientation(
                    body_id,
                    [float(center_xy[0]), float(center_xy[1]), z_base + float(z_offset)],
                    [0, 0, 0, 1],
                    physicsClientId=self.CLIENT,
                )

    ################################################################################

    def _addObstacles(self):
        z = self.WALL_HEIGHT / 2.0
        half = self.ARENA_SIZE_XY_M / 2.0
        thickness = self.WALL_THICKNESS

        wall_x_half_extents = [half, thickness / 2.0, self.WALL_HEIGHT / 2.0]
        wall_y_half_extents = [thickness / 2.0, half, self.WALL_HEIGHT / 2.0]
        col_x = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_x_half_extents, physicsClientId=self.CLIENT)
        col_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_y_half_extents, physicsClientId=self.CLIENT)
        vis_x = p.createVisualShape(p.GEOM_BOX, halfExtents=wall_x_half_extents, rgbaColor=[0.55, 0.55, 0.55, 1.0], physicsClientId=self.CLIENT)
        vis_y = p.createVisualShape(p.GEOM_BOX, halfExtents=wall_y_half_extents, rgbaColor=[0.55, 0.55, 0.55, 1.0], physicsClientId=self.CLIENT)
        wall_offset = half + thickness / 2.0  # 内表面在 ±half
        p.createMultiBody(0, col_x, vis_x, [0.0, +wall_offset, z], [0, 0, 0, 1], physicsClientId=self.CLIENT)
        p.createMultiBody(0, col_x, vis_x, [0.0, -wall_offset, z], [0, 0, 0, 1], physicsClientId=self.CLIENT)
        p.createMultiBody(0, col_y, vis_y, [+wall_offset, 0.0, z], [0, 0, 0, 1], physicsClientId=self.CLIENT)
        p.createMultiBody(0, col_y, vis_y, [-wall_offset, 0.0, z], [0, 0, 0, 1], physicsClientId=self.CLIENT)

        self._obstacle_positions_xy = np.zeros((self.OBSTACLE_COUNT, 2), dtype=np.float32)
        self._obstacle_velocities_xy = np.zeros((self.OBSTACLE_COUNT, 2), dtype=np.float32)
        self._obstacle_direction_steps = np.zeros((self.OBSTACLE_COUNT,), dtype=np.int32)
        self._obstacle_radii = self._rng.uniform(
            self.MIN_OBSTACLE_SIZE_XY_M / 2.0,
            self.MAX_OBSTACLE_SIZE_XY_M / 2.0,
            size=(self.OBSTACLE_COUNT,),
        ).astype(np.float32)
        self._obstacle_body_ids = []
        self._target_positions = np.zeros((self.TARGET_COUNT, 3), dtype=np.float32)
        self._target_velocities_xy = np.zeros((self.TARGET_COUNT, 2), dtype=np.float32)
        self._target_direction_steps = np.zeros((self.TARGET_COUNT,), dtype=np.int32)
        self._target_body_ids = []
        target_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.TARGET_RADIUS_M,
            length=self.TARGET_HEIGHT_M,
            rgbaColor=[0.15, 0.75, 0.20, 1.0],
            physicsClientId=self.CLIENT,
        )

        for idx in range(self.OBSTACLE_COUNT):
            obstacle_radius = float(self._obstacle_radii[idx])
            obstacle_xy = self._sample_free_xy(
                obstacle_radius,
                existing_xy=self._obstacle_positions_xy[:idx],
                existing_radii=self._obstacle_radii[:idx],
            )
            self._obstacle_positions_xy[idx, :] = obstacle_xy
            self._obstacle_velocities_xy[idx, :] = self._sample_entity_velocity(self._obstacle_speed_limit_mps())
            obstacle_collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=obstacle_radius,
                height=self.OBSTACLE_HEIGHT_M,
                physicsClientId=self.CLIENT,
            )
            obstacle_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=obstacle_radius,
                length=self.OBSTACLE_HEIGHT_M,
                rgbaColor=[0.85, 0.45, 0.20, 1.0],
                physicsClientId=self.CLIENT,
            )
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=obstacle_collision,
                baseVisualShapeIndex=obstacle_visual,
                basePosition=[float(obstacle_xy[0]), float(obstacle_xy[1]), self.OBSTACLE_Z_M],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )
            self._obstacle_body_ids.append(body_id)

        for idx in range(self.TARGET_COUNT):
            target_xy = self._sample_free_xy(
                self.TARGET_RADIUS_M,
                existing_xy=self._target_positions[:idx, :2],
                existing_radii=np.full((idx,), self.TARGET_RADIUS_M, dtype=np.float32),
                extra_xy=self._obstacle_positions_xy,
                extra_radii=self._obstacle_radii,
            )
            self._target_positions[idx, :] = np.array([target_xy[0], target_xy[1], self.TARGET_Z_M], dtype=np.float32)
            self._target_velocities_xy[idx, :] = self._sample_entity_velocity(self._target_speed_limit_mps())
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=target_visual,
                basePosition=self._target_positions[idx, :].tolist(),
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )
            self._target_body_ids.append(body_id)

    ################################################################################

    def _obstacle_speed_limit_mps(self) -> float:
        return 0.2 * self._velocity_scale()

    def _target_speed_limit_mps(self) -> float:
        return 0.4 * self._velocity_scale()

    def _sample_entity_velocity(self, fixed_speed: float) -> np.ndarray:
        if fixed_speed <= 0.0:
            return np.zeros(2, dtype=np.float32)
        angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
        return fixed_speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    def _sample_free_xy(
        self,
        radius: float,
        existing_xy: np.ndarray,
        existing_radii: Optional[np.ndarray] = None,
        extra_xy: Optional[np.ndarray] = None,
        extra_radii: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        half = self.ARENA_SIZE_XY_M / 2.0
        margin = radius + max(self.UAV_RADIUS_M, self.TARGET_RADIUS_M) + 0.05
        low = -half + margin
        high = half - margin
        drone_xy = self.pos[:, :2] if hasattr(self, "pos") and self.pos.shape[0] == self.NUM_DRONES else self.INIT_XYZS[:, :2]

        candidates = []
        if existing_xy is not None and existing_xy.size > 0:
            points = np.asarray(existing_xy, dtype=np.float32).reshape(-1, 2)
            if existing_radii is None:
                radii = np.full((points.shape[0],), radius, dtype=np.float32)
            else:
                radii = np.asarray(existing_radii, dtype=np.float32).reshape(-1)
            candidates.append((points, radii))
        if extra_xy is not None and extra_xy.size > 0:
            points = np.asarray(extra_xy, dtype=np.float32).reshape(-1, 2)
            if extra_radii is None:
                radii = np.full((points.shape[0],), radius, dtype=np.float32)
            else:
                radii = np.asarray(extra_radii, dtype=np.float32).reshape(-1)
            candidates.append((points, radii))
        drone_points = np.asarray(drone_xy, dtype=np.float32).reshape(-1, 2)
        drone_radii = np.full((drone_points.shape[0],), self.UAV_RADIUS_M, dtype=np.float32)
        candidates.append((drone_points, drone_radii))

        for _ in range(5000):
            xy = np.array([
                float(self._rng.uniform(low, high)),
                float(self._rng.uniform(low, high)),
            ], dtype=np.float32)
            if np.linalg.norm(xy) < self.OBSTACLE_KEEPOUT_CENTER_M:
                continue

            valid = True
            for pts, pts_radii in candidates:
                if pts.size == 0:
                    continue
                dists = np.linalg.norm(pts - xy.reshape(1, 2), axis=1)
                min_dists = radius + pts_radii + 0.05
                if np.any(dists < min_dists):
                    valid = False
                    break
            if valid:
                return xy

        return np.array([0.0, 0.0], dtype=np.float32)

    ################################################################################

    def _advance_bodies(
        self,
        positions_xy: np.ndarray,
        velocities_xy: np.ndarray,
        direction_steps: np.ndarray,
        radii: np.ndarray,
        body_ids: list[int],
        z_value: float,
    ):
        if positions_xy.shape[0] == 0:
            return

        half = self.ARENA_SIZE_XY_M / 2.0
        dt = float(self.CTRL_TIMESTEP)
        for idx in range(positions_xy.shape[0]):
            pos = positions_xy[idx, :].copy()
            vel = velocities_xy[idx, :].copy()
            if direction_steps[idx] >= self.ENTITY_DIRECTION_HOLD_STEPS or not np.any(np.abs(vel) > 0.0):
                speed = float(np.linalg.norm(vel))
                vel = self._sample_entity_velocity(speed)
                direction_steps[idx] = 0
            next_pos = pos + vel * dt
            limit = half - float(radii[idx]) - 0.05
            for axis in range(2):
                if next_pos[axis] > limit or next_pos[axis] < -limit:
                    vel[axis] *= -1.0
                    next_pos[axis] = np.clip(next_pos[axis], -limit, limit)
            positions_xy[idx, :] = next_pos
            velocities_xy[idx, :] = vel
            direction_steps[idx] += 1
            p.resetBasePositionAndOrientation(
                body_ids[idx],
                [float(next_pos[0]), float(next_pos[1]), float(z_value)],
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )

    def _update_dynamic_entities(self):
        self._advance_bodies(
            self._obstacle_positions_xy,
            self._obstacle_velocities_xy,
            self._obstacle_direction_steps,
            self._obstacle_radii,
            self._obstacle_body_ids,
            self.OBSTACLE_Z_M,
        )
        self._advance_bodies(
            self._target_positions[:, :2],
            self._target_velocities_xy,
            self._target_direction_steps,
            np.full((self.TARGET_COUNT,), self.TARGET_RADIUS_M, dtype=np.float32),
            self._target_body_ids,
            self.TARGET_Z_M,
        )
        if self.TARGET_COUNT > 0:
            self._target_positions[:, 2] = self.TARGET_Z_M

    ################################################################################

    def _actionSpace(self):
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4 if self.ACT_TYPE == ActionType.RPM else self.ACTION_DIM
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            raise ValueError("Unsupported action type for OurRLAviary")

        act_lower_bound = np.array([-1 * np.ones(size) for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1 * np.ones(size) for _ in range(self.NUM_DRONES)])
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _preprocessAction(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)

        if action.shape[0] < self.NUM_DRONES:
            padded_action = np.zeros((self.NUM_DRONES, action.shape[1]), dtype=np.float32)
            padded_action[:action.shape[0], :] = action
            action = padded_action
        elif action.shape[0] > self.NUM_DRONES:
            action = action[: self.NUM_DRONES, :]

        self._update_dynamic_entities()
        if self.ACT_TYPE == ActionType.VEL:
            self.action_buffer.append(action)
            rpm = np.zeros((self.NUM_DRONES, 4))
            for i in range(self.NUM_DRONES):
                target_xy = action[i, :]
                vel_cmd = np.array([target_xy[0], target_xy[1], self.VEL_Z_COMPONENT], dtype=np.float32)
                norm = np.linalg.norm(vel_cmd)
                if norm > 0:
                    v_unit_vector = vel_cmd / norm
                else:
                    v_unit_vector = np.zeros(3, dtype=np.float32)

                state = self._getDroneStateVector(i)
                target_pos_hold = state[0:3].copy()
                target_pos_hold[2] = self.INIT_XYZS[i, 2]
                rpm_i, _, _ = self.ctrl[i].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=target_pos_hold,
                    target_rpy=np.array([0, 0, state[9]], dtype=np.float32),
                    target_vel=self.SPEED_LIMIT * self.VEL_SPEED_SCALE * v_unit_vector,
                )
                rpm[i, :] = rpm_i
            return rpm

        return super()._preprocessAction(action[: self.NUM_DRONES, :])

    ################################################################################

    def _observationSpace(self):
        self_low = np.array([-1.0] * self.SELF_STATE_DIM, dtype=np.float32)
        self_high = np.array([1.0] * self.SELF_STATE_DIM, dtype=np.float32)
        teammate_low = np.array([-1.0] * self.TEAMMATE_STATE_DIM, dtype=np.float32)
        teammate_high = np.array([1.0] * self.TEAMMATE_STATE_DIM, dtype=np.float32)
        target_low = np.array([-1.0] * self.TARGET_STATE_DIM, dtype=np.float32)
        target_high = np.array([1.0] * self.TARGET_STATE_DIM, dtype=np.float32)
        obstacle_slot_low = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        obstacle_slot_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        obstacle_low = np.tile(obstacle_slot_low, self.OBSTACLE_SLOTS).astype(np.float32)
        obstacle_high = np.tile(obstacle_slot_high, self.OBSTACLE_SLOTS).astype(np.float32)

        return spaces.Dict(
            {
                "self_state": spaces.Box(
                    low=np.tile(self_low, (self.NUM_DRONES, 1)).astype(np.float32),
                    high=np.tile(self_high, (self.NUM_DRONES, 1)).astype(np.float32),
                    dtype=np.float32,
                ),
                "teammate_state": spaces.Box(
                    low=np.tile(teammate_low, (self.NUM_DRONES, 1)).astype(np.float32),
                    high=np.tile(teammate_high, (self.NUM_DRONES, 1)).astype(np.float32),
                    dtype=np.float32,
                ),
                "target_state": spaces.Box(
                    low=np.tile(target_low, (self.NUM_DRONES, 1)).astype(np.float32),
                    high=np.tile(target_high, (self.NUM_DRONES, 1)).astype(np.float32),
                    dtype=np.float32,
                ),
                "obstacle_state": spaces.Box(
                    low=np.tile(obstacle_low, (self.NUM_DRONES, 1)).astype(np.float32),
                    high=np.tile(obstacle_high, (self.NUM_DRONES, 1)).astype(np.float32),
                    dtype=np.float32,
                ),
            }
        )

    ################################################################################

    def _computeObs(self):
        self._process_target_captures()
        self._update_coverage_visuals()

        self_state = np.zeros((self.NUM_DRONES, self.SELF_STATE_DIM), dtype=np.float32)
        teammate_state = np.zeros((self.NUM_DRONES, self.TEAMMATE_STATE_DIM), dtype=np.float32)
        target_state = np.zeros((self.NUM_DRONES, self.TARGET_STATE_DIM), dtype=np.float32)
        obstacle_state = np.zeros((self.NUM_DRONES, self.OBSTACLE_STATE_DIM), dtype=np.float32)

        for i in range(self.NUM_DRONES):
            self_state[i, :] = self._build_self_state(i)
            teammate_state[i, :] = self._build_teammate_state(i).reshape(-1)
            target_state[i, :] = self._build_target_state(i).reshape(-1)
            obstacle_state[i, :] = self._build_obstacle_state(i).reshape(-1)

        return {
            "self_state": self_state,
            "teammate_state": teammate_state,
            "target_state": target_state,
            "obstacle_state": obstacle_state,
        }

    ################################################################################

    def _position_scale(self) -> np.ndarray:
        return np.array([
            max(1e-6, self.ARENA_SIZE_XY_M / 2.0),
            max(1e-6, self.ARENA_SIZE_XY_M / 2.0),
            max(1e-6, self.WALL_HEIGHT),
        ], dtype=np.float32)

    def _velocity_scale(self) -> float:
        if hasattr(self, "SPEED_LIMIT") and self.SPEED_LIMIT > 0:
            return float(self.SPEED_LIMIT)
        return max(1e-6, self.MAX_SPEED_KMH / 3.6)

    def _is_entity_visible(self, rel_pos: np.ndarray, entity_radius: float) -> bool:
        return float(np.linalg.norm(rel_pos)) <= (self.OBSERVATION_RADIUS_M + float(entity_radius))



    def _build_self_state(self, nth_drone: int) -> np.ndarray:
        pos_scale = self._position_scale()
        vel_scale = self._velocity_scale()
        pos_norm = np.clip(self.pos[nth_drone, :] / pos_scale, -1.0, 1.0)
        vel_norm = np.clip(self.vel[nth_drone, :] / vel_scale, -1.0, 1.0)
        return np.hstack([pos_norm, vel_norm]).astype(np.float32)

    def _build_teammate_state(self, nth_drone: int) -> np.ndarray:
        features = np.zeros((self.TEAMMATE_SLOTS, self.TEAMMATE_FEATURE_DIM), dtype=np.float32)
        pos_scale = self._position_scale()
        vel_scale = self._velocity_scale()
        for other_idx in range(self.NUM_DRONES):
            if other_idx == nth_drone:
                continue
            rel_pos = (self.pos[other_idx, :] - self.pos[nth_drone, :]).astype(np.float32)
            if float(np.linalg.norm(rel_pos)) <= self.OBSERVATION_RADIUS_M:
                rel_vel = (self.vel[other_idx, :] - self.vel[nth_drone, :]).astype(np.float32)
                features[other_idx, :] = np.hstack([
                    np.clip(rel_pos / pos_scale, -1.0, 1.0),
                    np.clip(rel_vel / vel_scale, -1.0, 1.0),
                ]).astype(np.float32)
        return features

    def _build_target_state(self, nth_drone: int) -> np.ndarray:
        features = np.zeros((self.TARGET_SLOTS, self.TARGET_FEATURE_DIM), dtype=np.float32)
        pos_scale = self._position_scale()
        for target_idx in range(self.TARGET_COUNT):
            rel_pos = (self._target_positions[target_idx, :] - self.pos[nth_drone, :]).astype(np.float32)
            if self._is_entity_visible(rel_pos, self.TARGET_RADIUS_M):
                features[target_idx, :] = np.clip(rel_pos / pos_scale, -1.0, 1.0)
        return features

    def _build_obstacle_state(self, nth_drone: int) -> np.ndarray:
        features = np.zeros((self.OBSTACLE_SLOTS, self.OBSTACLE_FEATURE_DIM), dtype=np.float32)
        if self.OBSTACLE_COUNT <= 0:
            return features

        pos_scale = self._position_scale()
        for obstacle_idx in range(self.OBSTACLE_COUNT):
            obstacle_pos = np.array([
                self._obstacle_positions_xy[obstacle_idx, 0],
                self._obstacle_positions_xy[obstacle_idx, 1],
                self.OBSTACLE_Z_M,
            ], dtype=np.float32)
            rel_pos = obstacle_pos - self.pos[nth_drone, :]
            if self._is_entity_visible(rel_pos, float(self._obstacle_radii[obstacle_idx])):
                features[obstacle_idx, 0:3] = np.clip(rel_pos / pos_scale, -1.0, 1.0)
                features[obstacle_idx, 3] = min(1.0, float(self._obstacle_radii[obstacle_idx]) / max(1e-6, self.OBSERVATION_RADIUS_M))
        return features

    ################################################################################

    def _process_target_captures(self):
        captures = []
        drone_captures = np.zeros(self.NUM_DRONES, dtype=np.int32)
        for target_idx in range(self.TARGET_COUNT):
            dists = np.linalg.norm(self.pos[:, :2] - self._target_positions[target_idx, :2], axis=1)
            covering = np.where(dists <= self.COVERAGE_RADIUS_M)[0]
            if len(covering) > 0:
                captures.append(target_idx)
                closest = covering[np.argmin(dists[covering])]
                drone_captures[closest] += 1

        self._last_target_captures = len(captures)
        self._last_drone_captures = drone_captures
        if not captures:
            return

        self._episode_target_captures += len(captures)
        self._targets_spawned += len(captures)
        for target_idx in captures:
            self._respawn_target(target_idx)

    def _respawn_target(self, target_idx: int):
        other_targets = np.delete(self._target_positions[:, :2], target_idx, axis=0)
        target_xy = self._sample_free_xy(
            self.TARGET_RADIUS_M,
            existing_xy=other_targets,
            existing_radii=np.full((other_targets.shape[0],), self.TARGET_RADIUS_M, dtype=np.float32),
            extra_xy=self._obstacle_positions_xy,
            extra_radii=self._obstacle_radii,
        )
        self._target_positions[target_idx, :] = np.array([target_xy[0], target_xy[1], self.TARGET_Z_M], dtype=np.float32)
        self._target_velocities_xy[target_idx, :] = self._sample_entity_velocity(self._target_speed_limit_mps())
        self._target_direction_steps[target_idx] = 0
        p.resetBasePositionAndOrientation(
            self._target_body_ids[target_idx],
            self._target_positions[target_idx, :].tolist(),
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT,
        )

    ################################################################################

    def _uav_pair_penalty(self, dist: float) -> float:
        collision_dist = 2.0 * self.UAV_RADIUS_M
        if dist <= collision_dist:
            return -20.0
        if dist < collision_dist + self.THREAT_ZONE_WIDTH_M:
            return -1.0 * (1.0 / (dist - collision_dist + 1.0) - 1.0 / (self.THREAT_ZONE_WIDTH_M + 1.0)) ** 2
        return 0.0

    def _obstacle_penalty(self, dist: float, obstacle_radius: float) -> float:
        min_safe_dist = self.UAV_RADIUS_M + obstacle_radius
        if dist <= min_safe_dist:
            return -20.0
        if dist < min_safe_dist + self.THREAT_ZONE_WIDTH_M:
            return -1.0 * (1.0 / (dist - min_safe_dist + 1.0) - 1.0 / (self.THREAT_ZONE_WIDTH_M + 1.0)) ** 2
        return 0.0

    def _target_attraction(self, dist: float) -> float:
        capture_dist = self.COVERAGE_RADIUS_M
        if dist < capture_dist + self.THREAT_ZONE_WIDTH_M:
            return 1.0 * (1.0 / (dist - capture_dist + 1.0) - 1.0 / (self.THREAT_ZONE_WIDTH_M + 1.0)) ** 2
        return 0.0

    def _boundary_penalty(self, dist: float) -> float:
        if dist <= 0:
            return -20.0
        if dist < self.THREAT_ZONE_WIDTH_M:
            return -1.0 * (1.0 / (dist + 1.0) - 1.0 / (self.THREAT_ZONE_WIDTH_M + 1.0)) ** 2
        return 0.0

    def _computeReward(self):
        rewards = np.zeros(self.NUM_DRONES, dtype=np.float32)

        # 目标覆盖奖励：归给最近的覆盖无人机
        if self._last_drone_captures is not None:
            rewards += 10.0 * self._last_drone_captures.astype(np.float32)

        # 目标吸引力：每个无人机对最近目标的 APF 引导奖励
        for drone_idx in range(self.NUM_DRONES):
            drone_pos = self.pos[drone_idx, :2]
            min_dist = float('inf')
            for target_idx in range(self.TARGET_COUNT):
                dist = float(np.linalg.norm(drone_pos - self._target_positions[target_idx, :2]))
                if dist < min_dist:
                    min_dist = dist
            rewards[drone_idx] += self._target_attraction(min_dist)

        # UAV 碰撞惩罚：双方各承担一半
        for i in range(self.NUM_DRONES - 1):
            for j in range(i + 1, self.NUM_DRONES):
                dist = float(np.linalg.norm(self.pos[i, :] - self.pos[j, :]))
                penalty = self._uav_pair_penalty(dist)
                rewards[i] += penalty / 2.0
                rewards[j] += penalty / 2.0

        # 障碍物惩罚：归给对应无人机
        for drone_idx in range(self.NUM_DRONES):
            drone_xy = self.pos[drone_idx, :2]
            for obstacle_idx in range(self.OBSTACLE_COUNT):
                dist = float(np.linalg.norm(drone_xy - self._obstacle_positions_xy[obstacle_idx, :]))
                rewards[drone_idx] += self._obstacle_penalty(dist, float(self._obstacle_radii[obstacle_idx]))

        # 边界惩罚：靠近边界的 APF 渐进惩罚（墙内表面在 ±half）
        half = self.ARENA_SIZE_XY_M / 2.0
        for drone_idx in range(self.NUM_DRONES):
            pos = self.pos[drone_idx, :]
            dist_to_boundary = min(
                half - abs(float(pos[0])) - self.UAV_RADIUS_M,
                half - abs(float(pos[1])) - self.UAV_RADIUS_M,
                float(pos[2]) - 0.05 - self.UAV_RADIUS_M,
                self.WALL_HEIGHT - float(pos[2]) - self.UAV_RADIUS_M,
            )
            rewards[drone_idx] += self._boundary_penalty(dist_to_boundary)

        self._last_drone_rewards = rewards
        return float(np.sum(rewards))

    ################################################################################

    def _count_uav_collisions(self) -> int:
        collisions = 0
        collision_dist = 2.0 * self.UAV_RADIUS_M
        for i in range(self.NUM_DRONES - 1):
            for j in range(i + 1, self.NUM_DRONES):
                if float(np.linalg.norm(self.pos[i, :] - self.pos[j, :])) <= collision_dist:
                    collisions += 1
        return collisions

    def _count_obstacle_collisions(self) -> int:
        collisions = 0
        for drone_idx in range(self.NUM_DRONES):
            drone_xy = self.pos[drone_idx, :2]
            for obstacle_idx in range(self.OBSTACLE_COUNT):
                min_safe_dist = self.UAV_RADIUS_M + float(self._obstacle_radii[obstacle_idx])
                if float(np.linalg.norm(drone_xy - self._obstacle_positions_xy[obstacle_idx, :])) <= min_safe_dist:
                    collisions += 1
        return collisions

    def _computeTerminated(self):
        self._last_uav_collision_count = self._count_uav_collisions()
        self._last_obstacle_collision_count = self._count_obstacle_collisions()
        self._last_out_of_bounds_count = self._count_out_of_bounds()
        if self._episode_target_captures >= self.COVERAGE_TARGET_COUNT:
            return True
        if self._last_uav_collision_count > 0:
            return True
        if self._last_obstacle_collision_count > 0:
            return True
        if self._last_out_of_bounds_count > 0:
            return True
        return False

    ################################################################################

    def _computeTruncated(self):
        if np.any(np.abs(self.rpy[:, 0]) > 1.0) or np.any(np.abs(self.rpy[:, 1]) > 1.0):
            return True
        return bool(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)

    ################################################################################

    def _computeInfo(self):
        coverage_ratio = min(1.0, float(self._episode_target_captures) / float(max(1, self.TARGET_COUNT)))
        return {
            "coverage_ratio": coverage_ratio,
            "target_capture_count": int(self._episode_target_captures),
            "coverage_target_count": int(self.COVERAGE_TARGET_COUNT),
            "num_drones": int(self.NUM_DRONES),
            "max_num_drones": int(self.MAX_NUM_DRONES),
            "target_count": int(self.TARGET_COUNT),
            "max_target_count": int(self.MAX_TARGET_COUNT),
            "obstacle_count": int(self.OBSTACLE_COUNT),
            "max_obstacle_count": int(self.MAX_OBSTACLE_COUNT),
            "team_new_cells": int(self._last_target_captures),
            "frontier_count": int(self.TARGET_COUNT),
            "uav_collision_count": int(self._last_uav_collision_count),
            "obstacle_collision_count": int(self._last_obstacle_collision_count),
            "out_of_bounds_count": int(self._last_out_of_bounds_count),
            "target_positions": self._target_positions.copy(),
            "obstacle_positions": self._obstacle_positions_xy.copy(),
            "adjacency": self._getAdjacencyMatrix().astype(np.float32),
            "drone_rewards": self._last_drone_rewards.copy() if self._last_drone_rewards is not None else np.zeros(self.NUM_DRONES, dtype=np.float32),
            "drone_captures": self._last_drone_captures.copy() if self._last_drone_captures is not None else np.zeros(self.NUM_DRONES, dtype=np.int32),
        }

    ################################################################################

    def _count_out_of_bounds(self) -> int:
        half = self.ARENA_SIZE_XY_M / 2.0
        r = self.UAV_RADIUS_M
        count = 0
        for i in range(self.NUM_DRONES):
            pos = self.pos[i, :]
            if (
                abs(float(pos[0])) + r >= half
                or abs(float(pos[1])) + r >= half
                or float(pos[2]) - r < 0.05
                or float(pos[2]) + r > self.WALL_HEIGHT
            ):
                count += 1
        return count

    ################################################################################

    def getTargetPositions(self):
        return self._target_positions.copy()

    def getObstaclePositions(self):
        return self._obstacle_positions_xy.copy()

    ################################################################################
    # Snapshot: save / restore full environment state
    ################################################################################

    def get_snapshot(self) -> dict:
        """Capture the complete mutable state of the environment.

        All state is saved in pure Python / NumPy — no ``p.saveState()`` is
        used, so snapshots are portable across PyBullet clients and don't
        leak memory.

        Returns a plain dict that can be stored externally (e.g. in a
        Go-Explore archive cell) and later restored with
        :meth:`restore_snapshot`.
        """
        return {
            # -- Python-cached kinematics --
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "quat": self.quat.copy(),
            "rpy": self.rpy.copy(),
            "ang_v": self.ang_v.copy(),
            "last_clipped_action": self.last_clipped_action.copy(),
            "rpy_rates": self.rpy_rates.copy() if hasattr(self, "rpy_rates") else None,
            # -- base counters / buffers --
            "step_counter": int(self.step_counter),
            "action_buffer": [np.array(a, copy=True) for a in self.action_buffer],
            # -- RNG (controls entity direction changes & respawn) --
            "rng_state": self._rng.get_state(),
            # -- target state --
            "target_positions": self._target_positions.copy(),
            "target_velocities_xy": self._target_velocities_xy.copy(),
            "target_direction_steps": self._target_direction_steps.copy(),
            # -- obstacle state --
            "obstacle_positions_xy": self._obstacle_positions_xy.copy(),
            "obstacle_velocities_xy": self._obstacle_velocities_xy.copy(),
            "obstacle_direction_steps": self._obstacle_direction_steps.copy(),
            "obstacle_radii": self._obstacle_radii.copy(),
            # -- episode bookkeeping --
            "episode_target_captures": int(self._episode_target_captures),
            "targets_spawned": int(self._targets_spawned),
            "last_target_captures": int(self._last_target_captures),
            "last_drone_captures": (
                self._last_drone_captures.copy()
                if self._last_drone_captures is not None
                else None
            ),
            "last_drone_rewards": (
                self._last_drone_rewards.copy()
                if self._last_drone_rewards is not None
                else None
            ),
            "episode_counter": int(self._episode_counter),
            # -- PID controller integrator state --
            "ctrl_states": [
                {
                    "control_counter": int(c.control_counter),
                    "last_rpy": c.last_rpy.copy(),
                    "last_pos_e": c.last_pos_e.copy(),
                    "integral_pos_e": c.integral_pos_e.copy(),
                    "last_rpy_e": c.last_rpy_e.copy(),
                    "integral_rpy_e": c.integral_rpy_e.copy(),
                }
                for c in self.ctrl
            ] if hasattr(self, "ctrl") else [],
        }

    def restore_snapshot(self, snapshot: dict) -> None:
        """Restore environment state from a snapshot dict.

        All PyBullet bodies are repositioned manually via
        ``p.resetBasePositionAndOrientation`` / ``p.resetBaseVelocity``,
        so the snapshot is portable across PyBullet clients and sessions.

        Parameters
        ----------
        snapshot : dict
            A dict previously returned by :meth:`get_snapshot`.
        """
        # -- restore Python-cached kinematics --
        self.pos = snapshot["pos"].copy()
        self.vel = snapshot["vel"].copy()
        self.quat = snapshot["quat"].copy()
        self.rpy = snapshot["rpy"].copy()
        self.ang_v = snapshot["ang_v"].copy()
        self.last_clipped_action = snapshot["last_clipped_action"].copy()
        if snapshot.get("rpy_rates") is not None and hasattr(self, "rpy_rates"):
            self.rpy_rates = snapshot["rpy_rates"].copy()

        # -- base counters / buffers --
        self.step_counter = snapshot["step_counter"]
        self.action_buffer = deque(
            [np.array(a, copy=True) for a in snapshot["action_buffer"]],
            maxlen=self.ACTION_BUFFER_SIZE,
        )

        # -- RNG --
        self._rng.set_state(snapshot["rng_state"])

        # -- target state --
        self._target_positions = snapshot["target_positions"].copy()
        self._target_velocities_xy = snapshot["target_velocities_xy"].copy()
        self._target_direction_steps = snapshot["target_direction_steps"].copy()

        # -- obstacle state --
        self._obstacle_positions_xy = snapshot["obstacle_positions_xy"].copy()
        self._obstacle_velocities_xy = snapshot["obstacle_velocities_xy"].copy()
        self._obstacle_direction_steps = snapshot["obstacle_direction_steps"].copy()
        self._obstacle_radii = snapshot["obstacle_radii"].copy()

        # -- episode bookkeeping --
        self._episode_target_captures = snapshot["episode_target_captures"]
        self._targets_spawned = snapshot["targets_spawned"]
        self._last_target_captures = snapshot["last_target_captures"]
        self._last_drone_captures = (
            snapshot["last_drone_captures"].copy()
            if snapshot["last_drone_captures"] is not None
            else None
        )
        self._last_drone_rewards = (
            snapshot["last_drone_rewards"].copy()
            if snapshot["last_drone_rewards"] is not None
            else None
        )
        self._episode_counter = snapshot["episode_counter"]

        # -- PID controller integrator state --
        ctrl_states = snapshot.get("ctrl_states", [])
        if hasattr(self, "ctrl") and ctrl_states:
            for c, cs in zip(self.ctrl, ctrl_states):
                c.control_counter = cs["control_counter"]
                c.last_rpy = cs["last_rpy"].copy()
                c.last_pos_e = cs["last_pos_e"].copy()
                c.integral_pos_e = cs["integral_pos_e"].copy()
                c.last_rpy_e = cs["last_rpy_e"].copy()
                c.integral_rpy_e = cs["integral_rpy_e"].copy()

        # -- sync PyBullet bodies from restored Python state --
        for i in range(self.NUM_DRONES):
            p.resetBasePositionAndOrientation(
                self.DRONE_IDS[i],
                self.pos[i].tolist(),
                self.quat[i].tolist(),
                physicsClientId=self.CLIENT,
            )
            p.resetBaseVelocity(
                self.DRONE_IDS[i],
                self.vel[i].tolist(),
                self.ang_v[i].tolist(),
                physicsClientId=self.CLIENT,
            )
        for idx in range(self.OBSTACLE_COUNT):
            if idx < len(self._obstacle_body_ids):
                p.resetBasePositionAndOrientation(
                    self._obstacle_body_ids[idx],
                    [float(self._obstacle_positions_xy[idx, 0]),
                     float(self._obstacle_positions_xy[idx, 1]),
                     self.OBSTACLE_Z_M],
                    [0, 0, 0, 1],
                    physicsClientId=self.CLIENT,
                )
        for idx in range(self.TARGET_COUNT):
            if idx < len(self._target_body_ids):
                p.resetBasePositionAndOrientation(
                    self._target_body_ids[idx],
                    self._target_positions[idx].tolist(),
                    [0, 0, 0, 1],
                    physicsClientId=self.CLIENT,
                )