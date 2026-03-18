import numpy as np
import pybullet as p
from gymnasium import spaces
from typing import Optional

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


class GridRLAviary(BaseRLAviary):
    """Single-agent RL environment for lidar-based coverage in a walled grid arena."""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
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
        lidar_range_m: float = 2.0,
        lidar_num_rays: int = 72,
        coverage_grid_resolution_m: float = 0.2,
        coverage_include_in_obs: bool = False,
        obstacle_count: int = 10,
        obstacle_size_xy_m: float = 1.0,
        obstacle_height_m: Optional[float] = None,
        obstacle_seed: int = 101,
        randomize_obstacles_on_reset: bool = True,
        obstacle_keepout_center_m: float = 1.5,
        max_episode_len_sec: float = 30.0,
        coverage_completion_ratio: float = 0.95,
        collision_distance_m: float = 0.10,
        visualize_scan: bool = True,
        visualize_coverage: bool = False,
    ):
        if obs != ObservationType.KIN:
            raise ValueError("GridRLAviary currently supports ObservationType.KIN only")

        self.ARENA_SIZE_XY_M = float(arena_size_xy_m)
        self.WALL_HEIGHT = float(wall_height)
        self.WALL_THICKNESS = float(wall_thickness)
        self.LIDAR_RANGE_M = float(lidar_range_m)
        self.LIDAR_NUM_RAYS = int(lidar_num_rays)
        self.COVERAGE_GRID_RES_M = float(coverage_grid_resolution_m)
        self.COVERAGE_INCLUDE_IN_OBS = bool(coverage_include_in_obs)
        self.OBSTACLE_COUNT = int(obstacle_count)
        self.OBSTACLE_SIZE_XY_M = float(obstacle_size_xy_m)
        self.OBSTACLE_HEIGHT_M = float(obstacle_height_m if obstacle_height_m is not None else wall_height)
        self.BASE_OBSTACLE_SEED = int(obstacle_seed)
        self.OBSTACLE_SEED = int(obstacle_seed)
        self.RANDOMIZE_OBSTACLES_ON_RESET = bool(randomize_obstacles_on_reset)
        self.OBSTACLE_KEEPOUT_CENTER_M = float(obstacle_keepout_center_m)
        self.EPISODE_LEN_SEC = float(max_episode_len_sec)
        self.COVERAGE_COMPLETION_RATIO = float(coverage_completion_ratio)
        self.COLLISION_DISTANCE_M = float(collision_distance_m)
        self.VIS_SCAN = bool(visualize_scan)
        self.VIS_COVERAGE = bool(visualize_coverage)

        if self.ARENA_SIZE_XY_M <= 0:
            raise ValueError("arena_size_xy_m must be > 0")
        if self.WALL_HEIGHT <= 0:
            raise ValueError("wall_height must be > 0")
        if self.WALL_THICKNESS <= 0:
            raise ValueError("wall_thickness must be > 0")
        if self.LIDAR_RANGE_M <= 0:
            raise ValueError("lidar_range_m must be > 0")
        if self.LIDAR_NUM_RAYS < 8:
            raise ValueError("lidar_num_rays must be >= 8")
        if self.COVERAGE_GRID_RES_M <= 0:
            raise ValueError("coverage_grid_resolution_m must be > 0")
        if self.OBSTACLE_COUNT < 0:
            raise ValueError("obstacle_count must be >= 0")
        if self.OBSTACLE_SIZE_XY_M <= 0:
            raise ValueError("obstacle_size_xy_m must be > 0")
        if self.OBSTACLE_HEIGHT_M <= 0:
            raise ValueError("obstacle_height_m must be > 0")
        if self.EPISODE_LEN_SEC <= 0:
            raise ValueError("max_episode_len_sec must be > 0")
        if not 0 < self.COVERAGE_COMPLETION_RATIO <= 1.0:
            raise ValueError("coverage_completion_ratio must be in (0, 1]")
        if self.COLLISION_DISTANCE_M <= 0:
            raise ValueError("collision_distance_m must be > 0")

        half = self.ARENA_SIZE_XY_M / 2.0
        self.GRID_BOUNDS = ((-half, half), (-half, half))
        x_min, x_max = self.GRID_BOUNDS[0]
        y_min, y_max = self.GRID_BOUNDS[1]
        self.GRID_ROWS = int(np.ceil((y_max - y_min) / self.COVERAGE_GRID_RES_M))
        self.GRID_COLS = int(np.ceil((x_max - x_min) / self.COVERAGE_GRID_RES_M))
        self.MAP_SIZE = self.GRID_ROWS * self.GRID_COLS
        self.coverage_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.uint8)
        self._obstacle_positions_xy = np.zeros((0, 2), dtype=np.float32)
        self._ray_angles = np.linspace(0.0, 2.0 * np.pi, self.LIDAR_NUM_RAYS, endpoint=False).astype(np.float32)
        self._ray_vecs = np.stack([np.cos(self._ray_angles), np.sin(self._ray_angles)], axis=1).astype(np.float32)
        self._episode_counter = 0
        self._prev_covered_cells = 0
        self._last_new_covered_cells = 0
        self._last_scan_min = self.LIDAR_RANGE_M
        self._last_collision = False
        self._last_out_of_bounds = False

        if initial_xyzs is None:
            z0 = max(0.10, min(1.0, self.WALL_HEIGHT * 0.25))
            initial_xyzs = np.array([[0.0, 0.0, z0]], dtype=np.float32)

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            neighbourhood_radius=np.inf,
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

    ################################################################################

    def reset(self, seed: int = None, options: dict = None):
        if self.RANDOMIZE_OBSTACLES_ON_RESET:
            if seed is not None:
                self.OBSTACLE_SEED = int(seed)
            else:
                self.OBSTACLE_SEED = int(self.BASE_OBSTACLE_SEED + self._episode_counter)

        self.coverage_grid.fill(0)
        self._prev_covered_cells = 0
        self._last_new_covered_cells = 0
        self._last_scan_min = self.LIDAR_RANGE_M
        self._last_collision = False
        self._last_out_of_bounds = False
        self._clear_coverage_visuals()
        obs, info = super().reset(seed=seed, options=options)
        self._prev_covered_cells = int(self.coverage_grid.sum())
        self._episode_counter += 1
        return obs, info

    ################################################################################

    def _housekeeping(self):
        super()._housekeeping()
        self._dbg_scan_points = -1 * np.ones((self.NUM_DRONES,), dtype=np.int32)
        self._dbg_scan_lines = -1 * np.ones((self.NUM_DRONES, self.LIDAR_NUM_RAYS), dtype=np.int32)
        self._coverage_debug_items = []

    ################################################################################

    def _clear_coverage_visuals(self):
        if not hasattr(self, "_coverage_debug_items"):
            return
        if self.GUI:
            for item_id in self._coverage_debug_items:
                try:
                    p.removeUserDebugItem(item_id, physicsClientId=self.CLIENT)
                except Exception:
                    pass
        self._coverage_debug_items = []

    ################################################################################

    def _addObstacles(self):
        z = self.WALL_HEIGHT / 2.0
        half = self.ARENA_SIZE_XY_M / 2.0
        t = self.WALL_THICKNESS

        x_wall_half_extents = [half, t / 2.0, self.WALL_HEIGHT / 2.0]
        y_wall_half_extents = [t / 2.0, half, self.WALL_HEIGHT / 2.0]

        col_x = p.createCollisionShape(p.GEOM_BOX, halfExtents=x_wall_half_extents, physicsClientId=self.CLIENT)
        col_y = p.createCollisionShape(p.GEOM_BOX, halfExtents=y_wall_half_extents, physicsClientId=self.CLIENT)
        vis_x = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=x_wall_half_extents,
            rgbaColor=[0.6, 0.6, 0.6, 1.0],
            physicsClientId=self.CLIENT,
        )
        vis_y = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=y_wall_half_extents,
            rgbaColor=[0.6, 0.6, 0.6, 1.0],
            physicsClientId=self.CLIENT,
        )

        p.createMultiBody(0, col_x, vis_x, [0.0, +half, z], [0, 0, 0, 1], physicsClientId=self.CLIENT)
        p.createMultiBody(0, col_x, vis_x, [0.0, -half, z], [0, 0, 0, 1], physicsClientId=self.CLIENT)
        p.createMultiBody(0, col_y, vis_y, [+half, 0.0, z], [0, 0, 0, 1], physicsClientId=self.CLIENT)
        p.createMultiBody(0, col_y, vis_y, [-half, 0.0, z], [0, 0, 0, 1], physicsClientId=self.CLIENT)

        self._obstacle_positions_xy = np.zeros((0, 2), dtype=np.float32)
        if self.OBSTACLE_COUNT <= 0:
            return

        size = self.OBSTACLE_SIZE_XY_M
        h = self.OBSTACLE_HEIGHT_M
        obs_half_extents = [size / 2.0, size / 2.0, h / 2.0]
        col_o = p.createCollisionShape(p.GEOM_BOX, halfExtents=obs_half_extents, physicsClientId=self.CLIENT)
        vis_o = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=obs_half_extents,
            rgbaColor=[0.4, 0.4, 0.4, 1.0],
            physicsClientId=self.CLIENT,
        )

        rng = np.random.RandomState(self.OBSTACLE_SEED)
        margin = (size / 2.0) + self.WALL_THICKNESS + 0.05
        lo = -half + margin
        hi = +half - margin
        placed = []
        max_attempts = max(2000, self.OBSTACLE_COUNT * 500)
        for _ in range(max_attempts):
            if len(placed) >= self.OBSTACLE_COUNT:
                break
            ox = float(rng.uniform(lo, hi))
            oy = float(rng.uniform(lo, hi))
            if (ox * ox + oy * oy) < (self.OBSTACLE_KEEPOUT_CENTER_M ** 2):
                continue
            ok = True
            for px, py in placed:
                if (abs(ox - px) < size) and (abs(oy - py) < size):
                    ok = False
                    break
            if not ok:
                continue
            placed.append((ox, oy))

        if placed:
            self._obstacle_positions_xy = np.asarray(placed, dtype=np.float32)

        for ox, oy in placed:
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_o,
                baseVisualShapeIndex=vis_o,
                basePosition=[ox, oy, h / 2.0],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )

    ################################################################################

    def _observationSpace(self):
        half = self.ARENA_SIZE_XY_M / 2.0
        lo_state = np.array(
            [
                -half,
                -half,
                0.0,
                -np.pi,
                -np.pi,
                -np.pi,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
                -np.inf,
            ],
            dtype=np.float32,
        )
        hi_state = np.array(
            [
                half,
                half,
                self.WALL_HEIGHT,
                np.pi,
                np.pi,
                np.pi,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
            ],
            dtype=np.float32,
        )
        lo_scan = np.zeros(self.LIDAR_NUM_RAYS, dtype=np.float32)
        hi_scan = np.ones(self.LIDAR_NUM_RAYS, dtype=np.float32)
        lo_scalar = np.array([0.0], dtype=np.float32)
        hi_scalar = np.array([1.0], dtype=np.float32)

        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            act_dim = 4
        elif self.ACT_TYPE == ActionType.PID:
            act_dim = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            act_dim = 1
        else:
            raise ValueError("Unsupported action type")

        lo_action = -1.0 * np.ones(act_dim, dtype=np.float32)
        hi_action = 1.0 * np.ones(act_dim, dtype=np.float32)

        pieces_lo = [lo_state, lo_scan, lo_scalar, lo_action]
        pieces_hi = [hi_state, hi_scan, hi_scalar, hi_action]
        if self.COVERAGE_INCLUDE_IN_OBS:
            pieces_lo.append(np.zeros(self.MAP_SIZE, dtype=np.float32))
            pieces_hi.append(np.ones(self.MAP_SIZE, dtype=np.float32))

        return spaces.Box(
            low=np.hstack(pieces_lo).astype(np.float32),
            high=np.hstack(pieces_hi).astype(np.float32),
            dtype=np.float32,
        )

    ################################################################################

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        kin = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]]).astype(np.float32)
        scan, endpoints = self._lidar_scan_2d(0)
        self._update_coverage_from_scan(endpoints)
        coverage_ratio = np.array([float(self.coverage_grid.sum()) / float(self.MAP_SIZE)], dtype=np.float32)
        last_action = self.action_buffer[-1][0, :].astype(np.float32)
        pieces = [kin, scan.astype(np.float32), coverage_ratio, last_action]
        if self.COVERAGE_INCLUDE_IN_OBS:
            pieces.append(self.coverage_grid.reshape(-1).astype(np.float32))
        return np.hstack(pieces).astype(np.float32)

    ################################################################################

    def _lidar_scan_2d(self, nth_drone: int):
        pos = self.pos[nth_drone, :].astype(np.float32)
        yaw = float(self.rpy[nth_drone, 2])

        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        dirs_xy = (self._ray_vecs @ rot.T).astype(np.float32)
        dirs = np.hstack([dirs_xy, np.zeros((self.LIDAR_NUM_RAYS, 1), dtype=np.float32)])

        start = pos + dirs * 0.05
        end = pos + dirs * self.LIDAR_RANGE_M
        results = p.rayTestBatch(start.tolist(), end.tolist(), physicsClientId=self.CLIENT)

        dists = np.empty((self.LIDAR_NUM_RAYS,), dtype=np.float32)
        endpoints = np.empty((self.LIDAR_NUM_RAYS, 3), dtype=np.float32)
        colors = np.empty((self.LIDAR_NUM_RAYS, 3), dtype=np.float32)
        for k, res in enumerate(results):
            hit_uid = res[0]
            hit_fraction = float(res[2])
            frac = float(np.clip(hit_fraction, 0.0, 1.0))
            no_hit = (hit_uid < 0) or (hit_uid == self.DRONE_IDS[nth_drone]) or (frac >= 1.0)
            if no_hit:
                frac = 1.0
                colors[k] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                colors[k] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            endpoints[k] = start[k] + (end[k] - start[k]) * np.float32(frac)
            dists[k] = np.float32(frac)

        self._last_scan_min = float(dists.min()) * self.LIDAR_RANGE_M

        if self.GUI and self.VIS_SCAN:
            self._dbg_scan_points[nth_drone] = p.addUserDebugPoints(
                pointPositions=endpoints.tolist(),
                pointColorsRGB=colors.tolist(),
                pointSize=3,
                replaceItemUniqueId=int(self._dbg_scan_points[nth_drone]),
                physicsClientId=self.CLIENT,
            )
            for k in range(self.LIDAR_NUM_RAYS):
                self._dbg_scan_lines[nth_drone, k] = p.addUserDebugLine(
                    lineFromXYZ=start[k].tolist(),
                    lineToXYZ=endpoints[k].tolist(),
                    lineColorRGB=colors[k].tolist(),
                    lineWidth=1.0,
                    replaceItemUniqueId=int(self._dbg_scan_lines[nth_drone, k]),
                    physicsClientId=self.CLIENT,
                )

        return dists, endpoints

    ################################################################################

    def _grid_to_pos(self, c: int, r: int):
        x_min, _ = self.GRID_BOUNDS[0]
        y_min, _ = self.GRID_BOUNDS[1]
        x = x_min + c * self.COVERAGE_GRID_RES_M + self.COVERAGE_GRID_RES_M / 2.0
        y = y_min + r * self.COVERAGE_GRID_RES_M + self.COVERAGE_GRID_RES_M / 2.0
        return float(x), float(y)

    ################################################################################

    def _fill_polygon_mask(self, poly_xy: np.ndarray, mask: np.ndarray):
        if poly_xy.shape[0] < 3:
            return
        x = poly_xy[:, 0]
        y = poly_xy[:, 1]
        x_min = int(max(0, np.floor(np.min(x))))
        x_max = int(min(mask.shape[1] - 1, np.ceil(np.max(x))))
        y_min = int(max(0, np.floor(np.min(y))))
        y_max = int(min(mask.shape[0] - 1, np.ceil(np.max(y))))
        if x_max < x_min or y_max < y_min:
            return

        x_test, y_test = np.meshgrid(
            np.arange(x_min, x_max + 1, dtype=np.float32) + 0.5,
            np.arange(y_min, y_max + 1, dtype=np.float32) + 0.5,
        )
        inside = np.zeros_like(x_test, dtype=bool)
        j = poly_xy.shape[0] - 1
        eps = 1e-9
        for i in range(poly_xy.shape[0]):
            xi, yi = x[i], y[i]
            xj, yj = x[j], y[j]
            cond = ((yi > y_test) != (yj > y_test))
            x_intersect = (xj - xi) * (y_test - yi) / (yj - yi + eps) + xi
            inside ^= (cond & (x_test < x_intersect))
            j = i
        mask[y_min : y_max + 1, x_min : x_max + 1][inside] = 1

    ################################################################################

    def _update_coverage_from_scan(self, endpoints: np.ndarray):
        if endpoints.shape[0] < 3:
            self._last_new_covered_cells = 0
            return
        pos = self.pos[0, :]
        if pos[2] < 0.05:
            self._last_new_covered_cells = 0
            return

        x_min, _ = self.GRID_BOUNDS[0]
        y_min, _ = self.GRID_BOUNDS[1]
        poly = np.zeros((endpoints.shape[0], 2), dtype=np.float32)
        poly[:, 0] = (endpoints[:, 0] - x_min) / self.COVERAGE_GRID_RES_M
        poly[:, 1] = (endpoints[:, 1] - y_min) / self.COVERAGE_GRID_RES_M

        scan_mask = np.zeros_like(self.coverage_grid, dtype=np.uint8)
        self._fill_polygon_mask(poly, scan_mask)
        new_mask = (scan_mask == 1) & (self.coverage_grid == 0)
        self._last_new_covered_cells = int(new_mask.sum())
        if self._last_new_covered_cells == 0:
            return
        self.coverage_grid[new_mask] = 1

        if self.GUI and self.VIS_COVERAGE:
            rows, cols = np.where(new_mask)
            points = []
            colors = []
            for r, c in zip(rows.tolist(), cols.tolist()):
                x, y = self._grid_to_pos(c, r)
                points.append([x, y, 0.05])
                colors.append([0.0, 0.0, 1.0])
            if points:
                item_id = p.addUserDebugPoints(
                    pointPositions=points,
                    pointColorsRGB=colors,
                    pointSize=4,
                    physicsClientId=self.CLIENT,
                )
                self._coverage_debug_items.append(item_id)

    ################################################################################

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        covered_cells = int(self.coverage_grid.sum())
        new_cells = max(0, covered_cells - self._prev_covered_cells)
        self._prev_covered_cells = covered_cells

        coverage_reward = 2.0 * float(new_cells) / max(1.0, float(self.MAP_SIZE))
        clearance_bonus = 0.05 * min(1.0, self._last_scan_min / self.LIDAR_RANGE_M)
        tilt_penalty = 0.02 * (abs(float(state[7])) + abs(float(state[8])))
        angular_penalty = 0.002 * float(np.linalg.norm(state[13:16]))
        step_penalty = 0.01
        collision_penalty = 2.0 if self._has_collision() else 0.0
        out_of_bounds_penalty = 2.0 if self._is_out_of_bounds(state) else 0.0
        return coverage_reward + clearance_bonus - tilt_penalty - angular_penalty - step_penalty - collision_penalty - out_of_bounds_penalty

    ################################################################################

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if self._has_collision():
            return True
        if self._last_scan_min <= self.COLLISION_DISTANCE_M:
            return True
        if self._is_out_of_bounds(state):
            return True
        coverage_ratio = float(self.coverage_grid.sum()) / float(self.MAP_SIZE)
        return coverage_ratio >= self.COVERAGE_COMPLETION_RATIO

    ################################################################################

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if abs(float(state[7])) > 1.0 or abs(float(state[8])) > 1.0:
            return True
        return bool(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)

    ################################################################################

    def _computeInfo(self):
        covered_cells = int(self.coverage_grid.sum())
        return {
            "coverage_ratio": covered_cells / float(self.MAP_SIZE),
            "covered_cells": covered_cells,
            "new_covered_cells": int(self._last_new_covered_cells),
            "scan_min_m": float(self._last_scan_min),
            "near_collision": bool(self._last_scan_min <= self.COLLISION_DISTANCE_M),
            "collision": bool(self._last_collision),
            "out_of_bounds": bool(self._last_out_of_bounds),
            "obstacle_positions": self._obstacle_positions_xy.copy(),
        }

    ################################################################################

    def _has_collision(self):
        contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        self._last_collision = len(contacts) > 0
        return self._last_collision

    def _is_out_of_bounds(self, state: np.ndarray):
        half = self.ARENA_SIZE_XY_M / 2.0
        margin = self.WALL_THICKNESS + 0.05
        out = (
            abs(float(state[0])) > (half - margin)
            or abs(float(state[1])) > (half - margin)
            or float(state[2]) < 0.05
            or float(state[2]) > self.WALL_HEIGHT
        )
        self._last_out_of_bounds = bool(out)
        return self._last_out_of_bounds

    ################################################################################

    def getObstaclePositions(self):
        return self._obstacle_positions_xy.copy()

    def getLidarScan2D(self, nth_drone: int = 0):
        scan, _ = self._lidar_scan_2d(nth_drone)
        return scan.copy() * self.LIDAR_RANGE_M

    def getCoverageGrid(self):
        return self.coverage_grid.copy()