import numpy as np
import pybullet as p
from gymnasium import spaces
from typing import Optional

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class GridAviary(BaseAviary):
	"""Aviary with a 10x10m arena enclosed by walls + 2D lidar + coverage grid.

	- Arena: square (default 10m x 10m) enclosed by 4 static walls.
	- 2D lidar: rays cast on the XY plane (z = drone z), distances in meters.
	- Coverage grid: a global 2D grid over the arena, updated by filling the
	  lidar scan polygon; covered cells are shown as blue points in the GUI.

	The coverage logic is inspired by the provided `coverage_lidar_aviary.py`.
	"""

	################################################################################

	def __init__(
		self,
		drone_model: DroneModel = DroneModel.CF2X,
		num_drones: int = 1,
		neighbourhood_radius: float = np.inf,
		initial_xyzs=None,
		initial_rpys=None,
		physics: Physics = Physics.PYB,
		pyb_freq: int = 240,
		ctrl_freq: int = 240,
		gui: bool = False,
		record: bool = False,
		user_debug_gui: bool = True,
		output_folder: str = "results",
		arena_size_xy_m: float = 10.0,
		wall_height: float = 2.0,
		wall_thickness: float = 0.05,
		lidar_range_m: float = 2.0,
		lidar_num_rays: int = 72,
		coverage_grid_resolution_m: float = 0.2,
		coverage_include_in_obs: bool = False,
		visualize_scan: bool = True,
		visualize_coverage: bool = True,
		obstacle_count: int = 10,
		obstacle_size_xy_m: float = 1.0,
		obstacle_height_m: Optional[float] = None,
		obstacle_seed: int = 101,
		obstacle_keepout_center_m: float = 1.5,
		print_target_pos: bool = False,
		print_target_every_n_steps: int = 10,
		visualize_target_points: bool = True,
		target_point_size: float = 4.0,
	):
		# Will be populated in _addObstacles() after connecting to PyBullet.
		self._obstacle_positions_xy = np.zeros((0, 2), dtype=np.float32)
		self._target_pos = None
		self._grid_step_counter = 0

		self.ARENA_SIZE_XY_M = float(arena_size_xy_m)
		self.WALL_HEIGHT = float(wall_height)
		self.WALL_THICKNESS = float(wall_thickness)
		self.LIDAR_RANGE_M = float(lidar_range_m)
		self.LIDAR_NUM_RAYS = int(lidar_num_rays)
		self.COVERAGE_GRID_RES_M = float(coverage_grid_resolution_m)
		self.COVERAGE_INCLUDE_IN_OBS = bool(coverage_include_in_obs)
		self.VIS_SCAN = bool(visualize_scan)
		self.VIS_COVERAGE = bool(visualize_coverage)
		self.OBSTACLE_COUNT = int(obstacle_count)
		self.OBSTACLE_SIZE_XY_M = float(obstacle_size_xy_m)
		if obstacle_height_m is None:
			obstacle_height_m = wall_height
		self.OBSTACLE_HEIGHT_M = float(obstacle_height_m)
		self.OBSTACLE_SEED = int(obstacle_seed)
		self.OBSTACLE_KEEPOUT_CENTER_M = float(obstacle_keepout_center_m)
		self.PRINT_TARGET_POS = bool(print_target_pos)
		self.PRINT_TARGET_EVERY_N = int(print_target_every_n_steps)
		self.VIS_TARGET_POINTS = bool(visualize_target_points)
		self.TARGET_POINT_SIZE = float(target_point_size)

		if self.ARENA_SIZE_XY_M <= 0:
			raise ValueError("arena_size_xy_m must be > 0")
		if self.WALL_HEIGHT <= 0:
			raise ValueError("wall_height must be > 0")
		if self.WALL_THICKNESS <= 0:
			raise ValueError("wall_thickness must be > 0")
		if self.LIDAR_RANGE_M <= 0:
			raise ValueError("lidar_range_m must be > 0")
		if self.LIDAR_NUM_RAYS <= 2:
			raise ValueError("lidar_num_rays must be > 2")
		if self.COVERAGE_GRID_RES_M <= 0:
			raise ValueError("coverage_grid_resolution_m must be > 0")
		if self.OBSTACLE_COUNT < 0:
			raise ValueError("obstacle_count must be >= 0")
		if self.OBSTACLE_SIZE_XY_M <= 0:
			raise ValueError("obstacle_size_xy_m must be > 0")
		if self.OBSTACLE_HEIGHT_M <= 0:
			raise ValueError("obstacle_height_m must be > 0")
		if self.OBSTACLE_KEEPOUT_CENTER_M < 0:
			raise ValueError("obstacle_keepout_center_m must be >= 0")
		if self.PRINT_TARGET_EVERY_N <= 0:
			raise ValueError("print_target_every_n_steps must be > 0")

		# Global coverage grid bounds: match arena bounds.
		half = self.ARENA_SIZE_XY_M / 2.0
		self.GRID_BOUNDS = ((-half, half), (-half, half))
		x_min, x_max = self.GRID_BOUNDS[0]
		y_min, y_max = self.GRID_BOUNDS[1]
		self.GRID_ROWS = int(np.ceil((y_max - y_min) / self.COVERAGE_GRID_RES_M))
		self.GRID_COLS = int(np.ceil((x_max - x_min) / self.COVERAGE_GRID_RES_M))
		self.MAP_SIZE = self.GRID_ROWS * self.GRID_COLS
		self.coverage_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.uint8)

		# Precompute ray directions in sensor frame (unit vectors on XY plane).
		self._ray_angles = np.linspace(0.0, 2.0 * np.pi, self.LIDAR_NUM_RAYS, endpoint=False).astype(np.float32)
		self._ray_vecs = np.stack([np.cos(self._ray_angles), np.sin(self._ray_angles)], axis=1).astype(np.float32)

		if initial_xyzs is None:
			# Spawn drones near the arena center, slightly above the ground.
			z0 = max(0.10, min(1.0, self.WALL_HEIGHT * 0.25))
			side = self.ARENA_SIZE_XY_M
			margin = max(0.25, 0.05 * side)
			span = max(0.0, side - 2.0 * margin)
			offsets = []
			for i in range(num_drones):
				gx = (i % 3) - 1
				gy = (i // 3) - 1
				offsets.append([0.15 * gx, 0.15 * gy])
			initial_xyzs = np.array(
				[[np.clip(o[0], -span / 2.0, span / 2.0), np.clip(o[1], -span / 2.0, span / 2.0), z0] for o in offsets],
				dtype=np.float32,
			)

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
			obstacles=True,
			user_debug_gui=user_debug_gui,
			output_folder=output_folder,
		)

		# Target positions are set externally (e.g., by a controller script).
		self._target_pos = np.zeros((self.NUM_DRONES, 3), dtype=np.float32)
		self._target_points_xyz = [np.zeros((0, 3), dtype=np.float32) for _ in range(self.NUM_DRONES)]

	################################################################################

	def reset(self, seed: int = None, options: dict = None):
		# Reset coverage state + visuals, then reset physics.
		self._clear_coverage_visuals()
		self._clear_target_points_visuals()
		self.coverage_grid.fill(0)
		self._grid_step_counter = 0
		return super().reset(seed=seed, options=options)

	################################################################################

	def _housekeeping(self):
		"""Extends BaseAviary housekeeping with debug visualization handles."""
		super()._housekeeping()
		self._dbg_scan_points = -1 * np.ones((self.NUM_DRONES,), dtype=np.int32)
		self._dbg_scan_lines = -1 * np.ones((self.NUM_DRONES, self.LIDAR_NUM_RAYS), dtype=np.int32)
		self._coverage_debug_items = []
		self._dbg_target_points = -1 * np.ones((self.NUM_DRONES,), dtype=np.int32)

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

	def _clear_target_points_visuals(self):
		if not hasattr(self, "_dbg_target_points"):
			return
		if self.GUI:
			for i in range(self.NUM_DRONES):
				uid = int(self._dbg_target_points[i])
				if uid >= 0:
					try:
						p.removeUserDebugItem(uid, physicsClientId=self.CLIENT)
					except Exception:
						pass
		self._dbg_target_points = -1 * np.ones((self.NUM_DRONES,), dtype=np.int32)

	################################################################################

	def _addObstacles(self):
		"""Adds a square arena enclosed by 4 perimeter walls."""
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

		# Add N static obstacles of footprint 1m x 1m (default) inside the arena.
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

		# Sample non-overlapping positions; avoid a keep-out region around the center.
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
			for (px, py) in placed:
				# No-overlap for axis-aligned squares: require separation on at least one axis.
				if (abs(ox - px) < size) and (abs(oy - py) < size):
					ok = False
					break
			if not ok:
				continue
			placed.append((ox, oy))

		if len(placed) > 0:
			self._obstacle_positions_xy = np.asarray(placed, dtype=np.float32)

		for (ox, oy) in placed:
			p.createMultiBody(
				baseMass=0,
				baseCollisionShapeIndex=col_o,
				baseVisualShapeIndex=vis_o,
				basePosition=[ox, oy, h / 2.0],
				baseOrientation=[0, 0, 0, 1],
				physicsClientId=self.CLIENT,
			)

	################################################################################

	def getObstaclePositions(self):
		"""Returns obstacle centers in XY as (N,2) array in world frame."""
		return self._obstacle_positions_xy.copy()

	################################################################################

	def _actionSpace(self):
		act_lower_bound = np.array([[0.0, 0.0, 0.0, 0.0] for _ in range(self.NUM_DRONES)])
		act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for _ in range(self.NUM_DRONES)])
		return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

	################################################################################

	def _observationSpace(self):
		lo_state = np.array(
			[
				-np.inf,
				-np.inf,
				0.0,
				-1.0,
				-1.0,
				-1.0,
				-1.0,
				-np.pi,
				-np.pi,
				-np.pi,
				-np.inf,
				-np.inf,
				-np.inf,
				-np.inf,
				-np.inf,
				-np.inf,
				0.0,
				0.0,
				0.0,
				0.0,
			],
			dtype=np.float32,
		)
		hi_state = np.array(
			[
				np.inf,
				np.inf,
				np.inf,
				1.0,
				1.0,
				1.0,
				1.0,
				np.pi,
				np.pi,
				np.pi,
				np.inf,
				np.inf,
				np.inf,
				np.inf,
				np.inf,
				np.inf,
				self.MAX_RPM,
				self.MAX_RPM,
				self.MAX_RPM,
				self.MAX_RPM,
			],
			dtype=np.float32,
		)
		lo_scan = np.zeros(self.LIDAR_NUM_RAYS, dtype=np.float32)
		hi_scan = np.ones(self.LIDAR_NUM_RAYS, dtype=np.float32) * np.float32(self.LIDAR_RANGE_M)

		if self.COVERAGE_INCLUDE_IN_OBS:
			lo_map = np.zeros(self.MAP_SIZE, dtype=np.float32)
			hi_map = np.ones(self.MAP_SIZE, dtype=np.float32)
			obs_lower_bound = np.array([np.hstack([lo_state, lo_scan, lo_map]) for _ in range(self.NUM_DRONES)], dtype=np.float32)
			obs_upper_bound = np.array([np.hstack([hi_state, hi_scan, hi_map]) for _ in range(self.NUM_DRONES)], dtype=np.float32)
		else:
			obs_lower_bound = np.array([np.hstack([lo_state, lo_scan]) for _ in range(self.NUM_DRONES)], dtype=np.float32)
			obs_upper_bound = np.array([np.hstack([hi_state, hi_scan]) for _ in range(self.NUM_DRONES)], dtype=np.float32)
		return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

	################################################################################

	def _computeObs(self):
		self._grid_step_counter += 1
		obs = []
		for i in range(self.NUM_DRONES):
			state = self._getDroneStateVector(i).astype(np.float32)
			scan, endpoints = self._lidar_scan_2d(i)
			# Update coverage every step (and visualize if enabled).
			self._update_coverage_from_scan(i, endpoints)
			if self.COVERAGE_INCLUDE_IN_OBS:
				obs.append(np.hstack([state, scan.astype(np.float32), self.coverage_grid.reshape(-1).astype(np.float32)]))
			else:
				obs.append(np.hstack([state, scan.astype(np.float32)]))

		# Optional: print target positions from inside the environment.
		if self.PRINT_TARGET_POS and (self._grid_step_counter % self.PRINT_TARGET_EVERY_N == 0):
			try:
				if self._target_pos is not None and self._target_pos.shape[0] > 0:
					t = self._target_pos[0]
					print(f"[GridAviary] step={self._grid_step_counter} target_pos[0]=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f})")
			except Exception:
				pass
		return np.array(obs, dtype=np.float32)

	################################################################################

	def _lidar_scan_2d(self, nth_drone: int):
		"""Returns scan distances (N,) and ray endpoints (N,3) in world frame."""
		pos = self.pos[nth_drone, :].astype(np.float32)
		yaw = float(self.rpy[nth_drone, 2])
		max_dist = float(self.LIDAR_RANGE_M)

		c = float(np.cos(yaw))
		s = float(np.sin(yaw))
		R = np.array([[c, -s], [s, c]], dtype=np.float32)
		dirs_xy = (self._ray_vecs @ R.T).astype(np.float32)
		dirs = np.hstack([dirs_xy, np.zeros((self.LIDAR_NUM_RAYS, 1), dtype=np.float32)])

		start = pos + dirs * 0.05
		end = pos + dirs * max_dist
		results = p.rayTestBatch(start.tolist(), end.tolist(), physicsClientId=self.CLIENT)

		dists = np.empty((self.LIDAR_NUM_RAYS,), dtype=np.float32)
		endpoints = np.empty((self.LIDAR_NUM_RAYS, 3), dtype=np.float32)
		colors = np.empty((self.LIDAR_NUM_RAYS, 3), dtype=np.float32)
		for k, r in enumerate(results):
			hit_uid = r[0]
			hit_fraction = float(r[2])
			frac = float(np.clip(hit_fraction, 0.0, 1.0))
			no_hit = (hit_uid < 0) or (hit_uid == self.DRONE_IDS[nth_drone]) or (frac >= 1.0)
			if no_hit:
				frac = 1.0
				colors[k] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
			else:
				colors[k] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
			endpoints[k] = start[k] + (end[k] - start[k]) * np.float32(frac)
			dists[k] = np.float32(frac * max_dist)

		if self.GUI and self.VIS_SCAN:
			# Draw endpoints as a point cloud. Avoid calling with empty point list.
			if endpoints.shape[0] > 0:
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

	def _pos_to_grid(self, x: float, y: float):
		x_min, x_max = self.GRID_BOUNDS[0]
		y_min, y_max = self.GRID_BOUNDS[1]
		c = int((x - x_min) / self.COVERAGE_GRID_RES_M)
		r = int((y - y_min) / self.COVERAGE_GRID_RES_M)
		c = int(np.clip(c, 0, self.GRID_COLS - 1))
		r = int(np.clip(r, 0, self.GRID_ROWS - 1))
		return c, r

	def _grid_to_pos(self, c: int, r: int):
		x_min, x_max = self.GRID_BOUNDS[0]
		y_min, y_max = self.GRID_BOUNDS[1]
		x = x_min + c * self.COVERAGE_GRID_RES_M + self.COVERAGE_GRID_RES_M / 2.0
		y = y_min + r * self.COVERAGE_GRID_RES_M + self.COVERAGE_GRID_RES_M / 2.0
		return float(x), float(y)

	################################################################################

	def _fill_polygon_mask(self, poly_xy: np.ndarray, mask: np.ndarray):
		"""Fill polygon into mask using point-in-polygon ray casting.

		poly_xy: (N,2) in continuous grid coordinates (col,row)
		mask: (rows,cols) uint8, filled with 1 inside polygon.
		"""
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

		# Test points are cell centers in (col,row) space.
		X, Y = np.meshgrid(
			np.arange(x_min, x_max + 1, dtype=np.float32) + 0.5,
			np.arange(y_min, y_max + 1, dtype=np.float32) + 0.5,
		)
		inside = np.zeros_like(X, dtype=bool)
		j = poly_xy.shape[0] - 1
		eps = 1e-9
		for i in range(poly_xy.shape[0]):
			xi, yi = x[i], y[i]
			xj, yj = x[j], y[j]
			cond = ((yi > Y) != (yj > Y))
			x_intersect = (xj - xi) * (Y - yi) / (yj - yi + eps) + xi
			inside ^= (cond & (X < x_intersect))
			j = i
		mask[y_min : y_max + 1, x_min : x_max + 1][inside] = 1

	################################################################################

	def _update_coverage_from_scan(self, nth_drone: int, endpoints: np.ndarray):
		"""Update coverage grid by filling the scan polygon; visualize newly covered cells."""
		if endpoints.shape[0] < 3:
			return
		pos = self.pos[nth_drone, :]
		if pos[2] < 0.05:
			return

		# Build polygon vertices in grid coordinates (col,row).
		x_min, x_max = self.GRID_BOUNDS[0]
		y_min, y_max = self.GRID_BOUNDS[1]
		poly = np.zeros((endpoints.shape[0], 2), dtype=np.float32)
		poly[:, 0] = (endpoints[:, 0] - x_min) / self.COVERAGE_GRID_RES_M
		poly[:, 1] = (endpoints[:, 1] - y_min) / self.COVERAGE_GRID_RES_M

		scan_mask = np.zeros_like(self.coverage_grid, dtype=np.uint8)
		self._fill_polygon_mask(poly, scan_mask)
		new_mask = (scan_mask == 1) & (self.coverage_grid == 0)
		if not np.any(new_mask):
			return
		self.coverage_grid[new_mask] = 1

		if self.GUI and self.VIS_COVERAGE:
			rows, cols = np.where(new_mask)
			if rows.size == 0:
				return
			points = []
			colors = []
			for r, c in zip(rows.tolist(), cols.tolist()):
				x, y = self._grid_to_pos(c, r)
				points.append([x, y, 0.05])
				colors.append([0.0, 0.0, 1.0])
			if len(points) > 0:
				item_id = p.addUserDebugPoints(
					pointPositions=points,
					pointColorsRGB=colors,
					pointSize=4,
					physicsClientId=self.CLIENT,
				)
				self._coverage_debug_items.append(item_id)

	################################################################################

	def getLidarScan2D(self, nth_drone: int):
		"""Returns the 2D planar scan distances (N,)."""
		scan, _ = self._lidar_scan_2d(nth_drone)
		return scan.copy()

	def getCoverageGrid(self):
		"""Returns the global coverage grid over the arena (rows, cols), 0/1."""
		return self.coverage_grid.copy()

	################################################################################

	def setTargetPos(self, nth_drone: int, target_pos_xyz):
		"""Set the current target position (world frame) for debugging/logging."""
		if self._target_pos is None:
			self._target_pos = np.zeros((self.NUM_DRONES, 3), dtype=np.float32)
		t = np.asarray(target_pos_xyz, dtype=np.float32).reshape(3,)
		self._target_pos[int(nth_drone), :] = t

	def setTargetPoints(self, nth_drone: int, target_points_xyz, color_rgb=(1.0, 1.0, 0.0)):
		"""Set and (optionally) draw the full target point set as PyBullet debug points.

		This is meant for visualizing the planned path/targets (not just the current
		target). Requires GUI + visualize_target_points.
		"""
		i = int(nth_drone)
		pts = np.asarray(target_points_xyz, dtype=np.float32).reshape(-1, 3)
		self._target_points_xyz[i] = pts
		if not (self.GUI and self.VIS_TARGET_POINTS):
			return
		if pts.shape[0] == 0:
			return
		colors = np.tile(np.asarray(color_rgb, dtype=np.float32).reshape(1, 3), (pts.shape[0], 1))
		self._dbg_target_points[i] = p.addUserDebugPoints(
			pointPositions=pts.tolist(),
			pointColorsRGB=colors.tolist(),
			pointSize=float(self.TARGET_POINT_SIZE),
			replaceItemUniqueId=int(self._dbg_target_points[i]),
			physicsClientId=self.CLIENT,
		)

	################################################################################

	def _preprocessAction(self, action):
		return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

	def _computeReward(self):
		# Reward not used for this control-style env.
		return -1

	def _computeTerminated(self):
		return False

	def _computeTruncated(self):
		return False

	def _computeInfo(self):
		covered = int(np.sum(self.coverage_grid))
		info = {"coverage_ratio": covered / float(self.MAP_SIZE)}
		if self._target_pos is not None:
			info["target_pos"] = self._target_pos.copy()
		return info

