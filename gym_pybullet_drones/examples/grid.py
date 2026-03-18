"""Example: GridAviary 10x10m enclosed arena + 2D planar lidar + coverage grid.

This script uses `DSLPIDControl` to generate RPM actions, similarly to
`gym_pybullet_drones/examples/pid.py`.

Run:

    python gym_pybullet_drones/examples/grid.py

"""

import time
import math
import numpy as np

from gym_pybullet_drones.envs.GridAviary import GridAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.GridControl import GridControl
from gym_pybullet_drones.utils.utils import sync


DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 20


def make_lawnmower_waypoints(
    arena_size_xy_m: float,
    margin_m: float,
    spacing_m: float,
    z_m: float,
    obstacle_centers_xy: np.ndarray,
    obstacle_size_xy_m: float,
    obstacle_buffer_m: float,
) -> np.ndarray:
    """Generate a simple back-and-forth sweep that covers most of a square arena."""
    half = arena_size_xy_m / 2.0
    x_min = -half + margin_m + 2
    x_max = +half - margin_m - 2
    y_min = -half + margin_m
    y_max = +half - margin_m

    if spacing_m <= 0:
        raise ValueError("spacing_m must be > 0")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("margin_m too large for the arena")

    # Sweep along rows inside the boundary, not exactly on it, so the planner
    # always has room to detour without leaving the allowed region.
    y0 = y_min + 0.5 * spacing_m
    y1 = y_max - 0.5 * spacing_m
    if y1 < y0:
        y0 = (y_min + y_max) / 2.0
        y1 = y0
    ys = np.arange(y0, y1 + 1e-6, spacing_m, dtype=np.float32)
    waypoints = []
    left_to_right = True
    obs_half = (obstacle_size_xy_m / 2.0) + obstacle_buffer_m

    def _is_free(x: float, y: float) -> bool:
        if obstacle_centers_xy is None or obstacle_centers_xy.size == 0:
            return True
        dx = np.abs(obstacle_centers_xy[:, 0] - x)
        dy = np.abs(obstacle_centers_xy[:, 1] - y)
        # Axis-aligned keepout around each box.
        return bool(np.all((dx > obs_half) | (dy > obs_half)))

    for y in ys.tolist():
        if left_to_right:
            if _is_free(x_min, y):
                waypoints.append([x_min, y, z_m])
            if _is_free(x_max, y):
                waypoints.append([x_max, y, z_m])
        else:
            if _is_free(x_max, y):
                waypoints.append([x_max, y, z_m])
            if _is_free(x_min, y):
                waypoints.append([x_min, y, z_m])
        left_to_right = not left_to_right
    return np.asarray(waypoints, dtype=np.float32)


def densify_polyline_xy(poly_xyz: np.ndarray, step_m: float) -> np.ndarray:
    """Densify a polyline by inserting points every ~step_m (in XY)."""
    if poly_xyz.shape[0] < 2:
        return poly_xyz[:, :2].copy()
    if step_m <= 0:
        raise ValueError("step_m must be > 0")

    pts_xy = []
    for i in range(poly_xyz.shape[0] - 1):
        a = poly_xyz[i, :2].astype(np.float32)
        b = poly_xyz[i + 1, :2].astype(np.float32)
        seg = b - a
        dist = float(np.linalg.norm(seg))
        if dist < 1e-9:
            continue
        n = max(1, int(math.ceil(dist / step_m)))
        for k in range(n):
            t = float(k) / float(n)
            p = a * (1.0 - t) + b * t
            pts_xy.append([float(p[0]), float(p[1])])
    pts_xy.append([float(poly_xyz[-1, 0]), float(poly_xyz[-1, 1])])
    return np.asarray(pts_xy, dtype=np.float32)


def _build_occupancy_grid(
    arena_size_xy_m: float,
    margin_m: float,
    grid_res_m: float,
    obstacle_centers_xy: np.ndarray,
    obstacle_size_xy_m: float,
    obstacle_buffer_m: float,
) -> tuple[np.ndarray, dict]:
    """Build a boolean occupancy grid (True=blocked) with wall margin + expanded obstacles."""
    half = arena_size_xy_m / 2.0
    x_min = -half
    x_max = +half
    y_min = -half
    y_max = +half

    cols = int(math.ceil((x_max - x_min) / grid_res_m))
    rows = int(math.ceil((y_max - y_min) / grid_res_m))
    occ = np.zeros((rows, cols), dtype=bool)

    # Block outside the allowed inner region (keep margin from walls).
    inner_x_min = -half + margin_m
    inner_x_max = +half - margin_m
    inner_y_min = -half + margin_m
    inner_y_max = +half - margin_m

    xs = x_min + (np.arange(cols, dtype=np.float32) + 0.5) * grid_res_m
    ys = y_min + (np.arange(rows, dtype=np.float32) + 0.5) * grid_res_m
    X, Y = np.meshgrid(xs, ys)
    occ |= (X < inner_x_min) | (X > inner_x_max) | (Y < inner_y_min) | (Y > inner_y_max)

    # Block expanded obstacle rectangles.
    if obstacle_centers_xy is not None and obstacle_centers_xy.size > 0:
        obs_half = (obstacle_size_xy_m / 2.0) + obstacle_buffer_m
        for ox, oy in obstacle_centers_xy.astype(np.float32):
            occ |= (np.abs(X - ox) <= obs_half) & (np.abs(Y - oy) <= obs_half)

    meta = {
        "x_min": x_min,
        "y_min": y_min,
        "grid_res_m": grid_res_m,
        "rows": rows,
        "cols": cols,
    }
    return occ, meta


def _world_to_grid(xy: np.ndarray, meta: dict) -> tuple[int, int]:
    x_min = float(meta["x_min"])
    y_min = float(meta["y_min"])
    res = float(meta["grid_res_m"])
    cols = int(meta["cols"])
    rows = int(meta["rows"])
    c = int((float(xy[0]) - x_min) / res)
    r = int((float(xy[1]) - y_min) / res)
    c = int(np.clip(c, 0, cols - 1))
    r = int(np.clip(r, 0, rows - 1))
    return r, c


def _grid_to_world(rc: tuple[int, int], meta: dict) -> np.ndarray:
    r, c = rc
    x_min = float(meta["x_min"])
    y_min = float(meta["y_min"])
    res = float(meta["grid_res_m"])
    x = x_min + (c + 0.5) * res
    y = y_min + (r + 0.5) * res
    return np.array([x, y], dtype=np.float32)


def _astar(occ: np.ndarray, start_rc: tuple[int, int], goal_rc: tuple[int, int]) -> list[tuple[int, int]]:
    """A* on a 2D grid with 8-connectivity. occ=True means blocked."""
    rows, cols = occ.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    if occ[sr, sc] or occ[gr, gc]:
        return []

    def h(r: int, c: int) -> float:
        return math.hypot(gr - r, gc - c)

    # Neighbors: (dr, dc, cost)
    nbrs = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]

    import heapq

    open_heap = []
    heapq.heappush(open_heap, (h(sr, sc), 0.0, (sr, sc)))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    gscore = { (sr, sc): 0.0 }
    closed = set()

    while open_heap:
        _, gcur, (r, c) = heapq.heappop(open_heap)
        if (r, c) in closed:
            continue
        if (r, c) == (gr, gc):
            # Reconstruct
            path = [(r, c)]
            while (r, c) in came_from:
                (r, c) = came_from[(r, c)]
                path.append((r, c))
            path.reverse()
            return path

        closed.add((r, c))
        for dr, dc, step_cost in nbrs:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            if occ[rr, cc]:
                continue

            # Prevent corner-cutting through diagonal gaps.
            if dr != 0 and dc != 0:
                if occ[r, cc] or occ[rr, c]:
                    continue

            ng = gcur + step_cost
            if ng < gscore.get((rr, cc), float("inf")):
                gscore[(rr, cc)] = ng
                came_from[(rr, cc)] = (r, c)
                heapq.heappush(open_heap, (ng + h(rr, cc), ng, (rr, cc)))

    return []


def plan_path_xy_via_astar(
    goals_xy: np.ndarray,
    occ: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Plan a continuous XY path that visits the goals in order, using A* per segment."""
    if goals_xy.shape[0] < 2:
        return goals_xy.astype(np.float32)

    def _nearest_free(rc: tuple[int, int], max_radius: int = 30) -> tuple[int, int] | None:
        r0, c0 = rc
        if not occ[r0, c0]:
            return (r0, c0)
        rows, cols = occ.shape
        # BFS expanding outward
        from collections import deque

        q = deque()
        q.append((r0, c0))
        seen = set([(r0, c0)])
        while q:
            r, c = q.popleft()
            if not occ[r, c]:
                return (r, c)
            if abs(r - r0) > max_radius or abs(c - c0) > max_radius:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        continue
                    if (rr, cc) in seen:
                        continue
                    seen.add((rr, cc))
                    q.append((rr, cc))
        return None

    path_xy = []

    # Snap the first start to a free cell.
    start_rc_raw = _world_to_grid(goals_xy[0], meta)
    start_rc = _nearest_free(start_rc_raw)
    if start_rc is None:
        raise RuntimeError("A* start is inside blocked region (no nearby free cell)")

    for i in range(goals_xy.shape[0] - 1):
        b = goals_xy[i + 1]
        goal_rc_raw = _world_to_grid(b, meta)
        goal_rc = _nearest_free(goal_rc_raw)
        if goal_rc is None:
            raise RuntimeError(f"A* goal {i+1} is inside blocked region (no nearby free cell)")

        rc_path = _astar(occ, start_rc, goal_rc)
        if not rc_path:
            raise RuntimeError(f"A* failed between goal {i} and {i+1}")
        seg_xy = np.stack([_grid_to_world(rc, meta) for rc in rc_path], axis=0)
        if i > 0:
            seg_xy = seg_xy[1:, :]
        path_xy.append(seg_xy)
        start_rc = goal_rc
    return np.concatenate(path_xy, axis=0).astype(np.float32)


def run(
    drone=DEFAULT_DRONE,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
):
    env = GridAviary(
        drone_model=drone,
        num_drones=1,
        physics=physics,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=False,
        user_debug_gui=False,
        arena_size_xy_m=10.0,
        wall_height=2.0,
        wall_thickness=0.05,
        lidar_range_m=2.0,
        lidar_num_rays=72,
        coverage_grid_resolution_m=0.2,
        coverage_include_in_obs=False,
        visualize_scan=True,
        visualize_coverage=True,
        print_target_pos=True,
        print_target_every_n_steps=10,
    )

    ctrl = GridControl(drone_model=drone)

    start = time.time()
    obs, info = env.reset()

    # Keep altitude constant (do not command z changes).
    z_hold = float(obs[0][2])

    # Match pid.py semantics: keep yaw fixed (do not steer yaw toward waypoints).
    # State vector format is (20,) = pos(3) + quat(4) + rpy(3) + vel(3) + ang_v(3) + last_rpm(4).
    yaw_hold = float(obs[0][9])

    # Clarify the coordinate frame: GridAviary spawns at z0=wall_height*0.25 (here 0.5m), ground is z=0.
    print(f"Initial altitude z_hold={z_hold:.3f} m (GridAviary default spawn height)")

    # Plan after reset so obstacle positions are available.
    waypoint_spacing_m = 1.0
    # Larger margin reduces wall hits due to tracking overshoot.
    waypoint_margin_m = 1.5
    obstacle_buffer_m = 0.6
    obstacle_centers_xy = env.getObstaclePositions()
    waypoints = make_lawnmower_waypoints(
        arena_size_xy_m=10.0,
        margin_m=waypoint_margin_m,
        spacing_m=waypoint_spacing_m,
        z_m=0.0,
        obstacle_centers_xy=obstacle_centers_xy,
        obstacle_size_xy_m=1.0,
        obstacle_buffer_m=obstacle_buffer_m,
    )
    if waypoints.shape[0] < 2:
        raise RuntimeError("Not enough waypoints after obstacle filtering")

    # Convert the sweep endpoints into an obstacle-avoiding path.
    plan_grid_res_m = 0.20
    occ, meta = _build_occupancy_grid(
        arena_size_xy_m=10.0,
        margin_m=waypoint_margin_m,
        grid_res_m=plan_grid_res_m,
        obstacle_centers_xy=obstacle_centers_xy,
        obstacle_size_xy_m=1.0,
        obstacle_buffer_m=obstacle_buffer_m,
    )
    # Important: include the current start position as the first goal.
    # Otherwise the first target will jump to the first sweep waypoint (near a corner).
    start_xy = obs[0][0:2].astype(np.float32)
    wp_xy = waypoints[:, :2].astype(np.float32)
    if wp_xy.shape[0] > 0 and float(np.linalg.norm(wp_xy[0] - start_xy)) < 1e-3:
        goals_xy = wp_xy
    else:
        goals_xy = np.vstack([start_xy.reshape(1, 2), wp_xy]).astype(np.float32)
    planned_xy = plan_path_xy_via_astar(goals_xy, occ, meta)
    # Force the very first planned point to be exactly the initial XY.
    # A* operates on grid cell centers, so otherwise planned_xy[0] can be offset by ~grid_res/2.
    if planned_xy.shape[0] > 0:
        planned_xy[0, :] = start_xy
    planned_xyz = np.hstack([planned_xy, np.zeros((planned_xy.shape[0], 1), dtype=np.float32)])

    # Dense target points: one target per control step.
    # IMPORTANT: with ctrl_freq=48Hz, max_target_step_m=0.10 implies ~4.8 m/s,
    # which is often too fast and looks like "flying the wrong way" due to lag/overshoot.
    desired_speed_mps = 1.0
    max_target_step_m = float(desired_speed_mps) / float(control_freq_hz)
    dense_step_m = max_target_step_m
    for _ in range(10):
        target_xy = densify_polyline_xy(planned_xyz, step_m=dense_step_m)
        if target_xy.shape[0] < 2:
            break
        diffs = np.diff(target_xy, axis=0)
        max_step = float(np.max(np.linalg.norm(diffs, axis=1))) if diffs.shape[0] > 0 else 0.0
        if max_step <= max_target_step_m * 1.01:
            break
        dense_step_m *= 0.5
    else:
        # Fall back: keep whatever we got after iterations.
        pass
    if target_xy.shape[0] < 2:
        raise RuntimeError("Not enough dense target points")

    # Match the target point count to the number of control steps.
    # This keeps the target sequence length deterministic (same as pid.py: 48Hz * 10s = 480).
    num_ctrl_steps = int(duration_sec * control_freq_hz)
    if num_ctrl_steps < 2:
        raise ValueError("duration_sec * control_freq_hz must be >= 2")
    if target_xy.shape[0] > num_ctrl_steps:
        target_xy = target_xy[:num_ctrl_steps, :]
    elif target_xy.shape[0] < num_ctrl_steps:
        pad_n = num_ctrl_steps - target_xy.shape[0]
        target_xy = np.vstack([target_xy, np.repeat(target_xy[-1:, :], pad_n, axis=0)]).astype(np.float32)

    diffs = np.diff(target_xy, axis=0)
    step_norms = np.linalg.norm(diffs, axis=1) if diffs.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
    max_step = float(np.max(step_norms)) if step_norms.shape[0] > 0 else 0.0
    min_step = float(np.min(step_norms)) if step_norms.shape[0] > 0 else 0.0
    start_to_tgt0 = float(np.linalg.norm(target_xy[0] - start_xy))
    print(
        f"Generated {target_xy.shape[0]} target points (ctrl_steps={num_ctrl_steps}, step_m={dense_step_m:.3f}, max_step={max_step:.3f}, max_allowed={max_target_step_m:.3f}, implied_speed<={max_target_step_m*control_freq_hz:.2f} m/s)"
    )
    print(f"start_xy=({start_xy[0]:+.3f},{start_xy[1]:+.3f})  tgt[0]=({target_xy[0,0]:+.3f},{target_xy[0,1]:+.3f})  dist={start_to_tgt0:.3f} m")
    if step_norms.shape[0] > 0:
        print(f"target step stats: min={min_step:.3f} m  max={max_step:.3f} m  mean={float(step_norms.mean()):.3f} m")
    preview_n = min(20, target_xy.shape[0])
    for k in range(preview_n):
        print(f"  tgt[{k:04d}] = ({target_xy[k,0]:+.2f}, {target_xy[k,1]:+.2f}, z={z_hold:.2f})")
    if target_xy.shape[0] > preview_n:
        print("  ...")
        for k in range(max(preview_n, target_xy.shape[0] - 5), target_xy.shape[0]):
            print(f"  tgt[{k:04d}] = ({target_xy[k,0]:+.2f}, {target_xy[k,1]:+.2f}, z={z_hold:.2f})")

    # Draw all target points in PyBullet as debug points (yellow).
    target_xyz = np.hstack([target_xy, np.full((target_xy.shape[0], 1), z_hold, dtype=np.float32)])
    env.setTargetPoints(0, target_xyz, color_rgb=(1.0, 1.0, 0.0))

    # Warm-up hover: avoid any first-step transient by holding position briefly.
    # This mirrors pid.py where the first waypoint equals the initial pose.
    warmup_steps = int(0.5 * control_freq_hz)
    warmup_steps = int(np.clip(warmup_steps, 0, int(duration_sec * control_freq_hz) - 1))
    hover_pos = np.array([start_xy[0], start_xy[1], z_hold], dtype=np.float32)

    try:
        for i in range(int(duration_sec * control_freq_hz)):
            # GridAviary obs is: state(20) + lidar + (optional) coverage; controller expects state(20)
            state = obs[0][:20]

            pos = state[0:3]
            rpy = state[7:10]

            if i < warmup_steps:
                target_pos = hover_pos
                tgt_idx = 0
            else:
                path_idx = min(i - warmup_steps, target_xy.shape[0] - 1)
                tgt_idx = int(path_idx)
                target_pos = np.array([target_xy[tgt_idx, 0], target_xy[tgt_idx, 1], z_hold], dtype=np.float32)

            # Debug: directly compare current position to the active target.
            # Print the first few steps and any large deviation events.
            pos_err = (target_pos - pos).astype(np.float32)
            pos_err_xy = float(np.linalg.norm(pos_err[0:2]))
            pos_err_xyz = float(np.linalg.norm(pos_err))
            yaw_err = float(yaw_hold - rpy[2])
            if i < 30 or pos_err_xyz > 0.75:
                mode = "WARMUP" if i < warmup_steps else "TRACK"
                print(
                    f"[DBG] i={i:04d} {mode} tgt_idx={tgt_idx:04d} "
                    f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
                    f"tgt=({target_pos[0]:+.3f},{target_pos[1]:+.3f},{target_pos[2]:+.3f}) "
                    f"err_xy={pos_err_xy:.3f} err_xyz={pos_err_xyz:.3f} yaw={rpy[2]:+.3f} yaw_err={yaw_err:+.3f}"
                )

            # Let the environment log/print the current target (debug).
            env.setTargetPos(0, target_pos)

            # Keep a fixed yaw (same intent as pid.py passing INIT_RPYS).
            target_rpy = np.array([0.0, 0.0, yaw_hold], dtype=np.float32)

            rpm, _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=state,
                target_pos=target_pos,
                target_rpy=target_rpy,
            )
            action = np.array([rpm], dtype=np.float32)

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"env.step() failed: {e}")
                break

            if i % 10 == 0:
                scan = env.getLidarScan2D(0)
                grid = env.getCoverageGrid()
                print(f"step={i:04d} scan_min={scan.min():.3f}m covered_cells={int(grid.sum())}")

            if gui:
                sync(i, start, env.CTRL_TIMESTEP)

            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    run()
