"""GridControl: velocity-limited outer loop + DSLPIDControl inner loop.

This follows the idea in your `stable_controller.py`:
- Do NOT jump directly between position waypoints.
- Convert waypoint tracking into a *bounded velocity* command.
- Integrate that velocity to form a smooth virtual target position/yaw.
- Feed-forward the target velocity into DSLPIDControl.

This makes the first few steps much less likely to drift/flip.
"""

from __future__ import annotations

import math
import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


def _wrap_pi(angle: float) -> float:
	"""Wrap angle to [-pi, pi]."""
	return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class GridControl(BaseControl):
	"""Waypoint follower using a cmd_vel-style outer loop."""

	def __init__(self, drone_model: DroneModel, g: float = 9.8):
		# BaseControl.__init__ calls self.reset(); initialize attributes used by reset() first.
		self._pid = None
		self._sp_pos = None
		self._sp_yaw = 0.0
		self._last_sp_pos = None
		self._last_rpm = None

		super().__init__(drone_model=drone_model, g=g)
		if self.DRONE_MODEL not in (DroneModel.CF2X, DroneModel.CF2P):
			raise ValueError("GridControl requires DroneModel.CF2X or DroneModel.CF2P")

		self._pid = DSLPIDControl(drone_model=drone_model, g=g)

		# Limits (similar spirit to your stable_controller).
		self.MAX_VEL_XY = 1.0  # m/s
		self.MAX_VEL_Z = 0.6  # m/s
		self.MAX_YAW_RATE = 0.8  # rad/s

		# Keep the virtual setpoint within the 10x10 arena (default in GridAviary).
		self.XY_LIMIT = 4.8
		self.Z_MIN = 0.15
		self.Z_MAX = 2.0

		self.reset()

	def reset(self):
		super().reset()
		if self._pid is not None:
			self._pid.reset()
		self._sp_pos = None
		self._sp_yaw = 0.0
		self._last_sp_pos = None
		self._last_rpm = None

	def computeControl(
		self,
		control_timestep,
		cur_pos,
		cur_quat,
		cur_vel,
		cur_ang_vel,
		target_pos,
		target_rpy=np.zeros(3),
		target_vel=np.zeros(3),
		target_rpy_rates=np.zeros(3),
	):
		"""Compute motor RPMs for a single drone.

		Inputs are treated as *waypoint commands* (position + desired yaw).
		The controller internally generates a smooth virtual setpoint.
		"""
		self.control_counter += 1

		dt = float(control_timestep)
		dt = max(dt, 1e-4)

		cur_pos = np.asarray(cur_pos, dtype=np.float32).reshape(3)
		cur_vel = np.asarray(cur_vel, dtype=np.float32).reshape(3)
		cur_ang_vel = np.asarray(cur_ang_vel, dtype=np.float32).reshape(3)
		cmd_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
		cmd_rpy = np.asarray(target_rpy, dtype=np.float32).reshape(3)

		cur_yaw = float(p.getEulerFromQuaternion(cur_quat)[2])
		cmd_yaw = float(cmd_rpy[2])

		# Initialize setpoint at current state to avoid first-step jumps.
		if self._sp_pos is None:
			self._sp_pos = cur_pos.copy()
			self._last_sp_pos = cur_pos.copy()
			self._sp_yaw = cur_yaw
			self._last_rpm = None

		# Compute desired velocity towards the commanded waypoint.
		delta = cmd_pos - self._sp_pos
		vel_cmd = (delta / dt).astype(np.float32)

		# Limit horizontal speed by norm.
		vxy = float(np.linalg.norm(vel_cmd[:2]))
		if vxy > self.MAX_VEL_XY and vxy > 1e-9:
			vel_cmd[0:2] *= float(self.MAX_VEL_XY / vxy)
		vel_cmd[2] = float(np.clip(vel_cmd[2], -self.MAX_VEL_Z, self.MAX_VEL_Z))

		# Integrate to update smooth virtual position setpoint.
		self._sp_pos = (self._sp_pos + vel_cmd * dt).astype(np.float32)
		self._sp_pos[0] = float(np.clip(self._sp_pos[0], -self.XY_LIMIT, self.XY_LIMIT))
		self._sp_pos[1] = float(np.clip(self._sp_pos[1], -self.XY_LIMIT, self.XY_LIMIT))
		self._sp_pos[2] = float(np.clip(self._sp_pos[2], self.Z_MIN, self.Z_MAX))

		# Yaw setpoint: limit yaw rate.
		yaw_err = _wrap_pi(cmd_yaw - self._sp_yaw)
		yaw_rate_cmd = float(np.clip(yaw_err / dt, -self.MAX_YAW_RATE, self.MAX_YAW_RATE))
		self._sp_yaw = _wrap_pi(self._sp_yaw + yaw_rate_cmd * dt)

		# Feed-forward target velocity in world frame.
		sp_vel_world = (self._sp_pos - self._last_sp_pos) / dt
		self._last_sp_pos = self._sp_pos.copy()

		# Inner PID (DSLPIDControl) does the heavy lifting.
		rpm, pos_e, yaw_e = self._pid.computeControl(
			control_timestep=dt,
			cur_pos=cur_pos,
			cur_quat=cur_quat,
			cur_vel=cur_vel,
			cur_ang_vel=cur_ang_vel,
			target_pos=self._sp_pos,
			target_rpy=np.array([0.0, 0.0, self._sp_yaw], dtype=np.float32),
			target_vel=sp_vel_world.astype(np.float32),
			target_rpy_rates=np.array([0.0, 0.0, yaw_rate_cmd], dtype=np.float32),
		)

		# Optional: limit instantaneous RPM changes (helps with "first step" weirdness).
		rpm = np.asarray(rpm, dtype=np.float32).reshape(4)
		# if self._last_rpm is None:
		# 	self._last_rpm = rpm.copy()
		# else:
		# 	max_drpm = 8000.0 * dt
		# 	rpm = np.clip(rpm, self._last_rpm - max_drpm, self._last_rpm + max_drpm)
		# 	self._last_rpm = rpm.copy()

		return rpm.astype(np.float32), np.asarray(pos_e, dtype=np.float32), float(yaw_e)
