"""PettingZoo ParallelEnv wrapper for OurRLAviary.

Converts the centralized OurRLAviary (single Gymnasium env with multi-drone
obs/action arrays) into a PettingZoo ParallelEnv where each drone is an
independent agent with its own observation, action, and reward.

Usage:
    from gym_pybullet_drones.envs.OurRLAviary_PettingZoo import OurRLAviaryPZ

    env = OurRLAviaryPZ(num_drones=4)
    observations, infos = env.reset()
    # observations = {"drone_0": {...}, "drone_1": {...}, ...}
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from gym_pybullet_drones.envs.OurRLAviary import OurRLAviary


class OurRLAviaryPZ(ParallelEnv):
    """PettingZoo ParallelEnv wrapper around OurRLAviary.

    All drones act simultaneously (parallel API). Each agent ``"drone_i"``
    receives its own Dict observation, continuous action, and scalar reward.

    Parameters
    ----------
    All keyword arguments are forwarded to :class:`OurRLAviary`.
    """

    metadata = {"render_modes": ["human"], "name": "OurRLAviaryPZ-v0"}

    def __init__(self, **env_kwargs: Any):
        super().__init__()
        self._env = OurRLAviary(**env_kwargs)
        self._num_drones = self._env.NUM_DRONES

        # Agent naming
        self.possible_agents = [f"drone_{i}" for i in range(self._num_drones)]
        self.agents = list(self.possible_agents)
        self._agent_idx = {name: i for i, name in enumerate(self.possible_agents)}

        # Build per-agent spaces from the centralized env
        self._per_agent_obs_space = self._build_per_agent_obs_space()
        self._per_agent_act_space = self._build_per_agent_act_space()

    # ------------------------------------------------------------------
    # Space helpers
    # ------------------------------------------------------------------

    def _build_per_agent_obs_space(self) -> spaces.Dict:
        """Extract per-agent observation space from the (NUM_DRONES, dim) spaces."""
        central_space: spaces.Dict = self._env.observation_space
        per_agent = {}
        for key, box in central_space.spaces.items():
            # box.shape == (NUM_DRONES, feature_dim)
            feature_dim = box.shape[1]
            per_agent[key] = spaces.Box(
                low=box.low[0, :feature_dim].copy(),
                high=box.high[0, :feature_dim].copy(),
                shape=(feature_dim,),
                dtype=np.float32,
            )
        return spaces.Dict(per_agent)

    def _build_per_agent_act_space(self) -> spaces.Box:
        """Extract per-agent action space from the (NUM_DRONES, act_dim) space."""
        central_space: spaces.Box = self._env.action_space
        act_dim = central_space.shape[1]
        return spaces.Box(
            low=central_space.low[0, :act_dim].copy(),
            high=central_space.high[0, :act_dim].copy(),
            shape=(act_dim,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # PettingZoo API – spaces
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        return self._per_agent_obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        return self._per_agent_act_space

    # ------------------------------------------------------------------
    # PettingZoo API – reset / step / render / close
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Reset the environment and return per-agent observations and infos."""
        self.agents = list(self.possible_agents)
        central_obs, central_info = self._env.reset(seed=seed, options=options)

        observations = self._split_obs(central_obs)
        infos = {agent: central_info for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, dict],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Execute one environment step with per-agent actions.

        Parameters
        ----------
        actions : dict[str, np.ndarray]
            Mapping from agent name to its action array of shape ``(act_dim,)``.

        Returns
        -------
        observations, rewards, terminations, truncations, infos
            All are dicts keyed by agent name.
        """
        # Assemble centralized action array (NUM_DRONES, act_dim)
        act_dim = self._per_agent_act_space.shape[0]
        central_action = np.zeros((self._num_drones, act_dim), dtype=np.float32)
        for agent_name, act in actions.items():
            idx = self._agent_idx[agent_name]
            central_action[idx, :] = np.asarray(act, dtype=np.float32)

        # Step the centralized environment
        central_obs, _reward_sum, terminated, truncated, central_info = self._env.step(central_action)

        # Per-agent observations
        observations = self._split_obs(central_obs)

        # Per-agent rewards (from the internally stored per-drone rewards)
        drone_rewards = getattr(self._env, "_last_drone_rewards", None)
        if drone_rewards is None:
            drone_rewards = np.full(self._num_drones, _reward_sum / self._num_drones, dtype=np.float32)
        rewards = {
            agent: float(drone_rewards[self._agent_idx[agent]])
            for agent in self.agents
        }

        # Termination / truncation: shared across all agents
        terminations = {agent: bool(terminated) for agent in self.agents}
        truncations = {agent: bool(truncated) for agent in self.agents}

        # Infos: shared base info + per-drone fields
        drone_captures = central_info.get("drone_captures", np.zeros(self._num_drones, dtype=np.int32))
        infos = {}
        for agent in self.agents:
            idx = self._agent_idx[agent]
            agent_info = dict(central_info)
            agent_info["drone_reward"] = float(drone_rewards[idx])
            agent_info["drone_capture_count"] = int(drone_captures[idx])
            infos[agent] = agent_info

        # Clear agents when episode ends (PettingZoo convention)
        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Observation splitting
    # ------------------------------------------------------------------

    def _split_obs(self, central_obs: dict[str, np.ndarray]) -> dict[str, dict]:
        """Split centralized (NUM_DRONES, dim) observations into per-agent dicts."""
        observations = {}
        for agent in self.agents:
            idx = self._agent_idx[agent]
            agent_obs = {}
            for key, arr in central_obs.items():
                agent_obs[key] = arr[idx, :].copy().astype(np.float32)
            observations[agent] = agent_obs
        return observations

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> OurRLAviary:
        """Access the underlying OurRLAviary environment."""
        return self._env

    @property
    def num_drones(self) -> int:
        return self._num_drones

    def state(self) -> np.ndarray:
        """Return a global state vector (for centralized critic / CTDE).

        Concatenates all drone positions, velocities, target positions,
        and obstacle positions into a single flat vector.
        """
        parts = [
            self._env.pos.reshape(-1),                         # all drone positions
            self._env.vel.reshape(-1),                         # all drone velocities
            self._env._target_positions.reshape(-1),           # all target positions
            self._env._obstacle_positions_xy.reshape(-1),      # all obstacle positions
            self._env._obstacle_radii.reshape(-1),             # all obstacle radii
        ]
        return np.concatenate(parts).astype(np.float32)
