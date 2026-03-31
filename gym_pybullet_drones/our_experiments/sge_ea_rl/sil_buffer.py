"""Self-Imitation Learning (SIL) replay buffer for SGE-EA ablation.

Identical to sge_ea_mamba_rl/sil_buffer.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class SILBuffer:
    """Fixed-capacity circular replay buffer for SIL."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self._obs: List[Dict[str, np.ndarray]] = []
        self._actions: List[np.ndarray] = []
        self._returns: List[float] = []
        self._ptr = 0

    def __len__(self) -> int:
        return len(self._obs)

    def load_demo(
        self,
        obs_list: List[Dict[str, np.ndarray]],
        action_list: List[np.ndarray],
        returns: List[float],
    ) -> None:
        for obs, action, ret in zip(obs_list, action_list, returns):
            self._add_one(
                {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()},
                np.asarray(action, dtype=np.float32),
                float(ret),
            )

    def add_trajectory(
        self,
        obs_list: List[Dict[str, np.ndarray]],
        action_list: List[np.ndarray],
        reward_list: List[float],
        gamma: float = 0.99,
    ) -> None:
        T = len(reward_list)
        returns = np.zeros(T, dtype=np.float64)
        running = 0.0
        for t in reversed(range(T)):
            running = reward_list[t] + gamma * running
            returns[t] = running
        for obs, action, ret in zip(obs_list, action_list, returns):
            self._add_one(
                {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()},
                np.asarray(action, dtype=np.float32),
                float(ret),
            )

    def _add_one(self, obs: Dict[str, np.ndarray], action: np.ndarray,
                 ret: float) -> None:
        if len(self._obs) < self.capacity:
            self._obs.append(obs)
            self._actions.append(action)
            self._returns.append(ret)
        else:
            self._obs[self._ptr] = obs
            self._actions[self._ptr] = action
            self._returns[self._ptr] = ret
        self._ptr = (self._ptr + 1) % self.capacity

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Dict[str, torch.Tensor]]:
        if len(self) == 0:
            return None
        n = min(batch_size, len(self))
        indices = np.random.choice(len(self), size=n, replace=False)

        obs_batch: Dict[str, torch.Tensor] = {}
        keys = list(self._obs[0].keys())
        for k in keys:
            arrs = [self._obs[i][k] for i in indices]
            obs_batch[k] = torch.as_tensor(
                np.stack(arrs), dtype=torch.float32, device=device)

        actions = torch.as_tensor(
            np.stack([self._actions[i] for i in indices]),
            dtype=torch.float32, device=device)
        returns = torch.as_tensor(
            np.array([self._returns[i] for i in indices]),
            dtype=torch.float32, device=device)

        return {"obs": obs_batch, "action": actions, "return": returns}
