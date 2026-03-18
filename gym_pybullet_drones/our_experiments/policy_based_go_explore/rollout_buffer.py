"""GAE-compatible rollout buffer with goal, phase, and valid-mask support.

Design note — **distributed / parameter-sharing**
--------------------------------------------------
The ``n_agents`` parameter should be set to ``n_envs × num_drones``.  Each
drone is treated as an independent agent in the batch dimension.  Rewards and
dones from the *environment* are broadcast to every drone in that environment
by the caller before calling :meth:`add`.
"""

from __future__ import annotations

from typing import Dict, Generator, Optional, Tuple

import numpy as np
import torch


class GoExploreRolloutBuffer:
    """Stores transitions from parallel Go-Explore rollouts.

    After each rollout, call :meth:`compute_returns` and then iterate
    mini-batches via :meth:`get_batches`.

    Parameters
    ----------
    n_steps : int
        Maximum number of time-steps per rollout (return + explore).
    n_agents : int
        Total number of independent agents = ``n_envs × num_drones``.
    obs_keys_shapes : dict[str, tuple[int, ...]]
        Per-drone feature shapes (e.g. ``{"self_state": (6,), ...}``).
    goal_dim : int
    action_dim : int
    gru_hidden : int
    num_gru_layers : int
    """

    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_keys_shapes: Dict[str, Tuple[int, ...]],
        goal_dim: int = 2,
        action_dim: int = 2,
        gru_hidden: int = 128,
        num_gru_layers: int = 1,
    ) -> None:
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.gru_hidden = gru_hidden
        self.num_gru_layers = num_gru_layers
        self.obs_keys_shapes = obs_keys_shapes

        self._ptr = 0
        self._full = False

        # pre-allocate
        self.obs: Dict[str, np.ndarray] = {}
        for key, shape in obs_keys_shapes.items():
            self.obs[key] = np.zeros((n_steps, n_agents, *shape), dtype=np.float32)

        self.goals = np.zeros((n_steps, n_agents, goal_dim), dtype=np.float32)
        self.phases = np.zeros((n_steps, n_agents, 1), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_agents, 1), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.values = np.zeros((n_steps, n_agents, 1), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.valid_masks = np.ones((n_steps, n_agents), dtype=np.float32)
        self.gru_hiddens = np.zeros(
            (n_steps, num_gru_layers, n_agents, gru_hidden), dtype=np.float32
        )

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    # ── recording ────────────────────────────────────────────────

    def reset(self) -> None:
        self._ptr = 0
        self._full = False
        self.advantages = None
        self.returns = None

    def add(
        self,
        obs: Dict[str, np.ndarray],
        goal: np.ndarray,
        phase: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: np.ndarray,
        valid_mask: np.ndarray,
        gru_hidden: np.ndarray,
    ) -> None:
        """Record one time-step across all agents.

        All arrays should have ``n_agents`` as their first dimension.
        """
        idx = self._ptr
        for key in self.obs:
            self.obs[key][idx] = obs[key]
        self.goals[idx] = goal
        self.phases[idx] = phase
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = done
        self.valid_masks[idx] = valid_mask
        self.gru_hiddens[idx] = gru_hidden

        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    @property
    def size(self) -> int:
        return self.n_steps if self._full else self._ptr

    # ── GAE computation ──────────────────────────────────────────

    def compute_returns(
        self,
        last_value: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_value : ndarray (n_agents, 1)
            Bootstrap value from the last observation.
        """
        T = self.size
        self.advantages = np.zeros((T, self.n_agents), dtype=np.float32)
        self.returns = np.zeros((T, self.n_agents), dtype=np.float32)

        last_gae = np.zeros(self.n_agents, dtype=np.float32)
        next_value = last_value.squeeze(-1)

        for t in reversed(range(T)):
            non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_value * non_terminal
                - self.values[t].squeeze(-1)
            )
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            last_gae = last_gae * self.valid_masks[t]
            self.advantages[t] = last_gae
            next_value = self.values[t].squeeze(-1)

        self.returns = self.advantages + self.values[:T].squeeze(-1)

    # ── mini-batch iteration ─────────────────────────────────────

    def get_batches(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Generator[dict, None, None]:
        """Flatten (T, n_agents) → (T*n_agents,) and yield shuffled mini-batches."""
        assert self.advantages is not None, "call compute_returns first"
        T = self.size
        total = T * self.n_agents

        def _flat(arr: np.ndarray) -> np.ndarray:
            return arr[:T].reshape(total, *arr.shape[2:])

        flat_obs = {k: _flat(v) for k, v in self.obs.items()}
        flat_goals = _flat(self.goals)
        flat_phases = _flat(self.phases)
        flat_actions = _flat(self.actions)
        flat_log_probs = _flat(self.log_probs)
        flat_values = _flat(self.values)
        flat_advantages = self.advantages[:T].reshape(total)
        flat_returns = self.returns[:T].reshape(total)
        flat_valid_masks = self.valid_masks[:T].reshape(total)
        flat_gru_hiddens = self.gru_hiddens[:T].reshape(
            total, self.num_gru_layers, self.gru_hidden
        )

        # normalise advantages
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        indices = np.arange(total)
        np.random.shuffle(indices)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]

            batch = {
                "obs": {k: torch.as_tensor(v[idx], dtype=torch.float32, device=device) for k, v in flat_obs.items()},
                "goal": torch.as_tensor(flat_goals[idx], dtype=torch.float32, device=device),
                "phase": torch.as_tensor(flat_phases[idx], dtype=torch.float32, device=device),
                "action": torch.as_tensor(flat_actions[idx], dtype=torch.float32, device=device),
                "old_log_prob": torch.as_tensor(flat_log_probs[idx], dtype=torch.float32, device=device),
                "old_value": torch.as_tensor(flat_values[idx], dtype=torch.float32, device=device),
                "advantage": torch.as_tensor(flat_advantages[idx], dtype=torch.float32, device=device),
                "return": torch.as_tensor(flat_returns[idx], dtype=torch.float32, device=device),
                "valid_mask": torch.as_tensor(flat_valid_masks[idx], dtype=torch.float32, device=device),
                "gru_hidden": torch.as_tensor(
                    flat_gru_hiddens[idx], dtype=torch.float32, device=device
                ).permute(1, 0, 2).contiguous(),  # (layers, batch, hidden)
            }
            yield batch
