"""Standard PPO rollout buffer for Go-Explore Phase 2."""

from __future__ import annotations

from typing import Dict, Generator, Optional, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Stores transitions and computes GAE returns.

    Parameters
    ----------
    n_steps : int
    n_agents : int
        ``n_envs × num_drones``.
    obs_keys_shapes : dict
    action_dim : int
    gru_hidden : int
    num_gru_layers : int
    """

    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_keys_shapes: Dict[str, Tuple[int, ...]],
        action_dim: int = 2,
        gru_hidden: int = 128,
        num_gru_layers: int = 1,
    ) -> None:
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gru_hidden = gru_hidden
        self.num_gru_layers = num_gru_layers

        self._ptr = 0
        self._full = False

        self.obs: Dict[str, np.ndarray] = {
            k: np.zeros((n_steps, n_agents, *s), dtype=np.float32)
            for k, s in obs_keys_shapes.items()
        }
        self.actions = np.zeros((n_steps, n_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_agents, 1), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.values = np.zeros((n_steps, n_agents, 1), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.gru_hiddens = np.zeros(
            (n_steps, num_gru_layers, n_agents, gru_hidden), dtype=np.float32
        )

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._ptr = 0
        self._full = False
        self.advantages = None
        self.returns = None

    def add(self, obs, action, log_prob, reward, value, done, gru_hidden) -> None:
        idx = self._ptr
        for k in self.obs:
            self.obs[k][idx] = obs[k]
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = done
        self.gru_hiddens[idx] = gru_hidden
        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True

    @property
    def size(self) -> int:
        return self.n_steps if self._full else self._ptr

    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95) -> None:
        T = self.size
        self.advantages = np.zeros((T, self.n_agents), dtype=np.float32)
        last_gae = np.zeros(self.n_agents, dtype=np.float32)
        next_val = last_value.squeeze(-1)
        for t in reversed(range(T)):
            nt = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * nt - self.values[t].squeeze(-1)
            last_gae = delta + gamma * gae_lambda * nt * last_gae
            self.advantages[t] = last_gae
            next_val = self.values[t].squeeze(-1)
        self.returns = self.advantages + self.values[:T].squeeze(-1)

    def get_batches(self, batch_size, device=torch.device("cpu")) -> Generator:
        assert self.advantages is not None
        T = self.size
        total = T * self.n_agents

        def _f(a):
            return a[:T].reshape(total, *a.shape[2:])

        flat_obs = {k: _f(v) for k, v in self.obs.items()}
        flat_act = _f(self.actions)
        flat_lp = _f(self.log_probs)
        flat_adv = self.advantages[:T].reshape(total)
        flat_ret = self.returns[:T].reshape(total)
        flat_hid = self.gru_hiddens[:T].reshape(total, self.num_gru_layers, self.gru_hidden)

        adv_std = flat_adv.std() + 1e-8
        flat_adv = (flat_adv - flat_adv.mean()) / adv_std

        idx = np.arange(total)
        np.random.shuffle(idx)
        for s in range(0, total, batch_size):
            e = min(s + batch_size, total)
            b = idx[s:e]
            yield {
                "obs": {k: torch.as_tensor(v[b], dtype=torch.float32, device=device) for k, v in flat_obs.items()},
                "action": torch.as_tensor(flat_act[b], dtype=torch.float32, device=device),
                "old_log_prob": torch.as_tensor(flat_lp[b], dtype=torch.float32, device=device),
                "advantage": torch.as_tensor(flat_adv[b], dtype=torch.float32, device=device),
                "return": torch.as_tensor(flat_ret[b], dtype=torch.float32, device=device),
                "gru_hidden": torch.as_tensor(flat_hid[b], dtype=torch.float32, device=device).permute(1, 0, 2).contiguous(),
            }
