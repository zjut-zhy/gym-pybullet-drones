"""Goal-conditioned actor-critic with per-key MLP encoders and a GRU core.

Architecture (distributed / parameter-sharing)
-----------------------------------------------
Each drone is treated as an independent agent sharing the same policy weights.
The ``NUM_DRONES`` dimension is folded into the **batch** dimension by the
caller so that the network always sees inputs of shape ``(batch, feat_dim)``.

1. Each observation key (self_state, action_history, teammate_state,
   target_state, obstacle_state) is independently encoded by a small MLP.
2. The encodings are concatenated and projected to ``obs_embed_dim``.
3. Goal (2-D) and phase (1-D) are encoded by a tiny MLP.
4. [obs_embed, goal_embed] are concatenated and fed through a GRU.
5. Two heads branch off the GRU output: policy (Gaussian) and value (scalar).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


# ─── small helper ────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
        nn.ReLU(inplace=True),
    )


# ─── Observation encoder ────────────────────────────────────────────────────

class ObsEncoder(nn.Module):
    """Encode a **single-drone** observation dict into a fixed-size vector.

    The caller is responsible for flattening the ``(n_envs, num_drones, feat)``
    tensors into ``(n_envs * num_drones, feat)`` before calling this module.

    Parameters
    ----------
    self_state_dim : int   (default 6)
    action_history_dim : int
    teammate_state_dim : int
    target_state_dim : int
    obstacle_state_dim : int
    embed_dim : int        (default 128)
    """

    def __init__(
        self,
        self_state_dim: int = 6,
        action_history_dim: int = 60,   # ACTION_BUFFER_SIZE * ACTION_DIM
        teammate_state_dim: int = 48,   # MAX_NUM_DRONES * 6
        target_state_dim: int = 54,     # MAX_TARGET_COUNT * 3
        obstacle_state_dim: int = 24,   # MAX_OBSTACLE_COUNT * 4
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.self_enc = _mlp(self_state_dim, 64, 64)
        self.act_enc = _mlp(action_history_dim, 64, 32)
        self.team_enc = _mlp(teammate_state_dim, 64, 64)
        self.tgt_enc = _mlp(target_state_dim, 64, 64)
        self.obs_enc = _mlp(obstacle_state_dim, 64, 32)
        self.proj = nn.Linear(64 + 32 + 64 + 64 + 32, embed_dim)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : dict of str → Tensor
            Each value has shape ``(batch, feat_dim)`` where *batch* already
            contains the flattened drone dimension.

        Returns
        -------
        Tensor (batch, embed_dim)
        """
        s = self.self_enc(obs["self_state"])
        a = self.act_enc(obs["action_history"])
        t = self.team_enc(obs["teammate_state"])
        g = self.tgt_enc(obs["target_state"])
        o = self.obs_enc(obs["obstacle_state"])

        cat = torch.cat([s, a, t, g, o], dim=-1)
        return self.proj(cat)


# ─── Goal-conditioned actor-critic ───────────────────────────────────────────

class GoalConditionedActorCritic(nn.Module):
    """Actor-critic with GRU core and goal conditioning.

    All inputs should have shape ``(batch, *)`` where *batch* =
    ``n_envs × num_drones`` (the drone dimension is already flattened).

    Parameters
    ----------
    obs_encoder : ObsEncoder
        Pre-built observation encoder.
    goal_dim : int
        Dimensionality of the goal vector (2).
    action_dim : int
        Dimensionality of the continuous action space (2 for VEL).
    gru_hidden : int
        Size of the GRU hidden state.
    num_gru_layers : int
        Number of stacked GRU layers.
    goal_embed_dim : int
        Internal dimension for the goal+phase encoder.
    """

    def __init__(
        self,
        obs_encoder: ObsEncoder,
        goal_dim: int = 2,
        action_dim: int = 2,
        gru_hidden: int = 128,
        num_gru_layers: int = 1,
        goal_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_dim = action_dim
        self.gru_hidden = gru_hidden
        self.num_gru_layers = num_gru_layers

        # goal + phase → embedding
        self.goal_encoder = _mlp(goal_dim + 1, 32, goal_embed_dim)  # +1 for phase

        obs_embed_dim = obs_encoder.proj.out_features
        gru_input_dim = obs_embed_dim + goal_embed_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # policy head
        self.policy_mean = nn.Linear(gru_hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # value head
        self.value_head = nn.Linear(gru_hidden, 1)

    # ── forward helpers ──────────────────────────────────────────

    def _encode(
        self,
        obs: Dict[str, torch.Tensor],
        goal: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        obs_embed = self.obs_encoder(obs)                    # (batch, obs_embed)
        goal_phase = torch.cat([goal, phase], dim=-1)        # (batch, 3)
        goal_embed = self.goal_encoder(goal_phase)           # (batch, goal_embed)
        return torch.cat([obs_embed, goal_embed], dim=-1)    # (batch, gru_in)

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return a zeroed GRU hidden state.

        Parameters
        ----------
        batch_size : int
            Should be ``n_envs * num_drones`` so that each drone keeps its
            own recurrent state.
        """
        device = next(self.parameters()).device
        return torch.zeros(
            self.num_gru_layers, batch_size, self.gru_hidden,
            device=device,
        )

    # ── action sampling (rollout) ────────────────────────────────

    @torch.no_grad()
    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        goal: torch.Tensor,
        phase: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action for a single time-step (no grad).

        All inputs have shape ``(batch, *)`` where batch = n_envs × num_drones.

        Returns (action, log_prob, value, new_hidden).
        """
        x = self._encode(obs, goal, phase)                  # (batch, gru_in)
        x = x.unsqueeze(1)                                   # (batch, 1, gru_in)
        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(1)                          # (batch, gru_h)

        mean = self.policy_mean(gru_out)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.tanh(action)                           # bound to [-1, 1]
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        value = self.value_head(gru_out)
        return action, log_prob, value, new_hidden

    # ── action evaluation (PPO update) ───────────────────────────

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        goal: torch.Tensor,
        phase: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-prob, entropy and value for stored transitions.

        Parameters have shape ``(batch, *)``.  GRU is run with ``seq_len=1``.

        Returns (log_prob, entropy, value) each ``(batch, 1)``.
        """
        x = self._encode(obs, goal, phase)
        x = x.unsqueeze(1)
        gru_out, _ = self.gru(x, hidden)
        gru_out = gru_out.squeeze(1)

        mean = self.policy_mean(gru_out)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.value_head(gru_out)
        return log_prob, entropy, value
