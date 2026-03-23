"""Goal-conditioned actor-critic with per-key MLP encoders and a GRU core.

Adapted for single-agent OurSingleRLAviary (no teammate_state, no action_history).

Architecture
------------
1. Each observation key (self_state, target_state, obstacle_state)
   is independently encoded by a small MLP.
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


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
        nn.ReLU(inplace=True),
    )


class ObsEncoder(nn.Module):
    """Encode a single-drone observation dict into a fixed-size vector.

    Expected keys: self_state, target_state, obstacle_state.
    """

    def __init__(
        self,
        self_state_dim: int = 6,
        target_state_dim: int = 54,
        obstacle_state_dim: int = 24,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.self_enc = _mlp(self_state_dim, 64, 64)
        self.tgt_enc = _mlp(target_state_dim, 64, 64)
        self.obs_enc = _mlp(obstacle_state_dim, 64, 32)
        self.proj = nn.Linear(64 + 64 + 32, embed_dim)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        s = self.self_enc(obs["self_state"])
        g = self.tgt_enc(obs["target_state"])
        o = self.obs_enc(obs["obstacle_state"])
        cat = torch.cat([s, g, o], dim=-1)
        return self.proj(cat)


class GoalConditionedActorCritic(nn.Module):
    """Actor-critic with GRU core and goal conditioning."""

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

        self.goal_encoder = _mlp(goal_dim + 1, 32, goal_embed_dim)  # +1 for phase

        obs_embed_dim = obs_encoder.proj.out_features
        gru_input_dim = obs_embed_dim + goal_embed_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        self.policy_mean = nn.Linear(gru_hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(gru_hidden, 1)

    def _encode(
        self,
        obs: Dict[str, torch.Tensor],
        goal: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        obs_embed = self.obs_encoder(obs)
        goal_phase = torch.cat([goal, phase], dim=-1)
        goal_embed = self.goal_encoder(goal_phase)
        return torch.cat([obs_embed, goal_embed], dim=-1)

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros(
            self.num_gru_layers, batch_size, self.gru_hidden,
            device=device,
        )

    @torch.no_grad()
    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        goal: torch.Tensor,
        phase: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._encode(obs, goal, phase)
        x = x.unsqueeze(1)
        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(1)

        mean = self.policy_mean(gru_out)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = torch.tanh(dist.sample())
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = self.value_head(gru_out)
        return action, log_prob, value, new_hidden

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        goal: torch.Tensor,
        phase: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
