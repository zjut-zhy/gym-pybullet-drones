"""Standard actor-critic for Go-Explore Phase 2 robustification.

No goal conditioning -- the policy learns a direct obs -> action mapping.
Adapted for single-agent OurSingleRLAviary (no teammate_state, no action_history).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.5


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
        nn.ReLU(inplace=True),
    )


class ObsEncoder(nn.Module):
    """Per-key MLP encoder for single-agent observations.

    Expected obs keys: self_state, target_state, obstacle_state.
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
        return self.proj(torch.cat([s, g, o], dim=-1))


class ActorCritic(nn.Module):
    """GRU-based actor-critic without goal conditioning."""

    def __init__(
        self,
        obs_encoder: ObsEncoder,
        action_dim: int = 2,
        gru_hidden: int = 128,
        num_gru_layers: int = 1,
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_dim = action_dim
        self.gru_hidden = gru_hidden
        self.num_gru_layers = num_gru_layers

        obs_embed_dim = obs_encoder.proj.out_features
        self.gru = nn.GRU(
            input_size=obs_embed_dim,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
        )
        self.policy_mean = nn.Linear(gru_hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(gru_hidden, 1)

    def _get_std(self) -> torch.Tensor:
        """Clamped std to prevent divergence."""
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros(self.num_gru_layers, batch_size, self.gru_hidden, device=device)

    @torch.no_grad()
    def get_action(
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.obs_encoder(obs).unsqueeze(1)
        gru_out, new_hidden = self.gru(x, hidden)
        gru_out = gru_out.squeeze(1)

        mean = self.policy_mean(gru_out)
        std = self._get_std().expand_as(mean)
        dist = Normal(mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        # Correct log_prob for tanh squashing: log π(a) = log π(z) - Σ log(1 - tanh²(z))
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        value = self.value_head(gru_out)
        return action, log_prob, value, new_hidden

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.obs_encoder(obs).unsqueeze(1)
        gru_out, _ = self.gru(x, hidden)
        gru_out = gru_out.squeeze(1)

        mean = self.policy_mean(gru_out)
        std = self._get_std().expand_as(mean)
        dist = Normal(mean, std)
        # Inverse tanh to recover the pre-squash sample
        raw_actions = torch.atanh(actions.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(raw_actions) - torch.log(1.0 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.value_head(gru_out)
        return log_prob, entropy, value
