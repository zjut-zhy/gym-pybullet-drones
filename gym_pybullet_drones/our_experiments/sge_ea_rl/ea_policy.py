"""EA-only Actor-Critic policy for SB3 PPO (ablation: no Mamba).

This module provides the Entity Attention encoder without the Mamba
temporal modeling stack. Used as an ablation baseline to isolate
the contribution of spatial entity attention vs temporal modeling.

Architecture:
    obs_dict → EntityAttentionEncoder → (B, d_model)
    → features (B, d_model)  -- NO Mamba stack, direct output
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entity Attention Encoder (reused from ea_mamba_policy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
        nn.ReLU(inplace=True),
    )


class EntityCrossAttention(nn.Module):
    """Cross-attention: query from self_state, keys/values from entity tokens."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        d_attn: int = 32,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_attn // n_heads
        assert d_attn % n_heads == 0

        self.W_q = nn.Linear(query_dim, d_attn, bias=False)
        self.W_k = nn.Linear(key_dim, d_attn, bias=False)
        self.W_v = nn.Linear(key_dim, d_attn, bias=False)
        self.out_proj = nn.Linear(d_attn, d_attn, bias=False)

        self.scale = math.sqrt(self.d_head)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = keys.shape

        q = self.W_q(query)
        k = self.W_k(keys)
        v = self.W_v(keys)

        q = q.view(B, self.n_heads, 1, self.d_head)
        k = k.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).squeeze(2)
        out = out.reshape(B, -1)

        return self.out_proj(out)


class EntityAttentionEncoder(nn.Module):
    """Entity-Attention encoder for structured observations.

    Same architecture as in ea_mamba_policy.py, duplicated here for
    package independence in the ablation study.
    """

    def __init__(
        self,
        self_state_dim: int = 6,
        target_state_dim: int = 54,
        obstacle_state_dim: int = 24,
        n_targets: int = 18,
        n_obstacles: int = 6,
        target_feat_dim: int = 3,
        obstacle_feat_dim: int = 4,
        d_self: int = 64,
        d_k: int = 32,
        d_attn_target: int = 64,
        d_attn_obstacle: int = 32,
        n_heads: int = 2,
        d_model: int = 128,
    ) -> None:
        super().__init__()

        self.n_targets = n_targets
        self.n_obstacles = n_obstacles
        self.target_feat_dim = target_feat_dim
        self.obstacle_feat_dim = obstacle_feat_dim
        self.d_model = d_model

        self.self_enc = _mlp(self_state_dim, 64, d_self)
        self.target_token_enc = _mlp(target_feat_dim, 32, d_k)
        self.obstacle_token_enc = _mlp(obstacle_feat_dim, 32, d_k)

        self.target_attn = EntityCrossAttention(
            query_dim=d_self, key_dim=d_k,
            d_attn=d_attn_target, n_heads=n_heads,
        )
        self.obstacle_attn = EntityCrossAttention(
            query_dim=d_self, key_dim=d_k,
            d_attn=d_attn_obstacle, n_heads=n_heads,
        )

        fusion_dim = d_self + d_attn_target + d_attn_obstacle
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = obs["self_state"].shape[0]

        self_feat = self.self_enc(obs["self_state"])

        target_flat = obs["target_state"]
        target_tokens = target_flat.view(B, self.n_targets, self.target_feat_dim)
        target_encoded = self.target_token_enc(target_tokens)

        obstacle_flat = obs["obstacle_state"]
        obstacle_tokens = obstacle_flat.view(B, self.n_obstacles, self.obstacle_feat_dim)
        obstacle_encoded = self.obstacle_token_enc(obstacle_tokens)

        attended_target = self.target_attn(self_feat, target_encoded)
        attended_obstacle = self.obstacle_attn(self_feat, obstacle_encoded)

        fused = torch.cat([self_feat, attended_target, attended_obstacle], dim=-1)
        return self.proj(fused)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EA Feature Extractor for SB3 (NO Mamba -- ablation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EAFeatureExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor: Entity Attention ONLY (no Mamba).

    Architecture:
        obs_dict → EntityAttentionEncoder → (B, d_model)
        → features (B, d_model)

    No temporal modeling -- processes each timestep independently.
    Used as an ablation baseline.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        d_model: int = 128,
        d_k: int = 32,
        d_attn_target: int = 64,
        d_attn_obstacle: int = 32,
        n_heads: int = 2,
        n_targets: int = 18,
        n_obstacles: int = 6,
        target_feat_dim: int = 3,
        obstacle_feat_dim: int = 4,
    ) -> None:
        super().__init__(observation_space, features_dim=d_model)

        spaces = observation_space.spaces
        self_state_dim = spaces["self_state"].shape[-1]
        target_state_dim = spaces["target_state"].shape[-1]
        obstacle_state_dim = spaces["obstacle_state"].shape[-1]

        self.ea_encoder = EntityAttentionEncoder(
            self_state_dim=self_state_dim,
            target_state_dim=target_state_dim,
            obstacle_state_dim=obstacle_state_dim,
            n_targets=n_targets,
            n_obstacles=n_obstacles,
            target_feat_dim=target_feat_dim,
            obstacle_feat_dim=obstacle_feat_dim,
            d_self=64,
            d_k=d_k,
            d_attn_target=d_attn_target,
            d_attn_obstacle=d_attn_obstacle,
            n_heads=n_heads,
            d_model=d_model,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.ea_encoder(observations)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Custom SB3 Actor-Critic Policy (EA only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EAActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy using Entity Attention only (no Mamba).

    Drop-in replacement for MultiInputPolicy. Ablation baseline.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        # Entity Attention config
        ea_d_model: int = 128,
        ea_d_k: int = 32,
        ea_d_attn_target: int = 64,
        ea_d_attn_obstacle: int = 32,
        ea_n_heads: int = 2,
        n_targets: int = 18,
        n_obstacles: int = 6,
        target_feat_dim: int = 3,
        obstacle_feat_dim: int = 4,
        # Standard SB3
        net_arch: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> None:
        self._ea_cfg = dict(
            d_model=ea_d_model,
            d_k=ea_d_k,
            d_attn_target=ea_d_attn_target,
            d_attn_obstacle=ea_d_attn_obstacle,
            n_heads=ea_n_heads,
            n_targets=n_targets,
            n_obstacles=n_obstacles,
            target_feat_dim=target_feat_dim,
            obstacle_feat_dim=obstacle_feat_dim,
        )

        if net_arch is None:
            net_arch = dict(pi=[128, 64], vf=[128, 64])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            features_extractor_class=EAFeatureExtractor,
            features_extractor_kwargs=self._ea_cfg,
            **kwargs,
        )
