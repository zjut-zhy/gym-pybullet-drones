"""EA-Mamba Actor-Critic policy for SB3 PPO (SEAM-RL).

This module provides:
    1. EntityAttentionEncoder -- cross-attention over per-entity tokens
    2. EAMambaFeatureExtractor -- SB3-compatible: EA encoder + Mamba stack
    3. EAMambaActorCriticPolicy -- drop-in replacement for SB3 MultiInputPolicy

Key difference from sge_mamba_rl/mamba_policy.py:
    - Old: flat MLP per observation key (target_state as 54d vector)
    - New: reshape into per-entity tokens (18×3 for targets, 6×4 for obstacles)
           and use cross-attention with self_state as query.

Set ``USE_REAL_MAMBA = True`` on a CUDA + Linux machine with ``mamba-ssm``.
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
#  Mamba toggle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_REAL_MAMBA: bool = False  # Set True on GPU server with mamba-ssm

if USE_REAL_MAMBA:
    from mamba_ssm import Mamba3 as _Mamba3Block  # noqa: F401
else:
    _Mamba3Block = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fallback SSM (CPU / Windows dev -- identical to sge_mamba_rl)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _FallbackSSM(nn.Module):
    """Minimal selective SSM (S4D-style) for local dev."""

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        with torch.no_grad():
            self.dt_proj.bias.uniform_(0.001, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = torch.silu(x_conv)

        A = -torch.exp(self.A_log)
        dt_bc = self.x_proj(x_conv)
        B_mat, C_mat = dt_bc.chunk(2, dim=-1)
        dt = torch.softplus(self.dt_proj(x_conv))

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)

        h = torch.zeros(B, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * torch.silu(z)
        return self.out_proj(y)

    def step(
        self, x_t: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = x_t.shape[0]
        device, dtype = x_t.device, x_t.dtype

        if state is None:
            state = {
                "h": torch.zeros(B, self.d_inner, self.d_state,
                                 device=device, dtype=dtype),
                "conv_buf": torch.zeros(B, self.d_inner, self.d_conv - 1,
                                        device=device, dtype=dtype),
            }

        xz = self.in_proj(x_t)
        x_inner, z = xz.chunk(2, dim=-1)

        conv_buf = state["conv_buf"]
        conv_in = torch.cat([conv_buf, x_inner.unsqueeze(-1)], dim=-1)
        weight = self.conv1d.weight
        bias = self.conv1d.bias
        x_conv = (conv_in * weight.squeeze(1)).sum(dim=-1) + bias
        x_conv = torch.silu(x_conv)
        new_conv_buf = conv_in[:, :, 1:]

        A = -torch.exp(self.A_log)
        dt_bc = self.x_proj(x_conv)
        B_mat, C_mat = dt_bc.chunk(2, dim=-1)
        dt = torch.softplus(self.dt_proj(x_conv))

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(1)

        h = state["h"]
        h = dA * h + dB * x_conv.unsqueeze(-1)
        y = (h * C_mat.unsqueeze(1)).sum(dim=-1)

        y = y + x_conv * self.D.unsqueeze(0)
        y = y * torch.silu(z)
        out = self.out_proj(y)

        return out, {"h": h, "conv_buf": new_conv_buf}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entity Attention Encoder (spatial dimension)
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
    """Cross-attention: query from self_state, keys/values from entity tokens.

    Supports multi-head attention for richer representation.
    Empty entity slots (all-zero tokens) naturally get near-zero attention
    weights after softmax, providing implicit masking.
    """

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
        """
        Args:
            query: (B, query_dim) -- self_state encoding
            keys:  (B, N, key_dim) -- N entity token encodings

        Returns:
            attended: (B, d_attn)
        """
        B, N, _ = keys.shape

        # Project
        q = self.W_q(query)                    # (B, d_attn)
        k = self.W_k(keys)                     # (B, N, d_attn)
        v = self.W_v(keys)                     # (B, N, d_attn)

        # Reshape for multi-head: (B, n_heads, *, d_head)
        q = q.view(B, self.n_heads, 1, self.d_head)
        k = k.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, heads, 1, N)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = (attn @ v).squeeze(2)  # (B, heads, d_head)
        out = out.reshape(B, -1)     # (B, d_attn)

        return self.out_proj(out)


class EntityAttentionEncoder(nn.Module):
    """Entity-Attention encoder for structured observations.

    Replaces the flat-MLP ObsEncoder with per-entity token encoding
    followed by cross-attention, explicitly modeling entity importance.

    Architecture:
        self_state (6d) → SelfEncoder → query (64d)
        target_state (54d) → reshape 18×3 → TokenEncoder → 18×d_k
            → CrossAttention(query, keys) → attended_target (d_attn)
        obstacle_state (24d) → reshape 6×4 → TokenEncoder → 6×d_k
            → CrossAttention(query, keys) → attended_obs (d_attn)
        Concat(query + attended_target + attended_obs) → Projection → d_model
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

        # Per-entity encoders
        self.self_enc = _mlp(self_state_dim, 64, d_self)
        self.target_token_enc = _mlp(target_feat_dim, 32, d_k)
        self.obstacle_token_enc = _mlp(obstacle_feat_dim, 32, d_k)

        # Cross-attention modules
        self.target_attn = EntityCrossAttention(
            query_dim=d_self, key_dim=d_k,
            d_attn=d_attn_target, n_heads=n_heads,
        )
        self.obstacle_attn = EntityCrossAttention(
            query_dim=d_self, key_dim=d_k,
            d_attn=d_attn_obstacle, n_heads=n_heads,
        )

        # Fusion projection
        fusion_dim = d_self + d_attn_target + d_attn_obstacle
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs: dict with keys self_state, target_state, obstacle_state

        Returns:
            embedding: (B, d_model)
        """
        B = obs["self_state"].shape[0]

        # Encode self state → query
        self_feat = self.self_enc(obs["self_state"])  # (B, d_self)

        # Reshape target_state from (B, 54) to (B, 18, 3) and encode tokens
        target_flat = obs["target_state"]  # (B, 54)
        target_tokens = target_flat.view(B, self.n_targets, self.target_feat_dim)
        target_encoded = self.target_token_enc(target_tokens)  # (B, 18, d_k)

        # Reshape obstacle_state from (B, 24) to (B, 6, 4) and encode tokens
        obstacle_flat = obs["obstacle_state"]  # (B, 24)
        obstacle_tokens = obstacle_flat.view(B, self.n_obstacles, self.obstacle_feat_dim)
        obstacle_encoded = self.obstacle_token_enc(obstacle_tokens)  # (B, 6, d_k)

        # Cross-attention
        attended_target = self.target_attn(self_feat, target_encoded)    # (B, d_attn_target)
        attended_obstacle = self.obstacle_attn(self_feat, obstacle_encoded)  # (B, d_attn_obstacle)

        # Fusion
        fused = torch.cat([self_feat, attended_target, attended_obstacle], dim=-1)
        return self.proj(fused)  # (B, d_model)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EA-Mamba Feature Extractor for SB3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EAMambaFeatureExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor: Entity Attention + Mamba stack.

    Architecture:
        obs_dict → EntityAttentionEncoder → (B, d_model)
        → [MambaBlock + LayerNorm + Residual] × n_layers
        → features (B, d_model)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 2,
        d_k: int = 32,
        d_attn_target: int = 64,
        d_attn_obstacle: int = 32,
        n_heads: int = 2,
        n_targets: int = 18,
        n_obstacles: int = 6,
        target_feat_dim: int = 3,
        obstacle_feat_dim: int = 4,
        # Mamba3-specific (ignored by fallback)
        headdim: int = 32,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        chunk_size: int = 16,
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

        self.d_model = d_model
        self.n_layers = n_layers

        # Build Mamba layers
        self.mamba_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for _ in range(n_layers):
            if USE_REAL_MAMBA and _Mamba3Block is not None:
                layer = _Mamba3Block(
                    d_model=d_model,
                    d_state=d_state,
                    headdim=headdim,
                    is_mimo=is_mimo,
                    mimo_rank=mimo_rank,
                    chunk_size=chunk_size,
                    is_outproj_norm=False,
                    dtype=torch.bfloat16,
                )
            else:
                layer = _FallbackSSM(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=4,
                    expand=2,
                )
            self.mamba_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(d_model))

        # Hidden state buffer for recurrent inference
        self._hidden: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None
        self._batch_size: int = 0

    def reset_hidden(self, batch_size: int = 1) -> None:
        self._hidden = [None] * self.n_layers
        self._batch_size = batch_size

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Entity Attention encoding
        x = self.ea_encoder(observations)  # (B, d_model)
        B = x.shape[0]

        if self._hidden is None or self._batch_size != B:
            self.reset_hidden(B)

        if self.training:
            x_seq = x.unsqueeze(1)  # (B, 1, d_model)
            for layer, norm in zip(self.mamba_layers, self.layer_norms):
                if USE_REAL_MAMBA:
                    x_bf16 = x_seq.to(torch.bfloat16)
                    out = layer(x_bf16).to(x_seq.dtype)
                else:
                    out = layer(x_seq)
                x_seq = norm(x_seq + out)
            return x_seq.squeeze(1)
        else:
            h = x
            for i, (layer, norm) in enumerate(
                zip(self.mamba_layers, self.layer_norms)
            ):
                if USE_REAL_MAMBA:
                    h_bf16 = h.unsqueeze(1).to(torch.bfloat16)
                    out = layer(h_bf16).squeeze(1).to(h.dtype)
                    h = norm(h + out)
                else:
                    out, new_state = layer.step(h, self._hidden[i])
                    self._hidden[i] = new_state
                    h = norm(h + out)
            return h


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Custom SB3 Actor-Critic Policy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EAMambaActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy using EA-Mamba as the feature extractor.

    Drop-in replacement for MultiInputPolicy in SB3's PPO.

    policy_kwargs can include:
        mamba_d_model, mamba_d_state, mamba_n_layers, mamba_headdim,
        ea_d_k, ea_d_attn_target, ea_d_attn_obstacle, ea_n_heads,
        n_targets, n_obstacles, target_feat_dim, obstacle_feat_dim,
        net_arch
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        # Mamba config
        mamba_d_model: int = 128,
        mamba_d_state: int = 64,
        mamba_n_layers: int = 2,
        mamba_headdim: int = 32,
        # Entity Attention config
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
        self._ea_mamba_cfg = dict(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            n_layers=mamba_n_layers,
            headdim=mamba_headdim,
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
            features_extractor_class=EAMambaFeatureExtractor,
            features_extractor_kwargs=self._ea_mamba_cfg,
            **kwargs,
        )

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        if episode_start is not None and np.any(episode_start):
            self.features_extractor.reset_hidden(batch_size=1)
        return super().predict(observation, state, episode_start, deterministic)
