"""Mamba3-based Actor-Critic policy for SB3 PPO.

This module provides:
    1. ObsEncoder        -- per-key MLP encoder for Dict observations
    2. MambaFeatureExtractor -- SB3-compatible feature extractor using Mamba3
    3. MambaActorCriticPolicy -- drop-in replacement for SB3's MultiInputPolicy

Set ``USE_REAL_MAMBA = True`` on a CUDA + Linux machine with ``mamba-ssm``
installed from source.  When False, a pure-PyTorch fallback SSM is used
for local development / debugging on Windows / CPU.

Mamba3 install (on GPU server):
    MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
        git+https://github.com/state-spaces/mamba.git --no-build-isolation
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Mamba3 toggle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_REAL_MAMBA: bool = False  # Set True on GPU server with mamba-ssm

if USE_REAL_MAMBA:
    from mamba_ssm import Mamba3 as _Mamba3Block  # noqa: F401
else:
    _Mamba3Block = None  # will use fallback


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Fallback: Pure-PyTorch S4D-like SSM for CPU / Windows development
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _FallbackSSM(nn.Module):
    """Minimal selective SSM (S4D-style) for local dev when mamba-ssm is unavailable.

    NOT intended for final training -- only for code-path validation.
    Operates on (batch, length, d_model) tensors, same interface as Mamba3.
    Also supports step-by-step inference via ``step(x_t, state)`` for
    recurrent rollout.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,  # absorb unused Mamba3 params
    ) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.d_conv = d_conv

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        # Depthwise causal conv1d
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        # SSM parameters (input-dependent via projections)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        # A parameter (log-space, negative real)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        with torch.no_grad():
            self.dt_proj.bias.uniform_(0.001, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward: (B, L, D) -> (B, L, D)."""
        B, L, D = x.shape

        # Input projection → (B, L, 2*d_inner) → split
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal conv1d
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal: trim right padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = torch.silu(x_conv)

        # SSM: discretize and scan
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        dt_bc = self.x_proj(x_conv)  # (B, L, d_state*2)
        B_mat, C_mat = dt_bc.chunk(2, dim=-1)  # each (B, L, d_state)
        dt = torch.softplus(self.dt_proj(x_conv))  # (B, L, d_inner)

        # Discretize A → dA = exp(A * dt)
        # dt: (B,L,d_inner), A: (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,d_inner,d_state)
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)  # (B,L,d_inner,d_state)

        # Sequential scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Skip + gate + output
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * torch.silu(z)
        return self.out_proj(y)

    def step(
        self, x_t: torch.Tensor, state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single-step recurrent inference: (B, D) -> (B, D), new_state.

        ``state`` contains 'h' (SSM hidden, (B, d_inner, d_state)) and
        'conv_buf' (causal conv buffer, (B, d_inner, d_conv-1)).
        """
        B = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype

        if state is None:
            state = {
                "h": torch.zeros(B, self.d_inner, self.d_state, device=device, dtype=dtype),
                "conv_buf": torch.zeros(B, self.d_inner, self.d_conv - 1, device=device, dtype=dtype),
            }

        # Input proj
        xz = self.in_proj(x_t)  # (B, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)

        # Causal conv1d (manual step)
        conv_buf = state["conv_buf"]
        conv_in = torch.cat([conv_buf, x_inner.unsqueeze(-1)], dim=-1)  # (B, d_inner, d_conv)
        # Apply conv weights manually
        weight = self.conv1d.weight  # (d_inner, 1, d_conv)
        bias = self.conv1d.bias      # (d_inner,)
        x_conv = (conv_in * weight.squeeze(1)).sum(dim=-1) + bias  # (B, d_inner)
        x_conv = torch.silu(x_conv)
        new_conv_buf = conv_in[:, :, 1:]  # shift left

        # SSM step
        A = -torch.exp(self.A_log)
        dt_bc = self.x_proj(x_conv)
        B_mat, C_mat = dt_bc.chunk(2, dim=-1)  # each (B, d_state)
        dt = torch.softplus(self.dt_proj(x_conv))  # (B, d_inner)

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(1)  # (B, d_inner, d_state)

        h = state["h"]
        h = dA * h + dB * x_conv.unsqueeze(-1)
        y = (h * C_mat.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)

        # Skip + gate + output
        y = y + x_conv * self.D.unsqueeze(0)
        y = y * torch.silu(z)
        out = self.out_proj(y)

        new_state = {"h": h, "conv_buf": new_conv_buf}
        return out, new_state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Observation Encoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
        nn.ReLU(inplace=True),
    )


class ObsEncoder(nn.Module):
    """Per-key MLP encoder for Dict observations.

    Expected keys: self_state, target_state, obstacle_state.
    Output: (batch, d_model) embedding.
    """

    def __init__(
        self,
        self_state_dim: int = 6,
        target_state_dim: int = 54,
        obstacle_state_dim: int = 24,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.self_enc = _mlp(self_state_dim, 64, 64)
        self.tgt_enc = _mlp(target_state_dim, 64, 64)
        self.obs_enc = _mlp(obstacle_state_dim, 64, 32)
        self.proj = nn.Linear(64 + 64 + 32, d_model)
        self.d_model = d_model

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(batch, ...) per key -> (batch, d_model)."""
        s = self.self_enc(obs["self_state"])
        g = self.tgt_enc(obs["target_state"])
        o = self.obs_enc(obs["obstacle_state"])
        return self.proj(torch.cat([s, g, o], dim=-1))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Mamba Feature Extractor for SB3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MambaFeatureExtractor(BaseFeaturesExtractor):
    """SB3-compatible feature extractor with stacked Mamba3 blocks.

    Architecture:
        obs_dict → ObsEncoder → (B, d_model)
        → [Mamba3Block + LayerNorm + Residual] × n_layers
        → features (B, d_model)

    During SB3 rollout, each call processes a single timestep.
    The Mamba blocks maintain internal recurrent state via a hidden
    state buffer managed by this class.  The hidden state is reset
    whenever a new rollout begins (detected by ``self.training`` flag
    or explicit call to ``reset_hidden``).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 2,
        # Mamba3-specific (ignored by fallback)
        headdim: int = 32,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        chunk_size: int = 16,
    ) -> None:
        super().__init__(observation_space, features_dim=d_model)

        # Infer obs dims from space
        spaces = observation_space.spaces
        self_state_dim = spaces["self_state"].shape[-1]
        target_state_dim = spaces["target_state"].shape[-1]
        obstacle_state_dim = spaces["obstacle_state"].shape[-1]

        self.obs_encoder = ObsEncoder(
            self_state_dim=self_state_dim,
            target_state_dim=target_state_dim,
            obstacle_state_dim=obstacle_state_dim,
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
        """Reset all layer hidden states (call at episode boundaries)."""
        self._hidden = [None] * self.n_layers
        self._batch_size = batch_size

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process observations through ObsEncoder + Mamba stack.

        Handles both:
        - Training: (B, d_model) single-step, uses recurrent ``step()``
        - SB3 internal: automatic shape handling
        """
        # Encode obs dict → (B, d_model)
        x = self.obs_encoder(observations)
        B = x.shape[0]

        # Reset hidden if batch size changed (new rollout)
        if self._hidden is None or self._batch_size != B:
            self.reset_hidden(B)

        if self.training:
            # During training, SB3 PPO passes the entire rollout buffer
            # as a flat batch (n_steps * n_envs, obs_dim).
            # We process each sample independently (no sequence modeling).
            # The recurrent benefit comes during inference / rollout collection.
            #
            # For proper sequence training, one would need to reshape into
            # (n_envs, n_steps, d_model) -- but SB3's PPO shuffles minibatches,
            # breaking temporal order. So we treat it as a deep MLP-like
            # feature extractor during the gradient step.
            x_seq = x.unsqueeze(1)  # (B, 1, d_model)
            for layer, norm in zip(self.mamba_layers, self.layer_norms):
                if USE_REAL_MAMBA:
                    # Mamba3 expects bf16 input
                    x_bf16 = x_seq.to(torch.bfloat16)
                    out = layer(x_bf16).to(x_seq.dtype)
                else:
                    out = layer(x_seq)
                x_seq = norm(x_seq + out)
            return x_seq.squeeze(1)  # (B, d_model)
        else:
            # Inference / rollout: step-by-step recurrent
            h = x  # (B, d_model)
            for i, (layer, norm) in enumerate(zip(self.mamba_layers, self.layer_norms)):
                if USE_REAL_MAMBA:
                    # Mamba3 step-by-step: wrap as (B, 1, D) sequence
                    h_bf16 = h.unsqueeze(1).to(torch.bfloat16)
                    out = layer(h_bf16).squeeze(1).to(h.dtype)
                    h = norm(h + out)
                else:
                    out, new_state = layer.step(h, self._hidden[i])
                    self._hidden[i] = new_state
                    h = norm(h + out)
            return h  # (B, d_model)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Custom SB3 Actor-Critic Policy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MambaActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy using Mamba3 as the feature extractor.

    Drop-in replacement for ``MultiInputPolicy`` in SB3's PPO.

    Usage:
        model = PPO(MambaActorCriticPolicy, env, policy_kwargs={...})

    policy_kwargs can include:
        mamba_d_model     : int   (default 128)
        mamba_d_state     : int   (default 64)
        mamba_n_layers    : int   (default 2)
        mamba_headdim     : int   (default 32)
        mamba_is_mimo     : bool  (default True)
        mamba_mimo_rank   : int   (default 4)
        mamba_chunk_size  : int   (default 16)
        net_arch          : dict  (actor/critic MLP arch after features)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        # Mamba config (passed via policy_kwargs)
        mamba_d_model: int = 128,
        mamba_d_state: int = 64,
        mamba_n_layers: int = 2,
        mamba_headdim: int = 32,
        mamba_is_mimo: bool = True,
        mamba_mimo_rank: int = 4,
        mamba_chunk_size: int = 16,
        # Standard SB3 policy kwargs
        net_arch: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> None:
        # Store Mamba config before super().__init__ calls _build()
        self._mamba_cfg = dict(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            n_layers=mamba_n_layers,
            headdim=mamba_headdim,
            is_mimo=mamba_is_mimo,
            mimo_rank=mamba_mimo_rank,
            chunk_size=mamba_chunk_size,
        )

        if net_arch is None:
            # Small actor/critic heads on top of Mamba features
            net_arch = dict(pi=[128, 64], vf=[128, 64])

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            features_extractor_class=MambaFeatureExtractor,
            features_extractor_kwargs=self._mamba_cfg,
            **kwargs,
        )

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        """Override to reset Mamba hidden state on episode boundaries."""
        if episode_start is not None and np.any(episode_start):
            self.features_extractor.reset_hidden(batch_size=1)
        return super().predict(observation, state, episode_start, deterministic)
