"""PPO + SIL trainer for Go-Explore Phase 2.

Combines:
  - Standard clipped PPO (online policy gradient)
  - Self-Imitation Learning loss (offline / high-return imitation)

The SIL loss is only applied when a SIL buffer is provided and
``sil_coef > 0``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from gym_pybullet_drones.our_experiments.go_explore.networks import ActorCritic
from gym_pybullet_drones.our_experiments.go_explore.rollout_buffer import RolloutBuffer
from gym_pybullet_drones.our_experiments.go_explore.sil_buffer import SILBuffer


class PPOTrainer:
    def __init__(
        self,
        model: ActorCritic,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 256,
        sil_coef: float = 0.1,
        sil_batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sil_coef = sil_coef
        self.sil_batch_size = sil_batch_size
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    def update(
        self,
        buffer: RolloutBuffer,
        sil_buffer: Optional[SILBuffer] = None,
    ) -> Dict[str, float]:
        """Run PPO + SIL update.

        Parameters
        ----------
        buffer : RolloutBuffer
            Online rollout data for PPO.
        sil_buffer : SILBuffer, optional
            Replay buffer with demo / high-return transitions for SIL.
        """
        tot_pl, tot_vl, tot_ent, tot_sil, tot_loss, n = (
            0.0, 0.0, 0.0, 0.0, 0.0, 0)

        for _ in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size, self.device):
                new_lp, entropy, new_val = self.model.evaluate_actions(
                    batch["obs"], batch["action"], batch["gru_hidden"],
                )

                # --- NaN guard ---
                if torch.isnan(new_lp).any() or torch.isnan(new_val).any():
                    continue

                ratio = (new_lp - batch["old_log_prob"]).exp()
                ratio = ratio.clamp(0.0, 10.0)

                adv = batch["advantage"].unsqueeze(-1)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - self.clip_eps,
                                 1 + self.clip_eps) * adv
                p_loss = -torch.min(s1, s2).mean()
                v_loss = (new_val - batch["return"].unsqueeze(-1)).pow(2).mean()
                ent_loss = -entropy.mean()
                loss = (p_loss
                        + self.vf_coef * v_loss
                        + self.entropy_coef * ent_loss)

                # --- SIL loss ---
                sil_loss_val = torch.tensor(0.0, device=self.device)
                if (sil_buffer is not None and self.sil_coef > 0
                        and len(sil_buffer) > 0):
                    sil_batch = sil_buffer.sample_batch(
                        self.sil_batch_size, self.device)
                    if sil_batch is not None:
                        sil_loss_val = self._compute_sil_loss(sil_batch)
                        loss = loss + self.sil_coef * sil_loss_val

                # --- NaN/Inf guard on total loss ---
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                tot_pl += p_loss.item()
                tot_vl += v_loss.item()
                tot_ent += entropy.mean().item()
                tot_sil += sil_loss_val.item()
                tot_loss += loss.item()
                n += 1

        n = max(n, 1)
        return {
            "policy_loss": tot_pl / n,
            "value_loss": tot_vl / n,
            "entropy": tot_ent / n,
            "sil_loss": tot_sil / n,
            "total_loss": tot_loss / n,
        }

    def _compute_sil_loss(
        self, sil_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """SIL loss: clipped advantage-weighted policy imitation.

        loss = -E[ max(0, R_demo - V(s)) * log π(a_demo | s) ]

        Only imitates when the demo return exceeds the current value estimate.
        """
        # We use a zero hidden state for SIL (single-step evaluation)
        hidden = self.model.initial_hidden(
            sil_batch["action"].shape[0]).to(self.device)

        sil_lp, _, sil_val = self.model.evaluate_actions(
            sil_batch["obs"], sil_batch["action"], hidden)

        demo_returns = sil_batch["return"].unsqueeze(-1)
        advantage = (demo_returns - sil_val.detach()).clamp(min=0.0)

        # Policy imitation loss (weighted by positive advantage)
        policy_sil = -(advantage * sil_lp).mean()
        # Value regression toward demo returns (only where demo is better)
        value_sil = (advantage * (demo_returns - sil_val).pow(2)).mean()

        return policy_sil + 0.5 * value_sil
