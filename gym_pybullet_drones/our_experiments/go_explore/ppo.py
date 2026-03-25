"""Standard PPO trainer for Go-Explore Phase 2."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from gym_pybullet_drones.our_experiments.go_explore.networks import ActorCritic
from gym_pybullet_drones.our_experiments.go_explore.rollout_buffer import RolloutBuffer


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
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        tot_pl, tot_vl, tot_ent, tot_loss, n = 0.0, 0.0, 0.0, 0.0, 0
        for _ in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size, self.device):
                new_lp, entropy, new_val = self.model.evaluate_actions(
                    batch["obs"], batch["action"], batch["gru_hidden"],
                )

                # --- NaN guard: skip bad batches to prevent poisoning ---
                if torch.isnan(new_lp).any() or torch.isnan(new_val).any():
                    continue

                ratio = (new_lp - batch["old_log_prob"]).exp()
                # Clamp ratio to prevent extreme updates
                ratio = ratio.clamp(0.0, 10.0)

                adv = batch["advantage"].unsqueeze(-1)
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                p_loss = -torch.min(s1, s2).mean()
                v_loss = (new_val - batch["return"].unsqueeze(-1)).pow(2).mean()
                loss = p_loss + self.vf_coef * v_loss - self.entropy_coef * entropy.mean()

                # --- NaN guard on loss ---
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                tot_pl += p_loss.item()
                tot_vl += v_loss.item()
                tot_ent += entropy.mean().item()
                tot_loss += loss.item()
                n += 1
        n = max(n, 1)
        return {"policy_loss": tot_pl / n, "value_loss": tot_vl / n,
                "entropy": tot_ent / n, "total_loss": tot_loss / n}
