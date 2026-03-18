"""PPO trainer with valid-mask support for Policy-Based Go-Explore."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from gym_pybullet_drones.our_experiments.policy_based_go_explore.config import GoExploreConfig
from gym_pybullet_drones.our_experiments.policy_based_go_explore.networks import GoalConditionedActorCritic
from gym_pybullet_drones.our_experiments.policy_based_go_explore.rollout_buffer import GoExploreRolloutBuffer


class PPOTrainer:
    """Runs the PPO clipped-objective update on collected rollouts.

    Parameters
    ----------
    model : GoalConditionedActorCritic
        The policy + value network.
    config : GoExploreConfig
        Hyper-parameters (lr, clip_eps, entropy_coef, …).
    """

    def __init__(
        self,
        model: GoalConditionedActorCritic,
        config: GoExploreConfig,
    ) -> None:
        self.model = model
        self.cfg = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.device = torch.device(config.device)

    def update(self, buffer: GoExploreRolloutBuffer) -> Dict[str, float]:
        """Run *n_epochs* of PPO mini-batch updates on the buffer.

        Returns a dict of aggregated loss metrics.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        n_updates = 0

        for _epoch in range(self.cfg.n_epochs):
            for batch in buffer.get_batches(self.cfg.batch_size, device=self.device):
                obs = batch["obs"]
                goal = batch["goal"]
                phase = batch["phase"]
                actions = batch["action"]
                old_log_probs = batch["old_log_prob"]
                advantages = batch["advantage"]
                returns = batch["return"]
                valid_mask = batch["valid_mask"]
                gru_hidden = batch["gru_hidden"]

                # forward pass
                new_log_probs, entropy, new_values = self.model.evaluate_actions(
                    obs, goal, phase, actions, gru_hidden,
                )

                # ── policy loss (PPO clip) ───────────────────────
                ratio = (new_log_probs - old_log_probs).exp()
                adv = advantages.unsqueeze(-1)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2)

                # ── value loss ───────────────────────────────────
                value_loss = (new_values - returns.unsqueeze(-1)).pow(2)

                # ── entropy bonus ────────────────────────────────
                entropy_bonus = entropy

                # ── combine with valid mask ──────────────────────
                mask = valid_mask.unsqueeze(-1)  # (batch, 1)
                loss = (
                    policy_loss * mask
                    + self.cfg.vf_coef * value_loss * mask
                    - self.cfg.entropy_coef * entropy_bonus * mask
                ).mean()

                # ── optimise ─────────────────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.mean().item()
                total_value_loss += value_loss.mean().item()
                total_entropy_loss += entropy_bonus.mean().item()
                total_loss += loss.item()
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy_loss / n_updates,
            "total_loss": total_loss / n_updates,
        }
