"""
TRLOO-augmented GRPOTrainer — fixes the 25% gradient shrinkage from GRPO self-inclusion bias.

Dr. Kernel (arXiv 2602.05885) proves that GRPO's advantage estimation includes
the current sample in its own baseline, causing E[gradient] = (1 - 1/N) * true_gradient.
With G=4, gradients are systematically 25% too small.

Fix: scale advantages by N/(N-1) after GRPO computes them. This is the TRLOO
(Turn-level Reinforce Leave-One-Out) correction — mathematically equivalent to
computing the baseline from the other G-1 samples.

MARS return-to-go was considered but dropped: with outcome-only rewards (no per-turn
signal), MARS degenerates to standard trajectory-level GRPO (see ALPHXIV analysis).
"""
from __future__ import annotations

import torch
from trl import GRPOTrainer, GRPOConfig


class TRLOOGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with TRLOO advantage correction.

    Drop-in replacement: just swap GRPOTrainer → TRLOOGRPOTrainer.
    Everything else (reward_funcs, rollout_func, config) stays the same.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trloo_enabled = True

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantages with TRLOO N/(N-1) correction.

        TRL's GRPOTrainer computes advantages as:
            A_i = (r_i - mean(r)) / (std(r) + eps)

        This includes sample i in its own baseline, causing (1-1/N) gradient shrinkage.
        We apply the correction after the base computation.
        """
        if torch.isnan(rewards).any():
            return self._compute_masked_advantages(rewards)

        # Let parent compute vanilla GRPO advantages
        advantages = super()._compute_advantages(rewards)

        if not self._trloo_enabled:
            return advantages

        # Apply TRLOO correction per group
        # rewards shape: (batch_size * num_generations,)
        # Each group of num_generations consecutive entries shares a prompt
        G = self.args.num_generations
        if G <= 1:
            return advantages

        scale = G / (G - 1.0)

        # Scale all advantages — the N/(N-1) factor is uniform within each group
        advantages = advantages * scale

        return advantages

    def _compute_masked_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages while masking NaN rewards."""
        G = int(self.args.num_generations)
        if G <= 0 or rewards.numel() % G != 0:
            raise ValueError(
                f"Expected rewards shaped as groups of {G}, got tensor with {rewards.numel()} elements."
            )

        grouped_rewards = rewards.view(-1, G)
        grouped_advantages = torch.zeros_like(grouped_rewards)
        masked_count = int(torch.isnan(grouped_rewards).sum().item())
        eps = 1e-6

        for group_idx in range(grouped_rewards.shape[0]):
            group = grouped_rewards[group_idx]
            valid_mask = torch.isfinite(group)
            valid_count = int(valid_mask.sum().item())
            if valid_count == 0:
                continue

            valid_rewards = group[valid_mask]
            mean = valid_rewards.mean()
            std = valid_rewards.std(unbiased=False)
            if float(std.item()) == 0.0:
                valid_advantages = torch.zeros_like(valid_rewards)
            else:
                valid_advantages = (valid_rewards - mean) / (std + eps)

            if self._trloo_enabled and valid_count > 1:
                valid_advantages = valid_advantages * (valid_count / (valid_count - 1.0))

            grouped_advantages[group_idx, valid_mask] = valid_advantages

        if masked_count:
            print(
                f"[trloo] masked_invalid_rewards={masked_count} "
                f"groups={grouped_rewards.shape[0]} G={G}"
            )

        return grouped_advantages.view_as(rewards)


def create_trloo_trainer(
    model,
    tokenizer,
    reward_funcs,
    train_dataset,
    config: GRPOConfig,
    rollout_func=None,
) -> TRLOOGRPOTrainer:
    """Factory function to create a TRLOO-augmented GRPO trainer.

    Args:
        model: The model to train (with LoRA already applied).
        tokenizer: The tokenizer / processing_class.
        reward_funcs: Reward function(s) for GRPO.
        train_dataset: HF Dataset with 'prompt' column.
        config: GRPOConfig with training hyperparameters.
        rollout_func: Optional custom rollout (for multi-turn OpenEnv).

    Returns:
        TRLOOGRPOTrainer ready to .train()
    """
    kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "reward_funcs": reward_funcs,
        "args": config,
        "train_dataset": train_dataset,
    }
    if rollout_func is not None:
        kwargs["rollout_func"] = rollout_func

    return TRLOOGRPOTrainer(**kwargs)
