from __future__ import annotations

import torch

from drl_lab.algorithms.vpg.networks import CategoricalPolicy


def clipped_policy_loss(
    policy: CategoricalPolicy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PPO-Clip policy loss and approximate KL.

    Returns:
        loss: scalar tensor
        ratio: shape ``[batch]`` probability ratio for diagnostics
    """
    if actions.shape != advantages.shape or actions.shape != old_log_probs.shape:
        raise ValueError("actions, advantages, and old_log_probs must have shape [batch]")
    if clip_ratio <= 0.0:
        raise ValueError("clip_ratio must be positive")

    new_log_probs = policy.log_prob(obs, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    loss = -torch.min(unclipped, clipped).mean()
    return loss, ratio


def approx_kl(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
    """Approximate KL(old || new) for early stopping diagnostics."""
    if old_log_probs.shape != new_log_probs.shape:
        raise ValueError("log-prob tensors must have the same shape")
    return (old_log_probs - new_log_probs).mean()


def entropy_bonus(policy: CategoricalPolicy, obs: torch.Tensor) -> torch.Tensor:
    """Mean categorical policy entropy."""
    entropy = policy.distribution(obs).entropy()  # type: ignore[no-untyped-call]
    if not isinstance(entropy, torch.Tensor):
        raise TypeError("expected entropy to be a tensor")
    return entropy.mean()
