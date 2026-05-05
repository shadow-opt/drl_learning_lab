from __future__ import annotations

import torch
from torch.nn import functional as F

from drl_lab.algorithms.vpg.networks import CategoricalPolicy, ValueFunction


def policy_loss(
    policy: CategoricalPolicy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Compute VPG actor loss: -E[log pi(a|s) * advantage]."""
    if actions.shape != advantages.shape:
        raise ValueError("actions and advantages must have the same shape")
    log_probs = policy.log_prob(obs, actions)
    return -(log_probs * advantages).mean()


def value_loss(
    value_function: ValueFunction,
    obs: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """Compute mean squared error value-function loss."""
    values = value_function(obs)
    if values.shape != returns.shape:
        raise ValueError("predicted values and returns must have the same shape")
    return F.mse_loss(values, returns)
