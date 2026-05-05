from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrajectoryBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    log_probs: torch.Tensor


def discount_cumsum(values: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute discounted cumulative sums.

    Input shape: ``[time]``.
    Output shape: ``[time]`` where output[t] = values[t] + discount * output[t+1].
    """
    if values.ndim != 1:
        raise ValueError(f"expected [time], got shape {tuple(values.shape)}")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0, 1]")

    result = torch.zeros_like(values)
    running_sum = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in range(values.shape[0] - 1, -1, -1):
        running_sum = values[index] + discount * running_sum
        result[index] = running_sum
    return result


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to mean 0 and standard deviation 1."""
    if advantages.ndim != 1:
        raise ValueError(f"expected [batch], got shape {tuple(advantages.shape)}")
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def make_trajectory_batch(
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    gamma: float,
    normalize_advantage: bool = True,
) -> TrajectoryBatch:
    """Build VPG training tensors from one trajectory.

    Shapes:
        obs: ``[time, obs_dim]``
        actions: ``[time]``
        rewards: ``[time]``
        values: ``[time]``
        log_probs: ``[time]``
    """
    if obs.ndim != 2:
        raise ValueError(f"expected obs [time, obs_dim], got {tuple(obs.shape)}")
    time = obs.shape[0]
    for name, tensor in {
        "actions": actions,
        "rewards": rewards,
        "values": values,
        "log_probs": log_probs,
    }.items():
        if tensor.shape != (time,):
            raise ValueError(f"expected {name} shape {(time,)}, got {tuple(tensor.shape)}")

    returns = discount_cumsum(rewards, gamma)
    advantages = returns - values
    if normalize_advantage:
        advantages = normalize_advantages(advantages)

    return TrajectoryBatch(
        obs=obs,
        actions=actions,
        returns=returns,
        advantages=advantages,
        log_probs=log_probs,
    )
