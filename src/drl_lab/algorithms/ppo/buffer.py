from __future__ import annotations

from dataclasses import dataclass

import torch

from drl_lab.algorithms.vpg.buffer import discount_cumsum, normalize_advantages


@dataclass(frozen=True)
class PPOTrajectoryBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE-lambda advantages and returns for one trajectory.

    Shapes:
        rewards: ``[time]``
        values: ``[time + 1]`` with a bootstrap value at the end
        dones: ``[time]`` where 1 means terminal
    """
    if rewards.ndim != 1 or dones.ndim != 1:
        raise ValueError("rewards and dones must have shape [time]")
    if values.shape != (rewards.shape[0] + 1,):
        raise ValueError("values must have shape [time + 1]")
    if dones.shape != rewards.shape:
        raise ValueError("dones must match rewards shape")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")

    deltas = rewards + gamma * values[1:] * (1.0 - dones) - values[:-1]
    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.shape[0] - 1, -1, -1):
        running_advantage = deltas[index] + gamma * lam * (1.0 - dones[index]) * running_advantage
        advantages[index] = running_advantage

    returns = advantages + values[:-1]
    return advantages, returns


def make_ppo_batch(
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    old_log_probs: torch.Tensor,
    gamma: float,
    lam: float,
    normalize_advantage: bool = True,
) -> PPOTrajectoryBatch:
    """Build PPO training tensors from one trajectory."""
    if obs.ndim != 2:
        raise ValueError(f"expected obs [time, obs_dim], got {tuple(obs.shape)}")
    time = obs.shape[0]
    for name, tensor in {
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "old_log_probs": old_log_probs,
    }.items():
        if tensor.shape != (time,):
            raise ValueError(f"expected {name} shape {(time,)}, got {tuple(tensor.shape)}")

    advantages, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
    if normalize_advantage:
        advantages = normalize_advantages(advantages)

    return PPOTrajectoryBatch(
        obs=obs,
        actions=actions,
        returns=returns,
        advantages=advantages,
        old_log_probs=old_log_probs,
    )


def reward_to_go_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Expose reward-to-go for PPO/VPG comparison exercises."""
    return discount_cumsum(rewards, gamma)
