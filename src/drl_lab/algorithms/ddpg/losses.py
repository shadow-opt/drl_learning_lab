from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBatch
from drl_lab.algorithms.ddpg.networks import ContinuousQNetwork, DeterministicActor


def critic_loss(
    critic: ContinuousQNetwork,
    target_actor: DeterministicActor,
    target_critic: ContinuousQNetwork,
    batch: ContinuousReplayBatch,
    gamma: float,
) -> torch.Tensor:
    """Compute DDPG Bellman critic loss."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    q_values = critic(batch.obs, batch.actions)
    with torch.no_grad():
        next_actions = target_actor(batch.next_obs)
        next_q = target_critic(batch.next_obs, next_actions)
        target = batch.rewards + gamma * (1.0 - batch.dones) * next_q
    return F.mse_loss(q_values, target)


def actor_loss(
    actor: DeterministicActor,
    critic: ContinuousQNetwork,
    obs: torch.Tensor,
) -> torch.Tensor:
    """Compute deterministic policy gradient actor objective."""
    actions = actor(obs)
    loss: torch.Tensor = -critic(obs, actions).mean()
    return loss


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak-average target parameters toward source parameters."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    with torch.no_grad():
        for source_param, target_param in zip(
            source.parameters(),
            target.parameters(),
            strict=True,
        ):
            target_param.mul_(1.0 - tau)
            target_param.add_(tau * source_param)
