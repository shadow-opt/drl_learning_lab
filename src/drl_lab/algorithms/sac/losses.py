from __future__ import annotations

import torch
from torch.nn import functional as F

from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBatch
from drl_lab.algorithms.sac.networks import SquashedGaussianActor
from drl_lab.algorithms.td3.networks import TwinContinuousQNetwork


def sac_critic_loss(
    critics: TwinContinuousQNetwork,
    target_actor: SquashedGaussianActor,
    target_critics: TwinContinuousQNetwork,
    batch: ContinuousReplayBatch,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    """Compute SAC twin-critic Bellman loss with entropy-regularized targets."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    q1, q2 = critics(batch.obs, batch.actions)
    with torch.no_grad():
        next_actions, next_log_probs = target_actor.sample(batch.next_obs)
        target_q1, target_q2 = target_critics(batch.next_obs, next_actions)
        min_target_q = torch.minimum(target_q1, target_q2)
        entropy_adjusted_target_q = min_target_q - alpha * next_log_probs
        target = batch.rewards + gamma * (1.0 - batch.dones) * entropy_adjusted_target_q

    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def sac_actor_loss(
    actor: SquashedGaussianActor,
    critics: TwinContinuousQNetwork,
    obs: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute SAC stochastic actor loss."""
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    actions, log_probs = actor.sample(obs)
    q1, q2 = critics(obs, actions)
    min_q = torch.minimum(q1, q2)
    loss: torch.Tensor = (alpha * log_probs - min_q).mean()
    return loss


def temperature_loss(
    log_alpha: torch.Tensor,
    log_probs: torch.Tensor,
    target_entropy: float,
) -> torch.Tensor:
    """Optimize SAC entropy temperature in log-space."""
    loss: torch.Tensor = -(log_alpha * (log_probs.detach() + target_entropy)).mean()
    return loss
