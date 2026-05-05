from __future__ import annotations

import torch
from torch.nn import functional as F

from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBatch
from drl_lab.algorithms.ddpg.networks import DeterministicActor
from drl_lab.algorithms.td3.networks import TwinContinuousQNetwork


def clipped_target_actions(
    target_actor: DeterministicActor,
    next_obs: torch.Tensor,
    target_noise: float,
    noise_clip: float,
) -> torch.Tensor:
    """Apply TD3 target policy smoothing and action clipping."""
    if target_noise < 0.0:
        raise ValueError("target_noise must be non-negative")
    if noise_clip < 0.0:
        raise ValueError("noise_clip must be non-negative")

    actions = target_actor(next_obs)
    if target_noise > 0.0:
        noise = torch.randn_like(actions) * target_noise
        noise = noise.clamp(-noise_clip, noise_clip)
        actions = actions + noise
    clipped_actions: torch.Tensor = actions.clamp(
        -target_actor.action_limit,
        target_actor.action_limit,
    )
    return clipped_actions


def td3_critic_loss(
    critics: TwinContinuousQNetwork,
    target_actor: DeterministicActor,
    target_critics: TwinContinuousQNetwork,
    batch: ContinuousReplayBatch,
    gamma: float,
    target_noise: float,
    noise_clip: float,
) -> torch.Tensor:
    """Compute TD3 twin-critic Bellman loss with min target Q."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    q1, q2 = critics(batch.obs, batch.actions)
    with torch.no_grad():
        next_actions = clipped_target_actions(
            target_actor,
            batch.next_obs,
            target_noise,
            noise_clip,
        )
        target_q1, target_q2 = target_critics(batch.next_obs, next_actions)
        min_target_q = torch.minimum(target_q1, target_q2)
        target = batch.rewards + gamma * (1.0 - batch.dones) * min_target_q

    return F.mse_loss(q1, target) + F.mse_loss(q2, target)
