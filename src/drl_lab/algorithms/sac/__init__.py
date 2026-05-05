"""Soft Actor-Critic core components."""

from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBatch, ContinuousReplayBuffer
from drl_lab.algorithms.ddpg.losses import soft_update
from drl_lab.algorithms.sac.losses import sac_actor_loss, sac_critic_loss, temperature_loss
from drl_lab.algorithms.sac.networks import SquashedGaussianActor
from drl_lab.algorithms.td3.networks import TwinContinuousQNetwork

__all__ = [
    "ContinuousReplayBatch",
    "ContinuousReplayBuffer",
    "SquashedGaussianActor",
    "TwinContinuousQNetwork",
    "sac_actor_loss",
    "sac_critic_loss",
    "soft_update",
    "temperature_loss",
]
