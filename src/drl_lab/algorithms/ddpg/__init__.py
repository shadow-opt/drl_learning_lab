"""Deep Deterministic Policy Gradient components."""

from drl_lab.algorithms.ddpg.agent import DDPGAgent
from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBatch, ContinuousReplayBuffer
from drl_lab.algorithms.ddpg.config import DDPGConfig
from drl_lab.algorithms.ddpg.losses import actor_loss, critic_loss, soft_update
from drl_lab.algorithms.ddpg.networks import ContinuousQNetwork, DeterministicActor

__all__ = [
    "ContinuousQNetwork",
    "ContinuousReplayBatch",
    "ContinuousReplayBuffer",
    "DDPGAgent",
    "DDPGConfig",
    "DeterministicActor",
    "actor_loss",
    "critic_loss",
    "soft_update",
]
