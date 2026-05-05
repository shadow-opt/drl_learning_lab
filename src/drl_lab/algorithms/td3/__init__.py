"""Twin Delayed Deep Deterministic Policy Gradient components."""

from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBatch, ContinuousReplayBuffer
from drl_lab.algorithms.ddpg.losses import actor_loss, soft_update
from drl_lab.algorithms.ddpg.networks import ContinuousQNetwork, DeterministicActor
from drl_lab.algorithms.td3.agent import TD3Agent
from drl_lab.algorithms.td3.config import TD3Config
from drl_lab.algorithms.td3.losses import clipped_target_actions, td3_critic_loss
from drl_lab.algorithms.td3.networks import TwinContinuousQNetwork

__all__ = [
    "ContinuousQNetwork",
    "ContinuousReplayBatch",
    "ContinuousReplayBuffer",
    "DeterministicActor",
    "TD3Agent",
    "TD3Config",
    "TwinContinuousQNetwork",
    "actor_loss",
    "clipped_target_actions",
    "soft_update",
    "td3_critic_loss",
]
