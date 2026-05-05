"""Proximal Policy Optimization components."""

from drl_lab.algorithms.ppo.buffer import PPOTrajectoryBatch, compute_gae
from drl_lab.algorithms.ppo.config import PPOConfig
from drl_lab.algorithms.ppo.losses import approx_kl, clipped_policy_loss, entropy_bonus
from drl_lab.algorithms.vpg.networks import CategoricalPolicy, ValueFunction

__all__ = [
    "CategoricalPolicy",
    "PPOConfig",
    "PPOTrajectoryBatch",
    "ValueFunction",
    "approx_kl",
    "clipped_policy_loss",
    "compute_gae",
    "entropy_bonus",
]
