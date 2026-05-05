from __future__ import annotations

import torch
from torch.nn import functional as F

from drl_lab.algorithms.dqn.buffer import ReplayBatch
from drl_lab.algorithms.dqn.networks import QNetwork


def dqn_loss(
    q_network: QNetwork,
    target_q_network: QNetwork,
    batch: ReplayBatch,
    gamma: float,
) -> torch.Tensor:
    """Compute the standard DQN one-step TD loss."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    q_values = q_network(batch.obs)
    chosen_q = q_values.gather(dim=1, index=batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_q_network(batch.next_obs).max(dim=1).values
        target = batch.rewards + gamma * (1.0 - batch.dones) * next_q

    return F.smooth_l1_loss(chosen_q, target)
