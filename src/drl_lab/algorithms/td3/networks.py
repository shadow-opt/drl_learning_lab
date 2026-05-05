from __future__ import annotations

import torch
from torch import nn

from drl_lab.algorithms.ddpg.networks import ContinuousQNetwork


class TwinContinuousQNetwork(nn.Module):
    """Pair of independent Q-networks used by TD3."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.q1 = ContinuousQNetwork(obs_dim, action_dim, hidden_sizes)
        self.q2 = ContinuousQNetwork(obs_dim, action_dim, hidden_sizes)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1: torch.Tensor = self.q1(obs, actions)
        q2: torch.Tensor = self.q2(obs, actions)
        return q1, q2

    def q1_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1: torch.Tensor = self.q1(obs, actions)
        return q1
