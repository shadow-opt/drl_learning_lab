from __future__ import annotations

import torch
from torch import nn

from drl_lab.common.networks import MLP


class QNetwork(nn.Module):
    """Q-network for discrete-action environments.

    Input shape: ``[batch, obs_dim]``.
    Output shape: ``[batch, n_actions]``.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = MLP(obs_dim, hidden_sizes or [64, 64], n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.net(obs)
        return output
