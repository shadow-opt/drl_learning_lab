from __future__ import annotations

import torch
from torch import nn

from drl_lab.common.networks import MLP


class DeterministicActor(nn.Module):
    """Bounded deterministic actor for continuous control.

    Input shape: ``obs`` is ``[batch, obs_dim]``.
    Output shape: actions are ``[batch, action_dim]`` in ``[-action_limit, action_limit]``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if action_limit <= 0:
            raise ValueError("action_limit must be positive")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.net = MLP(obs_dim, hidden_sizes or [256, 256], action_dim, activation=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw_actions = self.net(obs)
        return self.action_limit * torch.tanh(raw_actions)


class ContinuousQNetwork(nn.Module):
    """Q-network for continuous actions.

    Input shapes:
        obs: ``[batch, obs_dim]``
        actions: ``[batch, action_dim]``
    Output shape: ``[batch]``
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.net = MLP(obs_dim + action_dim, hidden_sizes or [256, 256], 1, activation=nn.ReLU)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or actions.ndim != 2:
            raise ValueError("obs and actions must both have rank 2")
        if obs.shape[0] != actions.shape[0]:
            raise ValueError("obs and actions batch dimensions must match")
        inputs = torch.cat([obs, actions], dim=-1)
        values: torch.Tensor = self.net(inputs).squeeze(-1)
        return values
