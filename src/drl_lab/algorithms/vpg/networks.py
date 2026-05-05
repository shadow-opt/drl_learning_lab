from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.distributions import Categorical

from drl_lab.common.networks import MLP


class CategoricalPolicy(nn.Module):
    """Categorical actor for discrete-action policy gradients.

    Input shape: ``obs`` is ``[batch, obs_dim]``.
    Output shape: logits are ``[batch, n_actions]``.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.logits_net = MLP(obs_dim, hidden_sizes or [64, 64], n_actions, activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.logits_net(obs)
        return logits

    def distribution(self, obs: torch.Tensor) -> Categorical:
        distribution: Categorical = Categorical(logits=self(obs))  # type: ignore[no-untyped-call]
        return distribution

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        log_probs = self.distribution(obs).log_prob(actions)  # type: ignore[no-untyped-call]
        return cast(torch.Tensor, log_probs)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample actions for a batch of observations.

        Input shape: ``[batch, obs_dim]``.
        Output shape: ``[batch]``.
        """
        actions = self.distribution(obs).sample()  # type: ignore[no-untyped-call]
        return cast(torch.Tensor, actions)


class ValueFunction(nn.Module):
    """State-value baseline.

    Input shape: ``obs`` is ``[batch, obs_dim]``.
    Output shape: values are ``[batch]``.
    """

    def __init__(self, obs_dim: int, hidden_sizes: list[int] | None = None) -> None:
        super().__init__()
        self.value_net = MLP(obs_dim, hidden_sizes or [64, 64], 1, activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        values: torch.Tensor = self.value_net(obs).squeeze(-1)
        return values
