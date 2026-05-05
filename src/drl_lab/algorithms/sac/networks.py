from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from drl_lab.common.networks import MLP


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor for continuous actions.

    Input shape: ``obs`` is ``[batch, obs_dim]``.
    Stochastic sample output:
        actions: ``[batch, action_dim]`` in ``[-action_limit, action_limit]``
        log_probs: ``[batch]`` with tanh correction applied.

    ``forward`` returns deterministic mean actions for inference/export.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        hidden_sizes: list[int] | None = None,
        log_std_bounds: tuple[float, float] = (-20.0, 2.0),
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        if action_limit <= 0:
            raise ValueError("action_limit must be positive")
        min_log_std, max_log_std = log_std_bounds
        if min_log_std >= max_log_std:
            raise ValueError("log_std_bounds must be increasing")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.net = MLP(obs_dim, hidden_sizes or [256, 256], action_dim * 2, activation=nn.ReLU)

    def distribution_params(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian mean and clamped log standard deviation."""
        params = self.net(obs)
        mean, log_std = torch.chunk(params, chunks=2, dim=-1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions with the reparameterization trick."""
        mean, log_std = self.distribution_params(obs)
        std = log_std.exp()
        normal = Normal(mean, std)  # type: ignore[no-untyped-call]
        pre_tanh = normal.rsample()
        squashed = torch.tanh(pre_tanh)
        actions = self.action_limit * squashed

        base_log_probs = normal.log_prob(pre_tanh)  # type: ignore[no-untyped-call]
        correction = torch.log1p(-squashed.pow(2) + 1e-6)
        log_probs: torch.Tensor = (base_log_probs - correction).sum(dim=-1)
        return actions, log_probs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_params(obs)
        actions: torch.Tensor = self.action_limit * torch.tanh(mean)
        return actions
