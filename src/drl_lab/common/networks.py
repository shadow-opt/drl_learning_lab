from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """A small fully connected network.

    Input shape: ``[batch, input_dim]``.
    Output shape: ``[batch, output_dim]``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        output_dim: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")

        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            if hidden_dim <= 0:
                raise ValueError("hidden sizes must be positive")
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected [batch, input_dim], got shape {tuple(x.shape)}")
        output: torch.Tensor = self.net(x)
        return output
