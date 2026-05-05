from __future__ import annotations

import torch

from drl_lab.common.networks import MLP
from drl_lab.common.seed import set_global_seed


def test_seed_repeats_model_initialization() -> None:
    set_global_seed(123)
    model_a = MLP(input_dim=3, hidden_sizes=[8], output_dim=2)
    params_a = [param.detach().clone() for param in model_a.parameters()]

    set_global_seed(123)
    model_b = MLP(input_dim=3, hidden_sizes=[8], output_dim=2)
    params_b = [param.detach().clone() for param in model_b.parameters()]

    for left, right in zip(params_a, params_b, strict=True):
        assert torch.equal(left, right)
