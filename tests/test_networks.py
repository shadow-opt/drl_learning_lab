from __future__ import annotations

import pytest
import torch

from drl_lab.common.networks import MLP


def test_mlp_forward_shape() -> None:
    model = MLP(input_dim=4, hidden_sizes=[8, 8], output_dim=2)
    output = model(torch.zeros(6, 4))
    assert output.shape == (6, 2)


def test_mlp_rejects_rank_one_input() -> None:
    model = MLP(input_dim=4, hidden_sizes=[8], output_dim=2)
    with pytest.raises(ValueError, match="expected"):
        model(torch.zeros(4))
