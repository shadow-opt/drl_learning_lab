from __future__ import annotations

import pytest
import torch

from drl_lab.algorithms.ddpg.networks import ContinuousQNetwork
from drl_lab.common.export import export_to_onnx, export_to_onnx_multi_input
from drl_lab.common.networks import MLP
from drl_lab.common.onnx_check import compare_pytorch_onnx, compare_pytorch_onnx_multi_input
from drl_lab.common.seed import set_global_seed


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_mlp_onnx_matches_pytorch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(11)
    model = MLP(input_dim=4, hidden_sizes=[8], output_dim=2)
    example_input = torch.randn(5, 4)
    onnx_path = tmp_path / "mlp.onnx"

    export_to_onnx(model, example_input, onnx_path)
    result = compare_pytorch_onnx(model, onnx_path, example_input, atol=1e-5)

    assert result.passed
    assert result.max_abs_diff <= 1e-5


def test_multi_input_onnx_matches_pytorch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(15)
    model = ContinuousQNetwork(obs_dim=3, action_dim=2, hidden_sizes=[8])
    obs = torch.randn(5, 3)
    actions = torch.randn(5, 2)
    onnx_path = export_to_onnx_multi_input(
        model,
        (obs, actions),
        tmp_path / "critic.onnx",
        input_names=["obs", "actions"],
        output_names=["q"],
    )
    result = compare_pytorch_onnx_multi_input(model, onnx_path, (obs, actions), atol=1e-5)

    assert result.passed
    assert result.max_abs_diff <= 1e-5
