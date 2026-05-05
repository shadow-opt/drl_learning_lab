from __future__ import annotations

import torch

from drl_lab.common.export import export_to_torch_export, export_to_torchscript
from drl_lab.common.networks import MLP
from drl_lab.common.onnx_check import (
    compare_pytorch_exported_program,
    compare_pytorch_torchscript,
)
from drl_lab.common.seed import set_global_seed


def test_torchscript_matches_pytorch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(62)
    model = MLP(input_dim=4, hidden_sizes=[8], output_dim=2)
    example_input = torch.randn(3, 4)
    path = export_to_torchscript(model, example_input, tmp_path / "model.ts")

    result = compare_pytorch_torchscript(model, path, example_input)

    assert result.passed
    assert result.max_abs_diff <= 1e-5


def test_torch_export_matches_pytorch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(63)
    model = MLP(input_dim=4, hidden_sizes=[8], output_dim=2)
    example_input = torch.randn(3, 4)
    path = export_to_torch_export(model, example_input, tmp_path / "model.pt2")

    result = compare_pytorch_exported_program(model, path, example_input)

    assert result.passed
    assert result.max_abs_diff <= 1e-5
