from __future__ import annotations

import torch

from drl_lab.common.export import export_to_onnx, export_to_torch_export, export_to_torchscript
from drl_lab.common.networks import MLP
from drl_lab.common.onnx_check import (
    compare_pytorch_exported_program,
    compare_pytorch_onnx,
    compare_pytorch_torchscript,
)
from drl_lab.common.seed import set_global_seed


def main() -> None:
    set_global_seed(61)
    model = MLP(input_dim=4, hidden_sizes=[16], output_dim=2)
    example_input = torch.randn(3, 4)

    onnx_path = export_to_onnx(model, example_input, "experiments/runs/export_demo/model.onnx")
    torchscript_path = export_to_torchscript(
        model,
        example_input,
        "experiments/runs/export_demo/model.ts",
    )
    torch_export_path = export_to_torch_export(
        model,
        example_input,
        "experiments/runs/export_demo/model.pt2",
    )

    print("onnx:", compare_pytorch_onnx(model, onnx_path, example_input))
    print("torchscript:", compare_pytorch_torchscript(model, torchscript_path, example_input))
    print(
        "torch_export:",
        compare_pytorch_exported_program(model, torch_export_path, example_input),
    )


if __name__ == "__main__":
    main()
