from __future__ import annotations

from pathlib import Path

import torch

from drl_lab.common.export import export_to_onnx
from drl_lab.common.networks import MLP
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def main() -> None:
    set_global_seed(7)
    model = MLP(input_dim=4, hidden_sizes=[16, 16], output_dim=2)
    example_input = torch.randn(3, 4, dtype=torch.float32)

    output_path = Path("experiments/runs/mlp_export_demo/model.onnx")
    export_to_onnx(model, example_input, output_path)
    result = compare_pytorch_onnx(model, output_path, example_input)

    print(f"exported={output_path}")
    print(f"max_abs_diff={result.max_abs_diff:.8f}")
    print(f"mean_abs_diff={result.mean_abs_diff:.8f}")
    print(f"passed={result.passed}")


if __name__ == "__main__":
    main()
