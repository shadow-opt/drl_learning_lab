from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def export_to_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str | Path,
    input_name: str = "input",
    output_name: str = "output",
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export an inference-only PyTorch module to ONNX."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        }

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (example_input,),
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
        )

    return path
