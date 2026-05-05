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


def export_to_onnx_multi_input(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    output_path: str | Path,
    input_names: list[str],
    output_names: list[str],
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export an inference-only PyTorch module with multiple tensor inputs."""
    if len(example_inputs) != len(input_names):
        raise ValueError("example_inputs and input_names must have the same length")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in [*input_names, *output_names]}

    with torch.inference_mode():
        torch.onnx.export(
            model,
            example_inputs,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    return path


def export_to_torchscript(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str | Path,
) -> Path:
    """Trace and save an inference-only TorchScript module."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(model, example_input)  # type: ignore[no-untyped-call]
        traced.save(str(path))

    return path


def export_to_torch_export(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str | Path,
) -> Path:
    """Export and save a PyTorch ExportedProgram."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    exported = torch.export.export(model, (example_input,))
    torch.export.save(exported, path)
    return path
