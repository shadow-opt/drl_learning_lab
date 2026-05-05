from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class ConsistencyResult:
    max_abs_diff: float
    mean_abs_diff: float
    passed: bool


def compare_pytorch_onnx(
    model: nn.Module,
    onnx_path: str | Path,
    example_input: torch.Tensor,
    atol: float = 1e-5,
) -> ConsistencyResult:
    """Compare a PyTorch model output with ONNXRuntime output."""
    model.eval()
    with torch.inference_mode():
        torch_output = model(example_input).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: example_input.detach().cpu().numpy()})[0]

    diff = np.abs(torch_output - onnx_output)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())
    return ConsistencyResult(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        passed=max_abs_diff <= atol,
    )


def compare_pytorch_onnx_multi_input(
    model: nn.Module,
    onnx_path: str | Path,
    example_inputs: tuple[torch.Tensor, ...],
    atol: float = 1e-5,
) -> ConsistencyResult:
    """Compare PyTorch and ONNXRuntime outputs for a multi-input module."""
    model.eval()
    with torch.inference_mode():
        torch_output_raw = model(*example_inputs)
    if isinstance(torch_output_raw, tuple):
        torch_outputs = [output.detach().cpu().numpy() for output in torch_output_raw]
    else:
        torch_outputs = [torch_output_raw.detach().cpu().numpy()]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_feed = {
        input_meta.name: tensor.detach().cpu().numpy()
        for input_meta, tensor in zip(session.get_inputs(), example_inputs, strict=True)
    }
    onnx_outputs = session.run(None, input_feed)

    diffs = [
        np.abs(torch_output - onnx_output)
        for torch_output, onnx_output in zip(torch_outputs, onnx_outputs, strict=True)
    ]
    max_abs_diff = float(max(diff.max() for diff in diffs))
    mean_abs_diff = float(np.mean([diff.mean() for diff in diffs]))
    return ConsistencyResult(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        passed=max_abs_diff <= atol,
    )


def _result_from_outputs(
    torch_output: NDArray[np.float32],
    exported_output: NDArray[np.float32],
    atol: float,
) -> ConsistencyResult:
    diff = np.abs(torch_output - exported_output)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())
    return ConsistencyResult(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        passed=max_abs_diff <= atol,
    )


def compare_pytorch_torchscript(
    model: nn.Module,
    torchscript_path: str | Path,
    example_input: torch.Tensor,
    atol: float = 1e-5,
) -> ConsistencyResult:
    """Compare a PyTorch model output with a TorchScript module output."""
    model.eval()
    with torch.inference_mode():
        torch_output = model(example_input).detach().cpu().numpy()
        scripted = torch.jit.load(  # type: ignore[no-untyped-call]
            str(torchscript_path),
            map_location=example_input.device,
        )
        scripted_output = scripted(example_input).detach().cpu().numpy()

    return _result_from_outputs(torch_output, scripted_output, atol)


def compare_pytorch_exported_program(
    model: nn.Module,
    exported_program_path: str | Path,
    example_input: torch.Tensor,
    atol: float = 1e-5,
) -> ConsistencyResult:
    """Compare a PyTorch model output with a torch.export ExportedProgram output."""
    model.eval()
    with torch.inference_mode():
        torch_output = model(example_input).detach().cpu().numpy()
        exported_program = torch.export.load(exported_program_path)
        exported_module = exported_program.module()
        exported_output = exported_module(example_input).detach().cpu().numpy()

    return _result_from_outputs(torch_output, exported_output, atol)
