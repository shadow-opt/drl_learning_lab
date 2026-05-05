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
