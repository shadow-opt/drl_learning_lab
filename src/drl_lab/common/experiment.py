from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def save_config_snapshot(config: object, run_dir: str | Path) -> Path:
    """Save a JSON snapshot of a dataclass-style experiment config."""
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")

    path = Path(run_dir) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(asdict(config))  # type: ignore[arg-type]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def save_environment_snapshot(run_dir: str | Path) -> Path:
    """Save lightweight runtime metadata needed to reproduce a small run."""
    path = Path(run_dir) / "environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "onnx": _package_version("onnx"),
        "onnxruntime": _package_version("onnxruntime"),
        "gymnasium": _package_version("gymnasium"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def save_run_snapshots(config: object, run_dir: str | Path) -> tuple[Path, Path]:
    """Save config and environment snapshots for an experiment run."""
    return save_config_snapshot(config, run_dir), save_environment_snapshot(run_dir)


def save_eval_report(run_dir: str | Path, metrics: Mapping[str, Any]) -> Path:
    """Save final evaluation metrics in a stable JSON artifact."""
    path = Path(run_dir) / "eval_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(dict(metrics)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def save_export_report(run_dir: str | Path, artifacts: Mapping[str, Any]) -> Path:
    """Save export paths, tensor metadata, and consistency check results."""
    path = Path(run_dir) / "export_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"artifacts": _json_ready(dict(artifacts))}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
