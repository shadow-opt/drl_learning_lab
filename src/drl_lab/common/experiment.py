from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, is_dataclass
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
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def save_run_snapshots(config: object, run_dir: str | Path) -> tuple[Path, Path]:
    """Save config and environment snapshots for an experiment run."""
    return save_config_snapshot(config, run_dir), save_environment_snapshot(run_dir)
