from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class CheckpointMetadata:
    step: int
    seed: int | None = None


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    metadata: CheckpointMetadata | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model state, optimizer state, and lightweight experiment metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata,
        "extra": extra or {},
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into a model and optionally an optimizer."""
    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])

    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    return payload
