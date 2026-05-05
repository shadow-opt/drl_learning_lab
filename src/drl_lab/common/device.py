from __future__ import annotations

import torch


def resolve_device(preferred: str | None = None) -> torch.device:
    """Resolve a user preference into a concrete torch device."""
    if preferred is None or preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def move_batch_to_device(
    batch: torch.Tensor | dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Move a tensor or tensor dictionary to a device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return {key: value.to(device) for key, value in batch.items()}
