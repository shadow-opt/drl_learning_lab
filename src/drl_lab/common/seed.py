from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedReport:
    seed: int
    deterministic_torch: bool
    cuda_available: bool


def set_global_seed(seed: int, deterministic_torch: bool = True) -> SeedReport:
    """Set Python, NumPy, and PyTorch seeds for repeatable experiments."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

    return SeedReport(
        seed=seed,
        deterministic_torch=deterministic_torch,
        cuda_available=torch.cuda.is_available(),
    )
