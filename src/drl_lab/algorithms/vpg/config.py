from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VPGConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0
    epochs: int = 10
    steps_per_epoch: int = 1_000
    gamma: float = 0.99
    policy_lr: float = 3e-3
    value_lr: float = 1e-3
    value_train_iters: int = 20
    hidden_size: int = 64
    eval_episodes: int = 5
    run_dir: Path = Path("experiments/runs/vpg_cartpole")
