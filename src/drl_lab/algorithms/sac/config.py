from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SACConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 0
    total_steps: int = 10_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    eval_frequency: int = 2_000
    eval_episodes: int = 5
    buffer_capacity: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: float | None = None
    max_grad_norm: float = 10.0
    hidden_size: int = 256
    run_dir: Path = Path("experiments/runs/sac_pendulum")
