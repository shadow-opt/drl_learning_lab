from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DQNConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0
    total_steps: int = 5_000
    learning_starts: int = 500
    train_frequency: int = 1
    target_update_frequency: int = 250
    eval_frequency: int = 1_000
    eval_episodes: int = 5
    buffer_capacity: int = 50_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    exploration_fraction: float = 0.4
    max_grad_norm: float = 10.0
    hidden_size: int = 128
    run_dir: Path = Path("experiments/runs/dqn_cartpole")
