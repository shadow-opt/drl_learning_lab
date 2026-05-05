from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PPOConfig:
    env_id: str = "CartPole-v1"
    seed: int = 0
    epochs: int = 10
    steps_per_epoch: int = 1_000
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    policy_train_iters: int = 20
    value_train_iters: int = 20
    target_kl: float = 0.03
    entropy_coef: float = 0.0
    hidden_size: int = 64
    eval_episodes: int = 5
    run_dir: Path = Path("experiments/runs/ppo_cartpole")
