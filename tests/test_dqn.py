from __future__ import annotations

import numpy as np
import pytest
import torch

from drl_lab.algorithms.dqn import QNetwork, ReplayBuffer
from drl_lab.algorithms.dqn.agent import linear_epsilon
from drl_lab.algorithms.dqn.config import DQNConfig
from drl_lab.algorithms.dqn.losses import dqn_loss
from drl_lab.algorithms.dqn.train import train
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def test_replay_buffer_shapes() -> None:
    buffer = ReplayBuffer(capacity=8, obs_dim=3, seed=0)
    for idx in range(4):
        obs = np.full(3, idx, dtype=np.float32)
        buffer.add(obs, action=idx % 2, reward=1.0, next_obs=obs + 1, done=False)

    batch = buffer.sample(batch_size=4)
    assert batch.obs.shape == (4, 3)
    assert batch.actions.shape == (4,)
    assert batch.rewards.dtype == torch.float32


def test_dqn_loss_is_finite() -> None:
    set_global_seed(4)
    buffer = ReplayBuffer(capacity=8, obs_dim=3, seed=0)
    for idx in range(6):
        obs = np.full(3, idx, dtype=np.float32)
        buffer.add(obs, action=idx % 2, reward=1.0, next_obs=obs + 1, done=idx == 5)

    q_network = QNetwork(obs_dim=3, n_actions=2, hidden_sizes=[8])
    target = QNetwork(obs_dim=3, n_actions=2, hidden_sizes=[8])
    target.load_state_dict(q_network.state_dict())
    loss = dqn_loss(q_network, target, buffer.sample(4), gamma=0.99)

    assert torch.isfinite(loss)


def test_q_network_onnx_consistency(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(9)
    q_network = QNetwork(obs_dim=4, n_actions=2, hidden_sizes=[8])
    example_input = torch.randn(3, 4)
    onnx_path = export_to_onnx(q_network, example_input, tmp_path / "q.onnx")
    result = compare_pytorch_onnx(q_network, onnx_path, example_input)

    assert result.passed


def test_linear_epsilon_reaches_floor() -> None:
    config = DQNConfig(total_steps=100, exploration_fraction=0.5)
    assert linear_epsilon(0, config) == config.start_epsilon
    assert linear_epsilon(100, config) == pytest.approx(config.end_epsilon)


def test_dqn_short_training_loop_writes_snapshots(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = DQNConfig(
        total_steps=12,
        learning_starts=4,
        batch_size=4,
        buffer_capacity=32,
        target_update_frequency=8,
        eval_frequency=100,
        eval_episodes=1,
        hidden_size=8,
        run_dir=tmp_path / "dqn",
    )

    train(config)

    assert (config.run_dir / "config.json").exists()
    assert (config.run_dir / "environment.json").exists()
