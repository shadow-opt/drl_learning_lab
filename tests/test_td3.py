from __future__ import annotations

import numpy as np
import torch

from drl_lab.algorithms.td3 import (
    ContinuousReplayBuffer,
    DeterministicActor,
    TD3Agent,
    TD3Config,
    TwinContinuousQNetwork,
    actor_loss,
    clipped_target_actions,
    td3_critic_loss,
)
from drl_lab.algorithms.td3.train import train
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def make_batch() -> tuple[ContinuousReplayBuffer, torch.device]:
    buffer = ContinuousReplayBuffer(capacity=8, obs_dim=3, action_dim=2, seed=0)
    for idx in range(6):
        obs = np.full(3, idx, dtype=np.float32)
        action = np.full(2, 0.1, dtype=np.float32)
        buffer.add(obs, action, reward=1.0, next_obs=obs + 1, done=idx == 5)
    return buffer, torch.device("cpu")


def test_twin_critic_shapes() -> None:
    critics = TwinContinuousQNetwork(obs_dim=3, action_dim=2, hidden_sizes=[8])
    q1, q2 = critics(torch.zeros(4, 3), torch.zeros(4, 2))

    assert q1.shape == (4,)
    assert q2.shape == (4,)


def test_clipped_target_actions_are_bounded() -> None:
    set_global_seed(82)
    actor = DeterministicActor(obs_dim=3, action_dim=2, action_limit=0.5, hidden_sizes=[8])
    actions = clipped_target_actions(actor, torch.randn(16, 3), target_noise=1.0, noise_clip=0.2)

    assert actions.shape == (16, 2)
    assert float(actions.abs().max()) <= 0.5


def test_td3_agent_delays_actor_updates() -> None:
    set_global_seed(85)
    config = TD3Config(hidden_size=8, policy_delay=2)
    agent = TD3Agent(
        obs_dim=3,
        action_dim=2,
        action_limit=1.0,
        config=config,
        device=torch.device("cpu"),
    )
    buffer, device = make_batch()
    batch = buffer.sample(4, device=device)

    first = agent.update(batch)
    second = agent.update(batch)

    assert first["actor_updated"] == 0.0
    assert second["actor_updated"] == 1.0


def test_td3_losses_are_finite() -> None:
    set_global_seed(83)
    buffer, device = make_batch()
    batch = buffer.sample(4, device=device)
    actor = DeterministicActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    target_actor = DeterministicActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    critics = TwinContinuousQNetwork(obs_dim=3, action_dim=2, hidden_sizes=[8])
    target_critics = TwinContinuousQNetwork(obs_dim=3, action_dim=2, hidden_sizes=[8])
    target_actor.load_state_dict(actor.state_dict())
    target_critics.load_state_dict(critics.state_dict())

    loss_q = td3_critic_loss(critics, target_actor, target_critics, batch, 0.99, 0.2, 0.5)
    loss_pi = actor_loss(actor, critics.q1, batch.obs)

    assert torch.isfinite(loss_q)
    assert torch.isfinite(loss_pi)


def test_td3_actor_onnx_consistency(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(84)
    actor = DeterministicActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    example_input = torch.randn(4, 3)
    onnx_path = export_to_onnx(actor, example_input, tmp_path / "actor.onnx")
    result = compare_pytorch_onnx(actor, onnx_path, example_input)

    assert result.passed


def test_td3_short_training_loop_writes_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = TD3Config(
        total_steps=12,
        learning_starts=4,
        batch_size=4,
        buffer_capacity=32,
        eval_frequency=100,
        eval_episodes=1,
        hidden_size=8,
        run_dir=tmp_path / "td3",
    )

    metrics = train(config)

    assert "last_eval_return" in metrics
    assert (config.run_dir / "metrics.csv").exists()
    assert (config.run_dir / "config.json").exists()
    assert (config.run_dir / "environment.json").exists()
    assert (config.run_dir / "actor.pt").exists()
    assert (config.run_dir / "critics.pt").exists()
    assert (config.run_dir / "actor.onnx").exists()
    assert (config.run_dir / "critics.onnx").exists()
