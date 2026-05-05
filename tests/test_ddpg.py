from __future__ import annotations

import numpy as np
import torch

from drl_lab.algorithms.ddpg import (
    ContinuousQNetwork,
    ContinuousReplayBuffer,
    DDPGAgent,
    DDPGConfig,
    DeterministicActor,
    actor_loss,
    critic_loss,
    soft_update,
)
from drl_lab.algorithms.ddpg.train import train
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def test_continuous_replay_buffer_shapes() -> None:
    buffer = ContinuousReplayBuffer(capacity=8, obs_dim=3, action_dim=2, seed=0)
    for idx in range(4):
        obs = np.full(3, idx, dtype=np.float32)
        action = np.full(2, 0.1, dtype=np.float32)
        buffer.add(obs, action, reward=1.0, next_obs=obs + 1, done=False)

    batch = buffer.sample(batch_size=4)

    assert batch.obs.shape == (4, 3)
    assert batch.actions.shape == (4, 2)
    assert batch.rewards.shape == (4,)


def test_actor_outputs_are_bounded() -> None:
    set_global_seed(72)
    actor = DeterministicActor(obs_dim=3, action_dim=2, action_limit=0.5, hidden_sizes=[8])
    actions = actor(torch.randn(16, 3))

    assert actions.shape == (16, 2)
    assert float(actions.abs().max()) <= 0.5


def test_ddpg_agent_actions_are_bounded() -> None:
    set_global_seed(76)
    config = DDPGConfig(hidden_size=8)
    agent = DDPGAgent(
        obs_dim=3,
        action_dim=2,
        action_limit=0.5,
        config=config,
        device=torch.device("cpu"),
    )

    action = agent.act(np.zeros(3, dtype=np.float32), noise_scale=2.0, rng=np.random.default_rng(0))

    assert action.shape == (2,)
    assert float(np.abs(action).max()) <= 0.5


def test_ddpg_losses_are_finite() -> None:
    set_global_seed(73)
    buffer = ContinuousReplayBuffer(capacity=8, obs_dim=3, action_dim=2, seed=0)
    for idx in range(6):
        obs = np.full(3, idx, dtype=np.float32)
        action = np.full(2, 0.1, dtype=np.float32)
        buffer.add(obs, action, reward=1.0, next_obs=obs + 1, done=idx == 5)

    actor = DeterministicActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    critic = ContinuousQNetwork(obs_dim=3, action_dim=2, hidden_sizes=[8])
    target_actor = DeterministicActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    target_critic = ContinuousQNetwork(obs_dim=3, action_dim=2, hidden_sizes=[8])
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    batch = buffer.sample(4)

    assert torch.isfinite(critic_loss(critic, target_actor, target_critic, batch, gamma=0.99))
    assert torch.isfinite(actor_loss(actor, critic, batch.obs))


def test_soft_update_changes_target_parameters() -> None:
    set_global_seed(74)
    source = DeterministicActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    target = DeterministicActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    before = [param.detach().clone() for param in target.parameters()]

    soft_update(source, target, tau=0.5)

    assert any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, target.parameters(), strict=True)
    )


def test_ddpg_actor_onnx_consistency(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(75)
    actor = DeterministicActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    example_input = torch.randn(4, 3)
    onnx_path = export_to_onnx(actor, example_input, tmp_path / "actor.onnx")
    result = compare_pytorch_onnx(actor, onnx_path, example_input)

    assert result.passed


def test_ddpg_short_training_loop_writes_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = DDPGConfig(
        total_steps=12,
        learning_starts=4,
        batch_size=4,
        buffer_capacity=32,
        eval_frequency=100,
        eval_episodes=1,
        hidden_size=8,
        run_dir=tmp_path / "ddpg",
    )

    metrics = train(config)

    assert "last_eval_return" in metrics
    assert (config.run_dir / "metrics.csv").exists()
    assert (config.run_dir / "config.json").exists()
    assert (config.run_dir / "environment.json").exists()
    assert (config.run_dir / "actor.pt").exists()
    assert (config.run_dir / "critic.pt").exists()
    assert (config.run_dir / "actor.onnx").exists()
