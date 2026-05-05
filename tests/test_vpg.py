from __future__ import annotations

import torch

from drl_lab.algorithms.vpg import CategoricalPolicy, ValueFunction, discount_cumsum
from drl_lab.algorithms.vpg.agent import VPGAgent
from drl_lab.algorithms.vpg.buffer import make_trajectory_batch, normalize_advantages
from drl_lab.algorithms.vpg.config import VPGConfig
from drl_lab.algorithms.vpg.losses import policy_loss, value_loss
from drl_lab.algorithms.vpg.train import train
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def test_discount_cumsum() -> None:
    values = torch.tensor([1.0, 1.0, 1.0])
    result = discount_cumsum(values, discount=0.9)
    assert torch.allclose(result, torch.tensor([2.71, 1.9, 1.0]))


def test_normalize_advantages() -> None:
    advantages = normalize_advantages(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_make_trajectory_batch_shapes() -> None:
    batch = make_trajectory_batch(
        obs=torch.zeros(4, 3),
        actions=torch.zeros(4, dtype=torch.int64),
        rewards=torch.ones(4),
        values=torch.zeros(4),
        log_probs=torch.zeros(4),
        gamma=1.0,
    )

    assert batch.obs.shape == (4, 3)
    assert batch.actions.shape == (4,)
    assert batch.returns.shape == (4,)
    assert batch.advantages.shape == (4,)


def test_vpg_losses_are_finite() -> None:
    set_global_seed(12)
    obs = torch.randn(6, 4)
    actions = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64)
    advantages = torch.randn(6)
    returns = torch.randn(6)
    policy = CategoricalPolicy(obs_dim=4, n_actions=2, hidden_sizes=[8])
    value_function = ValueFunction(obs_dim=4, hidden_sizes=[8])

    assert torch.isfinite(policy_loss(policy, obs, actions, advantages))
    assert torch.isfinite(value_loss(value_function, obs, returns))


def test_vpg_policy_onnx_consistency(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(14)
    policy = CategoricalPolicy(obs_dim=4, n_actions=2, hidden_sizes=[8])
    example_input = torch.randn(3, 4)
    onnx_path = export_to_onnx(policy, example_input, tmp_path / "policy.onnx")
    result = compare_pytorch_onnx(policy, onnx_path, example_input)

    assert result.passed


def test_vpg_agent_act_shapes() -> None:
    set_global_seed(20)
    agent = VPGAgent(obs_dim=4, n_actions=2, config=VPGConfig(), device=torch.device("cpu"))
    action, log_prob, value = agent.act(torch.zeros(4).numpy())

    assert action in {0, 1}
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_vpg_train_smoke(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = VPGConfig(
        epochs=1,
        steps_per_epoch=64,
        value_train_iters=2,
        eval_episodes=1,
        run_dir=tmp_path / "vpg",
    )
    metrics = train(config)

    assert "last_policy_loss" in metrics
    assert "last_value_loss" in metrics
    assert metrics["last_eval_return"] > 0.0
