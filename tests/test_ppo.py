from __future__ import annotations

import torch

from drl_lab.algorithms.ppo.buffer import compute_gae, make_ppo_batch
from drl_lab.algorithms.ppo.config import PPOConfig
from drl_lab.algorithms.ppo.losses import approx_kl, clipped_policy_loss, entropy_bonus
from drl_lab.algorithms.ppo.train import train
from drl_lab.algorithms.vpg import CategoricalPolicy
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def test_compute_gae_matches_discounted_returns_when_values_zero() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(4)
    dones = torch.zeros(3)
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)

    expected = torch.tensor([2.71, 1.9, 1.0])
    assert torch.allclose(advantages, expected)
    assert torch.allclose(returns, expected)


def test_compute_gae_stops_at_done() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(4)
    dones = torch.tensor([0.0, 1.0, 0.0])
    advantages, _ = compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)

    assert torch.allclose(advantages, torch.tensor([1.9, 1.0, 1.0]))


def test_make_ppo_batch_shapes() -> None:
    batch = make_ppo_batch(
        obs=torch.zeros(4, 3),
        actions=torch.zeros(4, dtype=torch.int64),
        rewards=torch.ones(4),
        values=torch.zeros(5),
        dones=torch.zeros(4),
        old_log_probs=torch.zeros(4),
        gamma=0.99,
        lam=0.95,
    )

    assert batch.obs.shape == (4, 3)
    assert batch.actions.shape == (4,)
    assert batch.returns.shape == (4,)
    assert batch.advantages.shape == (4,)
    assert batch.old_log_probs.shape == (4,)


def test_clipped_policy_loss_is_finite() -> None:
    set_global_seed(51)
    policy = CategoricalPolicy(obs_dim=4, n_actions=2, hidden_sizes=[8])
    obs = torch.randn(5, 4)
    actions = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64)
    old_log_probs = policy.log_prob(obs, actions).detach()
    advantages = torch.randn(5)

    loss, ratio = clipped_policy_loss(policy, obs, actions, advantages, old_log_probs, 0.2)

    assert torch.isfinite(loss)
    assert ratio.shape == (5,)


def test_ppo_diagnostics() -> None:
    old_log_probs = torch.tensor([-1.0, -2.0])
    new_log_probs = torch.tensor([-1.5, -1.0])
    assert torch.isclose(approx_kl(old_log_probs, new_log_probs), torch.tensor(-0.25))

    policy = CategoricalPolicy(obs_dim=4, n_actions=2, hidden_sizes=[8])
    entropy = entropy_bonus(policy, torch.zeros(3, 4))
    assert torch.isfinite(entropy)


def test_ppo_policy_onnx_consistency(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(52)
    policy = CategoricalPolicy(obs_dim=4, n_actions=2, hidden_sizes=[8])
    example_input = torch.randn(3, 4)
    onnx_path = export_to_onnx(policy, example_input, tmp_path / "policy.onnx")
    result = compare_pytorch_onnx(policy, onnx_path, example_input)

    assert result.passed


def test_ppo_train_smoke(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = PPOConfig(
        epochs=1,
        steps_per_epoch=64,
        policy_train_iters=2,
        value_train_iters=2,
        eval_episodes=1,
        run_dir=tmp_path / "ppo",
    )
    metrics = train(config)

    assert "last_policy_loss" in metrics
    assert "last_value_loss" in metrics
    assert "last_approx_kl" in metrics
    assert metrics["last_eval_return"] > 0.0
    assert (config.run_dir / "policy.onnx").exists()
    assert (config.run_dir / "value_function.onnx").exists()
