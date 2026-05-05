from __future__ import annotations

import numpy as np
import torch

from drl_lab.algorithms.sac import (
    ContinuousReplayBuffer,
    SquashedGaussianActor,
    TwinContinuousQNetwork,
    sac_actor_loss,
    sac_critic_loss,
    temperature_loss,
)
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def make_batch() -> tuple[ContinuousReplayBuffer, torch.device]:
    buffer = ContinuousReplayBuffer(capacity=8, obs_dim=3, action_dim=2, seed=0)
    for idx in range(6):
        obs = np.full(3, idx / 6, dtype=np.float32)
        action = np.full(2, 0.1, dtype=np.float32)
        buffer.add(obs, action, reward=1.0, next_obs=obs + 0.1, done=idx == 5)
    return buffer, torch.device("cpu")


def test_squashed_gaussian_actor_shapes_and_bounds() -> None:
    set_global_seed(91)
    actor = SquashedGaussianActor(obs_dim=3, action_dim=2, action_limit=0.5, hidden_sizes=[8])
    obs = torch.randn(16, 3)

    actions, log_probs = actor.sample(obs)

    assert actions.shape == (16, 2)
    assert log_probs.shape == (16,)
    assert float(actions.abs().max()) <= 0.5
    assert torch.isfinite(log_probs).all()


def test_sac_losses_are_finite() -> None:
    set_global_seed(92)
    buffer, device = make_batch()
    batch = buffer.sample(4, device=device)
    actor = SquashedGaussianActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    target_actor = SquashedGaussianActor(
        obs_dim=3,
        action_dim=2,
        action_limit=1.0,
        hidden_sizes=[8],
    )
    critics = TwinContinuousQNetwork(obs_dim=3, action_dim=2, hidden_sizes=[8])
    target_critics = TwinContinuousQNetwork(obs_dim=3, action_dim=2, hidden_sizes=[8])
    target_actor.load_state_dict(actor.state_dict())
    target_critics.load_state_dict(critics.state_dict())

    loss_q = sac_critic_loss(critics, target_actor, target_critics, batch, gamma=0.99, alpha=0.2)
    loss_pi = sac_actor_loss(actor, critics, batch.obs, alpha=0.2)
    _, log_probs = actor.sample(batch.obs)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    loss_alpha = temperature_loss(log_alpha, log_probs, target_entropy=-2.0)

    assert torch.isfinite(loss_q)
    assert torch.isfinite(loss_pi)
    assert torch.isfinite(loss_alpha)
    loss_alpha.backward()
    assert log_alpha.grad is not None


def test_sac_actor_onnx_consistency(tmp_path) -> None:  # type: ignore[no-untyped-def]
    set_global_seed(93)
    actor = SquashedGaussianActor(obs_dim=3, action_dim=2, action_limit=1.0, hidden_sizes=[8])
    example_input = torch.randn(4, 3)
    onnx_path = export_to_onnx(actor, example_input, tmp_path / "actor.onnx")
    result = compare_pytorch_onnx(actor, onnx_path, example_input)

    assert result.passed
