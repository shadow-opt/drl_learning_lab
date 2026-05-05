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


def main() -> None:
    set_global_seed(91)
    obs_dim = 3
    action_dim = 2
    buffer = ContinuousReplayBuffer(capacity=32, obs_dim=obs_dim, action_dim=action_dim, seed=91)
    for idx in range(16):
        obs = np.full(obs_dim, idx / 16, dtype=np.float32)
        action = np.full(action_dim, 0.1, dtype=np.float32)
        buffer.add(obs, action, reward=1.0, next_obs=obs + 0.1, done=False)

    actor = SquashedGaussianActor(obs_dim, action_dim, action_limit=1.0, hidden_sizes=[16])
    target_actor = SquashedGaussianActor(obs_dim, action_dim, action_limit=1.0, hidden_sizes=[16])
    critics = TwinContinuousQNetwork(obs_dim, action_dim, hidden_sizes=[16])
    target_critics = TwinContinuousQNetwork(obs_dim, action_dim, hidden_sizes=[16])
    target_actor.load_state_dict(actor.state_dict())
    target_critics.load_state_dict(critics.state_dict())

    batch = buffer.sample(batch_size=8)
    alpha = 0.2
    loss_q = sac_critic_loss(critics, target_actor, target_critics, batch, gamma=0.99, alpha=alpha)
    loss_pi = sac_actor_loss(actor, critics, batch.obs, alpha=alpha)
    _, log_probs = actor.sample(batch.obs)
    log_alpha = torch.tensor(0.0, requires_grad=True)
    loss_alpha = temperature_loss(log_alpha, log_probs, target_entropy=-float(action_dim))

    example_input = torch.zeros(3, obs_dim)
    onnx_path = export_to_onnx(actor, example_input, "experiments/runs/sac_core/actor.onnx")
    result = compare_pytorch_onnx(actor, onnx_path, example_input)

    print(f"critic_loss={float(loss_q.detach()):.6f}")
    print(f"actor_loss={float(loss_pi.detach()):.6f}")
    print(f"temperature_loss={float(loss_alpha.detach()):.6f}")
    print(f"sampled_log_prob_mean={float(log_probs.detach().mean()):.6f}")
    print(f"onnx_passed={result.passed}")


if __name__ == "__main__":
    main()
