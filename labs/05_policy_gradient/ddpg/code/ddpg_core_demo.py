from __future__ import annotations

import numpy as np
import torch

from drl_lab.algorithms.ddpg import (
    ContinuousQNetwork,
    ContinuousReplayBuffer,
    DeterministicActor,
    actor_loss,
    critic_loss,
    soft_update,
)
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def main() -> None:
    set_global_seed(71)
    obs_dim = 3
    action_dim = 2
    buffer = ContinuousReplayBuffer(capacity=32, obs_dim=obs_dim, action_dim=action_dim, seed=71)
    for idx in range(16):
        obs = np.full(obs_dim, idx / 16, dtype=np.float32)
        action = np.full(action_dim, 0.1, dtype=np.float32)
        buffer.add(obs, action, reward=1.0, next_obs=obs + 0.1, done=False)

    actor = DeterministicActor(obs_dim, action_dim, action_limit=1.0, hidden_sizes=[16])
    critic = ContinuousQNetwork(obs_dim, action_dim, hidden_sizes=[16])
    target_actor = DeterministicActor(obs_dim, action_dim, action_limit=1.0, hidden_sizes=[16])
    target_critic = ContinuousQNetwork(obs_dim, action_dim, hidden_sizes=[16])
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    batch = buffer.sample(batch_size=8)
    loss_q = critic_loss(critic, target_actor, target_critic, batch, gamma=0.99)
    loss_pi = actor_loss(actor, critic, batch.obs)
    soft_update(actor, target_actor, tau=0.1)

    example_input = torch.zeros(3, obs_dim)
    onnx_path = export_to_onnx(actor, example_input, "experiments/runs/ddpg_core/actor.onnx")
    result = compare_pytorch_onnx(actor, onnx_path, example_input)

    print(f"critic_loss={float(loss_q.detach()):.6f}")
    print(f"actor_loss={float(loss_pi.detach()):.6f}")
    print(f"onnx_passed={result.passed}")


if __name__ == "__main__":
    main()
