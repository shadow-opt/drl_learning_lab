from __future__ import annotations

import numpy as np
import torch
from torch.optim import Adam

from drl_lab.algorithms.dqn import QNetwork, ReplayBuffer
from drl_lab.algorithms.dqn.losses import dqn_loss
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def main() -> None:
    set_global_seed(21)
    obs_dim = 4
    n_actions = 2
    buffer = ReplayBuffer(capacity=64, obs_dim=obs_dim, seed=21)

    for idx in range(32):
        obs = np.full(obs_dim, idx / 32, dtype=np.float32)
        next_obs = obs + 0.1
        buffer.add(obs, idx % n_actions, 1.0, next_obs.astype(np.float32), done=False)

    q_network = QNetwork(obs_dim, n_actions, hidden_sizes=[16])
    target_q_network = QNetwork(obs_dim, n_actions, hidden_sizes=[16])
    target_q_network.load_state_dict(q_network.state_dict())
    optimizer = Adam(q_network.parameters(), lr=1e-3)

    batch = buffer.sample(batch_size=8)
    loss = dqn_loss(q_network, target_q_network, batch, gamma=0.99)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
    optimizer.step()

    example_input = torch.zeros(3, obs_dim)
    onnx_path = export_to_onnx(
        q_network,
        example_input,
        "experiments/runs/dqn_smoke/q_network.onnx",
    )
    result = compare_pytorch_onnx(q_network, onnx_path, example_input)

    print(f"loss={float(loss.detach()):.6f}")
    print(f"onnx_max_abs_diff={result.max_abs_diff:.8f}")
    print(f"onnx_passed={result.passed}")


if __name__ == "__main__":
    main()
