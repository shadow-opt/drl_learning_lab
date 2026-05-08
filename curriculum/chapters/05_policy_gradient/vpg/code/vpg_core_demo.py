from __future__ import annotations

import torch

from drl_lab.algorithms.vpg import (
    CategoricalPolicy,
    ValueFunction,
    discount_cumsum,
    policy_loss,
    value_loss,
)
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def main() -> None:
    set_global_seed(31)
    obs = torch.randn(5, 4)
    actions = torch.tensor([0, 1, 0, 1, 1], dtype=torch.int64)
    rewards = torch.ones(5)
    returns = discount_cumsum(rewards, discount=0.99)
    advantages = returns - returns.mean()

    policy = CategoricalPolicy(obs_dim=4, n_actions=2, hidden_sizes=[16])
    value_function = ValueFunction(obs_dim=4, hidden_sizes=[16])

    actor_loss = policy_loss(policy, obs, actions, advantages)
    critic_loss = value_loss(value_function, obs, returns)

    example_input = torch.zeros(3, 4)
    onnx_path = export_to_onnx(policy, example_input, "experiments/runs/vpg_core/policy.onnx")
    result = compare_pytorch_onnx(policy, onnx_path, example_input)

    print(f"actor_loss={float(actor_loss.detach()):.6f}")
    print(f"critic_loss={float(critic_loss.detach()):.6f}")
    print(f"onnx_passed={result.passed}")


if __name__ == "__main__":
    main()
