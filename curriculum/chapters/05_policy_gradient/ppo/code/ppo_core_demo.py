from __future__ import annotations

import torch

from drl_lab.algorithms.ppo import compute_gae
from drl_lab.algorithms.ppo.losses import approx_kl, clipped_policy_loss, entropy_bonus
from drl_lab.algorithms.vpg import CategoricalPolicy
from drl_lab.common.export import export_to_onnx
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


def main() -> None:
    set_global_seed(41)
    obs = torch.randn(6, 4)
    actions = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int64)
    rewards = torch.ones(6)
    values = torch.zeros(7)
    dones = torch.zeros(6)
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)

    policy = CategoricalPolicy(obs_dim=4, n_actions=2, hidden_sizes=[16])
    old_log_probs = policy.log_prob(obs, actions).detach()
    loss, ratio = clipped_policy_loss(
        policy,
        obs,
        actions,
        advantages,
        old_log_probs,
        clip_ratio=0.2,
    )
    new_log_probs = policy.log_prob(obs, actions)

    example_input = torch.zeros(3, 4)
    onnx_path = export_to_onnx(policy, example_input, "experiments/runs/ppo_core/policy.onnx")
    result = compare_pytorch_onnx(policy, onnx_path, example_input)

    print(f"loss={float(loss.detach()):.6f}")
    print(f"ratio_mean={float(ratio.mean().detach()):.6f}")
    print(f"return_0={float(returns[0]):.6f}")
    print(f"approx_kl={float(approx_kl(old_log_probs, new_log_probs).detach()):.6f}")
    print(f"entropy={float(entropy_bonus(policy, obs).detach()):.6f}")
    print(f"onnx_passed={result.passed}")


if __name__ == "__main__":
    main()
