from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.optim import Adam

from drl_lab.algorithms.vpg.config import VPGConfig
from drl_lab.algorithms.vpg.losses import policy_loss, value_loss
from drl_lab.algorithms.vpg.networks import CategoricalPolicy, ValueFunction


class VPGAgent:
    """Vanilla Policy Gradient agent for discrete action spaces."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: VPGConfig,
        device: torch.device,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config
        self.device = device

        hidden_sizes = [config.hidden_size, config.hidden_size]
        self.policy = CategoricalPolicy(obs_dim, n_actions, hidden_sizes).to(device)
        self.value_function = ValueFunction(obs_dim, hidden_sizes).to(device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.policy_lr)
        self.value_optimizer = Adam(self.value_function.parameters(), lr=config.value_lr)

    def act(self, obs: NDArray[np.float32]) -> tuple[int, float, float]:
        """Sample an action and return action, log prob, and value estimate."""
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"expected obs shape {(self.obs_dim,)}, got {obs.shape}")

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.policy.eval()
        self.value_function.eval()
        with torch.inference_mode():
            action_tensor = self.policy.act(obs_tensor)
            log_prob = self.policy.log_prob(obs_tensor, action_tensor)
            value = self.value_function(obs_tensor)
        return int(action_tensor.item()), float(log_prob.item()), float(value.item())

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, float]:
        self.policy.train()
        self.value_function.train()

        actor_loss = policy_loss(self.policy, obs, actions, advantages)
        self.policy_optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(actor_loss)
        self.policy_optimizer.step()

        last_value_loss = torch.zeros((), dtype=returns.dtype, device=returns.device)
        for _ in range(self.config.value_train_iters):
            last_value_loss = value_loss(self.value_function, obs, returns)
            self.value_optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(last_value_loss)
            self.value_optimizer.step()

        return {
            "policy_loss": float(actor_loss.detach()),
            "value_loss": float(last_value_loss.detach()),
        }
