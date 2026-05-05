from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.optim import Adam

from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBatch
from drl_lab.algorithms.ddpg.losses import soft_update
from drl_lab.algorithms.sac.config import SACConfig
from drl_lab.algorithms.sac.losses import sac_actor_loss, sac_critic_loss, temperature_loss
from drl_lab.algorithms.sac.networks import SquashedGaussianActor
from drl_lab.algorithms.td3.networks import TwinContinuousQNetwork


class SACAgent:
    """SAC agent with stochastic actor, twin critics, and learned temperature."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        config: SACConfig,
        device: torch.device,
    ) -> None:
        if config.initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.config = config
        self.device = device
        self.target_entropy = (
            config.target_entropy if config.target_entropy is not None else -float(action_dim)
        )

        hidden_sizes = [config.hidden_size, config.hidden_size]
        self.actor = SquashedGaussianActor(obs_dim, action_dim, action_limit, hidden_sizes).to(
            device
        )
        self.target_actor = SquashedGaussianActor(
            obs_dim,
            action_dim,
            action_limit,
            hidden_sizes,
        ).to(device)
        self.critics = TwinContinuousQNetwork(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_critics = TwinContinuousQNetwork(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = Adam(self.critics.parameters(), lr=config.critic_lr)
        self.log_alpha = torch.tensor(
            float(np.log(config.initial_alpha)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=config.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        alpha: torch.Tensor = self.log_alpha.exp()
        return alpha

    def act(
        self,
        obs: NDArray[np.float32],
        deterministic: bool = False,
    ) -> NDArray[np.float32]:
        """Select one bounded continuous action for a single observation."""
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"expected obs shape {(self.obs_dim,)}, got {obs.shape}")
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.inference_mode():
            if deterministic:
                action_tensor = self.actor(obs_tensor)
            else:
                action_tensor, _ = self.actor.sample(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
        clipped = np.clip(action, -self.action_limit, self.action_limit)
        action_array: NDArray[np.float32] = clipped.astype(np.float32)
        return action_array

    def update(self, batch: ContinuousReplayBatch) -> dict[str, float]:
        """Run one SAC critic, actor, temperature, and target-network update."""
        self.critics.train()
        self.actor.train()

        alpha_detached = float(self.alpha.detach())
        loss_q = sac_critic_loss(
            self.critics,
            self.target_actor,
            self.target_critics,
            batch,
            gamma=self.config.gamma,
            alpha=alpha_detached,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_q)
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        loss_pi = sac_actor_loss(
            self.actor,
            self.critics,
            batch.obs,
            alpha=alpha_detached,
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_pi)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        _, log_probs = self.actor.sample(batch.obs)
        loss_alpha = temperature_loss(self.log_alpha, log_probs, self.target_entropy)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_alpha)
        self.alpha_optimizer.step()

        soft_update(self.actor, self.target_actor, self.config.tau)
        soft_update(self.critics, self.target_critics, self.config.tau)

        return {
            "critic_loss": float(loss_q.detach()),
            "actor_loss": float(loss_pi.detach()),
            "temperature_loss": float(loss_alpha.detach()),
            "alpha": float(self.alpha.detach()),
        }
