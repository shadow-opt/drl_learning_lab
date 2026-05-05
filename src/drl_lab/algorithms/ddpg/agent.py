from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.optim import Adam

from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBatch
from drl_lab.algorithms.ddpg.config import DDPGConfig
from drl_lab.algorithms.ddpg.losses import actor_loss, critic_loss, soft_update
from drl_lab.algorithms.ddpg.networks import ContinuousQNetwork, DeterministicActor


class DDPGAgent:
    """DDPG agent with actor, critic, and Polyak-averaged target networks."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        config: DDPGConfig,
        device: torch.device,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.config = config
        self.device = device

        hidden_sizes = [config.hidden_size, config.hidden_size]
        self.actor = DeterministicActor(obs_dim, action_dim, action_limit, hidden_sizes).to(device)
        self.critic = ContinuousQNetwork(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_actor = DeterministicActor(obs_dim, action_dim, action_limit, hidden_sizes).to(
            device
        )
        self.target_critic = ContinuousQNetwork(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.critic_lr)

    def act(
        self,
        obs: NDArray[np.float32],
        noise_scale: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float32]:
        """Select one bounded continuous action for a single observation."""
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"expected obs shape {(self.obs_dim,)}, got {obs.shape}")
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.inference_mode():
            action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
        if noise_scale > 0.0:
            generator = rng or np.random.default_rng()
            noise = generator.normal(
                0.0,
                noise_scale * self.action_limit,
                self.action_dim,
            )
            action = action + noise
        clipped = np.clip(action, -self.action_limit, self.action_limit)
        action_array: NDArray[np.float32] = clipped.astype(np.float32)
        return action_array

    def update(self, batch: ContinuousReplayBatch) -> dict[str, float]:
        """Run one critic update, one actor update, and one target-network update."""
        self.critic.train()
        self.actor.train()

        loss_q = critic_loss(
            self.critic,
            self.target_actor,
            self.target_critic,
            batch,
            gamma=self.config.gamma,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_q)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        loss_pi = actor_loss(self.actor, self.critic, batch.obs)
        self.actor_optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_pi)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        soft_update(self.actor, self.target_actor, self.config.tau)
        soft_update(self.critic, self.target_critic, self.config.tau)

        return {
            "critic_loss": float(loss_q.detach()),
            "actor_loss": float(loss_pi.detach()),
        }
