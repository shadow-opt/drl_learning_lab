from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.optim import Adam

from drl_lab.algorithms.ddpg.buffer import ContinuousReplayBatch
from drl_lab.algorithms.ddpg.losses import actor_loss, soft_update
from drl_lab.algorithms.ddpg.networks import DeterministicActor
from drl_lab.algorithms.td3.config import TD3Config
from drl_lab.algorithms.td3.losses import td3_critic_loss
from drl_lab.algorithms.td3.networks import TwinContinuousQNetwork


class TD3Agent:
    """TD3 agent with delayed actor updates and twin critics."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        config: TD3Config,
        device: torch.device,
    ) -> None:
        if config.policy_delay <= 0:
            raise ValueError("policy_delay must be positive")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.config = config
        self.device = device
        self.update_count = 0

        hidden_sizes = [config.hidden_size, config.hidden_size]
        self.actor = DeterministicActor(obs_dim, action_dim, action_limit, hidden_sizes).to(device)
        self.target_actor = DeterministicActor(obs_dim, action_dim, action_limit, hidden_sizes).to(
            device
        )
        self.critics = TwinContinuousQNetwork(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_critics = TwinContinuousQNetwork(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = Adam(self.critics.parameters(), lr=config.critic_lr)

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
        """Run one TD3 update step."""
        self.critics.train()
        self.actor.train()
        self.update_count += 1

        loss_q = td3_critic_loss(
            self.critics,
            self.target_actor,
            self.target_critics,
            batch,
            gamma=self.config.gamma,
            target_noise=self.config.target_noise,
            noise_clip=self.config.noise_clip,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_q)
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        loss_pi_value = 0.0
        did_actor_update = self.update_count % self.config.policy_delay == 0
        if did_actor_update:
            loss_pi = actor_loss(self.actor, self.critics.q1, batch.obs)
            self.actor_optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss_pi)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()
            loss_pi_value = float(loss_pi.detach())

            soft_update(self.actor, self.target_actor, self.config.tau)
            soft_update(self.critics, self.target_critics, self.config.tau)

        return {
            "critic_loss": float(loss_q.detach()),
            "actor_loss": loss_pi_value,
            "actor_updated": float(did_actor_update),
        }
