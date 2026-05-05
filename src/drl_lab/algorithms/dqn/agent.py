from __future__ import annotations

import random

import numpy as np
import torch
from numpy.typing import NDArray
from torch.optim import Adam

from drl_lab.algorithms.dqn.buffer import ReplayBatch
from drl_lab.algorithms.dqn.config import DQNConfig
from drl_lab.algorithms.dqn.losses import dqn_loss
from drl_lab.algorithms.dqn.networks import QNetwork


class DQNAgent:
    """DQN agent with online and target Q-networks."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: DQNConfig,
        device: torch.device,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config
        self.device = device

        hidden_sizes = [config.hidden_size, config.hidden_size]
        self.q_network = QNetwork(obs_dim, n_actions, hidden_sizes).to(device)
        self.target_q_network = QNetwork(obs_dim, n_actions, hidden_sizes).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = Adam(self.q_network.parameters(), lr=config.lr)

    def act(self, obs: NDArray[np.float32], epsilon: float = 0.0) -> int:
        """Select an action for a single observation with epsilon-greedy exploration."""
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"expected obs shape {(self.obs_dim,)}, got {obs.shape}")
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.inference_mode():
            q_values = self.q_network(obs_tensor)
        return int(q_values.argmax(dim=1).item())

    def update(self, batch: ReplayBatch) -> float:
        self.q_network.train()
        loss = dqn_loss(self.q_network, self.target_q_network, batch, self.config.gamma)

        self.optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return float(loss.detach())

    def sync_target(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())


def linear_epsilon(step: int, config: DQNConfig) -> float:
    decay_steps = max(1, int(config.exploration_fraction * config.total_steps))
    progress = min(1.0, step / decay_steps)
    return config.start_epsilon + progress * (config.end_epsilon - config.start_epsilon)
