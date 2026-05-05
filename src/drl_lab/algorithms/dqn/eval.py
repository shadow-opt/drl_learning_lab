from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from drl_lab.algorithms.dqn.agent import DQNAgent
from drl_lab.algorithms.dqn.config import DQNConfig


def evaluate(agent: DQNAgent, config: DQNConfig, episodes: int | None = None) -> dict[str, float]:
    """Evaluate a DQN agent with greedy actions."""
    n_episodes = episodes or config.eval_episodes
    env = gym.make(config.env_id)
    returns: list[float] = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=config.seed + 10_000 + episode)
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(np.asarray(obs, dtype=np.float32), epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
        returns.append(total_reward)

    env.close()

    values = np.asarray(returns, dtype=np.float64)
    return {
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "episodes": float(n_episodes),
    }


def load_agent_for_eval(
    checkpoint_path: str,
    obs_dim: int,
    n_actions: int,
    config: DQNConfig,
    device: torch.device,
) -> DQNAgent:
    from drl_lab.common.checkpoint import load_checkpoint

    agent = DQNAgent(obs_dim, n_actions, config, device)
    load_checkpoint(checkpoint_path, agent.q_network, map_location=device)
    agent.sync_target()
    return agent
