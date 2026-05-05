from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from drl_lab.algorithms.ddpg.agent import DDPGAgent
from drl_lab.algorithms.ddpg.config import DDPGConfig


def evaluate(agent: DDPGAgent, config: DDPGConfig, episodes: int | None = None) -> dict[str, float]:
    """Evaluate a DDPG agent without exploration noise."""
    n_episodes = episodes or config.eval_episodes
    env = gym.make(config.env_id)
    returns: list[float] = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=config.seed + 10_000 + episode)
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(np.asarray(obs, dtype=np.float32), noise_scale=0.0)
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
    actor_checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    action_limit: float,
    config: DDPGConfig,
    device: torch.device,
) -> DDPGAgent:
    from drl_lab.common.checkpoint import load_checkpoint

    agent = DDPGAgent(obs_dim, action_dim, action_limit, config, device)
    load_checkpoint(actor_checkpoint_path, agent.actor, map_location=device)
    agent.target_actor.load_state_dict(agent.actor.state_dict())
    return agent
