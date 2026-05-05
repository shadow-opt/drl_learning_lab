from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from drl_lab.algorithms.ppo.agent import PPOAgent
from drl_lab.algorithms.ppo.config import PPOConfig


def evaluate(agent: PPOAgent, config: PPOConfig, episodes: int | None = None) -> dict[str, float]:
    """Evaluate a PPO policy with greedy action selection from logits."""
    n_episodes = episodes or config.eval_episodes
    env = gym.make(config.env_id)
    returns: list[float] = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=config.seed + 30_000 + episode)
        done = False
        total_reward = 0.0
        while not done:
            obs_array = np.asarray(obs, dtype=np.float32)
            obs_tensor = torch.as_tensor(
                obs_array,
                dtype=torch.float32,
                device=agent.device,
            ).unsqueeze(0)
            agent.policy.eval()
            with torch.inference_mode():
                logits = agent.policy(obs_tensor)
                action = int(logits.argmax(dim=1).item())
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
