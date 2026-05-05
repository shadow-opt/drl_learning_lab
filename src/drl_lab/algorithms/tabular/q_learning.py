from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from drl_lab.algorithms.tabular.gridworld import GridWorld


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 500
    max_steps_per_episode: int = 100
    alpha: float = 0.2
    gamma: float = 0.99
    epsilon: float = 0.2
    seed: int = 0


TabularControlConfig = QLearningConfig


def epsilon_greedy_action(
    q_values: NDArray[np.float64],
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Select an epsilon-greedy action from a Q-table."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[1]))
    return int(q_values[state].argmax())


def expected_epsilon_greedy_value(
    q_values: NDArray[np.float64],
    state: int,
    epsilon: float,
) -> float:
    """Compute E[Q(s, a)] under an epsilon-greedy policy."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    n_actions = q_values.shape[1]
    greedy_action = int(q_values[state].argmax())
    action_probs = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
    action_probs[greedy_action] += 1.0 - epsilon
    return float(np.dot(action_probs, q_values[state]))


def _validate_config(cfg: TabularControlConfig) -> None:
    if cfg.episodes <= 0:
        raise ValueError("episodes must be positive")
    if cfg.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive")
    if not 0.0 <= cfg.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= cfg.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= cfg.epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")


def q_learning(
    env: GridWorld,
    config: QLearningConfig | None = None,
) -> tuple[NDArray[np.float64], list[float]]:
    """Train tabular Q-learning on GridWorld."""
    cfg = config or QLearningConfig()
    _validate_config(cfg)

    rng = np.random.default_rng(cfg.seed)
    q_values = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    episode_returns: list[float] = []

    for _ in range(cfg.episodes):
        state = env.reset()
        total_reward = 0.0

        for _step in range(cfg.max_steps_per_episode):
            action = epsilon_greedy_action(q_values, state, cfg.epsilon, rng)
            next_state, reward, done = env.step(action)
            total_reward += reward

            target = reward
            if not done:
                target += cfg.gamma * float(q_values[next_state].max())
            q_values[state, action] += cfg.alpha * (target - q_values[state, action])

            state = next_state
            if done:
                break

        episode_returns.append(total_reward)

    return q_values, episode_returns


def sarsa(
    env: GridWorld,
    config: TabularControlConfig | None = None,
) -> tuple[NDArray[np.float64], list[float]]:
    """Train on-policy SARSA on GridWorld."""
    cfg = config or TabularControlConfig()
    _validate_config(cfg)

    rng = np.random.default_rng(cfg.seed)
    q_values = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    episode_returns: list[float] = []

    for _ in range(cfg.episodes):
        state = env.reset()
        action = epsilon_greedy_action(q_values, state, cfg.epsilon, rng)
        total_reward = 0.0

        for _step in range(cfg.max_steps_per_episode):
            next_state, reward, done = env.step(action)
            total_reward += reward

            target = reward
            if not done:
                next_action = epsilon_greedy_action(q_values, next_state, cfg.epsilon, rng)
                target += cfg.gamma * q_values[next_state, next_action]
            else:
                next_action = 0

            q_values[state, action] += cfg.alpha * (target - q_values[state, action])
            state = next_state
            action = next_action
            if done:
                break

        episode_returns.append(total_reward)

    return q_values, episode_returns


def expected_sarsa(
    env: GridWorld,
    config: TabularControlConfig | None = None,
) -> tuple[NDArray[np.float64], list[float]]:
    """Train Expected SARSA on GridWorld."""
    cfg = config or TabularControlConfig()
    _validate_config(cfg)

    rng = np.random.default_rng(cfg.seed)
    q_values = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    episode_returns: list[float] = []

    for _ in range(cfg.episodes):
        state = env.reset()
        total_reward = 0.0

        for _step in range(cfg.max_steps_per_episode):
            action = epsilon_greedy_action(q_values, state, cfg.epsilon, rng)
            next_state, reward, done = env.step(action)
            total_reward += reward

            target = reward
            if not done:
                target += cfg.gamma * expected_epsilon_greedy_value(
                    q_values,
                    next_state,
                    cfg.epsilon,
                )

            q_values[state, action] += cfg.alpha * (target - q_values[state, action])
            state = next_state
            if done:
                break

        episode_returns.append(total_reward)

    return q_values, episode_returns
