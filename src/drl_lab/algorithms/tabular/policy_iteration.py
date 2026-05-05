from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from drl_lab.algorithms.tabular.gridworld import GridWorld


def evaluate_policy(
    env: GridWorld,
    policy: NDArray[np.int64],
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> NDArray[np.float64]:
    """Evaluate a deterministic policy on a deterministic GridWorld."""
    if policy.shape != (env.n_states,):
        raise ValueError(f"expected policy shape {(env.n_states,)}, got {policy.shape}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    values = np.zeros(env.n_states, dtype=np.float64)
    for _ in range(max_iterations):
        delta = 0.0
        for state in range(env.n_states):
            old_value = values[state]
            if env.is_terminal(state):
                values[state] = 0.0
                continue

            next_state, reward, done = env.transition(state, int(policy[state]))
            bootstrap = 0.0 if done else gamma * values[next_state]
            values[state] = reward + bootstrap
            delta = max(delta, abs(old_value - values[state]))

        if delta < theta:
            break

    return values


def improve_policy(
    env: GridWorld,
    values: NDArray[np.float64],
    gamma: float = 0.99,
) -> NDArray[np.int64]:
    """Return the greedy deterministic policy for a value function."""
    if values.shape != (env.n_states,):
        raise ValueError(f"expected values shape {(env.n_states,)}, got {values.shape}")

    policy: NDArray[np.int64] = np.zeros(env.n_states, dtype=np.int64)
    for state in range(env.n_states):
        if env.is_terminal(state):
            continue

        action_values = np.zeros(env.n_actions, dtype=np.float64)
        for action in range(env.n_actions):
            next_state, reward, done = env.transition(state, action)
            bootstrap = 0.0 if done else gamma * values[next_state]
            action_values[action] = reward + bootstrap
        policy[state] = int(action_values.argmax())

    return policy


def policy_iteration(
    env: GridWorld,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 1_000,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Run deterministic policy iteration."""
    policy: NDArray[np.int64] = np.zeros(env.n_states, dtype=np.int64)

    for _ in range(max_iterations):
        values = evaluate_policy(env, policy, gamma=gamma, theta=theta)
        improved_policy = improve_policy(env, values, gamma=gamma)
        if np.array_equal(policy, improved_policy):
            return values, policy
        policy = improved_policy

    values = evaluate_policy(env, policy, gamma=gamma, theta=theta)
    return values, policy
