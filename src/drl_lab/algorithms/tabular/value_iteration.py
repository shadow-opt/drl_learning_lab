from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from drl_lab.algorithms.tabular.gridworld import GridWorld


def value_iteration(
    env: GridWorld,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Run deterministic value iteration.

    Returns:
        values: shape ``[n_states]``.
        policy: shape ``[n_states]`` with greedy action indices.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if theta <= 0:
        raise ValueError("theta must be positive")

    values = np.zeros(env.n_states, dtype=np.float64)
    policy = np.zeros(env.n_states, dtype=np.int64)

    for _ in range(max_iterations):
        delta = 0.0
        for state in range(env.n_states):
            if env.is_terminal(state):
                continue

            action_values = np.zeros(env.n_actions, dtype=np.float64)
            for action in range(env.n_actions):
                next_state, reward, done = env.transition(state, action)
                bootstrap = 0.0 if done else gamma * values[next_state]
                action_values[action] = reward + bootstrap

            best_value = float(action_values.max())
            delta = max(delta, abs(best_value - values[state]))
            values[state] = best_value
            policy[state] = int(action_values.argmax())

        if delta < theta:
            break

    return values, policy
