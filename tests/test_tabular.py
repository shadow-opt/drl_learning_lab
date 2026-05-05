from __future__ import annotations

import numpy as np

from drl_lab.algorithms.tabular import (
    GridWorld,
    QLearningConfig,
    policy_iteration,
    q_learning,
    value_iteration,
)


def test_gridworld_transition_reaches_terminal() -> None:
    env = GridWorld()
    next_state, reward, done = env.transition(14, 1)
    assert next_state == 15
    assert reward == 0.0
    assert done


def test_value_iteration_prefers_short_path() -> None:
    env = GridWorld()
    values, policy = value_iteration(env, gamma=1.0)

    assert values.shape == (env.n_states,)
    assert policy.shape == (env.n_states,)
    assert np.isclose(values[14], 0.0)
    assert policy[14] == 1


def test_policy_iteration_matches_value_iteration() -> None:
    env = GridWorld()
    value_values, _ = value_iteration(env, gamma=1.0)
    policy_values, policy = policy_iteration(env, gamma=1.0)

    assert np.allclose(policy_values, value_values)
    assert policy[14] == 1


def test_q_learning_learns_terminal_neighbor_action() -> None:
    env = GridWorld()
    q_values, returns = q_learning(
        env,
        QLearningConfig(episodes=500, max_steps_per_episode=50, alpha=0.3, gamma=1.0, seed=2),
    )

    assert q_values.shape == (env.n_states, env.n_actions)
    assert len(returns) == 500
    assert int(q_values[14].argmax()) == 1
