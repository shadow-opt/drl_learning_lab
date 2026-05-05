from __future__ import annotations

import numpy as np

from drl_lab.algorithms.tabular import (
    GridWorld,
    QLearningConfig,
    TabularControlConfig,
    expected_epsilon_greedy_value,
    expected_sarsa,
    monte_carlo_control,
    policy_iteration,
    q_learning,
    sarsa,
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


def test_monte_carlo_control_learns_terminal_neighbor_action() -> None:
    env = GridWorld()
    q_values, returns = monte_carlo_control(
        env,
        TabularControlConfig(
            episodes=1_000,
            max_steps_per_episode=50,
            gamma=1.0,
            epsilon=0.2,
            seed=5,
        ),
    )

    assert q_values.shape == (env.n_states, env.n_actions)
    assert len(returns) == 1_000
    assert int(q_values[14].argmax()) == 1


def test_expected_epsilon_greedy_value() -> None:
    q_values = np.asarray([[1.0, 3.0]], dtype=np.float64)
    value = expected_epsilon_greedy_value(q_values, state=0, epsilon=0.2)

    assert np.isclose(value, 2.8)


def test_sarsa_learns_terminal_neighbor_action() -> None:
    env = GridWorld()
    q_values, returns = sarsa(
        env,
        TabularControlConfig(
            episodes=600,
            max_steps_per_episode=50,
            alpha=0.3,
            gamma=1.0,
            epsilon=0.15,
            seed=3,
        ),
    )

    assert q_values.shape == (env.n_states, env.n_actions)
    assert len(returns) == 600
    assert int(q_values[14].argmax()) == 1


def test_expected_sarsa_learns_terminal_neighbor_action() -> None:
    env = GridWorld()
    q_values, returns = expected_sarsa(
        env,
        TabularControlConfig(
            episodes=600,
            max_steps_per_episode=50,
            alpha=0.3,
            gamma=1.0,
            epsilon=0.15,
            seed=4,
        ),
    )

    assert q_values.shape == (env.n_states, env.n_actions)
    assert len(returns) == 600
    assert int(q_values[14].argmax()) == 1
