from __future__ import annotations

import numpy as np

from drl_lab.algorithms.tabular import GridWorld, value_iteration


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
