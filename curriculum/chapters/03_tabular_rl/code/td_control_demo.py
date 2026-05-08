from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from drl_lab.algorithms.tabular import (
    GridWorld,
    TabularControlConfig,
    expected_sarsa,
    monte_carlo_control,
    q_learning,
    sarsa,
)

ControlFn = Callable[[GridWorld, TabularControlConfig], tuple[NDArray[np.float64], list[float]]]


def run(name: str, fn: ControlFn) -> None:
    env = GridWorld()
    q_values, returns = fn(
        env,
        TabularControlConfig(episodes=500, max_steps_per_episode=50, alpha=0.3, gamma=1.0, seed=4),
    )
    print(name)
    print("last_20_mean_return:", sum(returns[-20:]) / 20)
    print(q_values.argmax(axis=1).reshape(env.config.rows, env.config.cols))


def main() -> None:
    run("monte_carlo_control", monte_carlo_control)
    run("q_learning", q_learning)
    run("sarsa", sarsa)
    run("expected_sarsa", expected_sarsa)


if __name__ == "__main__":
    main()
