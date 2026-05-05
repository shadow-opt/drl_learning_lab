from __future__ import annotations

from drl_lab.algorithms.tabular import GridWorld, TabularControlConfig, monte_carlo_control


def main() -> None:
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
    greedy_policy = q_values.argmax(axis=1)
    print("last_20_mean_return:", sum(returns[-20:]) / 20)
    print("greedy_policy:")
    print(greedy_policy.reshape(env.config.rows, env.config.cols))


if __name__ == "__main__":
    main()
