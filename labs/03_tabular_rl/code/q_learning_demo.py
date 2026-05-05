from __future__ import annotations

from drl_lab.algorithms.tabular import GridWorld, QLearningConfig, q_learning


def main() -> None:
    env = GridWorld()
    q_values, returns = q_learning(
        env,
        QLearningConfig(episodes=300, epsilon=0.2, alpha=0.3, gamma=1.0, seed=3),
    )
    greedy_policy = q_values.argmax(axis=1)
    print("last_10_mean_return:", sum(returns[-10:]) / 10)
    print("greedy_policy:")
    print(greedy_policy.reshape(env.config.rows, env.config.cols))


if __name__ == "__main__":
    main()
