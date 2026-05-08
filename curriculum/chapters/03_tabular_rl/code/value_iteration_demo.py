from __future__ import annotations

from drl_lab.algorithms.tabular import GridWorld, value_iteration


def main() -> None:
    env = GridWorld()
    values, policy = value_iteration(env)
    print(values.reshape(env.config.rows, env.config.cols))
    print(policy.reshape(env.config.rows, env.config.cols))


if __name__ == "__main__":
    main()
