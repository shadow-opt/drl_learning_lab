from __future__ import annotations

from drl_lab.algorithms.tabular import GridWorld, value_iteration


def main() -> None:
    env = GridWorld()
    values, policy = value_iteration(env, gamma=1.0)
    print("action legend: 0=up, 1=right, 2=down, 3=left")
    print("values:")
    print(values.reshape(env.config.rows, env.config.cols))
    print("policy:")
    print(policy.reshape(env.config.rows, env.config.cols))


if __name__ == "__main__":
    main()
