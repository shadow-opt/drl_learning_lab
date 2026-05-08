from __future__ import annotations

from drl_lab.algorithms.tabular import GridWorld, policy_iteration


def main() -> None:
    env = GridWorld()
    values, policy = policy_iteration(env, gamma=1.0)
    print("values:")
    print(values.reshape(env.config.rows, env.config.cols))
    print("policy:")
    print(policy.reshape(env.config.rows, env.config.cols))


if __name__ == "__main__":
    main()
