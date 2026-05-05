from __future__ import annotations

import torch

from drl_lab.algorithms.trpo import conjugate_gradient, scale_step_to_kl


def main() -> None:
    fisher = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    gradient = torch.tensor([1.0, 2.0])
    natural_direction = conjugate_gradient(lambda vector: fisher @ vector, gradient, max_iters=10)
    step = scale_step_to_kl(natural_direction, lambda vector: fisher @ vector, max_kl=0.01)
    quadratic_kl = 0.5 * torch.dot(step, fisher @ step)

    print(f"natural_direction={natural_direction.tolist()}")
    print(f"scaled_step={step.tolist()}")
    print(f"quadratic_kl={float(quadratic_kl):.6f}")


if __name__ == "__main__":
    main()
