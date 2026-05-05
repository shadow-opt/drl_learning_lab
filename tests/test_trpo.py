from __future__ import annotations

import torch

from drl_lab.algorithms.trpo import conjugate_gradient, scale_step_to_kl


def test_conjugate_gradient_solves_positive_definite_system() -> None:
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])

    solution = conjugate_gradient(lambda vector: matrix @ vector, b, max_iters=10)

    assert torch.allclose(matrix @ solution, b, atol=1e-5)


def test_scale_step_to_kl_satisfies_quadratic_limit() -> None:
    fisher = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    direction = torch.tensor([1.0, 1.0])
    max_kl = 0.01

    step = scale_step_to_kl(direction, lambda vector: fisher @ vector, max_kl=max_kl)
    quadratic_kl = 0.5 * torch.dot(step, fisher @ step)

    assert torch.isclose(quadratic_kl, torch.tensor(max_kl), atol=1e-6)
