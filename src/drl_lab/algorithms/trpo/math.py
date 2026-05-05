from __future__ import annotations

from collections.abc import Callable

import torch


def conjugate_gradient(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    max_iters: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """Solve ``Ax = b`` using conjugate gradient and an implicit ``A @ v``."""
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    x = torch.zeros_like(b)
    residual = b.clone()
    direction = residual.clone()
    residual_dot = torch.dot(residual, residual)

    for _ in range(max_iters):
        avp = matrix_vector_product(direction)
        alpha = residual_dot / (torch.dot(direction, avp) + 1e-12)
        x = x + alpha * direction
        residual = residual - alpha * avp
        new_residual_dot = torch.dot(residual, residual)
        if float(new_residual_dot) < residual_tol:
            break
        beta = new_residual_dot / (residual_dot + 1e-12)
        direction = residual + beta * direction
        residual_dot = new_residual_dot
    return x


def scale_step_to_kl(
    step_direction: torch.Tensor,
    fisher_vector_product: Callable[[torch.Tensor], torch.Tensor],
    max_kl: float,
) -> torch.Tensor:
    """Scale a natural-gradient direction to satisfy a quadratic KL limit."""
    if max_kl <= 0.0:
        raise ValueError("max_kl must be positive")
    quadratic_kl = 0.5 * torch.dot(step_direction, fisher_vector_product(step_direction))
    scale = torch.sqrt(torch.as_tensor(max_kl, dtype=step_direction.dtype) / (quadratic_kl + 1e-12))
    scaled_step: torch.Tensor = scale * step_direction
    return scaled_step
