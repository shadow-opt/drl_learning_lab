from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from drl_lab.common.checkpoint import CheckpointMetadata, save_checkpoint
from drl_lab.common.experiment import save_run_snapshots
from drl_lab.common.export import export_to_onnx
from drl_lab.common.logging import CsvLogger
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


@dataclass(frozen=True)
class LinearRegressionConfig:
    seed: int = 42
    n_samples: int = 256
    input_dim: int = 3
    lr: float = 0.05
    epochs: int = 200
    run_dir: Path = Path("experiments/runs/linear_regression")


class LinearRegressor(nn.Module):
    """Linear regression model.

    Input shape: ``[batch, input_dim]``.
    Output shape: ``[batch, 1]``.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected [batch, input_dim], got {tuple(x.shape)}")
        output: torch.Tensor = self.linear(x)
        return output


def make_dataset(config: LinearRegressionConfig) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(config.n_samples, config.input_dim)
    true_w = torch.tensor([[2.0], [-3.0], [0.5]], dtype=torch.float32)
    true_b = torch.tensor([0.25], dtype=torch.float32)
    y = x @ true_w + true_b + 0.05 * torch.randn(config.n_samples, 1)
    return x, y


def train(config: LinearRegressionConfig) -> tuple[LinearRegressor, float]:
    set_global_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshots(config, config.run_dir)

    x, y = make_dataset(config)
    model = LinearRegressor(config.input_dim)
    optimizer = Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    metrics_path = config.run_dir / "metrics.csv"
    with CsvLogger(metrics_path, ["epoch", "loss"]) as logger:
        for epoch in range(config.epochs):
            model.train()
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            logger.log({"epoch": epoch, "loss": float(loss.detach())})

    final_loss = float(loss_fn(model(x), y).detach())
    save_checkpoint(
        config.run_dir / "model.pt",
        model,
        optimizer,
        metadata=CheckpointMetadata(step=config.epochs, seed=config.seed),
        extra={"final_loss": final_loss},
    )

    example_input = x[:4]
    onnx_path = export_to_onnx(model, example_input, config.run_dir / "model.onnx")
    result = compare_pytorch_onnx(model, onnx_path, example_input)
    if not result.passed:
        raise RuntimeError(f"ONNX consistency failed: {result}")

    return model, final_loss


def main() -> None:
    _, final_loss = train(LinearRegressionConfig())
    print(f"final_loss={final_loss:.6f}")


if __name__ == "__main__":
    main()
