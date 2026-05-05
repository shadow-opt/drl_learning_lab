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
from drl_lab.common.networks import MLP
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


@dataclass(frozen=True)
class BinaryClassifierConfig:
    seed: int = 123
    n_per_class: int = 128
    lr: float = 0.03
    epochs: int = 200
    run_dir: Path = Path("experiments/runs/binary_classifier")


def make_dataset(config: BinaryClassifierConfig) -> tuple[torch.Tensor, torch.Tensor]:
    class_0 = torch.randn(config.n_per_class, 2) * 0.4 + torch.tensor([-1.0, -1.0])
    class_1 = torch.randn(config.n_per_class, 2) * 0.4 + torch.tensor([1.0, 1.0])
    x = torch.cat([class_0, class_1], dim=0)
    y = torch.cat(
        [
            torch.zeros(config.n_per_class, 1),
            torch.ones(config.n_per_class, 1),
        ],
        dim=0,
    )
    return x, y


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
    return float((predictions == labels).float().mean())


def train(config: BinaryClassifierConfig) -> tuple[MLP, float]:
    set_global_seed(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshots(config, config.run_dir)

    x, y = make_dataset(config)
    model = MLP(input_dim=2, hidden_sizes=[16, 16], output_dim=1)
    optimizer = Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    with CsvLogger(config.run_dir / "metrics.csv", ["epoch", "loss", "accuracy"]) as logger:
        for epoch in range(config.epochs):
            model.train()
            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            logger.log(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "accuracy": accuracy_from_logits(logits.detach(), y),
                }
            )

    model.eval()
    with torch.inference_mode():
        final_accuracy = accuracy_from_logits(model(x), y)

    save_checkpoint(
        config.run_dir / "model.pt",
        model,
        optimizer,
        metadata=CheckpointMetadata(step=config.epochs, seed=config.seed),
        extra={"final_accuracy": final_accuracy},
    )

    example_input = x[:4]
    onnx_path = export_to_onnx(model, example_input, config.run_dir / "model.onnx")
    result = compare_pytorch_onnx(model, onnx_path, example_input)
    if not result.passed:
        raise RuntimeError(f"ONNX consistency failed: {result}")

    return model, final_accuracy


def main() -> None:
    _, final_accuracy = train(BinaryClassifierConfig())
    print(f"final_accuracy={final_accuracy:.4f}")


if __name__ == "__main__":
    main()
