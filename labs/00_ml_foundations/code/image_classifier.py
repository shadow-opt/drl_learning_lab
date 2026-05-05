from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from drl_lab.common.checkpoint import CheckpointMetadata, save_checkpoint
from drl_lab.common.device import resolve_device
from drl_lab.common.export import export_to_onnx
from drl_lab.common.logging import CsvLogger
from drl_lab.common.onnx_check import compare_pytorch_onnx
from drl_lab.common.seed import set_global_seed


@dataclass(frozen=True)
class ImageClassifierConfig:
    seed: int = 321
    num_classes: int = 10
    samples_per_class: int = 24
    batch_size: int = 32
    lr: float = 0.003
    epochs: int = 8
    device: str = "cpu"
    run_dir: Path = Path("experiments/runs/image_classifier")


class PatternImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Offline 28x28 image dataset with class-specific stripe patterns.

    Item shapes:
        image: ``[1, 28, 28]`` float32
        label: scalar int64
    """

    def __init__(self, num_classes: int, samples_per_class: int, seed: int) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive")

        generator = torch.Generator().manual_seed(seed)
        images: list[torch.Tensor] = []
        labels: list[int] = []
        for label in range(num_classes):
            base = self._base_pattern(label)
            for _ in range(samples_per_class):
                noise = 0.03 * torch.randn(1, 28, 28, generator=generator)
                image = (base + noise).clamp(0.0, 1.0)
                images.append(image)
                labels.append(label)

        self.images = torch.stack(images).to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]

    @staticmethod
    def _base_pattern(label: int) -> torch.Tensor:
        image = torch.zeros(1, 28, 28, dtype=torch.float32)
        row = 2 + (label * 2) % 20
        col = 2 + (label * 3) % 20
        image[:, row : row + 4, :] = 0.65
        image[:, :, col : col + 3] = 0.45
        image[:, 4 + label % 8 : 12 + label % 8, 4 + label : 8 + label] = 1.0
        return image


class SmallImageClassifier(nn.Module):
    """Small CNN for 28x28 grayscale classification.

    Input shape: ``[batch, 1, 28, 28]``.
    Output shape: logits ``[batch, num_classes]``.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"expected [batch, 1, 28, 28], got {tuple(images.shape)}")
        features = self.features(images)
        logits: torch.Tensor = self.classifier(features)
        return logits


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean())


ImageLoader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


def make_loaders(config: ImageClassifierConfig) -> tuple[ImageLoader, ImageLoader]:
    dataset = PatternImageDataset(
        num_classes=config.num_classes,
        samples_per_class=config.samples_per_class,
        seed=config.seed,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(
    model: SmallImageClassifier,
    loader: ImageLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.detach()) * int(labels.numel())
            total_correct += int((logits.argmax(dim=-1) == labels).sum())
            total_examples += int(labels.numel())
    if total_examples == 0:
        raise RuntimeError("empty evaluation loader")
    return total_loss / total_examples, total_correct / total_examples


def train(config: ImageClassifierConfig) -> tuple[SmallImageClassifier, float]:
    set_global_seed(config.seed)
    device = resolve_device(config.device)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(config)

    model = SmallImageClassifier(config.num_classes).to(device=device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    with CsvLogger(
        config.run_dir / "metrics.csv",
        ["epoch", "train_loss", "val_loss", "val_accuracy"],
    ) as logger:
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            seen = 0
            for images, labels in train_loader:
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)
                logits = model(images)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.detach()) * int(labels.numel())
                seen += int(labels.numel())

            val_loss, val_accuracy = evaluate(model, val_loader, device)
            logger.log(
                {
                    "epoch": epoch,
                    "train_loss": running_loss / seen,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )

    _, final_accuracy = evaluate(model, val_loader, device)
    save_checkpoint(
        config.run_dir / "model.pt",
        model,
        optimizer,
        metadata=CheckpointMetadata(step=config.epochs, seed=config.seed),
        extra={"final_accuracy": final_accuracy},
    )

    example_input = next(iter(val_loader))[0][:4].to(device=device, dtype=torch.float32)
    onnx_path = export_to_onnx(model, example_input, config.run_dir / "model.onnx")
    result = compare_pytorch_onnx(model, onnx_path, example_input)
    if not result.passed:
        raise RuntimeError(f"ONNX consistency failed: {result}")

    return model, final_accuracy


def main() -> None:
    _, final_accuracy = train(ImageClassifierConfig())
    print(f"final_accuracy={final_accuracy:.4f}")


if __name__ == "__main__":
    main()
