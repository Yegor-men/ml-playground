"""Compare ordinary and depthwise-separable CNNs on MNIST."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# Experiment configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
NUM_EXAMPLES_TO_PLOT = 12
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ClassificationStats:
    loss: float
    accuracy: float


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            depth_multiplier: int = 1,
    ):
        super().__init__()
        hidden_channels = in_channels * depth_multiplier
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size,
                groups=in_channels,
            ),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def make_model(depthwise: bool) -> nn.Module:
    convolution = DepthwiseSeparableConv2d if depthwise else nn.Conv2d
    return nn.Sequential(
        convolution(1, 16, 5),
        nn.SiLU(),
        convolution(16, 32, 5),
        nn.SiLU(),
        nn.MaxPool2d(2),
        convolution(32, 64, 3),
        nn.SiLU(),
        nn.MaxPool2d(2),
        convolution(64, 128, 4),
        nn.Flatten(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 10),
    )


def make_dataloaders() -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(SEED)
    loader_options = {
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }
    return (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            **loader_options,
        ),
        DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            **loader_options,
        ),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_epoch(
        models: dict[str, nn.Module],
        dataloader: DataLoader,
        optimizers: dict[str, torch.optim.Optimizer] | None,
        epoch: int,
) -> dict[str, ClassificationStats]:
    training = optimizers is not None
    totals = {
        name: {"loss": 0.0, "correct": 0, "count": 0}
        for name in models
    }
    for model in models.values():
        model.train(training)

    label = "TRAIN" if training else "TEST"
    progress = tqdm(dataloader, desc=f"{label} E{epoch}", leave=False)
    for images, targets in progress:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        batch_size = targets.size(0)
        postfix = {}

        for name, model in models.items():
            if training:
                optimizers[name].zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                if training:
                    loss.backward()
                    optimizers[name].step()

            totals[name]["loss"] += float(loss.detach().cpu()) * batch_size
            totals[name]["correct"] += int(logits.argmax(dim=-1).eq(targets).sum())
            totals[name]["count"] += batch_size
            postfix[name] = f"{loss.detach().item():.3f}"

        progress.set_postfix(postfix)

    return {
        name: ClassificationStats(
            loss=values["loss"] / values["count"],
            accuracy=values["correct"] / values["count"],
        )
        for name, values in totals.items()
    }


def plot_history(history: dict[str, dict[str, list[float]]]):
    epochs = range(1, NUM_EPOCHS + 1)
    _, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, values in history.items():
        axes[0].plot(epochs, values["train_loss"], marker="o", label=f"{name} train")
        axes[0].plot(epochs, values["test_loss"], marker="o", linestyle="--", label=f"{name} test")
        axes[1].plot(epochs, values["train_accuracy"], marker="o", label=f"{name} train")
        axes[1].plot(
            epochs,
            values["test_accuracy"],
            marker="o",
            linestyle="--",
            label=f"{name} test",
        )

    axes[0].set_title("Cross-entropy loss")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    plt.tight_layout()


@torch.no_grad()
def plot_predictions(models: dict[str, nn.Module], test_loader: DataLoader):
    images, targets = next(iter(test_loader))
    images = images[:NUM_EXAMPLES_TO_PLOT].to(DEVICE)
    targets = targets[:NUM_EXAMPLES_TO_PLOT]
    predictions = {
        name: model.eval()(images).argmax(dim=-1).cpu()
        for name, model in models.items()
    }

    columns = 4
    rows = (len(images) + columns - 1) // columns
    _, axes = plt.subplots(rows, columns, figsize=(10, 2.4 * rows))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(images):
            continue
        target = int(targets[index])
        base = int(predictions["standard"][index])
        depthwise = int(predictions["depthwise"][index])
        axis.imshow(images[index].cpu().squeeze(), cmap="gray")
        axis.set_title(f"true {target} | standard {base}\ndepthwise {depthwise}")
    plt.suptitle("MNIST predictions after training")
    plt.tight_layout()


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train_loader, test_loader = make_dataloaders()
    models = {
        "standard": make_model(depthwise=False).to(DEVICE),
        "depthwise": make_model(depthwise=True).to(DEVICE),
    }
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        for name, model in models.items()
    }
    history = {
        name: {
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }
        for name in models
    }

    print(f"Device: {DEVICE}")
    for name, model in models.items():
        print(f"{name}: {parameter_count(model):,} trainable parameters")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_stats = run_epoch(models, train_loader, optimizers, epoch)
        test_stats = run_epoch(models, test_loader, None, epoch)
        for name in models:
            history[name]["train_loss"].append(train_stats[name].loss)
            history[name]["test_loss"].append(test_stats[name].loss)
            history[name]["train_accuracy"].append(train_stats[name].accuracy)
            history[name]["test_accuracy"].append(test_stats[name].accuracy)
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} | "
            + " | ".join(
                f"{name}: loss {test_stats[name].loss:.4f}, "
                f"accuracy {test_stats[name].accuracy:.2%}"
                for name in models
            )
        )

    plot_history(history)
    plot_predictions(models, test_loader)
    plt.show()


if __name__ == "__main__":
    main()
