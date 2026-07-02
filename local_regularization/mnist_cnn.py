from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from regularizer import local_weight_variance_loss
except ImportError:
    from .regularizer import local_weight_variance_loss


@dataclass
class ClassificationStats:
    loss: float
    accuracy: float
    regularization_loss: float


class MNISTConvNet(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one convolution width.")

        feature_layers: list[nn.Module] = []
        in_channels = 1
        for index, out_channels in enumerate(channels):
            feature_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            feature_layers.append(nn.SiLU())
            if index < 2:
                feature_layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        feature_layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.features = nn.Sequential(*feature_layers)
        self.classifier = nn.Linear(channels[-1], 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_int_list(raw_values: str) -> list[int]:
    if raw_values.strip() == "":
        return []
    return [int(value.strip()) for value in raw_values.split(",") if value.strip()]


def parse_float_list(raw_values: str) -> list[float]:
    if raw_values.strip() == "":
        return []
    return [float(value.strip()) for value in raw_values.split(",") if value.strip()]


def set_seed(seed: int, deterministic: bool):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def get_mnist_dataloaders(
    data_dir: Path,
    batch_size: int,
    train_limit: int | None,
    test_limit: int | None,
    num_workers: int,
    seed: int,
    download: bool,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=download,
        transform=transform,
    )

    if train_limit is not None:
        train_dataset = Subset(train_dataset, range(min(train_limit, len(train_dataset))))
    if test_limit is not None:
        test_dataset = Subset(test_dataset, range(min(test_limit, len(test_dataset))))

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataloader, test_dataloader


def make_optimizer(
    model: nn.Module,
    lr: float,
    optimizer_name: str,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_one_epoch(
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    dataloader: DataLoader,
    device: torch.device,
    epoch_index: int,
    regularization_strength: float,
    grad_clip: float | None,
) -> dict[str, ClassificationStats]:
    totals = {
        name: {"loss": 0.0, "correct": 0.0, "regularization": 0.0, "count": 0}
        for name in models
    }

    for model in models.values():
        model.train()

    progress = tqdm(dataloader, desc=f"TRAIN E{epoch_index}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)

        postfix = {}
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            task_loss = F.cross_entropy(logits, targets)
            regularization_loss = local_weight_variance_loss(model)
            loss = task_loss
            if name == "local_variance":
                loss = loss + regularization_strength * regularization_loss

            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = predictions.eq(targets).sum().item()
                totals[name]["loss"] += float(task_loss.detach().cpu()) * batch_size
                totals[name]["correct"] += correct
                totals[name]["regularization"] += float(regularization_loss.detach().cpu()) * batch_size
                totals[name]["count"] += batch_size
                postfix[f"{name}_loss"] = f"{float(task_loss.detach().cpu()):.3f}"
                postfix[f"{name}_acc"] = f"{correct / batch_size:.3f}"

        progress.set_postfix(postfix)

    return {
        name: ClassificationStats(
            loss=values["loss"] / values["count"],
            accuracy=values["correct"] / values["count"],
            regularization_loss=values["regularization"] / values["count"],
        )
        for name, values in totals.items()
    }


@torch.no_grad()
def evaluate(
    models: dict[str, nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    epoch_index: int,
    noise_alpha: float = 0.0,
) -> dict[str, ClassificationStats]:
    totals = {
        name: {"loss": 0.0, "correct": 0.0, "regularization": 0.0, "count": 0}
        for name in models
    }

    for model in models.values():
        model.eval()

    label = f"TEST  E{epoch_index}"
    if noise_alpha > 0.0:
        label = f"NOISE {noise_alpha:.2f}"
    progress = tqdm(dataloader, desc=label, leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if noise_alpha > 0.0:
            noise = torch.rand_like(images)
            images = torch.lerp(images, noise, noise_alpha)

        batch_size = targets.size(0)
        postfix = {}
        for name, model in models.items():
            logits = model(images)
            task_loss = F.cross_entropy(logits, targets)
            regularization_loss = local_weight_variance_loss(model)
            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(targets).sum().item()

            totals[name]["loss"] += float(task_loss.cpu()) * batch_size
            totals[name]["correct"] += correct
            totals[name]["regularization"] += float(regularization_loss.cpu()) * batch_size
            totals[name]["count"] += batch_size
            postfix[name] = f"{correct / batch_size:.3f}"

        progress.set_postfix(postfix)

    return {
        name: ClassificationStats(
            loss=values["loss"] / values["count"],
            accuracy=values["correct"] / values["count"],
            regularization_loss=values["regularization"] / values["count"],
        )
        for name, values in totals.items()
    }


def make_history(model_names: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "train_variance": [],
            "test_variance": [],
        }
        for name in model_names
    }


def append_history(
    history: dict[str, dict[str, list[float]]],
    train_stats: dict[str, ClassificationStats],
    test_stats: dict[str, ClassificationStats],
):
    for name in history:
        history[name]["train_loss"].append(train_stats[name].loss)
        history[name]["test_loss"].append(test_stats[name].loss)
        history[name]["train_accuracy"].append(train_stats[name].accuracy)
        history[name]["test_accuracy"].append(test_stats[name].accuracy)
        history[name]["train_variance"].append(train_stats[name].regularization_loss)
        history[name]["test_variance"].append(test_stats[name].regularization_loss)


def brief_epoch_summary(
    epoch: int,
    total_epochs: int,
    test_stats: dict[str, ClassificationStats],
) -> str:
    parts = [
        (
            f"{name}: loss {stats.loss:.4f}, "
            f"acc {stats.accuracy:.4f}, var {stats.regularization_loss:.6f}"
        )
        for name, stats in test_stats.items()
    ]
    return f"Epoch {epoch:02d}/{total_epochs:02d} | " + " | ".join(parts)


def show_plots():
    import matplotlib.pyplot as plt

    if plt.get_backend().lower() == "agg":
        for figure_number in plt.get_fignums():
            plt.figure(figure_number).canvas.draw()
    else:
        plt.show()


def plot_classification_history(
    history: dict[str, dict[str, list[float]]],
    title: str,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    epochs = range(1, len(next(iter(history.values()))["train_loss"]) + 1)
    panels = [
        (axes[0], "loss", "Cross-entropy loss"),
        (axes[1], "accuracy", "Classification accuracy"),
        (axes[2], "variance", "Mean local variance"),
    ]

    for ax, metric, label in panels:
        ax.clear()
        for name, values in history.items():
            ax.plot(epochs, values[f"train_{metric}"], marker="o", label=f"{name} train")
            ax.plot(
                epochs,
                values[f"test_{metric}"],
                marker="o",
                linestyle="--",
                label=f"{name} clean test",
            )
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()


def make_noise_history(model_names: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        name: {"loss": [], "accuracy": [], "variance": []}
        for name in model_names
    }


def append_noise_history(
    history: dict[str, dict[str, list[float]]],
    stats: dict[str, ClassificationStats],
):
    for name in history:
        history[name]["loss"].append(stats[name].loss)
        history[name]["accuracy"].append(stats[name].accuracy)
        history[name]["variance"].append(stats[name].regularization_loss)


def plot_noise_sweep(
    noise_alphas: list[float],
    history: dict[str, dict[str, list[float]]],
    title: str,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (axes[0], "loss", "Cross-entropy loss"),
        (axes[1], "accuracy", "Classification accuracy"),
        (axes[2], "variance", "Mean local variance"),
    ]

    for ax, metric, label in panels:
        for name, values in history.items():
            ax.plot(noise_alphas, values[metric], marker="o", label=name)
        ax.set_title(label)
        ax.set_xlabel("noise alpha")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side-by-side MNIST CNN comparison for local weight-variance regularization."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--channels", type=str, default="32,64,128")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regularization-strength", type=float, default=10.0)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--noise-alphas", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace):
    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)

    channels = parse_int_list(args.channels)
    train_dataloader, test_dataloader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        num_workers=args.num_workers,
        seed=args.seed,
        download=not args.no_download,
    )

    baseline = MNISTConvNet(channels).to(device)
    regularized = copy.deepcopy(baseline).to(device)
    models = {
        "baseline": baseline,
        "local_variance": regularized,
    }
    optimizers = {
        name: make_optimizer(model, lr=args.lr, optimizer_name=args.optimizer)
        for name, model in models.items()
    }

    history = make_history(list(models))
    plot_title = (
        f"MNIST CNN | channels={channels} | "
        f"local variance lambda={args.regularization_strength:g}"
    )
    final_test_stats = None

    for epoch in range(1, args.num_epochs + 1):
        train_stats = train_one_epoch(
            models=models,
            optimizers=optimizers,
            dataloader=train_dataloader,
            device=device,
            epoch_index=epoch,
            regularization_strength=args.regularization_strength,
            grad_clip=args.grad_clip,
        )
        test_stats = evaluate(
            models=models,
            dataloader=test_dataloader,
            device=device,
            epoch_index=epoch,
        )
        final_test_stats = test_stats
        append_history(history, train_stats, test_stats)

        print(brief_epoch_summary(epoch, args.num_epochs, test_stats))

    plot_classification_history(history, plot_title)

    noise_alphas = parse_float_list(args.noise_alphas)
    if noise_alphas and final_test_stats is not None:
        plotted_noise_alphas = [0.0]
        noise_history = make_noise_history(list(models))
        append_noise_history(noise_history, final_test_stats)
        for alpha in noise_alphas:
            noisy_stats = evaluate(
                models=models,
                dataloader=test_dataloader,
                device=device,
                epoch_index=args.num_epochs,
                noise_alpha=alpha,
            )
            plotted_noise_alphas.append(alpha)
            append_noise_history(noise_history, noisy_stats)

        plot_noise_sweep(
            noise_alphas=plotted_noise_alphas,
            history=noise_history,
            title="MNIST CNN noise sweep: image = (1 - alpha) * image + alpha * U(0, 1)",
        )

    show_plots()


if __name__ == "__main__":
    main(get_args())
