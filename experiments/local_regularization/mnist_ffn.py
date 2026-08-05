import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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

# Experiment configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BATCH_SIZE = 128
NUM_EPOCHS = 5
HIDDEN_SIZES = [512, 256]
LEARNING_RATE = 1e-3
REGULARIZATION_STRENGTH = 100.0
OPTIMIZER_NAME = "adamw"  # "adamw" or "sgd"
GRAD_CLIP = None
NOISE_ALPHAS = [0.25, 0.5, 0.75]
TRAIN_LIMIT = None
TEST_LIMIT = None
NUM_WORKERS = 0
NUM_EXAMPLES_TO_PLOT = 12
SEED = 0
DETERMINISTIC = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ClassificationStats:
    loss: float
    accuracy: float
    regularization_loss: float


class MNISTFeedForward(nn.Module):
    def __init__(self, hidden_sizes: list[int]):
        super().__init__()
        layer_sizes = [28 * 28, *hidden_sizes, 10]
        layers: list[nn.Module] = [nn.Flatten()]

        for in_features, out_features in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def get_mnist_dataloaders() -> tuple[DataLoader, DataLoader]:
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

    if TRAIN_LIMIT is not None:
        train_dataset = Subset(train_dataset, range(min(TRAIN_LIMIT, len(train_dataset))))
    if TEST_LIMIT is not None:
        test_dataset = Subset(test_dataset, range(min(TEST_LIMIT, len(test_dataset))))

    generator = torch.Generator().manual_seed(SEED)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )
    return train_dataloader, test_dataloader


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if OPTIMIZER_NAME == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    if OPTIMIZER_NAME == "sgd":
        return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")


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
                totals[name]["regularization"] += (
                        float(regularization_loss.detach().cpu()) * batch_size
                )
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


@torch.no_grad()
def plot_predictions(models: dict[str, nn.Module], test_dataloader: DataLoader):
    images, targets = next(iter(test_dataloader))
    images = images[:NUM_EXAMPLES_TO_PLOT].to(DEVICE)
    targets = targets[:NUM_EXAMPLES_TO_PLOT]
    predictions = {}
    for name, model in models.items():
        model.eval()
        predictions[name] = model(images).argmax(dim=-1).cpu()

    columns = 4
    rows = (len(images) + columns - 1) // columns
    _, axes = plt.subplots(rows, columns, figsize=(10, 2.5 * rows))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(images):
            continue
        axis.imshow(images[index].cpu().squeeze(), cmap="gray")
        axis.set_title(
            f"true {int(targets[index])} | base {int(predictions['baseline'][index])}\n"
            f"regularized {int(predictions['local_variance'][index])}"
        )
    plt.suptitle("FFN predictions after training")
    plt.tight_layout()


def main():
    set_seed()
    train_dataloader, test_dataloader = get_mnist_dataloaders()

    baseline = MNISTFeedForward(HIDDEN_SIZES).to(DEVICE)
    regularized = copy.deepcopy(baseline).to(DEVICE)
    models = {
        "baseline": baseline,
        "local_variance": regularized,
    }
    optimizers = {name: make_optimizer(model) for name, model in models.items()}

    history = make_history(list(models))
    plot_title = (
        f"MNIST FFN | hidden={HIDDEN_SIZES} | "
        f"local variance lambda={REGULARIZATION_STRENGTH:g}"
    )
    final_test_stats = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_stats = train_one_epoch(
            models=models,
            optimizers=optimizers,
            dataloader=train_dataloader,
            device=DEVICE,
            epoch_index=epoch,
            regularization_strength=REGULARIZATION_STRENGTH,
            grad_clip=GRAD_CLIP,
        )
        test_stats = evaluate(
            models=models,
            dataloader=test_dataloader,
            device=DEVICE,
            epoch_index=epoch,
        )
        final_test_stats = test_stats
        append_history(history, train_stats, test_stats)

        print(brief_epoch_summary(epoch, NUM_EPOCHS, test_stats))

    plot_classification_history(history, plot_title)

    if NOISE_ALPHAS and final_test_stats is not None:
        plotted_noise_alphas = [0.0]
        noise_history = make_noise_history(list(models))
        append_noise_history(noise_history, final_test_stats)
        for alpha in NOISE_ALPHAS:
            noisy_stats = evaluate(
                models=models,
                dataloader=test_dataloader,
                device=DEVICE,
                epoch_index=NUM_EPOCHS,
                noise_alpha=alpha,
            )
            plotted_noise_alphas.append(alpha)
            append_noise_history(noise_history, noisy_stats)

        plot_noise_sweep(
            noise_alphas=plotted_noise_alphas,
            history=noise_history,
            title="MNIST FFN noise sweep: image = (1 - alpha) * image + alpha * U(0, 1)",
        )

    plot_predictions(models, test_dataloader)
    show_plots()


if __name__ == "__main__":
    main()
