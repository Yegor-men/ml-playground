from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

try:
    from regularizer import local_weight_variance_loss
except ImportError:
    from .regularizer import local_weight_variance_loss


@dataclass
class BoundaryData:
    train_x: torch.Tensor
    train_clean_y: torch.Tensor
    train_observed_y: torch.Tensor
    flipped: torch.Tensor
    eval_x: torch.Tensor
    eval_clean_y: torch.Tensor
    grid_x: torch.Tensor
    grid_y: torch.Tensor
    grid_points: torch.Tensor
    grid_clean_y: torch.Tensor


@dataclass
class BoundaryStats:
    observed_train_loss: float
    observed_train_accuracy: float
    clean_train_accuracy: float
    clean_eval_accuracy: float
    regularization_loss: float


class BoundaryMLP(nn.Module):
    def __init__(self, hidden_sizes: list[int]):
        super().__init__()
        layer_sizes = [2, *hidden_sizes, 2]
        layers: list[nn.Module] = []
        for in_features, out_features in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_int_list(raw_values: str) -> list[int]:
    if raw_values.strip() == "":
        return []
    return [int(value.strip()) for value in raw_values.split(",") if value.strip()]


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


def clean_boundary_label(points: torch.Tensor) -> torch.Tensor:
    return points.sum(dim=-1).gt(0.0).to(torch.long)


def make_boundary_data(
    train_points: int,
    eval_points: int,
    grid_size: int,
    label_flip_probability: float,
    seed: int,
) -> BoundaryData:
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_x = 2.0 * torch.rand(train_points, 2, generator=generator) - 1.0
    train_clean_y = clean_boundary_label(train_x)
    flipped = torch.rand(train_points, generator=generator).lt(label_flip_probability)
    train_observed_y = torch.where(flipped, 1 - train_clean_y, train_clean_y)

    eval_x = 2.0 * torch.rand(eval_points, 2, generator=generator) - 1.0
    eval_clean_y = clean_boundary_label(eval_x)

    grid_axis = torch.linspace(-1.0, 1.0, grid_size)
    grid_y, grid_x = torch.meshgrid(grid_axis, grid_axis, indexing="ij")
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    grid_clean_y = clean_boundary_label(grid_points).reshape(grid_size, grid_size)

    return BoundaryData(
        train_x=train_x,
        train_clean_y=train_clean_y,
        train_observed_y=train_observed_y,
        flipped=flipped,
        eval_x=eval_x,
        eval_clean_y=eval_clean_y,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_points=grid_points,
        grid_clean_y=grid_clean_y,
    )


def move_data_to_device(data: BoundaryData, device: torch.device) -> BoundaryData:
    return BoundaryData(
        train_x=data.train_x.to(device),
        train_clean_y=data.train_clean_y.to(device),
        train_observed_y=data.train_observed_y.to(device),
        flipped=data.flipped.to(device),
        eval_x=data.eval_x.to(device),
        eval_clean_y=data.eval_clean_y.to(device),
        grid_x=data.grid_x.to(device),
        grid_y=data.grid_y.to(device),
        grid_points=data.grid_points.to(device),
        grid_clean_y=data.grid_clean_y.to(device),
    )


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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(target).to(torch.float32).mean().detach().cpu())


def train_one_epoch(
    models: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    data: BoundaryData,
    regularization_strength: float,
    grad_clip: float | None,
):
    for model in models.values():
        model.train()

    for name, model in models.items():
        optimizer = optimizers[name]
        optimizer.zero_grad(set_to_none=True)

        logits = model(data.train_x)
        task_loss = F.cross_entropy(logits, data.train_observed_y)
        regularization_loss = local_weight_variance_loss(model)
        loss = task_loss
        if name == "local_variance":
            loss = loss + regularization_strength * regularization_loss

        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


@torch.no_grad()
def evaluate(models: dict[str, nn.Module], data: BoundaryData) -> dict[str, BoundaryStats]:
    stats = {}
    for name, model in models.items():
        model.eval()
        train_logits = model(data.train_x)
        eval_logits = model(data.eval_x)
        stats[name] = BoundaryStats(
            observed_train_loss=float(F.cross_entropy(train_logits, data.train_observed_y).cpu()),
            observed_train_accuracy=accuracy_from_logits(train_logits, data.train_observed_y),
            clean_train_accuracy=accuracy_from_logits(train_logits, data.train_clean_y),
            clean_eval_accuracy=accuracy_from_logits(eval_logits, data.eval_clean_y),
            regularization_loss=float(local_weight_variance_loss(model).cpu()),
        )
    return stats


def make_history(model_names: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "observed_train_loss": [],
            "observed_train_accuracy": [],
            "clean_train_accuracy": [],
            "clean_eval_accuracy": [],
            "regularization_loss": [],
        }
        for name in model_names
    }


def append_history(
    history: dict[str, dict[str, list[float]]],
    stats: dict[str, BoundaryStats],
):
    for name in history:
        history[name]["observed_train_loss"].append(stats[name].observed_train_loss)
        history[name]["observed_train_accuracy"].append(stats[name].observed_train_accuracy)
        history[name]["clean_train_accuracy"].append(stats[name].clean_train_accuracy)
        history[name]["clean_eval_accuracy"].append(stats[name].clean_eval_accuracy)
        history[name]["regularization_loss"].append(stats[name].regularization_loss)


def brief_epoch_summary(
    epoch: int,
    total_epochs: int,
    stats: dict[str, BoundaryStats],
) -> str:
    parts = [
        (
            f"{name}: observed {values.observed_train_accuracy:.3f}, "
            f"clean eval {values.clean_eval_accuracy:.3f}, var {values.regularization_loss:.6f}"
        )
        for name, values in stats.items()
    ]
    return f"Epoch {epoch:04d}/{total_epochs:04d} | " + " | ".join(parts)


def show_plots():
    import matplotlib.pyplot as plt

    if plt.get_backend().lower() == "agg":
        for figure_number in plt.get_fignums():
            plt.figure(figure_number).canvas.draw()
    else:
        plt.show()


@torch.no_grad()
def plot_results(
    models: dict[str, nn.Module],
    plot_data: BoundaryData,
    device_data: BoundaryData,
    history: dict[str, dict[str, list[float]]],
    title: str,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    top_axes = axes[0]
    bottom_axes = axes[1]

    class_colors = ["tab:blue", "tab:orange"]
    cmap = "coolwarm"
    boundary_panels = [
        ("clean target", plot_data.grid_clean_y.cpu()),
    ]
    for name, model in models.items():
        model.eval()
        logits = model(device_data.grid_points)
        predicted = logits.argmax(dim=-1).reshape_as(device_data.grid_clean_y).cpu()
        boundary_panels.append((name, predicted))

    for ax, (panel_title, grid_values) in zip(top_axes, boundary_panels):
        ax.contourf(
            plot_data.grid_x,
            plot_data.grid_y,
            grid_values.to(torch.float32),
            levels=[-0.5, 0.5, 1.5],
            alpha=0.28,
            cmap=cmap,
        )
        ax.contour(
            plot_data.grid_x,
            plot_data.grid_y,
            grid_values.to(torch.float32),
            levels=[0.5],
            colors="black",
            linewidths=1.0,
        )
        for label in (0, 1):
            mask = plot_data.train_observed_y.eq(label) & ~plot_data.flipped
            ax.scatter(
                plot_data.train_x[mask, 0],
                plot_data.train_x[mask, 1],
                s=26,
                color=class_colors[label],
                edgecolor="white",
                linewidth=0.5,
                label=f"observed {label}" if panel_title == "clean target" else None,
            )
        flipped = plot_data.flipped
        ax.scatter(
            plot_data.train_x[flipped, 0],
            plot_data.train_x[flipped, 1],
            s=58,
            c=[class_colors[int(label)] for label in plot_data.train_observed_y[flipped].tolist()],
            marker="x",
            linewidth=1.8,
            label="flipped labels" if panel_title == "clean target" else None,
        )
        ax.set_title(panel_title)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        if panel_title == "clean target":
            ax.legend(fontsize=8)

    epochs = range(1, len(next(iter(history.values()))["observed_train_loss"]) + 1)
    metric_panels = [
        (bottom_axes[0], "observed_train_accuracy", "Observed-label train accuracy"),
        (bottom_axes[1], "clean_eval_accuracy", "Clean eval accuracy"),
        (bottom_axes[2], "regularization_loss", "Mean local variance"),
    ]
    for ax, metric, panel_title in metric_panels:
        for name, values in history.items():
            ax.plot(epochs, values[metric], label=name)
        ax.set_title(panel_title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noisy-label 2D boundary test for local weight-variance regularization."
    )
    parser.add_argument("--train-points", type=int, default=128)
    parser.add_argument("--eval-points", type=int, default=5_000)
    parser.add_argument("--grid-size", type=int, default=160)
    parser.add_argument("--label-flip-probability", type=float, default=0.20)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--hidden-sizes", type=str, default="64,64,64")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regularization-strength", type=float, default=30.0)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--summary-every", "--log-every", dest="summary_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace):
    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)

    plot_data = make_boundary_data(
        train_points=args.train_points,
        eval_points=args.eval_points,
        grid_size=args.grid_size,
        label_flip_probability=args.label_flip_probability,
        seed=args.seed,
    )
    device_data = move_data_to_device(plot_data, device)

    hidden_sizes = parse_int_list(args.hidden_sizes)
    baseline = BoundaryMLP(hidden_sizes).to(device)
    regularized = copy.deepcopy(baseline).to(device)
    models = {
        "baseline": baseline,
        "local_variance": regularized,
    }
    optimizers = {
        name: make_optimizer(model, lr=args.lr, optimizer_name=args.optimizer)
        for name, model in models.items()
    }

    parameter_count = count_parameters(baseline)
    history = make_history(list(models))
    title = (
        f"Noisy linear boundary | train={args.train_points} | "
        f"flips={args.label_flip_probability:g} | params/train={parameter_count / args.train_points:.1f} | "
        f"local variance lambda={args.regularization_strength:g}"
    )

    progress = tqdm(range(1, args.num_epochs + 1), desc="TRAIN", leave=False)
    for epoch in progress:
        train_one_epoch(
            models=models,
            optimizers=optimizers,
            data=device_data,
            regularization_strength=args.regularization_strength,
            grad_clip=args.grad_clip,
        )
        stats = evaluate(models, device_data)
        append_history(history, stats)

        should_summarize = (
            args.summary_every > 0
            and (epoch == 1 or epoch % args.summary_every == 0 or epoch == args.num_epochs)
        )
        if should_summarize:
            progress.set_postfix(
                baseline_clean=f"{stats['baseline'].clean_eval_accuracy:.3f}",
                local_clean=f"{stats['local_variance'].clean_eval_accuracy:.3f}",
                local_var=f"{stats['local_variance'].regularization_loss:.5f}",
            )
            progress.write(brief_epoch_summary(epoch, args.num_epochs, stats))

    plot_results(models, plot_data, device_data, history, title)
    show_plots()


if __name__ == "__main__":
    main(get_args())
