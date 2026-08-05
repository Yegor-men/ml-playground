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

# Experiment configuration
TRAIN_POINTS = 100
EVAL_POINTS = 5_000
GRID_SIZE = 160
LABEL_FLIP_PROBABILITY = 0.20
NUM_EPOCHS = 500
STATE_SIZE = 128
RESIDUAL_HIDDEN_SIZE = 128
NUM_BLOCKS = 3
RESIDUAL_SCALE = 1.0
LEARNING_RATE = 1e-3
REGULARIZATION_STRENGTH = 30.0
OPTIMIZER_NAME = "adamw"  # "adamw" or "sgd"
GRAD_CLIP = None
SUMMARY_EVERY = 100
SEED = 0
DETERMINISTIC = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class ResidualBoundaryBlock(nn.Module):
    def __init__(self, state_size: int, residual_hidden_size: int, residual_scale: float):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Linear(state_size, residual_hidden_size),
            nn.SiLU(),
            nn.Linear(residual_hidden_size, state_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.net(x)


class ResidualBoundaryMLP(nn.Module):
    def __init__(
            self,
            state_size: int,
            residual_hidden_size: int,
            num_blocks: int,
            residual_scale: float,
    ):
        super().__init__()
        if state_size < 1:
            raise ValueError("state_size must be positive.")
        if residual_hidden_size < 1:
            raise ValueError("residual_hidden_size must be positive.")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative.")

        self.input_projection = nn.Sequential(
            nn.Linear(2, state_size),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBoundaryBlock(
                    state_size=state_size,
                    residual_hidden_size=residual_hidden_size,
                    residual_scale=residual_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(state_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.blocks(x)
        return self.classifier(x)


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


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


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if OPTIMIZER_NAME == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    if OPTIMIZER_NAME == "sgd":
        return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")


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


def main():
    set_seed()
    plot_data = make_boundary_data(
        train_points=TRAIN_POINTS,
        eval_points=EVAL_POINTS,
        grid_size=GRID_SIZE,
        label_flip_probability=LABEL_FLIP_PROBABILITY,
        seed=SEED,
    )
    device_data = move_data_to_device(plot_data, DEVICE)

    baseline = ResidualBoundaryMLP(
        state_size=STATE_SIZE,
        residual_hidden_size=RESIDUAL_HIDDEN_SIZE,
        num_blocks=NUM_BLOCKS,
        residual_scale=RESIDUAL_SCALE,
    ).to(DEVICE)
    regularized = copy.deepcopy(baseline).to(DEVICE)
    models = {
        "baseline": baseline,
        "local_variance": regularized,
    }
    optimizers = {name: make_optimizer(model) for name, model in models.items()}

    parameter_count = count_parameters(baseline)
    history = make_history(list(models))
    title = (
        f"Noisy linear boundary | train={TRAIN_POINTS} | "
        f"residual={NUM_BLOCKS}x({STATE_SIZE}->{RESIDUAL_HIDDEN_SIZE}->{STATE_SIZE}) | "
        f"flips={LABEL_FLIP_PROBABILITY:g} | "
        f"params/train={parameter_count / TRAIN_POINTS:.1f} | "
        f"local variance lambda={REGULARIZATION_STRENGTH:g}"
    )

    progress = tqdm(range(1, NUM_EPOCHS + 1), desc="TRAIN", leave=False)
    for epoch in progress:
        train_one_epoch(
            models=models,
            optimizers=optimizers,
            data=device_data,
            regularization_strength=REGULARIZATION_STRENGTH,
            grad_clip=GRAD_CLIP,
        )
        stats = evaluate(models, device_data)
        append_history(history, stats)

        should_summarize = (
                SUMMARY_EVERY > 0
                and (epoch == 1 or epoch % SUMMARY_EVERY == 0 or epoch == NUM_EPOCHS)
        )
        if should_summarize:
            progress.set_postfix(
                baseline_clean=f"{stats['baseline'].clean_eval_accuracy:.3f}",
                local_clean=f"{stats['local_variance'].clean_eval_accuracy:.3f}",
                local_var=f"{stats['local_variance'].regularization_loss:.5f}",
            )
            progress.write(brief_epoch_summary(epoch, NUM_EPOCHS, stats))

    plot_results(models, plot_data, device_data, history, title)
    show_plots()


if __name__ == "__main__":
    main()
