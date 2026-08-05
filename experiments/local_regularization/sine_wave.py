import copy
import math
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
TRAIN_POINTS = 32
EVAL_POINTS = 256
X_MIN = 0.0
X_MAX = 2.0 * math.pi
LABEL_NOISE = 0.15
NUM_EPOCHS = 2_000
HIDDEN_SIZES = [128, 128, 128]
LEARNING_RATE = 1e-3
REGULARIZATION_STRENGTH = 10.0
OPTIMIZER_NAME = "adamw"  # "adamw" or "sgd"
GRAD_CLIP = None
SUMMARY_EVERY = 200
SEED = 0
DETERMINISTIC = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RegressionStats:
    observed_train_mse: float
    clean_train_mse: float
    clean_eval_mse: float
    regularization_loss: float


class SineNet(nn.Module):
    def __init__(self, hidden_sizes: list[int]):
        super().__init__()
        layer_sizes = [1, *hidden_sizes, 1]
        layers: list[nn.Module] = []
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


def make_sine_data(
        train_points: int,
        eval_points: int,
        x_min: float,
        x_max: float,
        label_noise: float,
        seed: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_x = x_min + (x_max - x_min) * torch.rand(train_points, 1, generator=generator)
    train_x = train_x.sort(dim=0).values
    train_y_clean = torch.sin(train_x)
    train_y_observed = train_y_clean + label_noise * torch.randn(
        train_y_clean.shape,
        generator=generator,
    )

    eval_x = torch.linspace(x_min, x_max, eval_points).unsqueeze(1)
    eval_y_clean = torch.sin(eval_x)

    return (train_x, train_y_observed, train_y_clean), (eval_x, eval_y_clean)


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if OPTIMIZER_NAME == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    if OPTIMIZER_NAME == "sgd":
        return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_one_epoch(
        models: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        regularization_strength: float,
        grad_clip: float | None,
) -> None:
    x, y_observed, _ = train_tensors
    for model in models.values():
        model.train()

    for name, model in models.items():
        optimizer = optimizers[name]
        optimizer.zero_grad(set_to_none=True)

        prediction = model(x)
        observed_loss = F.mse_loss(prediction, y_observed)
        regularization_loss = local_weight_variance_loss(model)
        loss = observed_loss
        if name == "local_variance":
            loss = loss + regularization_strength * regularization_loss

        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


@torch.no_grad()
def evaluate(
        models: dict[str, nn.Module],
        train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eval_tensors: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, RegressionStats]:
    train_x, train_y_observed, train_y_clean = train_tensors
    eval_x, eval_y_clean = eval_tensors

    stats = {}
    for name, model in models.items():
        model.eval()
        train_prediction = model(train_x)
        eval_prediction = model(eval_x)
        regularization_loss = local_weight_variance_loss(model)
        stats[name] = RegressionStats(
            observed_train_mse=float(F.mse_loss(train_prediction, train_y_observed).cpu()),
            clean_train_mse=float(F.mse_loss(train_prediction, train_y_clean).cpu()),
            clean_eval_mse=float(F.mse_loss(eval_prediction, eval_y_clean).cpu()),
            regularization_loss=float(regularization_loss.cpu()),
        )
    return stats


def make_history(model_names: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "observed_train_mse": [],
            "clean_train_mse": [],
            "clean_eval_mse": [],
            "regularization_loss": [],
        }
        for name in model_names
    }


def append_history(
        history: dict[str, dict[str, list[float]]],
        stats: dict[str, RegressionStats],
):
    for name in history:
        history[name]["observed_train_mse"].append(stats[name].observed_train_mse)
        history[name]["clean_train_mse"].append(stats[name].clean_train_mse)
        history[name]["clean_eval_mse"].append(stats[name].clean_eval_mse)
        history[name]["regularization_loss"].append(stats[name].regularization_loss)


def brief_epoch_summary(
        epoch: int,
        total_epochs: int,
        stats: dict[str, RegressionStats],
) -> str:
    parts = [
        (
            f"{name}: noisy {values.observed_train_mse:.6f}, "
            f"eval {values.clean_eval_mse:.6f}, var {values.regularization_loss:.6f}"
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
def plot_regression_results(
        models: dict[str, nn.Module],
        eval_tensors: tuple[torch.Tensor, torch.Tensor],
        train_plot_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eval_plot_tensors: tuple[torch.Tensor, torch.Tensor],
        history: dict[str, dict[str, list[float]]],
        title: str,
):
    import matplotlib.pyplot as plt

    train_x, train_y_observed, train_y_clean = train_plot_tensors
    eval_x, eval_y_clean = eval_plot_tensors

    fig, axes_grid = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes_grid.flatten()

    axes[0].plot(
        eval_x.squeeze(1),
        eval_y_clean.squeeze(1),
        color="black",
        linewidth=2.0,
        label="clean sine",
    )
    axes[0].scatter(
        train_x.squeeze(1),
        train_y_observed.squeeze(1),
        color="tab:red",
        s=30,
        alpha=0.75,
        label="noisy train points",
    )
    axes[0].scatter(
        train_x.squeeze(1),
        train_y_clean.squeeze(1),
        color="black",
        s=14,
        alpha=0.5,
        label="train clean target",
    )
    for name, model in models.items():
        model.eval()
        prediction = model(eval_tensors[0]).detach().cpu()
        axes[0].plot(eval_x.squeeze(1), prediction.squeeze(1), linewidth=1.8, label=name)
    axes[0].set_title("Function fit")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    epochs = range(1, len(next(iter(history.values()))["observed_train_mse"]) + 1)
    for name, values in history.items():
        axes[1].plot(epochs, values["observed_train_mse"], label=f"{name} noisy train")
        axes[1].plot(epochs, values["clean_eval_mse"], linestyle="--", label=f"{name} clean eval")
    axes[1].set_title("Noisy train and clean eval MSE")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    for name, values in history.items():
        axes[2].plot(epochs, values["clean_train_mse"], label=name)
    axes[2].set_title("Clean train-point MSE")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    for name, values in history.items():
        axes[3].plot(epochs, values["regularization_loss"], label=name)
    axes[3].set_title("Mean local variance")
    axes[3].set_xlabel("epoch")
    axes[3].grid(alpha=0.25)
    axes[3].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()


def main():
    set_seed()
    train_plot_tensors, eval_plot_tensors = make_sine_data(
        train_points=TRAIN_POINTS,
        eval_points=EVAL_POINTS,
        x_min=X_MIN,
        x_max=X_MAX,
        label_noise=LABEL_NOISE,
        seed=SEED,
    )
    train_tensors = tuple(tensor.to(DEVICE) for tensor in train_plot_tensors)
    eval_tensors = tuple(tensor.to(DEVICE) for tensor in eval_plot_tensors)

    baseline = SineNet(HIDDEN_SIZES).to(DEVICE)
    regularized = copy.deepcopy(baseline).to(DEVICE)
    models = {
        "baseline": baseline,
        "local_variance": regularized,
    }
    optimizers = {name: make_optimizer(model) for name, model in models.items()}

    parameter_count = count_parameters(baseline)
    history = make_history(list(models))
    plot_title = (
        f"Sine fit | hidden={HIDDEN_SIZES} | train points={TRAIN_POINTS} | "
        f"noise={LABEL_NOISE:g} | params/train={parameter_count / TRAIN_POINTS:.1f} | "
        f"local variance lambda={REGULARIZATION_STRENGTH:g}"
    )

    progress = tqdm(range(1, NUM_EPOCHS + 1), desc="TRAIN", leave=False)
    for epoch in progress:
        train_one_epoch(
            models=models,
            optimizers=optimizers,
            train_tensors=train_tensors,
            regularization_strength=REGULARIZATION_STRENGTH,
            grad_clip=GRAD_CLIP,
        )
        stats = evaluate(models, train_tensors, eval_tensors)
        append_history(history, stats)
        should_summarize = (
                SUMMARY_EVERY > 0
                and (epoch == 1 or epoch % SUMMARY_EVERY == 0 or epoch == NUM_EPOCHS)
        )
        if should_summarize:
            progress.set_postfix(
                baseline_eval=f"{stats['baseline'].clean_eval_mse:.5f}",
                local_eval=f"{stats['local_variance'].clean_eval_mse:.5f}",
                local_var=f"{stats['local_variance'].regularization_loss:.5f}",
            )

        if should_summarize:
            progress.write(brief_epoch_summary(epoch, NUM_EPOCHS, stats))

    plot_regression_results(
        models=models,
        eval_tensors=eval_tensors,
        train_plot_tensors=train_plot_tensors,
        eval_plot_tensors=eval_plot_tensors,
        history=history,
        title=plot_title,
    )
    show_plots()


if __name__ == "__main__":
    main()
