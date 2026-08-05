"""Train an MNIST classifier with binary parameter-jitter updates.

Each update perturbs every model parameter with an independent binary sign,
measures the resulting loss, and correlates the loss surprise with those signs.
A small image-conditioned model predicts the expected perturbed loss and acts as
a variance-reducing baseline.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# Experiment configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET_NAME = "mnist"  # "mnist" or "fashion-mnist"
DOWNLOAD_DATA = True
TRAIN_LIMIT = 10_000
EVAL_LIMIT = 10_000

HIDDEN_SIZES = [128]
NUM_CLASSES = 10

NUM_GENERATIONS = 1_000
SAMPLES_PER_UPDATE = 1
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 1_024

EXPLORATION_SCALE = 0.01
EXPLORATION_DECAY = 1.0
OPTIMIZER_NAME = "sgd"  # "sgd" or "adamw"
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
ADAM_BETAS = (0.99, 0.9999)
WEIGHT_DECAY = 1e-5
GRAD_CLIP = None

BASELINE_LEARNING_RATE = 1e-3
BASELINE_WEIGHT_DECAY = 0.0
BASELINE_GRAD_CLIP = None

EVAL_EVERY = 250
PRINT_EVERY = 100
NUM_EXAMPLES_TO_PLOT = 16
NUM_WORKERS = 0
SEED = 0
DETERMINISTIC = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ClassificationStats:
    loss: float
    accuracy: float


@dataclass
class EvolutionStepStats:
    exploration_scale: float
    baseline_loss: float
    gradient_norm: float
    max_abs_gradient: float
    update_norm: float
    observed_loss: float
    observed_accuracy: float
    loss_surprise: float
    abs_loss_surprise: float


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = 28 * 28

        for hidden_size in HIDDEN_SIZES:
            layers.extend([nn.Linear(in_features, hidden_size), nn.SiLU()])
            in_features = hidden_size

        layers.append(nn.Linear(in_features, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class LossBaseline(nn.Module):
    """Predict the expected perturbed loss for an individual image."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(images)).squeeze(-1)


def validate_config():
    if DATASET_NAME not in {"mnist", "fashion-mnist"}:
        raise ValueError("DATASET_NAME must be 'mnist' or 'fashion-mnist'.")
    if any(hidden_size <= 0 for hidden_size in HIDDEN_SIZES):
        raise ValueError("Every hidden size must be positive.")
    if NUM_GENERATIONS < 1 or SAMPLES_PER_UPDATE < 1:
        raise ValueError("Generation and sample counts must be positive.")
    if BATCH_SIZE < 1 or EVAL_BATCH_SIZE < 1:
        raise ValueError("Batch sizes must be positive.")
    if EXPLORATION_SCALE <= 0.0 or EXPLORATION_DECAY <= 0.0:
        raise ValueError("Exploration scale and decay must be positive.")
    if LEARNING_RATE <= 0.0 or BASELINE_LEARNING_RATE <= 0.0:
        raise ValueError("Learning rates must be positive.")
    if OPTIMIZER_NAME not in {"sgd", "adamw"}:
        raise ValueError("OPTIMIZER_NAME must be 'sgd' or 'adamw'.")


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def make_dataset(train: bool) -> Dataset:
    dataset_class = {
        "mnist": datasets.MNIST,
        "fashion-mnist": datasets.FashionMNIST,
    }[DATASET_NAME]
    return dataset_class(
        root=DATA_DIR,
        train=train,
        download=DOWNLOAD_DATA,
        transform=transforms.ToTensor(),
    )


def limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return Subset(dataset, range(min(limit, len(dataset))))


def make_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = limit_dataset(make_dataset(train=True), TRAIN_LIMIT)
    eval_dataset = limit_dataset(make_dataset(train=False), EVAL_LIMIT)

    generator = torch.Generator().manual_seed(SEED)
    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, train_eval_loader, eval_loader


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if OPTIMIZER_NAME == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )


def softplus_inverse(value: float) -> float:
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


@torch.no_grad()
def initialize_baseline(model: LossBaseline, initial_loss: float):
    linear = model.net[-1]
    linear.weight.zero_()
    linear.bias.fill_(softplus_inverse(initial_loss))


@torch.no_grad()
def evaluate_batch(
        model: nn.Module,
        images: torch.Tensor,
        targets: torch.Tensor,
) -> ClassificationStats:
    model.eval()
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    predictions = logits.argmax(dim=-1)
    return ClassificationStats(
        loss=float(loss.cpu()),
        accuracy=float(predictions.eq(targets).float().mean().cpu()),
    )


@torch.no_grad()
def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
) -> ClassificationStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in dataloader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        logits = model(images)

        total_loss += float(F.cross_entropy(logits, targets, reduction="sum").cpu())
        total_correct += int(logits.argmax(dim=-1).eq(targets).sum().cpu())
        total_examples += targets.size(0)

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return ClassificationStats(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


def next_train_batch(
        dataloader: DataLoader,
        iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, Iterator]:
    try:
        images, targets = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        images, targets = next(iterator)

    return (
        images.to(DEVICE, non_blocking=True),
        targets.to(DEVICE, non_blocking=True),
        iterator,
    )


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and torch.is_floating_point(parameter)
    ]


def make_binary_signs(parameters: list[nn.Parameter]) -> list[torch.Tensor]:
    signs = []
    for parameter in parameters:
        sign = torch.empty_like(parameter).bernoulli_(0.5)
        signs.append(sign.mul_(2.0).sub_(1.0))
    return signs


@torch.no_grad()
def add_scaled_signs(
        parameters: list[nn.Parameter],
        signs: list[torch.Tensor],
        scale: float,
):
    for parameter, sign in zip(parameters, signs):
        parameter.add_(sign, alpha=scale)


def tensor_list_norm(tensors: list[torch.Tensor]) -> float:
    squared_norm = sum(float(tensor.detach().square().sum().cpu()) for tensor in tensors)
    return math.sqrt(squared_norm)


def tensor_list_max_abs(tensors: list[torch.Tensor]) -> float:
    return max(
        (float(tensor.detach().abs().max().cpu()) for tensor in tensors),
        default=0.0,
    )


def exploration_scale_for_generation(generation: int) -> float:
    return EXPLORATION_SCALE * EXPLORATION_DECAY ** (generation - 1)


def binary_jitter_step(
        model: nn.Module,
        baseline_model: LossBaseline,
        parameters: list[nn.Parameter],
        images: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        baseline_optimizer: torch.optim.Optimizer,
        generation: int,
) -> EvolutionStepStats:
    exploration_scale = exploration_scale_for_generation(generation)
    gradient_estimate = [torch.zeros_like(parameter) for parameter in parameters]
    baseline_predictions = []
    observed_losses = []
    observed_accuracies = []
    loss_surprises = []
    num_observations = images.size(0) * SAMPLES_PER_UPDATE

    model.eval()
    baseline_model.train()
    baseline_optimizer.zero_grad(set_to_none=True)

    for image, target in zip(images, targets):
        image = image.unsqueeze(0)
        target = target.unsqueeze(0)

        for _ in range(SAMPLES_PER_UPDATE):
            baseline_prediction = baseline_model(image).squeeze()
            baseline_value = float(baseline_prediction.detach().cpu())
            signs = make_binary_signs(parameters)

            add_scaled_signs(parameters, signs, exploration_scale)
            observed_stats = evaluate_batch(model, image, target)
            add_scaled_signs(parameters, signs, -exploration_scale)

            loss_surprise = observed_stats.loss - baseline_value
            for estimate, sign in zip(gradient_estimate, signs):
                estimate.add_(sign, alpha=loss_surprise / num_observations)

            baseline_target = torch.full_like(baseline_prediction, observed_stats.loss)
            baseline_loss = F.mse_loss(
                baseline_prediction,
                baseline_target,
                reduction="sum",
            )
            (baseline_loss / num_observations).backward()

            baseline_predictions.append(baseline_value)
            observed_losses.append(observed_stats.loss)
            observed_accuracies.append(observed_stats.accuracy)
            loss_surprises.append(loss_surprise)

    if BASELINE_GRAD_CLIP is not None:
        nn.utils.clip_grad_norm_(baseline_model.parameters(), BASELINE_GRAD_CLIP)
    baseline_optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    for parameter, estimate in zip(parameters, gradient_estimate):
        parameter.grad = estimate

    if GRAD_CLIP is not None:
        nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)

    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    gradient_norm = tensor_list_norm(gradients)
    max_abs_gradient = tensor_list_max_abs(gradients)
    parameters_before_step = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    updates = [
        parameter.detach() - before
        for parameter, before in zip(parameters, parameters_before_step)
    ]

    return EvolutionStepStats(
        exploration_scale=exploration_scale,
        baseline_loss=sum(baseline_predictions) / len(baseline_predictions),
        gradient_norm=gradient_norm,
        max_abs_gradient=max_abs_gradient,
        update_norm=tensor_list_norm(updates),
        observed_loss=sum(observed_losses) / len(observed_losses),
        observed_accuracy=sum(observed_accuracies) / len(observed_accuracies),
        loss_surprise=sum(loss_surprises) / len(loss_surprises),
        abs_loss_surprise=(
                sum(abs(surprise) for surprise in loss_surprises) / len(loss_surprises)
        ),
    )


def make_history() -> dict[str, list[float]]:
    return {
        "generation": [],
        "observed_loss": [],
        "batch_loss": [],
        "baseline_loss": [],
        "eval_loss": [],
        "observed_accuracy": [],
        "batch_accuracy": [],
        "eval_accuracy": [],
        "loss_surprise": [],
        "abs_loss_surprise": [],
        "gradient_norm": [],
        "update_norm": [],
    }


def record_history(
        history: dict[str, list[float]],
        generation: int,
        batch_stats: ClassificationStats,
        eval_stats: ClassificationStats | None,
        step_stats: EvolutionStepStats,
):
    history["generation"].append(generation)
    history["observed_loss"].append(step_stats.observed_loss)
    history["batch_loss"].append(batch_stats.loss)
    history["baseline_loss"].append(step_stats.baseline_loss)
    history["eval_loss"].append(eval_stats.loss if eval_stats else math.nan)
    history["observed_accuracy"].append(step_stats.observed_accuracy)
    history["batch_accuracy"].append(batch_stats.accuracy)
    history["eval_accuracy"].append(eval_stats.accuracy if eval_stats else math.nan)
    history["loss_surprise"].append(step_stats.loss_surprise)
    history["abs_loss_surprise"].append(step_stats.abs_loss_surprise)
    history["gradient_norm"].append(step_stats.gradient_norm)
    history["update_norm"].append(step_stats.update_norm)


def should_evaluate(generation: int) -> bool:
    return generation == NUM_GENERATIONS or (
            EVAL_EVERY > 0 and (generation == 1 or generation % EVAL_EVERY == 0)
    )


def should_print(generation: int) -> bool:
    return generation == NUM_GENERATIONS or (
            PRINT_EVERY > 0 and (generation == 1 or generation % PRINT_EVERY == 0)
    )


def plot_history(history: dict[str, list[float]]):
    generations = history["generation"]
    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(generations, history["observed_loss"], label="perturbed")
    axes[0, 0].plot(generations, history["batch_loss"], label="post-update batch")
    axes[0, 0].plot(generations, history["baseline_loss"], label="baseline")
    axes[0, 0].plot(generations, history["eval_loss"], label="evaluation")
    axes[0, 0].set_title("Cross-entropy loss")
    axes[0, 0].legend()

    axes[0, 1].plot(generations, history["observed_accuracy"], label="perturbed")
    axes[0, 1].plot(generations, history["batch_accuracy"], label="post-update batch")
    axes[0, 1].plot(generations, history["eval_accuracy"], label="evaluation")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(generations, history["loss_surprise"], label="signed")
    axes[1, 0].plot(generations, history["abs_loss_surprise"], label="absolute")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1, 0].set_title("Loss surprise")
    axes[1, 0].legend()

    axes[1, 1].plot(generations, history["gradient_norm"], label="gradient")
    axes[1, 1].plot(generations, history["update_norm"], label="update")
    axes[1, 1].set_title("Vector norms")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("generation")
        axis.grid(alpha=0.25)
    plt.tight_layout()


@torch.no_grad()
def plot_predictions(model: nn.Module, eval_loader: DataLoader):
    model.eval()
    images, targets = next(iter(eval_loader))
    images = images[:NUM_EXAMPLES_TO_PLOT].to(DEVICE)
    targets = targets[:NUM_EXAMPLES_TO_PLOT]
    probabilities = model(images).softmax(dim=-1).cpu()
    confidences, predictions = probabilities.max(dim=-1)

    num_columns = math.ceil(math.sqrt(len(images)))
    num_rows = math.ceil(len(images) / num_columns)
    _, axes = plt.subplots(num_rows, num_columns, figsize=(10, 2.5 * num_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= len(images):
            continue

        target = int(targets[index])
        prediction = int(predictions[index])
        title_color = "green" if prediction == target else "red"
        axis.imshow(images[index].cpu().squeeze(), cmap="gray")
        axis.set_title(
            f"true {target} | pred {prediction}\n{confidences[index]:.1%}",
            color=title_color,
        )

    plt.suptitle("Evaluation examples after training")
    plt.tight_layout()


def main():
    validate_config()
    set_seed()
    train_loader, train_eval_loader, eval_loader = make_dataloaders()

    model = MNISTClassifier().to(DEVICE)
    baseline_model = LossBaseline().to(DEVICE)
    parameters = trainable_parameters(model)
    optimizer = make_optimizer(model)
    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=BASELINE_LEARNING_RATE,
        weight_decay=BASELINE_WEIGHT_DECAY,
    )

    initial_train_stats = evaluate_model(model, train_eval_loader)
    latest_eval_stats = evaluate_model(model, eval_loader)
    initialize_baseline(baseline_model, initial_train_stats.loss)

    print(f"Device: {DEVICE}")
    print(f"Data: {DATASET_NAME} from {DATA_DIR}")
    print(
        f"Model: 784 -> {HIDDEN_SIZES} -> {NUM_CLASSES} | "
        f"optimizer: {OPTIMIZER_NAME}, lr={LEARNING_RATE:g}"
    )
    print(
        f"Initial: train loss {initial_train_stats.loss:.4f}, "
        f"accuracy {initial_train_stats.accuracy:.2%} | "
        f"eval loss {latest_eval_stats.loss:.4f}, "
        f"accuracy {latest_eval_stats.accuracy:.2%}"
    )

    history = make_history()
    train_iterator = iter(train_loader)
    progress = tqdm(range(1, NUM_GENERATIONS + 1), desc="Binary jitter")

    for generation in progress:
        images, targets, train_iterator = next_train_batch(
            train_loader,
            train_iterator,
        )
        step_stats = binary_jitter_step(
            model,
            baseline_model,
            parameters,
            images,
            targets,
            optimizer,
            baseline_optimizer,
            generation,
        )
        batch_stats = evaluate_batch(model, images, targets)
        eval_stats = None
        if should_evaluate(generation):
            latest_eval_stats = evaluate_model(model, eval_loader)
            eval_stats = latest_eval_stats

        record_history(history, generation, batch_stats, eval_stats, step_stats)
        progress.set_postfix(
            loss=f"{batch_stats.loss:.3f}",
            accuracy=f"{batch_stats.accuracy:.1%}",
            eval_accuracy=f"{latest_eval_stats.accuracy:.1%}",
        )

        if should_print(generation):
            tqdm.write(
                f"Generation {generation:4d}/{NUM_GENERATIONS}: "
                f"observed loss {step_stats.observed_loss:.4f}, "
                f"batch loss {batch_stats.loss:.4f}, "
                f"batch accuracy {batch_stats.accuracy:.2%}, "
                f"eval accuracy {latest_eval_stats.accuracy:.2%}, "
                f"gradient norm {step_stats.gradient_norm:.4f}"
            )

    print(
        f"Final evaluation: loss {latest_eval_stats.loss:.4f}, "
        f"accuracy {latest_eval_stats.accuracy:.2%}"
    )
    plot_history(history)
    plot_predictions(model, eval_loader)
    plt.show()


if __name__ == "__main__":
    main()
