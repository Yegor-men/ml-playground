"""Train a sigmoid MLP using reward-modulated, randomly masked activity updates.

A neuron's signed preactivation is its activity memory, shared by all incoming
parameters. Average masked per-example updates; never compute gradients or
perturb the model to evaluate a trial configuration.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# Experiment configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET_NAME = "mnist"  # "mnist" or "fashion-mnist"
TRAIN_LIMIT = 10_000  # None uses the entire dataset.
EVAL_LIMIT = 10_000
HIDDEN_SIZES = [128]
NUM_CLASSES = 10
INIT_STDEV = 0.01
NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 1e-2
UPDATE_PROBABILITY = 0.5  # Independent for each example, weight, and bias.
ACTIVITY_MODE = "value"  # "value": signed preactivation; "sign": polarity only.
MASK_CHUNK_SIZE = 16  # Bound temporary mask memory without changing the rule.
INPUT_EPSILON = 1e-3  # Map [0, 1] pixels into (0, 1).
NUM_EXAMPLES_TO_PLOT = 16
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActivityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = [28 * 28, *HIDDEN_SIZES, NUM_CLASSES]
        self.layers = nn.ModuleList(
            nn.Linear(inputs, outputs) for inputs, outputs in zip(sizes, sizes[1:])
        )
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=INIT_STDEV)
            nn.init.zeros_(layer.bias)
        self.requires_grad_(False)

    def forward(self, images, record_activity=False):
        values = INPUT_EPSILON + (1 - 2 * INPUT_EPSILON) * images.flatten(1)
        activities = []
        for index, layer in enumerate(self.layers):
            preactivation = layer(values)
            if record_activity:
                activities.append(preactivation)
            values = preactivation.sigmoid() if index < len(self.layers) - 1 else preactivation
        return (values, activities) if record_activity else values


def make_dataloaders():
    dataset_class = {"mnist": datasets.MNIST, "fashion-mnist": datasets.FashionMNIST}[
        DATASET_NAME
    ]
    loaders = []
    for train, limit in [(True, TRAIN_LIMIT), (False, EVAL_LIMIT)]:
        dataset = dataset_class(
            root=DATA_DIR, train=train, download=True, transform=transforms.ToTensor()
        )
        if limit is not None:
            dataset = Subset(dataset, range(min(limit, len(dataset))))
        loaders.append(DataLoader(
            dataset, batch_size=TRAIN_BATCH_SIZE if train else EVAL_BATCH_SIZE,
            shuffle=train,
        ))
    return loaders


def previous_loss_advantages(losses, baseline_loss):
    """Use the previous image's loss, including across batch boundaries."""
    expected = torch.cat((losses.new_tensor([baseline_loss]), losses[:-1]))
    return expected - losses, losses[-1].item()


@torch.no_grad()
def apply_activity_updates(model, activities, advantages):
    batch_size = len(advantages)
    for layer, activity in zip(model.layers, activities):
        trace = activity.sign() if ACTIVITY_MODE == "sign" else activity
        signals = advantages[:, None] * trace
        weight_delta = torch.zeros_like(layer.weight)
        bias_delta = torch.zeros_like(layer.bias)
        for chunk in signals.split(MASK_CHUNK_SIZE):
            # Each example proposes its own imperfect step, before averaging.
            weight_mask = torch.rand(
                len(chunk), *layer.weight.shape, device=layer.weight.device
            ) < UPDATE_PROBABILITY
            bias_mask = torch.rand(
                len(chunk), *layer.bias.shape, device=layer.bias.device
            ) < UPDATE_PROBABILITY
            weight_delta.add_((chunk[:, :, None] * weight_mask).sum(dim=0))
            bias_delta.add_((chunk * bias_mask).sum(dim=0))
        # No division by p: lower p reduces coverage and expected movement.
        layer.weight.add_(weight_delta, alpha=LEARNING_RATE / batch_size)
        layer.bias.add_(bias_delta, alpha=LEARNING_RATE / batch_size)


@torch.no_grad()
def train_epoch(model, loader, baseline_loss):
    model.train()
    total_loss = total_correct = total_abs_advantage = total_examples = 0
    progress = tqdm(loader, desc="Activity updates", leave=False)
    for images, targets in progress:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        logits, activities = model(images, record_activity=True)
        losses = F.cross_entropy(logits, targets, reduction="none")
        if not torch.isfinite(losses).all():
            raise RuntimeError("Nonfinite loss: try a smaller learning rate or sign activity.")
        advantages, baseline_loss = previous_loss_advantages(losses, baseline_loss)
        apply_activity_updates(model, activities, advantages)
        total_examples += len(targets)
        total_loss += losses.sum().item()
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_abs_advantage += advantages.abs().sum().item()
        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            accuracy=f"{total_correct / total_examples:.2%}",
            advantage=f"{total_abs_advantage / total_examples:.3g}",
        )
    tqdm.write(
        f"Next expected loss: {baseline_loss:.4f} | "
        f"Mean absolute advantage: {total_abs_advantage / total_examples:.4g}"
    )
    return baseline_loss, total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = total_correct = total_examples = 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        logits = model(images)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_examples += len(targets)
    return total_loss / total_examples, total_correct / total_examples


def plot_history(history):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    for axis, metric, label in zip(axes, ("loss", "accuracy"), ("Cross-entropy", "Accuracy")):
        for split in ("train", "eval"):
            axis.plot(epochs, history[f"{split}_{metric}"], marker=".", label=split)
        axis.set(xlabel="Epoch", ylabel=label)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Training history — epoch {len(history['train_loss'])}")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


@torch.no_grad()
def plot_predictions(model, loader):
    model.eval()
    images, targets = next(iter(loader))
    images, targets = images[:NUM_EXAMPLES_TO_PLOT], targets[:NUM_EXAMPLES_TO_PLOT]
    confidences, predictions = model(images.to(DEVICE)).softmax(dim=1).cpu().max(dim=1)
    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(10, 2.5 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
            axis.set_title(
                f"True {targets[index].item()} | predicted {predictions[index].item()}\n"
                f"Confidence {confidences[index]:.1%}",
                color="green" if targets[index] == predictions[index] else "red",
            )
    figure.suptitle("Test examples after training")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def main():
    if not 0 <= UPDATE_PROBABILITY <= 1:
        raise ValueError("UPDATE_PROBABILITY must be between zero and one.")
    if ACTIVITY_MODE not in {"value", "sign"}:
        raise ValueError("ACTIVITY_MODE must be 'value' or 'sign'.")
    if not 0 < INPUT_EPSILON < 0.5:
        raise ValueError("INPUT_EPSILON must be between zero and 0.5.")
    if min(TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, MASK_CHUNK_SIZE, NUM_EPOCHS,
           NUM_EXAMPLES_TO_PLOT) < 1 or LEARNING_RATE <= 0:
        raise ValueError("Batch sizes, epochs, example count, and learning rate must be positive.")
    torch.manual_seed(SEED)
    train_loader, eval_loader = make_dataloaders()
    model = ActivityMLP().to(DEVICE)
    baseline_loss = math.log(NUM_CLASSES)
    history = {key: [] for key in (
        "train_loss", "train_accuracy", "eval_loss", "eval_accuracy"
    )}
    print(f"Device: {DEVICE} | Dataset: {DATASET_NAME} from {DATA_DIR}")
    print(f"Sigmoid MLP: {[784, *HIDDEN_SIZES, NUM_CLASSES]} | Activity: {ACTIVITY_MODE}")
    print(f"Step scale: {LEARNING_RATE:g} | Update probability: {UPDATE_PROBABILITY:g}")
    print(f"Train: {len(train_loader.dataset)} | Test: {len(eval_loader.dataset)}")
    print(f"Initial expected loss: {baseline_loss:.4f} | Baseline: previous image loss")
    for epoch in range(1, NUM_EPOCHS + 1):
        baseline_loss, train_loss, train_accuracy = train_epoch(model, train_loader, baseline_loss)
        eval_loss, eval_accuracy = evaluate(model, eval_loader)
        for key, value in zip(history, (train_loss, train_accuracy, eval_loss, eval_accuracy)):
            history[key].append(value)
        tqdm.write(
            f"Epoch {epoch}/{NUM_EPOCHS} | Train loss {train_loss:.4f}, "
            f"accuracy {train_accuracy:.2%} | Test loss {eval_loss:.4f}, "
            f"accuracy {eval_accuracy:.2%}"
        )
        plot_history(history)
    plot_predictions(model, eval_loader)


if __name__ == "__main__":
    main()
