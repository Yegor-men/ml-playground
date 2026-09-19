"""Test learned input averaging plus a SiLU delta using ordinary backpropagation.

Each layer computes x @ softmax(average_logits, dim=0) + silu(linear(x)).
Columns of the averaging matrix sum to one: each output averages its inputs.
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

from diagnostics import (
    PROBE_BATCH_SIZE,
    GradientHistory,
    plot_averaging_weights,
    plot_branch_ablations,
    plot_ideal_inputs,
    plot_local_smoothness,
)

# Experiment configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET_NAME = "fashion-mnist"  # "mnist" or "fashion-mnist"
TRAIN_LIMIT = None  # None uses the entire training split.
EVAL_LIMIT = None
HIDDEN_SIZES = [512, 128, 64, 512, 1024, 512, 1024, 512, 1024, 512]
NUM_CLASSES = 10
INIT_STDEV = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 1_024
LEARNING_RATE = 1e-3
NUM_EXAMPLES_TO_PLOT = 16
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AveragingLayer(nn.Module):
    """Learn a convex combination of inputs and an additive nonlinear delta."""

    def __init__(self, in_features, out_features, init_stdev=INIT_STDEV):
        super().__init__()
        self.average_logits = nn.Parameter(torch.empty(in_features, out_features))
        self.linear = nn.Linear(in_features, out_features)
        nn.init.normal_(self.average_logits, mean=0.0, std=init_stdev)
        nn.init.normal_(self.linear.weight, mean=0.0, std=init_stdev)
        nn.init.normal_(self.linear.bias, mean=0.0, std=init_stdev)

    def forward(self, inputs):
        # average_logits is [input, output], so normalize over the input axis.
        average = inputs @ self.average_logits.softmax(dim=0)
        return average + F.silu(self.linear(inputs))


def make_model():
    sizes = [28 * 28, *HIDDEN_SIZES, NUM_CLASSES]
    layers = [nn.Flatten()]
    for in_features, out_features in zip(sizes, sizes[1:]):
        layers.append(AveragingLayer(in_features, out_features))
    # The final layer's output is used directly as cross-entropy logits.
    return nn.Sequential(*layers).to(DEVICE)


def make_dataloaders():
    dataset_class = {"mnist": datasets.MNIST, "fashion-mnist": datasets.FashionMNIST}[
        DATASET_NAME
    ]
    loaders = []
    class_names = None
    for train, limit in [(True, TRAIN_LIMIT), (False, EVAL_LIMIT)]:
        dataset = dataset_class(
            root=DATA_DIR, train=train, download=True, transform=transforms.ToTensor()
        )
        class_names = dataset.classes
        if limit is not None:
            dataset = Subset(dataset, range(min(limit, len(dataset))))
        loaders.append(DataLoader(
            dataset,
            batch_size=BATCH_SIZE if train else EVAL_BATCH_SIZE,
            shuffle=train,
            generator=torch.Generator().manual_seed(SEED),
            pin_memory=DEVICE.type == "cuda",
        ))
    return *loaders, class_names


def train_epoch(model, loader, optimizer, epoch, gradient_history=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    for images, targets in progress:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        if gradient_history is not None:
            gradient_history.record(epoch)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        total_correct += logits.detach().argmax(dim=1).eq(targets).sum().item()
        total_examples += len(targets)
        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            accuracy=f"{total_correct / total_examples:.2%}",
            refresh=False,
        )
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        logits = model(images)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_examples += len(targets)
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def print_activation_stats(model, images, label):
    """Check whether output levels survive while differences between features shrink."""
    model.eval()
    activations = images.to(DEVICE).flatten(start_dim=1)
    print(f"{label} activations (same {len(images)} test images):")
    for index, layer in enumerate(model[1:], start=1):
        activations = layer(activations)
        feature_std = activations.std(dim=1, unbiased=False).mean().item()
        example_std = activations.std(dim=0, unbiased=False).mean().item()
        print(
            f"  Layer {index}: mean {activations.mean().item():.4f} | "
            f"std across features {feature_std:.2e} | "
            f"std across examples {example_std:.2e}"
        )


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"], label="Test")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["test_accuracy"], label="Test")
    axes[1].set_ylabel("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.25)
    plt.tight_layout()


@torch.no_grad()
def plot_predictions(model, loader, class_names):
    model.eval()
    # Read from the dataset so the grid size does not depend on evaluation batch size.
    examples = [loader.dataset[index] for index in range(
        min(NUM_EXAMPLES_TO_PLOT, len(loader.dataset))
    )]
    images = torch.stack([image for image, _ in examples])
    targets = [target for _, target in examples]
    probabilities = model(images.to(DEVICE)).softmax(dim=1).cpu()
    confidences, predictions = probabilities.max(dim=1)
    num_columns = math.ceil(math.sqrt(len(examples)))
    num_rows = math.ceil(len(examples) / num_columns)
    _, axes = plt.subplots(num_rows, num_columns, figsize=(12, 3 * num_rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(examples):
            continue
        target, prediction = targets[index], predictions[index].item()
        axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
        axis.set_title(
            f"True: {class_names[target]}\n"
            f"Pred: {class_names[prediction]} ({confidences[index]:.1%})",
            color="green" if target == prediction else "red",
            fontsize=9,
        )
    plt.suptitle(f"Averaging MLP — {DATASET_NAME} test predictions")
    plt.tight_layout()


def main():
    torch.manual_seed(SEED)
    train_loader, test_loader, class_names = make_dataloaders()
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    initial_average_weights = [
        layer.average_logits.detach().softmax(dim=0).cpu().clone() for layer in model[1:]
    ]
    probe_examples = [train_loader.dataset[index] for index in range(
        min(PROBE_BATCH_SIZE, len(train_loader.dataset))
    )]
    probe_images = torch.stack([image for image, _ in probe_examples]).to(DEVICE)
    probe_targets = torch.tensor([target for _, target in probe_examples], device=DEVICE)
    gradient_history = GradientHistory(model, probe_images, probe_targets)
    diagnostic_images, _ = next(iter(test_loader))
    initial_loss, initial_accuracy = evaluate(model, test_loader)
    print(f"Device: {DEVICE} | Dataset: {DATASET_NAME} from {DATA_DIR}")
    print(
        f"Averaging MLP: {[784, *HIDDEN_SIZES, NUM_CLASSES]} | "
        f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    print(f"Initial test: loss {initial_loss:.4f}, accuracy {initial_accuracy:.2%}")
    print_activation_stats(model, diagnostic_images, "Initial")
    history = {key: [] for key in (
        "train_loss", "train_accuracy", "test_loss", "test_accuracy"
    )}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, epoch, gradient_history)
        test_loss, test_accuracy = evaluate(model, test_loader)
        for key, value in zip(history, (train_loss, train_accuracy, test_loss, test_accuracy)):
            history[key].append(value)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}: train loss {train_loss:.4f}, "
            f"accuracy {train_accuracy:.2%} | test loss {test_loss:.4f}, "
            f"accuracy {test_accuracy:.2%}"
        )
        gradient_history.end_epoch(model)

    print_activation_stats(model, diagnostic_images, "Final")
    plot_history(history)
    plot_predictions(model, test_loader, class_names)
    print("Analyzing learned averaging weights and inference-only ablations...")
    plot_averaging_weights(model, initial_average_weights)
    plot_branch_ablations(model, test_loader)
    gradient_history.plot()
    print("Optimizing class inputs and probing local gradient smoothness...")
    plot_ideal_inputs(model, class_names)
    plot_local_smoothness(model, probe_images, probe_targets)
    plt.show()


if __name__ == "__main__":
    main()
