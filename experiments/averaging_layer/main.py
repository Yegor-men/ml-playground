"""Compare averaging, plain, and projected-residual MLPs with backpropagation.

All three use the same widths, SiLU branches, batches, and Adam settings.
For a controlled comparison, even the output layer uses the same SiLU branch
as the original averaging model (rather than a conventional linear-only head).
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
HIDDEN_SIZES = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
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


class PlainLayer(nn.Module):
    def __init__(self, in_features, out_features, init_stdev=INIT_STDEV):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.normal_(self.linear.weight, mean=0.0, std=init_stdev)
        nn.init.normal_(self.linear.bias, mean=0.0, std=init_stdev)

    def forward(self, inputs):
        return F.silu(self.linear(inputs))


class ResidualLayer(PlainLayer):
    def __init__(self, in_features, out_features, init_stdev=INIT_STDEV):
        super().__init__(in_features, out_features, init_stdev)
        if in_features == out_features:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(in_features, out_features, bias=False)
            # Preserve expected per-feature variance across changing widths.
            # Near-zero skip weights would defeat the purpose of the skip path.
            nn.init.normal_(self.skip.weight, std=in_features ** -0.5)

    def forward(self, inputs):
        return self.skip(inputs) + super().forward(inputs)


def make_model(layer_type=AveragingLayer):
    sizes = [28 * 28, *HIDDEN_SIZES, NUM_CLASSES]
    layers = [nn.Flatten()]
    for in_features, out_features in zip(sizes, sizes[1:]):
        layers.append(layer_type(in_features, out_features))
    # The final layer's output is used directly as cross-entropy logits.
    return nn.Sequential(*layers).to(DEVICE)


def make_models():
    models = {
        "Averaging": make_model(AveragingLayer),
        "Plain": make_model(PlainLayer),
        "Residual": make_model(ResidualLayer),
    }
    # Match the actual initial tensors, not just their distributions.
    for name in ("Plain", "Residual"):
        for source, target in zip(models["Averaging"][1:], models[name][1:]):
            target.linear.load_state_dict(source.linear.state_dict())
    return models


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


def train_epoch(models, loader, optimizers, epoch, gradient_history=None):
    for model in models.values():
        model.train()
    totals = {name: [0.0, 0] for name in models}
    total_examples = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    for images, targets in progress:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            if name == "Averaging" and gradient_history is not None:
                gradient_history.record(epoch)
            optimizer.step()
            totals[name][0] += loss.item() * len(targets)
            totals[name][1] += logits.detach().argmax(dim=1).eq(targets).sum().item()
        total_examples += len(targets)
        progress.set_postfix(
            {name: f"{correct / total_examples:.1%}" for name, (_, correct) in totals.items()},
            refresh=False,
        )
    return {name: (loss / total_examples, correct / total_examples)
            for name, (loss, correct) in totals.items()}


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


def plot_history(histories):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for color, (name, history) in zip(("C0", "C1", "C2"), histories.items()):
        epochs = range(1, len(history["train_loss"]) + 1)
        for axis, metric in zip(axes, ("loss", "accuracy")):
            axis.plot(epochs, history[f"train_{metric}"], "--", color=color, label=f"{name} train")
            axis.plot(epochs, history[f"test_{metric}"], color=color, label=f"{name} test")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_ylabel("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.25)
    plt.tight_layout()


@torch.no_grad()
def plot_predictions(models, loader, class_names):
    # Read from the dataset so the grid size does not depend on evaluation batch size.
    examples = [loader.dataset[index] for index in range(
        min(NUM_EXAMPLES_TO_PLOT, len(loader.dataset))
    )]
    images = torch.stack([image for image, _ in examples])
    targets = [target for _, target in examples]
    predictions = {}
    for name, model in models.items():
        model.eval()
        predictions[name] = model(images.to(DEVICE)).softmax(dim=1).cpu().max(dim=1)
    num_columns = math.ceil(math.sqrt(len(examples)))
    num_rows = math.ceil(len(examples) / num_columns)
    _, axes = plt.subplots(num_rows, num_columns, figsize=(14, 4 * num_rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(examples):
            continue
        axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
        lines = [f"True: {class_names[targets[index]]}"]
        for name, (confidences, labels) in predictions.items():
            prediction = labels[index].item()
            lines.append(f"{name}: {class_names[prediction]} ({confidences[index]:.0%})")
        axis.set_title("\n".join(lines), fontsize=8)
    plt.suptitle(f"Same test images — {DATASET_NAME}")
    plt.tight_layout(rect=(0, 0, 1, 0.95))


def main():
    torch.manual_seed(SEED)
    train_loader, test_loader, class_names = make_dataloaders()
    models = make_models()
    model = models["Averaging"]
    optimizers = {name: torch.optim.Adam(candidate.parameters(), lr=LEARNING_RATE)
                  for name, candidate in models.items()}
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
    print(f"Device: {DEVICE} | Dataset: {DATASET_NAME} from {DATA_DIR}")
    print(f"Widths: {[784, *HIDDEN_SIZES, NUM_CLASSES]} | matched batches and nonlinear weights")
    print("Residual skips: identity at equal widths, learned variance-scaled projection otherwise.")
    print("All models retain the original SiLU output branch; parameter counts differ.")
    for name, candidate in models.items():
        initial_loss, initial_accuracy = evaluate(candidate, test_loader)
        print(f"{name}: {sum(p.numel() for p in candidate.parameters()):,} parameters | "
              f"initial test loss {initial_loss:.4f}, accuracy {initial_accuracy:.2%}")
        print_activation_stats(candidate, diagnostic_images, f"{name} initial")
    histories = {name: {key: [] for key in (
        "train_loss", "train_accuracy", "test_loss", "test_accuracy"
    )} for name in models}

    for epoch in range(1, NUM_EPOCHS + 1):
        training = train_epoch(models, train_loader, optimizers, epoch, gradient_history)
        for name, candidate in models.items():
            train_loss, train_accuracy = training[name]
            test_loss, test_accuracy = evaluate(candidate, test_loader)
            for key, value in zip(histories[name], (train_loss, train_accuracy, test_loss, test_accuracy)):
                histories[name][key].append(value)
            print(f"Epoch {epoch}/{NUM_EPOCHS} {name}: train loss {train_loss:.4f}, "
                  f"accuracy {train_accuracy:.2%} | test loss {test_loss:.4f}, "
                  f"accuracy {test_accuracy:.2%}")
        gradient_history.end_epoch(model)

    for name, candidate in models.items():
        print_activation_stats(candidate, diagnostic_images, f"{name} final")
    plot_history(histories)
    plot_predictions(models, test_loader, class_names)
    print("Analyzing learned averaging weights and inference-only ablations...")
    plot_averaging_weights(model, initial_average_weights)
    plot_branch_ablations(model, test_loader)
    gradient_history.plot()
    print("Optimizing class inputs and probing local gradient smoothness...")
    for name, candidate in models.items():
        plot_ideal_inputs(candidate, class_names)
        plt.gcf().suptitle(f"{name}: optimized class inputs (same initialization and penalties)")
    plot_local_smoothness(model, probe_images, probe_targets)
    plt.show()


if __name__ == "__main__":
    main()
