"""Static visualizations and diagnostics for the averaging-layer experiment."""

import math

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

# Diagnostic settings (independent of training hyperparameters).
NEURONS_TO_PLOT = 8
GRADIENT_SAMPLES_PER_TENSOR = 16
PROBE_BATCH_SIZE = 128
PERTURBATION_STDEVS = [1e-5, 1e-4, 1e-3, 1e-2]
PERTURBATION_DIRECTIONS = 8
IDEAL_INPUT_STEPS = 300
IDEAL_INPUT_LEARNING_RATE = 0.05
IDEAL_INPUT_L2 = 0.1
IDEAL_INPUT_SMOOTHNESS = 0.2
DIAGNOSTIC_SEED = 123


def cosine(first, second):
    denominator = first.norm() * second.norm()
    if denominator.item() == 0:
        return math.nan  # A zero gradient has no defined direction.
    return (first.flatten().dot(second.flatten()) / denominator).item()


class GradientHistory:
    """Sample fixed parameter coordinates; use full gradients for the fixed probe."""

    def __init__(self, model, probe_images, probe_targets):
        self.probe_images = probe_images
        self.probe_targets = probe_targets
        self.parameters = list(model.named_parameters())
        generator = torch.Generator().manual_seed(DIAGNOSTIC_SEED)
        self.indices = [
            torch.randperm(parameter.numel(), generator=generator)[
                :GRADIENT_SAMPLES_PER_TENSOR
            ].to(parameter.device)
            for _, parameter in self.parameters
        ]
        self.is_average = torch.cat([
            torch.full((len(indices),), "average_logits" in name, dtype=torch.bool)
            for (name, _), indices in zip(self.parameters, self.indices)
        ])
        self.rows = []
        self.epochs = []
        self.probe_cosines = []
        self.previous_probe = self.probe_gradients(model)

    def record(self, epoch):
        # Read existing .grad after backward; do not alter the optimizer's input.
        self.rows.append(torch.cat([
            parameter.grad.detach().flatten()[indices]
            for (_, parameter), indices in zip(self.parameters, self.indices)
        ]).cpu())
        self.epochs.append(epoch)

    def probe_gradients(self, model):
        loss = F.cross_entropy(model(self.probe_images), self.probe_targets)
        gradients = torch.autograd.grad(loss, [parameter for _, parameter in self.parameters])
        return [torch.cat([
            gradient.detach().flatten()
            for (name, _), gradient in zip(self.parameters, gradients)
            if ("average_logits" in name) == is_average
        ]) for is_average in (True, False)]

    def end_epoch(self, model):
        current = self.probe_gradients(model)
        similarities = [cosine(old, new) for old, new in zip(self.previous_probe, current)]
        self.probe_cosines.append(similarities)
        self.previous_probe = current
        print(
            "  Fixed-training-batch gradient cosine vs previous epoch: "
            f"averaging {similarities[0]:.3f}, delta {similarities[1]:.3f}"
        )

    def plot(self):
        gradients = torch.stack(self.rows)
        epochs = torch.tensor(self.epochs)
        unique_epochs = epochs.unique(sorted=True)
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for group, mask in enumerate((self.is_average, ~self.is_average)):
            label = "Averaging logits" if group == 0 else "Linear delta"
            adjacent, consistency, rms = [], [], []
            for epoch in unique_epochs:
                values = gradients[epochs == epoch][:, mask]
                similarities = [cosine(a, b) for a, b in zip(values[:-1], values[1:])]
                adjacent.append(torch.tensor(similarities).nanmean().item() if similarities else math.nan)
                parameter_rms = values.square().mean(dim=0).sqrt()
                active = parameter_rms > 0
                score = values.mean(dim=0).abs()[active] / parameter_rms[active]
                consistency.append(score.mean().item() if active.any() else math.nan)
                rms.append(values.square().mean().sqrt().item())
            axes[0, 0].plot(unique_epochs, adjacent, label=label)
            axes[0, 1].plot(unique_epochs, consistency, label=label)
            axes[1, 0].plot(unique_epochs, rms, label=label)
            axes[1, 1].plot(unique_epochs, [row[group] for row in self.probe_cosines], label=label)
        titles = [
            "Consecutive training batches: gradient cosine (sampled)",
            "Per-parameter |mean gradient| / RMS (sampled)",
            "Training-gradient RMS (sampled)",
            "Same training batch: gradient cosine across epochs (full)",
        ]
        for axis, title in zip(axes.flat, titles):
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            axis.legend()
        axes[0, 0].set_ylim(-1.05, 1.05)
        axes[0, 1].set_ylim(0, 1.05)
        axes[1, 0].set_yscale("symlog", linthresh=1e-12)
        axes[1, 1].set_ylim(-1.05, 1.05)

        # Normalize each trace only for display; sampling retains the same coordinates.
        scale = gradients.square().mean(dim=0).sqrt().clamp_min(1e-30)
        stride = max(1, math.ceil(len(gradients) / 1_000))
        figure, axis = plt.subplots(figsize=(12, 6), layout="constrained")
        shown = axis.imshow(
            (gradients[::stride] / scale).T, aspect="auto", cmap="RdBu_r",
            vmin=-3, vmax=3, extent=(1, len(gradients), gradients.shape[1], 0),
        )
        offset, centers, labels = 0, [], []
        for (name, _), indices in zip(self.parameters, self.indices):
            centers.append(offset + len(indices) / 2)
            labels.append(name)
            offset += len(indices)
            axis.axhline(offset, color="black", linewidth=0.4)
        axis.set_yticks(centers, labels, fontsize=8)
        axis.set_xlabel("Training batch")
        axis.set_title("Fixed sampled parameters: signed gradients throughout training")
        figure.colorbar(shown, ax=axis, label="Gradient / that parameter's training RMS (clipped to ±3)")


@torch.no_grad()
def plot_averaging_weights(model, initial_weights):
    layers = list(model)[1:]
    figure, axes = plt.subplots(len(layers), 2, figsize=(13, 3 * len(layers)), squeeze=False, layout="constrained")
    for index, (layer, initial) in enumerate(zip(layers, initial_weights)):
        weights = layer.average_logits.softmax(dim=0).cpu()
        relative = (weights.clamp_min(1e-30) * weights.shape[0]).log2()
        limit = max(relative.abs().max().item(), 1e-3)
        shown = axes[index, 0].imshow(relative.T, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[index, 0].set_title(f"Layer {index + 1}: learned averaging weights")
        axes[index, 0].set_xlabel("Input feature")
        axes[index, 0].set_ylabel("Output neuron")
        figure.colorbar(shown, ax=axes[index, 0], label="log₂(weight / uniform weight)")
        effective = (-(weights * weights.clamp_min(1e-30).log()).sum(dim=0)).exp()
        initial_effective = (-(initial * initial.clamp_min(1e-30).log()).sum(dim=0)).exp()
        axes[index, 1].plot(initial_effective, label="Initial")
        axes[index, 1].plot(effective, label="Trained")
        axes[index, 1].axhline(weights.shape[0], color="gray", linestyle="--", label="Uniform")
        axes[index, 1].set_ylim(0, weights.shape[0] * 1.05)
        axes[index, 1].set_title("Effective number of inputs (exp entropy)")
        axes[index, 1].set_xlabel("Output neuron")
        axes[index, 1].legend()
        change = (weights - initial).abs().sum(dim=0).mean().item()
        diversity = (weights - weights.mean(dim=1, keepdim=True)).abs().sum(dim=0).mean().item()
        print(
            f"  Averaging layer {index + 1}: effective inputs {effective.mean():.1f}/{weights.shape[0]} | "
            f"mean L1 change {change:.3f} | mean L1 distance from average neuron {diversity:.3f}"
        )

    first = layers[0]
    count = min(NEURONS_TO_PLOT, first.average_logits.shape[1])
    selected = torch.linspace(0, first.average_logits.shape[1] - 1, count).long()
    maps = (first.average_logits.softmax(dim=0).cpu()[:, selected].T.clamp_min(1e-30) * 784).log2()
    delta = first.linear.weight.detach().cpu()[selected]
    figure, axes = plt.subplots(math.ceil(count / 2), 4, figsize=(13, 3 * math.ceil(count / 2)), squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.axis("off")
    for index, neuron in enumerate(selected):
        average_axis, delta_axis = list(axes.flat)[2 * index:2 * index + 2]
        average_plot = average_axis.imshow(maps[index].reshape(28, 28), cmap="RdBu_r", vmin=-max(maps.abs().max().item(), 1e-3), vmax=max(maps.abs().max().item(), 1e-3))
        delta_plot = delta_axis.imshow(delta[index].reshape(28, 28), cmap="RdBu_r", vmin=-max(delta.abs().max().item(), 1e-8), vmax=max(delta.abs().max().item(), 1e-8))
        average_axis.set_title(f"Neuron {neuron}: averaging")
        delta_axis.set_title(f"Neuron {neuron}: linear delta")
    figure.colorbar(average_plot, ax=list(axes.flat)[::2], label="log₂(averaging weight / uniform)", shrink=0.6)
    figure.colorbar(delta_plot, ax=list(axes.flat)[1::2], label="Linear weight", shrink=0.6)
    figure.suptitle("First-layer pixel weights — evenly spaced neurons, not selected for appearance")


@torch.no_grad()
def plot_branch_ablations(model, loader):
    layers = list(model)[1:]
    conditions = [("Trained", None), *[(f"Uniform L{i + 1}", i) for i in range(len(layers))],
                  ("Uniform all", "uniform"), ("No averaging", "no_average"), ("No delta", "no_delta")]
    results = []
    device = next(model.parameters()).device
    for label, condition in conditions:
        total_loss = total_correct = total_examples = 0
        for images, targets in loader:
            values, targets = images.to(device).flatten(1), targets.to(device)
            for index, layer in enumerate(layers):
                if condition == "uniform" or condition == index:
                    average = values.mean(dim=1, keepdim=True).expand(-1, layer.linear.out_features)
                else:
                    average = values @ layer.average_logits.softmax(dim=0)
                delta = F.silu(layer.linear(values))
                values = (0 if condition == "no_average" else average) + (0 if condition == "no_delta" else delta)
            total_loss += F.cross_entropy(values, targets, reduction="sum").item()
            total_correct += values.argmax(dim=1).eq(targets).sum().item()
            total_examples += len(targets)
        results.append((total_loss / total_examples, total_correct / total_examples))
        print(f"  Ablation {label}: loss {results[-1][0]:.4f}, accuracy {results[-1][1]:.2%}")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for index, (axis, title) in enumerate(zip(axes, ("Test loss", "Test accuracy"))):
        values = [row[index] for row in results]
        bars = axis.barh([label for label, _ in conditions], values)
        labels = [f"{value:.4f}" if index == 0 else f"{value:.2%}" for value in values]
        axis.bar_label(bars, labels=labels, padding=3, fontsize=8)
        axis.set_xlim(0, max(values) * 1.15 if index == 0 else 1.15)
        axis.invert_yaxis()
        axis.set_title(title)
    figure.suptitle("Inference-only ablations — no retraining; zero averaging logits give uniform weights")


def plot_ideal_inputs(model, class_names):
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(DIAGNOSTIC_SEED)
    count = len(class_names)
    images = (0.1 * torch.rand(count, 1, 28, 28, generator=generator)).to(device).requires_grad_()
    optimizer = torch.optim.Adam([images], lr=IDEAL_INPUT_LEARNING_RATE)
    for _ in range(IDEAL_INPUT_STEPS):
        logits = model(images)
        smoothness = (
            (images[:, :, 1:] - images[:, :, :-1]).square().mean()
            + (images[:, :, :, 1:] - images[:, :, :, :-1]).square().mean()
        )
        # A class's score relative to the others avoids merely raising every logit.
        score = (logits.diagonal() - logits.mean(dim=1)).mean()
        loss = -score + IDEAL_INPUT_L2 * images.square().mean() + IDEAL_INPUT_SMOOTHNESS * smoothness
        images.grad, = torch.autograd.grad(loss, images)
        optimizer.step()
        with torch.no_grad():
            images.clamp_(0, 1)
    with torch.no_grad():
        probabilities = model(images).softmax(dim=1).cpu()
    figure, axes = plt.subplots(math.ceil(count / 5), 5, figsize=(13, 6), squeeze=False, layout="constrained")
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(images[index, 0].detach().cpu(), cmap="gray", vmin=0, vmax=1)
        predicted = probabilities[index].argmax().item()
        axis.set_title(
            f"Target: {class_names[index]} ({probabilities[index, index]:.1%})\n"
            f"Pred: {class_names[predicted]}", fontsize=9,
        )
    figure.suptitle("Optimized inputs: maximize relative class score with pixel and smoothness penalties")


def plot_local_smoothness(model, images, targets):
    """Probe a frozen model on fixed training images; never write into model weights."""
    # Double precision makes small finite differences distinguishable from rounding.
    parameters = {name: value.detach().double().requires_grad_() for name, value in model.named_parameters()}
    images = images.double()
    generator = torch.Generator(device=images.device).manual_seed(DIAGNOSTIC_SEED)

    def loss_at(values):
        logits = torch.func.functional_call(model, values, (images,))
        return F.cross_entropy(logits, targets)

    base_loss = loss_at(parameters)
    gradients = torch.autograd.grad(base_loss, tuple(parameters.values()))
    flat_gradient = torch.cat([gradient.flatten() for gradient in gradients])
    directions = [
        {name: torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
         for name, value in parameters.items()}
        for _ in range(PERTURBATION_DIRECTIONS)
    ]
    cosines, changes, errors, loss_changes = [], [], [], []
    for sigma in PERTURBATION_STDEVS:
        similarities, relative_changes, measured, predicted, differences = [], [], [], [], []
        for direction in directions:
            positive = {name: value + sigma * direction[name] for name, value in parameters.items()}
            positive_loss = loss_at(positive)
            positive_gradient = torch.cat([gradient.flatten() for gradient in torch.autograd.grad(positive_loss, tuple(positive.values()))])
            with torch.no_grad():
                negative_loss = loss_at({name: value - sigma * direction[name] for name, value in parameters.items()})
                measured.append(((positive_loss - negative_loss) / (2 * sigma)).item())
                predicted.append(sum((gradient * noise).sum().item() for gradient, noise in zip(gradients, direction.values())))
                similarities.append(cosine(flat_gradient, positive_gradient))
                relative_changes.append(((positive_gradient - flat_gradient).norm() / flat_gradient.norm().clamp_min(1e-30)).item())
                differences.append(abs(positive_loss.item() - base_loss.item()))
        measured = torch.tensor(measured, dtype=torch.float64)
        predicted = torch.tensor(predicted, dtype=torch.float64)
        relative_error = ((measured - predicted).square().mean().sqrt() / predicted.square().mean().sqrt().clamp_min(1e-30)).item()
        cosines.append(similarities)
        changes.append(relative_changes)
        errors.append(relative_error)
        loss_changes.append(differences)
        print(f"  Local probe σ={sigma:g}: gradient cosine {sum(similarities) / len(similarities):.4f} | relative directional error {relative_error:.2e}")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, values, title in zip(axes.flat, (cosines, changes, errors, loss_changes), (
        "Full gradient cosine after perturbation", "Relative gradient change: ‖g′ − g‖ / ‖g‖",
        "Antithetic directional derivative: relative RMS error", "Absolute change in fixed-batch loss",
    )):
        values = torch.tensor(values)
        if values.ndim == 2:
            axis.fill_between(PERTURBATION_STDEVS, values.min(dim=1).values, values.max(dim=1).values, alpha=0.2, label="Range over directions")
            values = values.mean(dim=1)
        axis.plot(PERTURBATION_STDEVS, values, marker="o")
        axis.set_xscale("log")
        axis.set_xlabel("Per-parameter Gaussian standard deviation")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylim(-1.05, 1.05)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        # These quantities are nonnegative; ordinary log axes avoid suggesting
        # that negative gradient-change magnitudes or errors are possible.
        if any((line.get_ydata() > 0).any() for line in axis.lines):
            axis.set_yscale("log", nonpositive="mask")
    figure.suptitle(f"Frozen-model local smoothness — same {len(images)} training images, {PERTURBATION_DIRECTIONS} directions, float64")
