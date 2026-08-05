"""Train a stateful Hodgkin-Huxley-inspired network on MNIST."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# Experiment configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HIDDEN_SIZE = 128
NUM_TIMESTEPS = 25
DELTA_T = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_WORKERS = 0
NUM_EXAMPLES_TO_PLOT = 12
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ClassificationStats:
    loss: float
    accuracy: float


class HodgkinHuxleyLayer(nn.Module):
    """A differentiable conductance-based recurrent layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.input_current = nn.Linear(in_features, out_features)
        self.spike_readout = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.Softplus(),
        )

        self.capacitance = nn.Parameter(torch.zeros(out_features))
        self.g_na = nn.Parameter(torch.zeros(out_features))
        self.g_k = nn.Parameter(torch.zeros(out_features))
        self.g_l = nn.Parameter(torch.zeros(out_features))
        self.e_na = nn.Parameter(torch.ones(out_features))
        self.e_k = nn.Parameter(-torch.ones(out_features))
        self.e_l = nn.Parameter(torch.zeros(out_features))

        self.v_half_m = nn.Parameter(torch.zeros(out_features))
        self.v_half_h = nn.Parameter(torch.zeros(out_features))
        self.v_half_n = nn.Parameter(torch.zeros(out_features))
        self.k_m = nn.Parameter(torch.zeros(out_features))
        self.k_h = nn.Parameter(torch.zeros(out_features))
        self.k_n = nn.Parameter(torch.zeros(out_features))
        self.tau_min_m = nn.Parameter(torch.zeros(out_features))
        self.tau_min_h = nn.Parameter(torch.zeros(out_features))
        self.tau_min_n = nn.Parameter(torch.zeros(out_features))
        self.tau_amp_m = nn.Parameter(torch.zeros(out_features))
        self.tau_amp_h = nn.Parameter(torch.zeros(out_features))
        self.tau_amp_n = nn.Parameter(torch.zeros(out_features))
        self.tau_width_m = nn.Parameter(torch.zeros(out_features))
        self.tau_width_h = nn.Parameter(torch.zeros(out_features))
        self.tau_width_n = nn.Parameter(torch.zeros(out_features))

        self.v: torch.Tensor | None = None
        self.m: torch.Tensor | None = None
        self.h: torch.Tensor | None = None
        self.n: torch.Tensor | None = None

    @staticmethod
    def steady_state(
            voltage: torch.Tensor,
            half_voltage: torch.Tensor,
            slope: torch.Tensor,
    ) -> torch.Tensor:
        safe_slope = F.softplus(slope) + 1e-6
        return torch.sigmoid((voltage - half_voltage) / safe_slope)

    @staticmethod
    def time_constant(
            voltage: torch.Tensor,
            minimum: torch.Tensor,
            amplitude: torch.Tensor,
            center: torch.Tensor,
            width: torch.Tensor,
    ) -> torch.Tensor:
        minimum = F.softplus(minimum) + 1e-4
        amplitude = F.softplus(amplitude)
        width = F.softplus(width) + 1e-4
        return minimum + amplitude * torch.exp(-(voltage - center).square() / width)

    @torch.no_grad()
    def reset_state(self, batch_size: int, reference: torch.Tensor):
        self.v = reference.new_zeros(batch_size, self.out_features)
        self.m = self.steady_state(self.v, self.v_half_m, self.k_m)
        self.h = self.steady_state(self.v, self.v_half_h, self.k_h)
        self.n = self.steady_state(self.v, self.v_half_n, self.k_n)

    def detach_state(self):
        self.v = self.v.detach()
        self.m = self.m.detach()
        self.h = self.h.detach()
        self.n = self.n.detach()

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        if self.v is None or self.v.size(0) != spikes.size(0):
            self.reset_state(spikes.size(0), spikes)

        m_inf = self.steady_state(self.v, self.v_half_m, self.k_m)
        h_inf = self.steady_state(self.v, self.v_half_h, self.k_h)
        n_inf = self.steady_state(self.v, self.v_half_n, self.k_n)
        tau_m = self.time_constant(
            self.v, self.tau_min_m, self.tau_amp_m, self.v_half_m, self.tau_width_m
        )
        tau_h = self.time_constant(
            self.v, self.tau_min_h, self.tau_amp_h, self.v_half_h, self.tau_width_h
        )
        tau_n = self.time_constant(
            self.v, self.tau_min_n, self.tau_amp_n, self.v_half_n, self.tau_width_n
        )

        self.m = (self.m + DELTA_T * (m_inf - self.m) / tau_m).clamp(0.0, 1.0)
        self.h = (self.h + DELTA_T * (h_inf - self.h) / tau_h).clamp(0.0, 1.0)
        self.n = (self.n + DELTA_T * (n_inf - self.n) / tau_n).clamp(0.0, 1.0)

        sodium = F.softplus(self.g_na) * self.m.pow(3) * self.h * (self.v - self.e_na)
        potassium = F.softplus(self.g_k) * self.n.pow(4) * (self.v - self.e_k)
        leakage = F.softplus(self.g_l) * (self.v - self.e_l)
        dv_dt = (
                        self.input_current(spikes) - sodium - potassium - leakage
                ) / (F.softplus(self.capacitance) + 1e-4)
        self.v = self.v + DELTA_T * dv_dt
        return self.spike_readout(self.v)


class HodgkinHuxleyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HodgkinHuxleyLayer(28 * 28, HIDDEN_SIZE),
                HodgkinHuxleyLayer(HIDDEN_SIZE, HIDDEN_SIZE),
                HodgkinHuxleyLayer(HIDDEN_SIZE, 10),
            ]
        )

    def reset_state(self, batch_size: int, reference: torch.Tensor):
        for layer in self.layers:
            layer.reset_state(batch_size, reference)

    def detach_state(self):
        for layer in self.layers:
            layer.detach_state()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spikes = images.flatten(start_dim=1)
        for layer in self.layers:
            spikes = layer(spikes)
        return spikes


def make_dataloaders() -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(SEED)
    options = {
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }
    return (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            **options,
        ),
        DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            **options,
        ),
    )


def run_epoch(
        model: HodgkinHuxleyClassifier,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
) -> ClassificationStats:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    split = "TRAIN" if training else "TEST"

    progress = tqdm(dataloader, desc=f"{split} E{epoch}", leave=False)
    for images, targets in progress:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        model.reset_state(images.size(0), images)
        if training:
            optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        logits = None
        for _ in range(NUM_TIMESTEPS):
            with torch.set_grad_enabled(training):
                logits = model(images)
                step_loss = F.cross_entropy(logits, targets)
                if training:
                    (step_loss / NUM_TIMESTEPS).backward()
                    model.detach_state()
            accumulated_loss += float(step_loss.detach().cpu())

        if training:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        batch_size = targets.size(0)
        batch_loss = accumulated_loss / NUM_TIMESTEPS
        batch_accuracy = logits.argmax(dim=-1).eq(targets).float().mean().item()
        total_loss += batch_loss * batch_size
        total_correct += int(logits.argmax(dim=-1).eq(targets).sum().cpu())
        total_examples += batch_size
        progress.set_postfix(loss=f"{batch_loss:.3f}", accuracy=f"{batch_accuracy:.1%}")

    return ClassificationStats(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


def plot_history(history: dict[str, list[float]]):
    epochs = range(1, NUM_EPOCHS + 1)
    _, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["test_loss"], marker="o", label="test")
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="train")
    axes[1].plot(epochs, history["test_accuracy"], marker="o", label="test")
    axes[0].set_title("Cross-entropy loss")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    plt.tight_layout()


@torch.no_grad()
def plot_predictions(model: HodgkinHuxleyClassifier, test_loader: DataLoader):
    images, targets = next(iter(test_loader))
    images = images[:NUM_EXAMPLES_TO_PLOT].to(DEVICE)
    targets = targets[:NUM_EXAMPLES_TO_PLOT]
    model.eval()
    model.reset_state(len(images), images)
    for _ in range(NUM_TIMESTEPS):
        logits = model(images)
    predictions = logits.argmax(dim=-1).cpu()

    columns = 4
    rows = (len(images) + columns - 1) // columns
    _, axes = plt.subplots(rows, columns, figsize=(10, 2.4 * rows))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(images):
            continue
        target = int(targets[index])
        prediction = int(predictions[index])
        axis.imshow(images[index].cpu().squeeze(), cmap="gray")
        axis.set_title(
            f"true {target} | pred {prediction}",
            color="green" if target == prediction else "red",
        )
    plt.suptitle("Hodgkin-Huxley network predictions")
    plt.tight_layout()


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train_loader, test_loader = make_dataloaders()
    model = HodgkinHuxleyClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

    print(f"Device: {DEVICE}")
    print(
        f"Model: 784 -> {HIDDEN_SIZE} -> {HIDDEN_SIZE} -> 10, "
        f"{NUM_TIMESTEPS} recurrent timesteps per image"
    )
    for epoch in range(1, NUM_EPOCHS + 1):
        train_stats = run_epoch(model, train_loader, optimizer, epoch)
        test_stats = run_epoch(model, test_loader, None, epoch)
        history["train_loss"].append(train_stats.loss)
        history["test_loss"].append(test_stats.loss)
        history["train_accuracy"].append(train_stats.accuracy)
        history["test_accuracy"].append(test_stats.accuracy)
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} | "
            f"train loss {train_stats.loss:.4f}, accuracy {train_stats.accuracy:.2%} | "
            f"test loss {test_stats.loss:.4f}, accuracy {test_stats.accuracy:.2%}"
        )

    plot_history(history)
    plot_predictions(model, test_loader)
    plt.show()


if __name__ == "__main__":
    main()
