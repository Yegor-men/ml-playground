"""Train a compact convolutional variational autoencoder on MNIST."""

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
IMAGE_SIZE = 32
LATENT_CHANNELS = 4
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
KL_WEIGHT = 0.1
NUM_WORKERS = 0
NUM_EXAMPLES_TO_PLOT = 10
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class VAEStats:
    loss: float
    reconstruction_loss: float
    kl_loss: float


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            DepthwiseSeparableConv2d(in_channels, out_channels),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            DepthwiseSeparableConv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            DepthwiseSeparableConv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            DepthwiseSeparableConv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 2 * LATENT_CHANNELS, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, 256, kernel_size=1),
            nn.SiLU(),
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64),
            UpsampleBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moments = self.encoder(images)
        return moments.chunk(2, dim=1)

    def sample(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mean
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + standard_deviation * torch.randn_like(standard_deviation)

    def forward(
            self,
            images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(images)
        latent = self.sample(mean, log_variance)
        return self.decoder(latent), mean, log_variance


def make_dataloaders() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )
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


def vae_loss(
        reconstruction: torch.Tensor,
        images: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.mse_loss(reconstruction, images)
    kl_loss = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).mean()
    loss = reconstruction_loss + KL_WEIGHT * kl_loss
    return loss, reconstruction_loss, kl_loss


def run_epoch(
        model: VariationalAutoencoder,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
) -> VAEStats:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "count": 0}
    split = "TRAIN" if training else "TEST"

    progress = tqdm(dataloader, desc=f"{split} E{epoch}", leave=False)
    for images, _ in progress:
        images = images.to(DEVICE, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstruction, mean, log_variance = model(images)
            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                images,
                mean,
                log_variance,
            )
            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["reconstruction"] += float(reconstruction_loss.detach().cpu()) * batch_size
        totals["kl"] += float(kl_loss.detach().cpu()) * batch_size
        totals["count"] += batch_size
        progress.set_postfix(
            reconstruction=f"{reconstruction_loss.detach().item():.4f}",
            kl=f"{kl_loss.detach().item():.4f}",
        )

    return VAEStats(
        loss=totals["loss"] / totals["count"],
        reconstruction_loss=totals["reconstruction"] / totals["count"],
        kl_loss=totals["kl"] / totals["count"],
    )


def plot_history(history: dict[str, list[float]]):
    epochs = range(1, NUM_EPOCHS + 1)
    _, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric, title in [
        (axes[0], "loss", "Total loss"),
        (axes[1], "reconstruction", "Reconstruction MSE"),
        (axes[2], "kl", "KL divergence"),
    ]:
        axis.plot(epochs, history[f"train_{metric}"], marker="o", label="train")
        axis.plot(epochs, history[f"test_{metric}"], marker="o", label="test")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    plt.tight_layout()


@torch.no_grad()
def plot_reconstructions(model: VariationalAutoencoder, test_loader: DataLoader):
    images, _ = next(iter(test_loader))
    images = images[:NUM_EXAMPLES_TO_PLOT].to(DEVICE)
    model.eval()
    reconstructions, _, _ = model(images)
    images = images.cpu()
    reconstructions = reconstructions.cpu()

    _, axes = plt.subplots(2, len(images), figsize=(1.7 * len(images), 4))
    for index in range(len(images)):
        axes[0, index].imshow(images[index].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, index].imshow(
            reconstructions[index].squeeze(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("reconstruction")
    plt.suptitle("MNIST reconstructions after training")
    plt.tight_layout()


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train_loader, test_loader = make_dataloaders()
    model = VariationalAutoencoder().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history = {
        f"{split}_{metric}": []
        for split in ("train", "test")
        for metric in ("loss", "reconstruction", "kl")
    }

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {DEVICE}")
    print(f"VAE parameters: {parameters:,}; latent shape: {LATENT_CHANNELS}x4x4")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_stats = run_epoch(model, train_loader, optimizer, epoch)
        test_stats = run_epoch(model, test_loader, None, epoch)
        for split, stats in (("train", train_stats), ("test", test_stats)):
            history[f"{split}_loss"].append(stats.loss)
            history[f"{split}_reconstruction"].append(stats.reconstruction_loss)
            history[f"{split}_kl"].append(stats.kl_loss)
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} | "
            f"train loss {train_stats.loss:.5f} "
            f"(recon {train_stats.reconstruction_loss:.5f}, kl {train_stats.kl_loss:.5f}) | "
            f"test loss {test_stats.loss:.5f} "
            f"(recon {test_stats.reconstruction_loss:.5f}, kl {test_stats.kl_loss:.5f})"
        )

    plot_history(history)
    plot_reconstructions(model, test_loader)
    plt.show()


if __name__ == "__main__":
    main()
