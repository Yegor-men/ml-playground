import time

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

BATCH_SIZE = 128
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def one_hot_encode(label):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


class OneHotMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(
            root='data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(
                    (128, 128),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True
                ),
                transforms.ToTensor()  # Converts to [C, H, W] in [0.0, 1.0]
            ])
        )

    def __getitem__(self, index):
        image, label = self.dataset[index]
        one_hot_label = one_hot_encode(label)
        return image, one_hot_label

    def __len__(self):
        return len(self.dataset)


train_dataset = OneHotMNIST(train=True)
test_dataset = OneHotMNIST(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class DSConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)


class DSUpscaleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            DSConv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, c_channels: int = 1, z_channels: int = 4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(c_channels, 16, 3, padding=1),
            nn.SiLU(),
            DSConv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            DSConv2d(32, 32, 4, 2, 1),
            nn.SiLU(),
            DSConv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            DSConv2d(64, 128, 3, padding=1),
            nn.SiLU(),
            DSConv2d(128, 128, 4, 2, 1),
            nn.SiLU(),
            DSConv2d(128, 256, 3, padding=1),
            nn.SiLU(),
            DSConv2d(256, 512, 3, padding=1),
            nn.SiLU(),
            DSConv2d(512, 512, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(512, z_channels * 2, 1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(z_channels, 512, 1),
            nn.SiLU(),
            DSUpscaleConv2d(512, 256),
            DSUpscaleConv2d(256, 128),
            DSUpscaleConv2d(128, 64),
            nn.Conv2d(64, c_channels, 1),
            nn.Sigmoid()
        )

    def encode_image(self, x):
        assert (x.size(-1) % 8 == 0) and (x.size(-2) % 8 == 0), "Image height and/or width is not divisible by 8"

        raw_latent = self.encoder(x)
        mu, logvar = torch.chunk(raw_latent, 2, dim=-3)

        if self.training:
            sigma = torch.exp(0.5 * logvar)
            z = mu + sigma * torch.randn_like(sigma)
            return z, mu, logvar
        else:
            z = mu
            return z

    def decode_latent(self, x):
        return self.decoder(x)


def print_trainable_parameters(model, name):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: {total_params:,}")


def kl_loss_fn(mu, logvar):
    return -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )


def show_reconstruction(base, recon, title=None):
    """
    base, recon: torch.Tensor
        Shape: [C, H, W] or [1, C, H, W]
        Range: [0, 1]
    """

    # Remove batch dimension if present
    if base.dim() == 4:
        base = base[0]
    if recon.dim() == 4:
        recon = recon[0]

    # Move to CPU and detach
    base = base.detach().cpu()
    recon = recon.detach().cpu()

    # Squared error
    error = (base - recon) ** 2
    error = error.mean(dim=0) if error.dim() == 3 else error

    # Convert to HWC for matplotlib
    def to_img(x):
        if x.dim() == 3:
            return x.permute(1, 2, 0)
        return x

    base_img = to_img(base)
    recon_img = to_img(recon)
    error_img = error

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(base_img, cmap="gray" if base_img.shape[-1] == 1 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_img, cmap="gray" if recon_img.shape[-1] == 1 else None)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(error_img, cmap="inferno")
    axes[2].set_title("Squared Error")
    axes[2].axis("off")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


vae = VAE(1, 4).to(device)
vae.train()

with torch.no_grad():
    print_trainable_parameters(vae, "VAE")
    print_trainable_parameters(vae.encoder, "VAE.encoder")
    print_trainable_parameters(vae.decoder, "VAE.decoder")

    dummy_image = torch.rand(1, 1, 512, 512).to(device)

    z, mu, logvar = vae.encode_image(dummy_image)
    reconstructed_image = vae.decode_latent(z)

    kl_loss = kl_loss_fn(mu, logvar)

    assert dummy_image.shape == reconstructed_image.shape, "reconstructed image not the same size"
    print(f"KL LOSS: {kl_loss}")

    time.sleep(0.1)

NUM_EPOCHS = 1

optimizer = torch.optim.AdamW(vae.parameters(), 1e-4)

train_recon_losses, train_kl_losses = [], []

for e in range(NUM_EPOCHS):
    vae.train()
    vae.zero_grad()

    train_recon_loss, train_kl_loss = 0, 0

    for image, label in tqdm(train_dataloader, total=len(train_dataloader), desc=f"E{e} - TRAIN"):
        image = image.to(device)

        z, mu, logvar = vae.encode_image(image)
        recon_image = vae.decode_latent(z)

        print(recon_image.min(), recon_image.mean(), recon_image.max())

        recon_loss = torch.nn.functional.mse_loss(recon_image, image)
        kl_loss = kl_loss_fn(mu, logvar)

        train_recon_losses.append(recon_loss.item())
        train_kl_losses.append(kl_loss.item())

        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()

        net_loss = recon_loss + kl_loss

        net_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_recon_loss /= len(train_dataloader)
    train_kl_loss /= len(train_dataloader)

    print(f"RECON: {train_recon_loss}, KL: {train_kl_loss}")

    plt.title("TRAIN")
    plt.plot(train_recon_losses, label="reconstruction")
    plt.plot(train_kl_losses, label="KL")
    plt.legend()
    plt.show()

    vae.eval()
    test_recon_loss = 0

    with torch.no_grad():
        for image, label in tqdm(test_dataloader, total=len(test_dataloader), desc=f"E{e} - TEST"):
            image = image.to(device)
            z = vae.encode_image(image)
            recon_image = vae.decode_latent(z)

            show_reconstruction(image, recon_image)

            recon_loss = torch.nn.functional.mse_loss(recon_image, image)

            test_recon_loss += recon_loss.item()

        test_recon_loss /= len(test_dataloader)

        print(f"TEST: {test_recon_loss}")
