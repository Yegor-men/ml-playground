import torch
from torch import nn


def get_mnist_dataloaders(image_size, batch_size):
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
                        (image_size, image_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True
                    ),
                    transforms.ToTensor()
                ])
            )

        def __getitem__(self, index):
            image, label = self.dataset[index]
            one_hot_label = one_hot_encode(label)
            return image, one_hot_label

        def __len__(self):
            return len(self.dataset)

    train_dataloader = DataLoader(OneHotMNIST(train=True), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(OneHotMNIST(train=False), batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


class VAE(nn.Module):
    def __init__(self, z_channels):
        super().__init__()

    def encode_image(self, x):
        assert (x.size(-1) % 8 == 0) and (x.size(-2) % 8 == 0), "Image height and/or width is not divisible by 8"
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-3)
        return mu, logvar

    def decode_latent(self, x):
        return self.decoder(x)

    def reparametrize(self, mu, logvar):
        if self.training:
            sigma = torch.exp(0.5 * logvar)
            z = mu + sigma * torch.randn_like(sigma)
            return z
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode_image(x)
        z = self.reparametrize(mu, logvar)
        reconstruction = self.decode_latent(z)
        return reconstruction, mu, logvar


def train_loop(
        num_epochs,
        train_dataloader,
        test_dataloader,
        beta,
        model,
        optimizer,
        device,
):
    def kld_loss_fn(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    from tqdm import tqdm

    for e in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        for image, label in tqdm(train_dataloader, total=len(train_dataloader), desc=f"E{e} - TRAIN"):
            image, label = image.to(device), label.to(device)

            recon_image, mu, logvar = model(image)

            recon_loss = torch.nn.functional.mse_loss(recon_image, image)
            kl_loss = kld_loss_fn(mu, logvar) * beta

            net_loss = recon_loss + kl_loss
            net_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        for image, label in tqdm(test_dataloader, total=len(test_dataloader), desc=f"E{e} - TEST"):
            image, label = image.to(device), label.to(device)

            recon_image, mu, logvar = model(image)

            recon_loss = torch.nn.functional.mse_loss(recon_image, image)
            kl_loss = kld_loss_fn(mu, logvar) * beta

            net_loss = recon_loss + kl_loss


# ======================================================================================================================


import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MNIST VAE training script")

    model_group = parser.add_argument_group("Model Hyperparameters")
    model_group.add_argument("--z_channels", type=int, default=4)

    data_group = parser.add_argument_group("Data Augmentation Settings")
    data_group.add_argument("--image_size", type=int, default=128)

    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--lr", type=float, default=1e-3)
    train_group.add_argument("--num_epochs", type=int, default=1)
    train_group.add_argument("--batch_size", type=int, default=8)
    train_group.add_argument("--beta", type=float, default=1.0, help="Beta value for KL divergence scaling")
    train_group.add_argument("--ema_decay", type=float, default=0.99)

    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--seed", type=int, default=0)
    misc_group.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto")

    return parser.parse_args()


def main(args):
    device = args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_dataloader, test_dataloader = get_mnist_dataloaders(args.image_size, args.batch_size)
    vae_model = VAE(args.z_channels).to(device)
    optimizer = torch.optim.AdamW(vae_model.parameters(), args.lr)

    train_loop(
        num_epochs=args.num_epochs,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        beta=args.beta,
        model=vae_model,
        optimizer=optimizer,
        device=device
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
