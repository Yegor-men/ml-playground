import torch
import torch.nn as nn
from typing import Optional

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class NegEuclidDist(nn.Module):
    """
    Negative squared Euclidean distance head using nn.Embedding to store prototypes.

    Args:
        in_features: dimension of each prototype (and last dim of input)
        out_features: number of prototypes (classes)
        init_std: if provided, prototypes are initialized ~N(0, init_std). If None,
                  the embedding's default init is used.
    Forward:
        x: tensor of shape [..., in_features]
        returns: tensor of shape [..., out_features] with values -||x - prototype||^2
    """

    def __init__(self, in_features: int, out_features: int, *, init_std: Optional[float] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # embedding holds prototypes as embedding.weight of shape (out_features, in_features)
        # Note: sparse=True is available, but not useful for dense vector-distance updates.
        self.prototypes = nn.Embedding(num_embeddings=out_features, embedding_dim=in_features)

        if init_std is not None:
            nn.init.normal_(self.prototypes.weight, mean=0.0, std=init_std)
        # else: keep nn.Embedding's default initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_features]
        returns: [..., out_features] with -squared-euclidean distances
        """
        if x.size(-1) != self.in_features:
            raise ValueError(f"Last dim of input must be {self.in_features}, got {x.size(-1)}")

        # use embedding.weight directly: shape (out_features, in_features)
        prot = self.prototypes.weight  # (K, H)
        # broadcasting subtraction: x[..., None, :] - prot[None, :, :] -> [..., K, H]
        diff = x.unsqueeze(-2) - prot
        dist2 = (diff * diff).sum(dim=-1)  # [..., out_features]
        return -dist2

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


NUM_CLASSES = 10
BATCH_SIZE = 64
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def one_hot_encode(label, num_classes=NUM_CLASSES):
    return torch.nn.functional.one_hot(torch.tensor(label), num_classes=num_classes).float()


class OneHotMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(
            root='data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),  # Converts to [C, H, W] in [0.0, 1.0]
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


class Enc(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0),  # 28-2=26
            nn.MaxPool2d(2, 2),  # 26/2=13
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=4),  # 13-3=10
            nn.MaxPool2d(2, 2),  # 10/2=5
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        return self.enc(x)


class ClassicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.dec = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 10),
            nn.Softmax(-1)
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)

        return x


class EuclidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        self.dec = nn.Sequential(
            nn.SiLU(),
            nn.Linear(64, 10),
            NegEuclidDist(10, 10),
            nn.Softmax(-1)
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)

        return x


classic_model = ClassicModel().to(device)
euclid_model = EuclidModel().to(device)

trainable_params = sum(p.numel() for p in classic_model.parameters() if p.requires_grad)
print(f"CLASSIC: {trainable_params:,}")

trainable_params = sum(p.numel() for p in euclid_model.parameters() if p.requires_grad)
print(f"EUCLID: {trainable_params:,}")

classic_optimizer = torch.optim.AdamW(classic_model.parameters(), 1e-3)
euclid_optimizer = torch.optim.AdamW(euclid_model.parameters(), 1e-3)

NUM_EPOCHS = 1

classic_losses, euclid_losses = [], []

from tqdm import tqdm
import matplotlib.pyplot as plt

for e in range(NUM_EPOCHS):
    classic_model.train()
    euclid_model.train()

    for image, label in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        image, label = image.to(device), label.to(device)

        classic_output = classic_model(image)
        euclid_output = euclid_model(image)

        classic_loss = nn.functional.mse_loss(classic_output, label)
        euclid_loss = nn.functional.mse_loss(euclid_output, label)

        classic_losses.append(classic_loss.item())
        euclid_losses.append(euclid_loss.item())

        classic_loss.backward()
        euclid_loss.backward()

        classic_optimizer.step()
        euclid_optimizer.step()

        classic_model.zero_grad()
        euclid_model.zero_grad()

    plt.title("Train")
    plt.plot(classic_losses, label="Classic")
    plt.plot(euclid_losses, label="Euclid")
    plt.legend()
    plt.show()

    classic_model.eval()
    euclid_model.eval()

    classic_test_loss = 0
    euclid_test_loss = 0

    with torch.no_grad():
        for image, label in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{e}"):
            image, label = image.to(device), label.to(device)

            classic_output = classic_model(image)
            euclid_output = euclid_model(image)

            classic_loss = nn.functional.mse_loss(classic_output, label)
            euclid_loss = nn.functional.mse_loss(euclid_output, label)

            classic_test_loss += classic_loss.item()
            euclid_test_loss += euclid_loss.item()

        classic_test_loss /= len(test_dataloader)
        euclid_test_loss /= len(test_dataloader)

    print(f"CLASSIC: {classic_test_loss}, EUCLID: {euclid_test_loss}")
