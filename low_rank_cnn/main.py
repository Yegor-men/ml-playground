import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

NUM_CLASSES = 10
BATCH_SIZE = 1
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


class LowRankConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, depth_mul=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_mul, kernel_size, groups=in_channels),
            nn.Conv2d(in_channels * depth_mul, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x)


base_model = nn.Sequential(
    nn.Conv2d(1, 16, 5),  # 28-4=24
    nn.SiLU(),
    nn.Conv2d(16, 32, 5),  # 24-4=20
    nn.SiLU(),
    nn.MaxPool2d(2, 2),  # 20/2=10
    nn.Conv2d(32, 64, 3),  # 10-2=8
    nn.SiLU(),
    nn.MaxPool2d(2, 2),  # 8/2=4
    nn.Conv2d(64, 128, 4),
    nn.Flatten(),
    nn.Linear(128, 128),
    nn.SiLU(),
    nn.Linear(128, 10),
    nn.Softmax(-1)
).to(device)

low_rank_model = nn.Sequential(
    nn.Conv2d(1, 16, 5),
    nn.SiLU(),
    LowRankConv2d(16, 32, 5),
    nn.SiLU(),
    nn.MaxPool2d(2, 2),
    LowRankConv2d(32, 64, 3),
    nn.SiLU(),
    nn.MaxPool2d(2, 2),  # 8/2=4
    LowRankConv2d(64, 128, 4),
    nn.Flatten(),
    nn.Linear(128, 128),
    nn.SiLU(),
    nn.Linear(128, 10),
    nn.Softmax(-1)
).to(device)


def print_trainable_parameters(model, name):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: {total_params:,}")


print_trainable_parameters(base_model, "BASE")
print_trainable_parameters(low_rank_model, "LOW RANK")

base_optimizer = torch.optim.AdamW(base_model.parameters(), 1e-3)
low_rank_optimizer = torch.optim.AdamW(low_rank_model.parameters(), 1e-3)

NUM_EPOCHS = 1

base_losses, low_rank_losses = [], []
smooth_base_losses, smooth_low_rank_losses = [0], [0]
loss_smoothness = 0.99

from tqdm import tqdm
import matplotlib.pyplot as plt

for e in range(NUM_EPOCHS):
    base_model.train()
    low_rank_model.train()

    for image, label in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        image, label = image.to(device), label.to(device)

        classic_output = base_model(image)
        euclid_output = low_rank_model(image)

        base_loss = nn.functional.mse_loss(classic_output, label)
        low_rank_loss = nn.functional.mse_loss(euclid_output, label)

        base_losses.append(base_loss.item())
        low_rank_losses.append(low_rank_loss.item())

        smooth_base_losses.append(smooth_base_losses[-1] * loss_smoothness + base_loss.item() * (1 - loss_smoothness))
        smooth_low_rank_losses.append(
            smooth_low_rank_losses[-1] * loss_smoothness + low_rank_loss.item() * (1 - loss_smoothness))

        base_loss.backward()
        low_rank_loss.backward()

        base_optimizer.step()
        low_rank_optimizer.step()

        base_model.zero_grad()
        low_rank_model.zero_grad()

    plt.title("RAW")
    plt.plot(base_losses, label="Base")
    plt.plot(low_rank_losses, label="Low Rank")
    plt.legend()
    plt.show()

    plt.title("SMOOTH")
    plt.plot(smooth_base_losses[1::], label="Base")
    plt.plot(smooth_low_rank_losses[1::], label="Low Rank")
    plt.legend()
    plt.show()

    base_model.eval()
    low_rank_model.eval()

    base_test_loss = 0
    low_rank_test_loss = 0

    with torch.no_grad():
        for image, label in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{e}"):
            image, label = image.to(device), label.to(device)

            classic_output = base_model(image)
            euclid_output = low_rank_model(image)

            base_loss = nn.functional.mse_loss(classic_output, label)
            low_rank_loss = nn.functional.mse_loss(euclid_output, label)

            base_test_loss += base_loss.item()
            low_rank_test_loss += low_rank_loss.item()

        base_test_loss /= len(test_dataloader)
        low_rank_test_loss /= len(test_dataloader)

    print(f"BASE: {base_test_loss}, LOW RANK: {low_rank_test_loss}")
