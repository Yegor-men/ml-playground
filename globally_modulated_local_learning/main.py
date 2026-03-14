import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.n1 = nn.Linear(in_features, out_features)

        hidden_features = in_features + out_features
        self.n2 = nn.Sequential(
            nn.Linear(in_features + out_features, 4 * hidden_features),
            nn.SiLU(),
            nn.Linear(4 * hidden_features, 1)
        )

        self.n1_output = None
        self.pred_loss = None

    def forward(self, x):
        output = self.n1(x)

        combined = torch.cat([x, output], dim=-1)

        pred_loss = self.n2(combined)

        self.pred_loss = pred_loss

        return output.detach()


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = Layer(784, 256)
        self.l2 = Layer(256, 256)
        self.l3 = Layer(256, 10)

    def update_params(self, real_loss):
        for layer in (self.l1, self.l2, self.l3):
            loss = nn.functional.mse_loss(layer.pred_loss, real_loss.detach())
            loss.backward()
            layer.pred_loss = None

    def forward(self, x):
        x = nn.functional.silu(self.l1(x))
        x = nn.functional.silu(self.l2(x))
        x = self.l3(x)
        return x


def get_mnist_dataloaders(batch_size=64):
    """Create MNIST dataloaders that save data to 'data' directory and return flattened images."""

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Transform to flatten images (28*28 = 784) and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 1D tensor
    ])

    # Download and load MNIST data
    train_dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


train_dataloader, test_dataloader = get_mnist_dataloaders(batch_size=100)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

# Lists to store metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

num_epochs = 10
for E in range(num_epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    epoch_train_correct = 0
    epoch_train_total = 0

    for (image, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{E}"):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()

        model_output = model(image)

        real_loss = nn.functional.cross_entropy(model_output, label, reduction='none').unsqueeze(1)

        # Calculate accuracy
        _, predicted = torch.max(model_output.data, 1)
        epoch_train_total += label.size(0)
        epoch_train_correct += (predicted == label).sum().item()

        # Update model parameters using the custom update method
        model.update_params(real_loss)
        optimizer.step()

        epoch_train_loss += real_loss.mean().item()

    # Evaluation phase
    model.eval()
    epoch_test_loss = 0.0
    epoch_test_correct = 0
    epoch_test_total = 0

    with torch.no_grad():
        for (image, label) in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{E}"):
            image, label = image.to(device), label.to(device)

            model_output = model(image)
            test_loss = nn.functional.cross_entropy(model_output, label, reduction='none').unsqueeze(1)

            # Calculate accuracy
            _, predicted = torch.max(model_output.data, 1)
            epoch_test_total += label.size(0)
            epoch_test_correct += (predicted == label).sum().item()

            epoch_test_loss += test_loss.mean().item()

    # Calculate averages
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    avg_train_acc = epoch_train_correct / epoch_train_total
    avg_test_loss = epoch_test_loss / len(test_dataloader)
    avg_test_acc = epoch_test_correct / epoch_test_total

    # Store metrics
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)
    test_losses.append(avg_test_loss)
    test_accuracies.append(avg_test_acc)

    # Print epoch results
    print(f"Epoch {E + 1}/{num_epochs}:")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
    print(f"  Test Loss:  {avg_test_loss:.4f}, Test Acc:  {avg_test_acc:.4f}")
    print()

# Plot results
plt.figure(figsize=(12, 5))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, 'r-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, 'r-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
