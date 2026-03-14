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

        self.n1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.SiLU(),
        )

        # n2_upstream: takes layer input x → predicts the true label (one-hot)
        # This approximates "what the label should be" from everything upstream + this point
        self.n2_upstream = nn.Sequential(
            nn.Linear(in_features, 4 * in_features),
            nn.SiLU(),
            nn.Linear(4 * in_features, 10)
        )

        # n2_downstream: takes layer output y → predicts the final model output
        # This approximates "what the rest of the network will output" from this point onward
        self.n2_downstream = nn.Sequential(
            nn.Linear(out_features, 4 * out_features),
            nn.SiLU(),
            nn.Linear(4 * out_features, 10)
        )

        # Storage for manual gradient flow
        self.last_input = None
        self.n1_output = None

    def forward(self, x):
        self.last_input = x
        self.n1_output = self.n1(x)
        return self.n1_output.detach()  # break global graph

    def update_layer(self, global_target, global_final):
        """
        1. Train the two critics directly:
           - n2_upstream(x) ← global_target (label)
           - n2_downstream(y) ← global_final (actual model output)
        2. Compute predicted_loss = MSE(n2_upstream(x), n2_downstream(y))
           This becomes a surrogate for the true global MSE(final, label)
        3. Backprop the surrogate gradient ONLY through n1 (via detached y)
        """
        # === Critic supervision (makes predicted loss match true loss) ===
        upstream_approx = self.n2_upstream(self.last_input)
        downstream_approx = self.n2_downstream(self.n1_output.detach())

        upstream_loss = nn.functional.mse_loss(upstream_approx, global_target.detach())
        upstream_loss.backward()

        downstream_loss = nn.functional.mse_loss(downstream_approx, global_final.detach())
        downstream_loss.backward()

        # === Surrogate gradient for n1 ===
        if self.n1_output is not None:
            detached_y = self.n1_output.detach().requires_grad_(True)

            # Recompute with fresh forward (params are still the same)
            upstream_val = self.n2_upstream(self.last_input)
            downstream_for_grad = self.n2_downstream(detached_y)

            # predicted_loss = MSE(upstream_view_of_label, downstream_view_of_final)
            # This is exactly the split you asked for
            pred_loss = nn.functional.mse_loss(
                upstream_val,
                downstream_for_grad,
                reduction='none'
            ).mean(dim=1, keepdim=True)

            # Gradient of predicted loss w.r.t. layer output y
            grad_wrt_y = torch.autograd.grad(
                outputs=pred_loss,
                inputs=detached_y,
                grad_outputs=torch.ones_like(pred_loss),
                create_graph=False,
                retain_graph=False
            )[0]

            # Apply surrogate gradient to n1 (no global graph)
            self.n1_output.backward(grad_wrt_y)

        # Cleanup
        self.last_input = None
        self.n1_output = None


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = Layer(784, 256)
        self.l2 = Layer(256, 256)
        self.l3 = Layer(256, 10)  # final layer outputs classification logits

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

    def update_params(self, global_target, global_final):
        # Each layer gets the SAME global target (label) and global final output
        # No global backprop anywhere — exactly as you wanted
        self.l1.update_layer(global_target, global_final)
        self.l2.update_layer(global_target, global_final)
        self.l3.update_layer(global_target, global_final)


# ================== Data (classification, one-hot) ==================
def get_mnist_dataloaders(batch_size=100):
    os.makedirs('data', exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        labels_onehot = torch.zeros(labels.size(0), 10)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        return images, labels_onehot

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader


train_dataloader, test_dataloader = get_mnist_dataloaders()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Metrics
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

num_epochs = 5
for E in range(num_epochs):
    # ===================== TRAINING =====================
    model.train()
    epoch_train_loss = 0.0
    epoch_train_correct = 0
    epoch_train_total = 0

    for image, label in tqdm(train_dataloader, desc=f"TRAIN E{E}"):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()

        model_output = model(image)

        # True global loss (exactly as you asked — this is the "actual mistake")
        real_loss = nn.functional.mse_loss(model_output, label, reduction='none').mean(dim=1, keepdim=True)

        # Accuracy
        _, predicted = torch.max(model_output, 1)
        _, true_labels = torch.max(label, 1)
        epoch_train_total += label.size(0)
        epoch_train_correct += (predicted == true_labels).sum().item()

        # Local per-layer update (n2_upstream + n2_downstream + surrogate on n1)
        model.update_params(label, model_output)

        optimizer.step()

        epoch_train_loss += real_loss.mean().item()

    # ===================== EVALUATION =====================
    model.eval()
    epoch_test_loss = 0.0
    epoch_test_correct = 0
    epoch_test_total = 0

    with torch.no_grad():
        for image, label in tqdm(test_dataloader, desc=f"TEST E{E}"):
            image, label = image.to(device), label.to(device)
            model_output = model(image)

            test_loss = nn.functional.mse_loss(model_output, label, reduction='none').mean(dim=1, keepdim=True)

            _, predicted = torch.max(model_output, 1)
            _, true_labels = torch.max(label, 1)
            epoch_test_total += label.size(0)
            epoch_test_correct += (predicted == true_labels).sum().item()

            epoch_test_loss += test_loss.mean().item()

    # Averages
    avg_train_loss = epoch_train_loss / len(train_dataloader)
    avg_train_acc = epoch_train_correct / epoch_train_total
    avg_test_loss = epoch_test_loss / len(test_dataloader)
    avg_test_acc = epoch_test_correct / epoch_test_total

    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)
    test_losses.append(avg_test_loss)
    test_accuracies.append(avg_test_acc)

    print(f"Epoch {E + 1}/{num_epochs}:")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
    print(f"  Test Loss:  {avg_test_loss:.4f} | Test Acc:  {avg_test_acc:.4f}")
    print()

# ===================== PLOTS =====================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, 'r-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, 'b-', label='Train Acc')
plt.plot(range(1, num_epochs + 1), test_accuracies, 'r-', label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
