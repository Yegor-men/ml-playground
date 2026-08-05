from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Experiment configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BATCH_SIZE = 128
NUM_EPOCHS = 5
HIDDEN_SIZES = [256, 256]
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0.0
OPTIMIZER_NAME = "adamw"  # "adamw" or "sgd"
CRITIC_STEPS = 1
ACTOR_STEPS = 1
GRAD_CLIP = 1.0
TRAIN_LIMIT = None
TEST_LIMIT = None
NUM_WORKERS = 0
NUM_EXAMPLES_TO_PLOT = 12
SEED = 0
DETERMINISTIC = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BatchUpdateStats:
    critic_loss: float
    critic_reward_pred: float
    actor_reward_pred: float


@dataclass
class EpochStats:
    reward: float
    accuracy: float
    critic_loss: float = 0.0
    critic_reward_pred: float = 0.0
    actor_reward_pred: float = 0.0


class LocalActorCriticLayer(nn.Module):
    """
    A vectorized layer of independent actor/critic neurons.

    For output neuron i:
      actor_i(x) = SiLU(w_i @ x + b_i)
      critic_i(y_i) = sigmoid(c_i * y_i)

    The layer caches only the input and output that the local neuron observed.
    Actor updates later recompute actor_i(cached_x), freeze critic_i, and ascend
    critic_i's predicted reward. No gradient is propagated through neighboring
    layers or through the global reward computation.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.actor_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.actor_bias = nn.Parameter(torch.empty(out_features))
        self.critic_weight = nn.Parameter(torch.empty(out_features))

        self.last_input: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.actor_weight, a=5 ** 0.5)
        fan_in = self.actor_weight.size(1)
        bound = fan_in ** -0.5 if fan_in > 0 else 0.0
        nn.init.uniform_(self.actor_bias, -bound, bound)

        # A small non-zero initial slope lets the actor receive a local update
        # before the critic has fully separated useful and useless outputs.
        nn.init.normal_(self.critic_weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach()
        output = self.actor_output(self.last_input)
        self.last_output = output.detach()
        return self.last_output

    def actor_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(F.linear(x, self.actor_weight, self.actor_bias))

    def critic_prediction(self, output: torch.Tensor, detach_critic: bool = False) -> torch.Tensor:
        if detach_critic:
            critic_weight = self.critic_weight.detach()
        else:
            critic_weight = self.critic_weight

        return torch.sigmoid(output * critic_weight.unsqueeze(0))

    def critic_parameters(self):
        yield self.critic_weight

    def actor_parameters(self):
        yield self.actor_weight
        yield self.actor_bias

    def clear_cache(self):
        self.last_input = None
        self.last_output = None


class LocalActorCriticMNIST(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include at least input and output sizes.")

        self.layers = nn.ModuleList(
            LocalActorCriticLayer(in_features, out_features)
            for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        return x

    def actor_parameters(self):
        for layer in self.layers:
            yield from layer.actor_parameters()

    def critic_parameters(self):
        for layer in self.layers:
            yield from layer.critic_parameters()

    def critic_loss(self, reward: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        losses = []
        predictions = []
        target = reward.detach()

        for layer in self.layers:
            if layer.last_output is None:
                raise RuntimeError("critic_loss called before a forward pass.")
            prediction = layer.critic_prediction(layer.last_output.detach())
            losses.append(F.mse_loss(prediction, target.expand_as(prediction)))
            predictions.append(prediction.detach().mean())

        return torch.stack(losses).mean(), torch.stack(predictions).mean()

    def actor_loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        objectives = []
        predictions = []

        for layer in self.layers:
            if layer.last_input is None:
                raise RuntimeError("actor_loss called before a forward pass.")
            output = layer.actor_output(layer.last_input)
            prediction = layer.critic_prediction(output, detach_critic=True)
            objectives.append(prediction.mean())
            predictions.append(prediction.detach().mean())

        predicted_reward = torch.stack(objectives).mean()
        return -predicted_reward, torch.stack(predictions).mean()

    def local_update(
            self,
            reward: torch.Tensor,
            actor_optimizer: torch.optim.Optimizer,
            critic_optimizer: torch.optim.Optimizer,
            critic_steps: int,
            actor_steps: int,
            grad_clip: float | None,
    ) -> BatchUpdateStats:
        reward = reward.detach()
        critic_loss_value = 0.0
        critic_pred_value = 0.0

        for _ in range(critic_steps):
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss, critic_prediction = self.critic_loss(reward)
            critic_loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(list(self.critic_parameters()), grad_clip)
            critic_optimizer.step()

            critic_loss_value = float(critic_loss.detach().cpu())
            critic_pred_value = float(critic_prediction.detach().cpu())

        actor_pred_value = 0.0
        for _ in range(actor_steps):
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss, actor_prediction = self.actor_loss()
            actor_loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(list(self.actor_parameters()), grad_clip)
            actor_optimizer.step()

            actor_pred_value = float(actor_prediction.detach().cpu())

        self.clear_cache()
        return BatchUpdateStats(
            critic_loss=critic_loss_value,
            critic_reward_pred=critic_pred_value,
            actor_reward_pred=actor_pred_value,
        )

    def clear_cache(self):
        for layer in self.layers:
            layer.clear_cache()


def one_hot_digit(label: int) -> torch.Tensor:
    return F.one_hot(torch.tensor(label), num_classes=10).to(torch.float32)


def get_mnist_dataloaders() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda image: image.flatten()),
        ]
    )

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
        target_transform=one_hot_digit,
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
        target_transform=one_hot_digit,
    )

    if TRAIN_LIMIT is not None:
        train_dataset = Subset(train_dataset, range(min(TRAIN_LIMIT, len(train_dataset))))
    if TEST_LIMIT is not None:
        test_dataset = Subset(test_dataset, range(min(TEST_LIMIT, len(test_dataset))))

    generator = torch.Generator().manual_seed(SEED)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )
    return train_dataloader, test_dataloader


def cosine_reward(logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    reward = F.cosine_similarity(probabilities, target_onehot, dim=-1, eps=1e-8)
    return reward.clamp(min=0.0, max=1.0).unsqueeze(1)


def accuracy_from_logits(logits: torch.Tensor, target_onehot: torch.Tensor) -> float:
    predicted = logits.argmax(dim=-1)
    target = target_onehot.argmax(dim=-1)
    return float(predicted.eq(target).to(torch.float32).mean().detach().cpu())


def make_optimizer(name: str, parameters, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = list(parameters)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(
        model: LocalActorCriticMNIST,
        dataloader: DataLoader,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch_index: int,
        critic_steps: int,
        actor_steps: int,
        grad_clip: float | None,
) -> EpochStats:
    model.train()
    reward_total = 0.0
    accuracy_total = 0.0
    critic_loss_total = 0.0
    critic_pred_total = 0.0
    actor_pred_total = 0.0
    batch_count = 0

    progress = tqdm(dataloader, desc=f"TRAIN E{epoch_index}", leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        reward = cosine_reward(logits, targets)
        batch_update = model.local_update(
            reward=reward,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            critic_steps=critic_steps,
            actor_steps=actor_steps,
            grad_clip=grad_clip,
        )

        batch_reward = float(reward.mean().detach().cpu())
        batch_accuracy = accuracy_from_logits(logits, targets)
        reward_total += batch_reward
        accuracy_total += batch_accuracy
        critic_loss_total += batch_update.critic_loss
        critic_pred_total += batch_update.critic_reward_pred
        actor_pred_total += batch_update.actor_reward_pred
        batch_count += 1

        progress.set_postfix(
            reward=f"{batch_reward:.4f}",
            acc=f"{batch_accuracy:.3f}",
            critic=f"{batch_update.critic_loss:.4f}",
        )

    return EpochStats(
        reward=reward_total / batch_count,
        accuracy=accuracy_total / batch_count,
        critic_loss=critic_loss_total / batch_count,
        critic_reward_pred=critic_pred_total / batch_count,
        actor_reward_pred=actor_pred_total / batch_count,
    )


@torch.no_grad()
def evaluate(
        model: LocalActorCriticMNIST,
        dataloader: DataLoader,
        device: torch.device,
        epoch_index: int,
) -> EpochStats:
    model.eval()
    reward_total = 0.0
    accuracy_total = 0.0
    batch_count = 0

    progress = tqdm(dataloader, desc=f"TEST  E{epoch_index}", leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        reward = cosine_reward(logits, targets)
        batch_reward = float(reward.mean().detach().cpu())
        batch_accuracy = accuracy_from_logits(logits, targets)

        reward_total += batch_reward
        accuracy_total += batch_accuracy
        batch_count += 1
        model.clear_cache()

        progress.set_postfix(reward=f"{batch_reward:.4f}", acc=f"{batch_accuracy:.3f}")

    return EpochStats(reward=reward_total / batch_count, accuracy=accuracy_total / batch_count)


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def plot_history(history: dict[str, list[float]]):
    epochs = range(1, NUM_EPOCHS + 1)
    _, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, history["train_reward"], marker="o", label="train")
    axes[0].plot(epochs, history["test_reward"], marker="o", label="test")
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="train")
    axes[1].plot(epochs, history["test_accuracy"], marker="o", label="test")
    axes[2].plot(epochs, history["critic_loss"], marker="o")
    axes[0].set_title("Cosine reward")
    axes[1].set_title("Accuracy")
    axes[2].set_title("Critic MSE")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        if axis is not axes[2]:
            axis.legend()
    plt.tight_layout()


@torch.no_grad()
def plot_predictions(model: LocalActorCriticMNIST, test_dataloader: DataLoader):
    images, targets = next(iter(test_dataloader))
    images = images[:NUM_EXAMPLES_TO_PLOT].to(DEVICE)
    targets = targets[:NUM_EXAMPLES_TO_PLOT]
    model.eval()
    predictions = model(images).argmax(dim=-1).cpu()
    model.clear_cache()

    columns = 4
    rows = (len(images) + columns - 1) // columns
    _, axes = plt.subplots(rows, columns, figsize=(10, 2.5 * rows))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(images):
            continue
        target = int(targets[index].argmax())
        prediction = int(predictions[index])
        axis.imshow(images[index].cpu().reshape(28, 28), cmap="gray")
        axis.set_title(
            f"true {target} | pred {prediction}",
            color="green" if target == prediction else "red",
        )
    plt.suptitle("Locally learned MNIST predictions")
    plt.tight_layout()


def main():
    set_seed()
    layer_sizes = [784, *HIDDEN_SIZES, 10]
    train_dataloader, test_dataloader = get_mnist_dataloaders()

    model = LocalActorCriticMNIST(layer_sizes).to(DEVICE)
    actor_optimizer = make_optimizer(
        OPTIMIZER_NAME,
        model.actor_parameters(),
        lr=ACTOR_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    critic_optimizer = make_optimizer(
        OPTIMIZER_NAME,
        model.critic_parameters(),
        lr=CRITIC_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Device: {DEVICE}")
    print(f"Layer sizes: {layer_sizes}")
    print(
        "Learning: "
        f"optimizer={OPTIMIZER_NAME}, actor_lr={ACTOR_LEARNING_RATE}, "
        f"critic_lr={CRITIC_LEARNING_RATE}, critic_steps={CRITIC_STEPS}, "
        f"actor_steps={ACTOR_STEPS}, critic_bias=False"
    )

    history = {
        "train_reward": [],
        "test_reward": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "critic_loss": [],
    }
    for epoch in range(1, NUM_EPOCHS + 1):
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=DEVICE,
            epoch_index=epoch,
            critic_steps=CRITIC_STEPS,
            actor_steps=ACTOR_STEPS,
            grad_clip=GRAD_CLIP,
        )
        test_stats = evaluate(model, test_dataloader, DEVICE, epoch)
        history["train_reward"].append(train_stats.reward)
        history["test_reward"].append(test_stats.reward)
        history["train_accuracy"].append(train_stats.accuracy)
        history["test_accuracy"].append(test_stats.accuracy)
        history["critic_loss"].append(train_stats.critic_loss)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} | "
            f"train reward {train_stats.reward:.4f} acc {train_stats.accuracy:.4f} | "
            f"test reward {test_stats.reward:.4f} acc {test_stats.accuracy:.4f} | "
            f"critic mse {train_stats.critic_loss:.5f} | "
            f"critic pred {train_stats.critic_reward_pred:.4f} | "
            f"actor pred {train_stats.actor_reward_pred:.4f}"
        )

    plot_history(history)
    plot_predictions(model, test_dataloader)
    plt.show()


if __name__ == "__main__":
    main()
