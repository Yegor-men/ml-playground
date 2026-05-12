import argparse
import copy
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms


@dataclass
class LossStats:
    loss: float
    pos_loss: float
    neg_loss: float
    pos_sim: float
    neg_sim: float


class BalancedClassBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: list[int],
        classes_per_batch: int,
        samples_per_class: int,
        batches_per_epoch: int,
        seed: int,
    ):
        self.labels = labels
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0

        class_to_indices = defaultdict(list)
        for index, label in enumerate(labels):
            class_to_indices[int(label)].append(index)

        self.class_to_indices = {
            label: indices
            for label, indices in class_to_indices.items()
            if len(indices) >= samples_per_class
        }
        self.classes = sorted(self.class_to_indices)
        if len(self.classes) < classes_per_batch:
            raise ValueError(
                f"Need at least {classes_per_batch} classes with "
                f"{samples_per_class} samples each, got {len(self.classes)}."
            )

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        for _ in range(self.batches_per_epoch):
            batch = []
            labels = rng.sample(self.classes, self.classes_per_batch)
            for label in labels:
                batch.extend(rng.sample(self.class_to_indices[label], self.samples_per_class))
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


class GlyphEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class TwinGlyphModel(nn.Module):
    def __init__(self, embedding_dim: int, clone_second_tower: bool):
        super().__init__()
        self.left = GlyphEncoder(embedding_dim)
        self.right = copy.deepcopy(self.left) if clone_second_tower else GlyphEncoder(embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.left(x), self.right(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self(x)
        return F.normalize(0.5 * (left + right), p=2, dim=-1)


def get_omniglot_labels(dataset: datasets.Omniglot) -> list[int]:
    if hasattr(dataset, "_flat_character_images"):
        return [int(label) for _, label in dataset._flat_character_images]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def get_omniglot_dataloaders(
    data_dir: Path,
    image_size: int,
    classes_per_batch: int,
    samples_per_class: int,
    train_batches_per_epoch: int,
    eval_batches: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, datasets.Omniglot]:
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.Omniglot(
        root=str(data_dir),
        background=True,
        download=True,
        transform=transform,
    )
    eval_dataset = datasets.Omniglot(
        root=str(data_dir),
        background=False,
        download=True,
        transform=transform,
    )

    train_sampler = BalancedClassBatchSampler(
        labels=get_omniglot_labels(train_dataset),
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        batches_per_epoch=train_batches_per_epoch,
        seed=seed,
    )
    eval_sampler = BalancedClassBatchSampler(
        labels=get_omniglot_labels(eval_dataset),
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        batches_per_epoch=eval_batches,
        seed=seed + 10_000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader, eval_dataset


def pairwise_cosine_matrix_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: float,
    negative_weight: float,
) -> tuple[torch.Tensor, LossStats]:
    similarities = left @ right.T
    same_class = labels[:, None].eq(labels[None, :])
    different_class = ~same_class

    pos_loss = (similarities[same_class] - 1.0).pow(2).mean()
    neg_loss = (similarities[different_class] + 1.0).pow(2).mean()
    loss = positive_weight * pos_loss + negative_weight * neg_loss

    stats = LossStats(
        loss=float(loss.detach().cpu()),
        pos_loss=float(pos_loss.detach().cpu()),
        neg_loss=float(neg_loss.detach().cpu()),
        pos_sim=float(similarities[same_class].detach().mean().cpu()),
        neg_sim=float(similarities[different_class].detach().mean().cpu()),
    )
    return loss, stats


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def set_requires_grad(model: nn.Module, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def mean_stats(stats: list[LossStats]) -> LossStats:
    count = len(stats)
    return LossStats(
        loss=sum(stat.loss for stat in stats) / count,
        pos_loss=sum(stat.pos_loss for stat in stats) / count,
        neg_loss=sum(stat.neg_loss for stat in stats) / count,
        pos_sim=sum(stat.pos_sim for stat in stats) / count,
        neg_sim=sum(stat.neg_sim for stat in stats) / count,
    )


def train_one_epoch(
    models: dict[str, TwinGlyphModel],
    ema_models: dict[str, TwinGlyphModel],
    optimizers: dict[str, torch.optim.Optimizer],
    train_loader: DataLoader,
    device: torch.device,
    ema_decay: float,
    positive_weight: float,
    negative_weight: float,
    epoch: int,
) -> dict[str, LossStats]:
    from tqdm import tqdm

    history = {name: [] for name in models}
    for model in models.values():
        model.train()

    progress = tqdm(train_loader, desc=f"train e{epoch}", total=len(train_loader))
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        postfix = {}
        for name, model in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            left, right = model(images)
            loss, stats = pairwise_cosine_matrix_loss(
                left=left,
                right=right,
                labels=labels,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
            )
            loss.backward()
            optimizers[name].step()
            update_ema(ema_models[name], model, ema_decay)
            history[name].append(stats)
            postfix[name] = f"{stats.loss:.3f}"
        progress.set_postfix(postfix)

    return {name: mean_stats(stats) for name, stats in history.items()}


@torch.no_grad()
def evaluate(
    models: dict[str, TwinGlyphModel],
    eval_loader: DataLoader,
    device: torch.device,
    positive_weight: float,
    negative_weight: float,
    epoch: int,
) -> dict[str, LossStats]:
    from tqdm import tqdm

    history = {name: [] for name in models}
    for model in models.values():
        model.eval()

    for images, labels in tqdm(eval_loader, desc=f"eval e{epoch}", total=len(eval_loader)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for name, model in models.items():
            left, right = model(images)
            _, stats = pairwise_cosine_matrix_loss(
                left=left,
                right=right,
                labels=labels,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
            )
            history[name].append(stats)

    return {name: mean_stats(stats) for name, stats in history.items()}


def append_history(
    history: dict[str, dict[str, list[float]]],
    train_stats: dict[str, LossStats],
    eval_stats: dict[str, LossStats],
):
    for name in train_stats:
        history[name]["train_loss"].append(train_stats[name].loss)
        history[name]["eval_loss"].append(eval_stats[name].loss)
        history[name]["train_pos_sim"].append(train_stats[name].pos_sim)
        history[name]["train_neg_sim"].append(train_stats[name].neg_sim)
        history[name]["eval_pos_sim"].append(eval_stats[name].pos_sim)
        history[name]["eval_neg_sim"].append(eval_stats[name].neg_sim)


def plot_loss_curves(
    history: dict[str, dict[str, list[float]]],
    output_dir: Path,
    show_plots: bool,
    save_plots: bool,
):
    import matplotlib.pyplot as plt

    epochs = range(1, len(next(iter(history.values()))["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], "train_loss", "Training loss"),
        (axes[0, 1], "eval_loss", "Evaluation loss"),
        (axes[1, 0], "eval_pos_sim", "Evaluation positive cosine similarity"),
        (axes[1, 1], "eval_neg_sim", "Evaluation negative cosine similarity"),
    ]

    for ax, metric, title in panels:
        for name, values in history.items():
            ax.plot(epochs, values[metric], marker="o", label=name)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(f"Omniglot twin encoder metrics after epoch {len(next(iter(history.values()))['train_loss'])}")
    fig.tight_layout()
    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "training_metrics.png", dpi=160)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


@torch.no_grad()
def collect_diagnostic_batch(
    dataset: datasets.Omniglot,
    classes_per_batch: int,
    samples_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampler = BalancedClassBatchSampler(
        labels=get_omniglot_labels(dataset),
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        batches_per_epoch=1,
        seed=seed,
    )
    indices = next(iter(sampler))
    images, labels = zip(*(dataset[index] for index in indices))
    return torch.stack(list(images)), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def plot_nearest_neighbor_diagnostics(
    model: TwinGlyphModel,
    name: str,
    dataset: datasets.Omniglot,
    device: torch.device,
    output_dir: Path,
    classes: int,
    samples_per_class: int,
    query_count: int,
    neighbor_count: int,
    seed: int,
    show_plots: bool,
    save_plots: bool,
):
    import matplotlib.pyplot as plt

    model.eval()
    images, labels = collect_diagnostic_batch(
        dataset=dataset,
        classes_per_batch=classes,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    embeddings = model.embed(images.to(device)).cpu()
    similarities = embeddings @ embeddings.T
    nearest_similarities = similarities.clone()
    nearest_similarities.fill_diagonal_(float("-inf"))

    rng = random.Random(seed)
    query_indices = rng.sample(range(len(images)), min(query_count, len(images)))

    columns = min(neighbor_count, len(images) - 1) + 1
    rows = len(query_indices)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.55, max(2.0, rows * 1.75)))
    if rows == 1:
        axes = axes[None, :]

    for row, query_index in enumerate(query_indices):
        neighbor_scores, neighbor_indices = nearest_similarities[query_index].topk(columns - 1)
        shown_images = [(query_index, 1.0)]
        shown_images.extend(
            (int(image_index), float(score))
            for image_index, score in zip(neighbor_indices.tolist(), neighbor_scores.tolist())
        )

        query_label = int(labels[query_index])
        for col, (image_index, score) in enumerate(shown_images):
            ax = axes[row, col]
            ax.imshow(images[image_index].squeeze(0), cmap="gray")
            image_label = int(labels[image_index])
            if col == 0:
                title = f"query\nclass {image_label}"
            else:
                marker = "same" if image_label == query_label else "diff"
                title = f"top {col} ({marker})\nclass {image_label}, cos {score:.2f}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    fig.suptitle(f"{name} EMA top-{columns - 1} eval nearest neighbors", fontsize=13)
    fig.tight_layout()
    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{name}_top_neighbors.png", dpi=170)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser(description="Omniglot twin-network glyph embedding experiment")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_dir", type=Path, default=Path("data"))
    data_group.add_argument("--image_size", type=int, default=64)
    data_group.add_argument("--classes_per_batch", type=int, default=16)
    data_group.add_argument("--samples_per_class", type=int, default=4)
    data_group.add_argument("--train_batches_per_epoch", type=int, default=300)
    data_group.add_argument("--eval_batches", type=int, default=80)
    data_group.add_argument("--num_workers", type=int, default=2)

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--embedding_dim", type=int, default=64)

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=20)
    train_group.add_argument("--lr", type=float, default=1e-4)
    train_group.add_argument("--weight_decay", type=float, default=1e-4)
    train_group.add_argument("--ema_decay", type=float, default=0.999)
    train_group.add_argument("--positive_weight", type=float, default=0.5)
    train_group.add_argument("--negative_weight", type=float, default=0.5)

    diagnostics_group = parser.add_argument_group("Diagnostics")
    diagnostics_group.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    diagnostics_group.add_argument("--diagnostic_classes", type=int, default=8)
    diagnostics_group.add_argument("--diagnostic_samples_per_class", type=int, default=4)
    diagnostics_group.add_argument("--diagnostic_queries", type=int, default=6)
    diagnostics_group.add_argument("--diagnostic_neighbors", type=int, default=10)
    diagnostics_group.add_argument("--save_plots", action="store_true")
    diagnostics_group.add_argument("--no_show_plots", action="store_false", dest="show_plots")
    diagnostics_group.set_defaults(show_plots=True)

    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument("--seed", type=int, default=0)
    misc_group.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")

    return parser.parse_args()


def main(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    try:
        import matplotlib.pyplot  # noqa: F401
        import tqdm  # noqa: F401
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "This experiment needs matplotlib and tqdm for plots/progress bars. "
            "Install them, then rerun siamese/main.py."
        ) from error

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    batch_size = args.classes_per_batch * args.samples_per_class
    print(f"Device: {device}")
    print(f"Batch shape: {args.classes_per_batch} classes x {args.samples_per_class} samples = {batch_size}")

    train_loader, eval_loader, eval_dataset = get_omniglot_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        train_batches_per_epoch=args.train_batches_per_epoch,
        eval_batches=args.eval_batches,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    models = {
        "deepcopy": TwinGlyphModel(args.embedding_dim, clone_second_tower=True).to(device),
        "random": TwinGlyphModel(args.embedding_dim, clone_second_tower=False).to(device),
    }
    ema_models = {name: copy.deepcopy(model).to(device) for name, model in models.items()}
    for ema_model in ema_models.values():
        set_requires_grad(ema_model, False)
        ema_model.eval()

    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for name, model in models.items()
    }

    history = {
        name: {
            "train_loss": [],
            "eval_loss": [],
            "train_pos_sim": [],
            "train_neg_sim": [],
            "eval_pos_sim": [],
            "eval_neg_sim": [],
        }
        for name in models
    }

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            models=models,
            ema_models=ema_models,
            optimizers=optimizers,
            train_loader=train_loader,
            device=device,
            ema_decay=args.ema_decay,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
            epoch=epoch,
        )
        eval_stats = evaluate(
            models=ema_models,
            eval_loader=eval_loader,
            device=device,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
            epoch=epoch,
        )
        append_history(history, train_stats, eval_stats)
        plot_loss_curves(
            history=history,
            output_dir=args.output_dir,
            show_plots=args.show_plots,
            save_plots=args.save_plots,
        )

        summary = []
        for name in models:
            summary.append(
                f"{name}: train={train_stats[name].loss:.4f}, eval={eval_stats[name].loss:.4f}, "
                f"eval pos cos={eval_stats[name].pos_sim:.3f}, eval neg cos={eval_stats[name].neg_sim:.3f}"
            )
        print(f"Epoch {epoch}: " + " | ".join(summary))

    for name, model in ema_models.items():
        plot_nearest_neighbor_diagnostics(
            model=model,
            name=name,
            dataset=eval_dataset,
            device=device,
            output_dir=args.output_dir,
            classes=args.diagnostic_classes,
            samples_per_class=args.diagnostic_samples_per_class,
            query_count=args.diagnostic_queries,
            neighbor_count=args.diagnostic_neighbors,
            seed=args.seed + 20_000,
            show_plots=args.show_plots,
            save_plots=args.save_plots,
        )

    if args.save_plots:
        print(f"Plots saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main(get_args())
