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


@dataclass
class RetrievalStats:
    top1_same: float
    top3_any_same: float
    mrr: float
    mean_same_rank: float


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
) -> tuple[DataLoader, DataLoader, datasets.Omniglot, datasets.Omniglot]:
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
    return train_loader, eval_loader, train_dataset, eval_dataset


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


def retrieval_stats_from_embeddings(embeddings: torch.Tensor, labels: torch.Tensor) -> RetrievalStats:
    similarities = embeddings @ embeddings.T
    similarities.fill_diagonal_(float("-inf"))

    same_class = labels[:, None].eq(labels[None, :])
    same_class.fill_diagonal_(False)

    sorted_indices = similarities.argsort(dim=1, descending=True)
    sorted_same = same_class.gather(dim=1, index=sorted_indices)
    has_same = sorted_same.any(dim=1)
    first_same_rank = sorted_same.to(torch.float32).argmax(dim=1).to(torch.float32) + 1.0
    missing_rank = torch.full_like(first_same_rank, fill_value=float(embeddings.size(0)))
    first_same_rank = torch.where(has_same, first_same_rank, missing_rank)

    top3_width = min(3, max(1, embeddings.size(0) - 1))
    reciprocal_rank = torch.where(has_same, 1.0 / first_same_rank, torch.zeros_like(first_same_rank))

    return RetrievalStats(
        top1_same=float(sorted_same[:, 0].to(torch.float32).mean().cpu()),
        top3_any_same=float(sorted_same[:, :top3_width].any(dim=1).to(torch.float32).mean().cpu()),
        mrr=float(reciprocal_rank.mean().cpu()),
        mean_same_rank=float(first_same_rank.mean().cpu()),
    )


def mean_retrieval_stats(stats: list[RetrievalStats]) -> RetrievalStats:
    count = len(stats)
    return RetrievalStats(
        top1_same=sum(stat.top1_same for stat in stats) / count,
        top3_any_same=sum(stat.top3_any_same for stat in stats) / count,
        mrr=sum(stat.mrr for stat in stats) / count,
        mean_same_rank=sum(stat.mean_same_rank for stat in stats) / count,
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
) -> tuple[dict[str, LossStats], dict[str, RetrievalStats]]:
    from tqdm import tqdm

    loss_history = {name: [] for name in models}
    retrieval_history = {name: [] for name in models}
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
            embeddings = F.normalize(0.5 * (left + right), p=2, dim=-1)
            retrieval_stats = retrieval_stats_from_embeddings(embeddings, labels)
            loss_history[name].append(stats)
            retrieval_history[name].append(retrieval_stats)

    return (
        {name: mean_stats(stats) for name, stats in loss_history.items()},
        {name: mean_retrieval_stats(stats) for name, stats in retrieval_history.items()},
    )


def append_history(
    history: dict[str, dict[str, list[float]]],
    train_stats: dict[str, LossStats],
    eval_stats: dict[str, LossStats],
    retrieval_stats: dict[str, RetrievalStats],
):
    for name in train_stats:
        history[name]["train_loss"].append(train_stats[name].loss)
        history[name]["eval_loss"].append(eval_stats[name].loss)
        history[name]["train_pos_sim"].append(train_stats[name].pos_sim)
        history[name]["train_neg_sim"].append(train_stats[name].neg_sim)
        history[name]["eval_pos_sim"].append(eval_stats[name].pos_sim)
        history[name]["eval_neg_sim"].append(eval_stats[name].neg_sim)
        history[name]["eval_top1_same"].append(retrieval_stats[name].top1_same)
        history[name]["eval_top3_any_same"].append(retrieval_stats[name].top3_any_same)
        history[name]["eval_mrr"].append(retrieval_stats[name].mrr)
        history[name]["eval_mean_same_rank"].append(retrieval_stats[name].mean_same_rank)


def plot_loss_curves(
    history: dict[str, dict[str, list[float]]],
    output_dir: Path,
    show_plots: bool,
    save_plots: bool,
):
    import matplotlib.pyplot as plt

    epochs = range(1, len(next(iter(history.values()))["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    panels = [
        (axes[0, 0], "train_loss", "Training loss"),
        (axes[0, 1], "eval_loss", "Evaluation loss"),
        (axes[1, 0], "eval_pos_sim", "Evaluation positive cosine similarity"),
        (axes[1, 1], "eval_neg_sim", "Evaluation negative cosine similarity"),
        (axes[0, 2], "eval_top1_same", "Eval-batch retrieval top-1 same-class"),
        (axes[0, 3], "eval_top3_any_same", "Eval-batch retrieval top-3 has same-class"),
        (axes[1, 2], "eval_mrr", "Eval-batch retrieval mean reciprocal rank"),
        (axes[1, 3], "eval_mean_same_rank", "Eval-batch retrieval mean first same-class rank"),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sampler = BalancedClassBatchSampler(
        labels=get_omniglot_labels(dataset),
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        batches_per_epoch=1,
        seed=seed,
    )
    indices = next(iter(sampler))
    images, labels = zip(*(dataset[index] for index in indices))
    return torch.stack(list(images)), torch.tensor(labels, dtype=torch.long), torch.tensor(indices, dtype=torch.long)


@torch.no_grad()
def plot_cross_split_nearest_neighbor_diagnostics(
    model: TwinGlyphModel,
    name: str,
    train_dataset: datasets.Omniglot,
    eval_dataset: datasets.Omniglot,
    device: torch.device,
    output_dir: Path,
    train_classes: int,
    eval_classes: int,
    samples_per_class: int,
    query_count: int,
    neighbor_count: int,
    seed: int,
    show_plots: bool,
    save_plots: bool,
):
    import matplotlib.pyplot as plt

    model.eval()
    eval_images, eval_labels, eval_indices = collect_diagnostic_batch(
        dataset=eval_dataset,
        classes_per_batch=eval_classes,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    train_images, train_labels, train_indices = collect_diagnostic_batch(
        dataset=train_dataset,
        classes_per_batch=train_classes,
        samples_per_class=samples_per_class,
        seed=seed + 1,
    )

    gallery_images = torch.cat([eval_images, train_images], dim=0)
    gallery_labels = torch.cat([eval_labels, train_labels], dim=0)
    gallery_indices = torch.cat([eval_indices, train_indices], dim=0)
    gallery_is_eval = torch.cat(
        [
            torch.ones(len(eval_images), dtype=torch.bool),
            torch.zeros(len(train_images), dtype=torch.bool),
        ],
        dim=0,
    )

    query_embeddings = model.embed(eval_images.to(device)).cpu()
    gallery_embeddings = model.embed(gallery_images.to(device)).cpu()
    similarities = query_embeddings @ gallery_embeddings.T

    same_eval_example = gallery_is_eval[None, :] & gallery_indices[None, :].eq(eval_indices[:, None])
    similarities = similarities.masked_fill(same_eval_example, float("-inf"))

    rng = random.Random(seed)
    query_indices = rng.sample(range(len(eval_images)), min(query_count, len(eval_images)))

    columns = min(neighbor_count, len(gallery_images) - 1) + 1
    rows = len(query_indices)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.55, max(2.0, rows * 1.75)))
    if rows == 1:
        axes = axes[None, :]

    for row, query_index in enumerate(query_indices):
        neighbor_scores, neighbor_indices = similarities[query_index].topk(columns - 1)
        shown_images = [("query", query_index, 1.0)]
        shown_images.extend(
            ("neighbor", int(image_index), float(score))
            for image_index, score in zip(neighbor_indices.tolist(), neighbor_scores.tolist())
        )

        query_label = int(eval_labels[query_index])
        for col, (kind, image_index, score) in enumerate(shown_images):
            ax = axes[row, col]
            if kind == "query":
                ax.imshow(eval_images[image_index].squeeze(0), cmap="gray")
                title = f"eval query\nclass {query_label}"
            else:
                ax.imshow(gallery_images[image_index].squeeze(0), cmap="gray")
                image_label = int(gallery_labels[image_index])
                split = "eval" if bool(gallery_is_eval[image_index]) else "train"
                marker = "same" if split == "eval" and image_label == query_label else "diff"
                title = f"top {col} ({split}, {marker})\nclass {image_label}, cos {score:.2f}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    fig.suptitle(f"{name} EMA eval queries vs mixed train+eval gallery", fontsize=13)
    fig.tight_layout()
    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{name}_cross_split_neighbors.png", dpi=170)
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
    diagnostics_group.add_argument("--diagnostic_eval_classes", type=int, default=8)
    diagnostics_group.add_argument("--diagnostic_train_classes", type=int, default=32)
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

    train_loader, eval_loader, train_dataset, eval_dataset = get_omniglot_dataloaders(
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
            "eval_top1_same": [],
            "eval_top3_any_same": [],
            "eval_mrr": [],
            "eval_mean_same_rank": [],
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
        eval_stats, retrieval_stats = evaluate(
            models=ema_models,
            eval_loader=eval_loader,
            device=device,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
            epoch=epoch,
        )
        append_history(history, train_stats, eval_stats, retrieval_stats)
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
                f"eval pos cos={eval_stats[name].pos_sim:.3f}, eval neg cos={eval_stats[name].neg_sim:.3f}, "
                f"top1={retrieval_stats[name].top1_same:.3f}, top3={retrieval_stats[name].top3_any_same:.3f}, "
                f"mrr={retrieval_stats[name].mrr:.3f}, rank={retrieval_stats[name].mean_same_rank:.2f}"
            )
        print(f"Epoch {epoch}: " + " | ".join(summary))

    for name, model in ema_models.items():
        plot_cross_split_nearest_neighbor_diagnostics(
            model=model,
            name=name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            device=device,
            output_dir=args.output_dir,
            train_classes=args.diagnostic_train_classes,
            eval_classes=args.diagnostic_eval_classes,
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
