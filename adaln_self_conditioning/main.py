from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


@dataclass
class ClassificationStats:
    loss: float
    accuracy: float
    auxiliary_loss: float = 0.0
    expert_usage: tuple[float, ...] | None = None


@dataclass
class DataBundle:
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    input_dim: int
    image_shape: tuple[int, int, int]
    num_classes: int
    display_name: str


class DenseFFN(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            residual_init_std: float,
    ):
        super().__init__()
        self.proj_up = nn.Linear(dim, hidden_dim)
        self.proj_down = nn.Linear(hidden_dim, dim)
        init_small_residual_projection(self.proj_down, residual_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_down(F.silu(self.proj_up(x)))


class SelfConditionedAdaLNFFN(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            conditioning_init_std: float,
            residual_init_std: float,
            conditioning_mode: str,
    ):
        super().__init__()
        validate_conditioning_mode(conditioning_mode)
        self.conditioning_mode = conditioning_mode
        self.proj_up = nn.Linear(dim, hidden_dim)
        self.gamma = (
            nn.Linear(dim, hidden_dim)
            if conditioning_mode in {"gamma", "gamma_beta"}
            else None
        )
        self.beta = (
            nn.Linear(dim, hidden_dim)
            if conditioning_mode in {"beta", "gamma_beta"}
            else None
        )
        self.proj_down = nn.Linear(hidden_dim, dim)

        if self.gamma is not None:
            init_conditioning_projection(self.gamma, conditioning_init_std)
        if self.beta is not None:
            init_conditioning_projection(self.beta, conditioning_init_std)
        init_small_residual_projection(self.proj_down, residual_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj_up(x)
        conditioned = projected
        if self.gamma is not None:
            conditioned = (1.0 + self.gamma(x)) * conditioned
        if self.beta is not None:
            conditioned = conditioned + self.beta(x)
        return self.proj_down(F.silu(conditioned))


class Top1MoEFFN(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            num_experts: int,
            residual_init_std: float,
            router_noise_std: float,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least 2.")

        self.num_experts = num_experts
        self.router_noise_std = router_noise_std
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                DenseFFN(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    residual_init_std=residual_init_std,
                )
                for _ in range(num_experts)
            ]
        )
        self.last_load_balance_loss: torch.Tensor | None = None
        self.last_router_z_loss: torch.Tensor | None = None
        self.last_tokens_per_expert: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_x = x.reshape(-1, x.size(-1))
        router_logits = self.router(flat_x)
        if self.training and self.router_noise_std > 0.0:
            router_logits = router_logits + self.router_noise_std * torch.randn_like(
                router_logits
            )

        router_probs = router_logits.softmax(dim=-1)
        expert_indices = router_probs.argmax(dim=-1)
        expert_one_hot = F.one_hot(
            expert_indices,
            num_classes=self.num_experts,
        ).to(router_probs.dtype)
        selected_gates = (router_probs * expert_one_hot).sum(dim=-1, keepdim=True)

        combined = flat_x.new_zeros(flat_x.shape)
        for expert_index, expert in enumerate(self.experts):
            selected = expert_indices == expert_index
            if bool(selected.any()):
                combined[selected] = expert(flat_x[selected]) * selected_gates[selected]

        tokens_per_expert = expert_one_hot.mean(dim=0)
        router_prob_per_expert = router_probs.mean(dim=0)
        self.last_tokens_per_expert = tokens_per_expert.detach()
        self.last_load_balance_loss = self.num_experts * torch.sum(
            tokens_per_expert.detach() * router_prob_per_expert
        )
        self.last_router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
        return combined.reshape_as(x)

    def auxiliary_loss(
            self,
            load_balance_weight: float,
            router_z_loss_weight: float,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        loss = parameter.new_zeros(())
        if self.last_load_balance_loss is not None:
            loss = loss + load_balance_weight * self.last_load_balance_loss
        if self.last_router_z_loss is not None:
            loss = loss + router_z_loss_weight * self.last_router_z_loss
        return loss


class ConvDenseFFN(nn.Module):
    def __init__(
            self,
            channels: int,
            hidden_channels: int,
            residual_init_std: float,
    ):
        super().__init__()
        self.proj_up = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.proj_down = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        init_small_residual_projection(self.proj_down, residual_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_down(F.silu(self.proj_up(x)))

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        up_weight = self.proj_up.weight.flatten(start_dim=1)
        down_weight = self.proj_down.weight.flatten(start_dim=1)
        x = F.linear(x, up_weight, self.proj_up.bias)
        x = F.silu(x)
        return F.linear(x, down_weight, self.proj_down.bias)


class ConvSelfConditionedAdaLNFFN(nn.Module):
    def __init__(
            self,
            channels: int,
            hidden_channels: int,
            conditioning_init_std: float,
            residual_init_std: float,
            conditioning_mode: str,
    ):
        super().__init__()
        validate_conditioning_mode(conditioning_mode)
        self.conditioning_mode = conditioning_mode
        self.proj_up = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.gamma = (
            nn.Conv2d(channels, hidden_channels, kernel_size=1)
            if conditioning_mode in {"gamma", "gamma_beta"}
            else None
        )
        self.beta = (
            nn.Conv2d(channels, hidden_channels, kernel_size=1)
            if conditioning_mode in {"beta", "gamma_beta"}
            else None
        )
        self.proj_down = nn.Conv2d(hidden_channels, channels, kernel_size=1)

        if self.gamma is not None:
            init_conditioning_projection(self.gamma, conditioning_init_std)
        if self.beta is not None:
            init_conditioning_projection(self.beta, conditioning_init_std)
        init_small_residual_projection(self.proj_down, residual_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj_up(x)
        conditioned = projected
        if self.gamma is not None:
            conditioned = (1.0 + self.gamma(x)) * conditioned
        if self.beta is not None:
            conditioned = conditioned + self.beta(x)
        return self.proj_down(F.silu(conditioned))


class Top1ConvMoEFFN(nn.Module):
    def __init__(
            self,
            channels: int,
            hidden_channels: int,
            num_experts: int,
            residual_init_std: float,
            router_noise_std: float,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least 2.")

        self.num_experts = num_experts
        self.router_noise_std = router_noise_std
        self.router = nn.Conv2d(channels, num_experts, kernel_size=1)
        self.experts = nn.ModuleList(
            [
                ConvDenseFFN(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    residual_init_std=residual_init_std,
                )
                for _ in range(num_experts)
            ]
        )
        self.last_load_balance_loss: torch.Tensor | None = None
        self.last_router_z_loss: torch.Tensor | None = None
        self.last_tokens_per_expert: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        router_logits = self.router(x)
        if self.training and self.router_noise_std > 0.0:
            router_logits = router_logits + self.router_noise_std * torch.randn_like(
                router_logits
            )

        flat_x = x.permute(0, 2, 3, 1).reshape(-1, channels)
        flat_logits = router_logits.permute(0, 2, 3, 1).reshape(-1, self.num_experts)
        router_probs = flat_logits.softmax(dim=-1)
        expert_indices = router_probs.argmax(dim=-1)
        expert_one_hot = F.one_hot(
            expert_indices,
            num_classes=self.num_experts,
        ).to(router_probs.dtype)
        selected_gates = (router_probs * expert_one_hot).sum(dim=-1, keepdim=True)

        combined = flat_x.new_zeros(flat_x.shape)
        for expert_index, expert in enumerate(self.experts):
            selected = expert_indices == expert_index
            if bool(selected.any()):
                combined[selected] = (
                        expert.forward_tokens(flat_x[selected]) * selected_gates[selected]
                )

        tokens_per_expert = expert_one_hot.mean(dim=0)
        router_prob_per_expert = router_probs.mean(dim=0)
        self.last_tokens_per_expert = tokens_per_expert.detach()
        self.last_load_balance_loss = self.num_experts * torch.sum(
            tokens_per_expert.detach() * router_prob_per_expert
        )
        self.last_router_z_loss = torch.logsumexp(flat_logits, dim=-1).square().mean()
        return combined.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

    def auxiliary_loss(
            self,
            load_balance_weight: float,
            router_z_loss_weight: float,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        loss = parameter.new_zeros(())
        if self.last_load_balance_loss is not None:
            loss = loss + load_balance_weight * self.last_load_balance_loss
        if self.last_router_z_loss is not None:
            loss = loss + router_z_loss_weight * self.last_router_z_loss
        return loss


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, ffn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, ffn: nn.Module):
        super().__init__()
        self.spatial_norm = ChannelLayerNorm(channels)
        self.spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.ffn_norm = ChannelLayerNorm(channels)
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.spatial(self.spatial_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class ResidualMLPClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            model_dim: int,
            depth: int,
            ffn_kind: str,
            expansion: int,
            conditioning_init_std: float,
            residual_init_std: float,
            conditioning_mode: str = "gamma_beta",
            num_experts: int = 2,
            router_noise_std: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if model_dim < 1:
            raise ValueError("model_dim must be positive.")
        if expansion < 1:
            raise ValueError("expansion must be positive.")

        hidden_dim = model_dim * expansion
        self.flatten = nn.Flatten()
        self.stem = nn.Linear(input_dim, model_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim=model_dim,
                    ffn=make_ffn(
                        ffn_kind=ffn_kind,
                        dim=model_dim,
                        hidden_dim=hidden_dim,
                        conditioning_init_std=conditioning_init_std,
                        residual_init_std=residual_init_std,
                        conditioning_mode=conditioning_mode,
                        num_experts=num_experts,
                        router_noise_std=router_noise_std,
                    ),
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(self.flatten(images))
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))

    def auxiliary_loss(
            self,
            moe_load_balance_weight: float,
            moe_router_z_loss_weight: float,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        losses = [
            module.auxiliary_loss(
                load_balance_weight=moe_load_balance_weight,
                router_z_loss_weight=moe_router_z_loss_weight,
            )
            for module in self.modules()
            if isinstance(module, Top1MoEFFN)
        ]
        if not losses:
            return parameter.new_zeros(())
        return torch.stack(losses).sum()

    def moe_expert_usage(self) -> torch.Tensor | None:
        usages = [
            module.last_tokens_per_expert
            for module in self.modules()
            if isinstance(module, Top1MoEFFN)
               and module.last_tokens_per_expert is not None
        ]
        if not usages:
            return None
        return torch.stack(usages).mean(dim=0)


class ResidualCNNClassifier(nn.Module):
    def __init__(
            self,
            image_shape: tuple[int, int, int],
            num_classes: int,
            model_dim: int,
            depth: int,
            ffn_kind: str,
            expansion: int,
            conditioning_init_std: float,
            residual_init_std: float,
            conditioning_mode: str = "gamma_beta",
            num_experts: int = 2,
            router_noise_std: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if model_dim < 1:
            raise ValueError("model_dim must be positive.")
        if expansion < 1:
            raise ValueError("expansion must be positive.")

        in_channels, _, _ = image_shape
        hidden_channels = model_dim * expansion
        self.stem = nn.Conv2d(in_channels, model_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    channels=model_dim,
                    ffn=make_conv_ffn(
                        ffn_kind=ffn_kind,
                        channels=model_dim,
                        hidden_channels=hidden_channels,
                        conditioning_init_std=conditioning_init_std,
                        residual_init_std=residual_init_std,
                        conditioning_mode=conditioning_mode,
                        num_experts=num_experts,
                        router_noise_std=router_noise_std,
                    ),
                )
                for _ in range(depth)
            ]
        )
        self.pool_after = make_pool_points(depth)
        self.final_norm = ChannelLayerNorm(model_dim)
        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        for block_index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if block_index in self.pool_after and min(x.shape[-2:]) > 1:
                x = F.avg_pool2d(x, kernel_size=2)
        x = self.final_norm(x).mean(dim=(-2, -1))
        return self.head(x)

    def auxiliary_loss(
            self,
            moe_load_balance_weight: float,
            moe_router_z_loss_weight: float,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        losses = [
            module.auxiliary_loss(
                load_balance_weight=moe_load_balance_weight,
                router_z_loss_weight=moe_router_z_loss_weight,
            )
            for module in self.modules()
            if isinstance(module, Top1ConvMoEFFN)
        ]
        if not losses:
            return parameter.new_zeros(())
        return torch.stack(losses).sum()

    def moe_expert_usage(self) -> torch.Tensor | None:
        usages = [
            module.last_tokens_per_expert
            for module in self.modules()
            if isinstance(module, Top1ConvMoEFFN)
               and module.last_tokens_per_expert is not None
        ]
        if not usages:
            return None
        return torch.stack(usages).mean(dim=0)


def make_ffn(
        ffn_kind: str,
        dim: int,
        hidden_dim: int,
        conditioning_init_std: float,
        residual_init_std: float,
        conditioning_mode: str,
        num_experts: int,
        router_noise_std: float,
) -> nn.Module:
    if ffn_kind == "dense":
        return DenseFFN(
            dim=dim,
            hidden_dim=hidden_dim,
            residual_init_std=residual_init_std,
        )
    if ffn_kind == "adaln":
        return SelfConditionedAdaLNFFN(
            dim=dim,
            hidden_dim=hidden_dim,
            conditioning_init_std=conditioning_init_std,
            residual_init_std=residual_init_std,
            conditioning_mode=conditioning_mode,
        )
    if ffn_kind == "moe":
        return Top1MoEFFN(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            residual_init_std=residual_init_std,
            router_noise_std=router_noise_std,
        )
    raise ValueError(f"Unsupported ffn_kind: {ffn_kind}")


def make_conv_ffn(
        ffn_kind: str,
        channels: int,
        hidden_channels: int,
        conditioning_init_std: float,
        residual_init_std: float,
        conditioning_mode: str,
        num_experts: int,
        router_noise_std: float,
) -> nn.Module:
    if ffn_kind == "dense":
        return ConvDenseFFN(
            channels=channels,
            hidden_channels=hidden_channels,
            residual_init_std=residual_init_std,
        )
    if ffn_kind == "adaln":
        return ConvSelfConditionedAdaLNFFN(
            channels=channels,
            hidden_channels=hidden_channels,
            conditioning_init_std=conditioning_init_std,
            residual_init_std=residual_init_std,
            conditioning_mode=conditioning_mode,
        )
    if ffn_kind == "moe":
        return Top1ConvMoEFFN(
            channels=channels,
            hidden_channels=hidden_channels,
            num_experts=num_experts,
            residual_init_std=residual_init_std,
            router_noise_std=router_noise_std,
        )
    raise ValueError(f"Unsupported ffn_kind: {ffn_kind}")


def validate_conditioning_mode(conditioning_mode: str):
    if conditioning_mode not in {"gamma_beta", "gamma", "beta"}:
        raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")


def make_pool_points(depth: int) -> set[int]:
    if depth < 3:
        return set()
    return {max(1, depth // 3), max(1, (2 * depth) // 3)}


def init_conditioning_projection(layer: nn.Module, std: float):
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_small_residual_projection(layer: nn.Module, std: float):
    nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def set_seed(seed: int, deterministic: bool):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def canonical_dataset_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    aliases = {
        "fashionmnist": "fashion_mnist",
        "fashion_mnist": "fashion_mnist",
        "mnist": "mnist",
        "cifar10": "cifar10",
        "cifar_10": "cifar10",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dataset: {name}")
    return aliases[normalized]


def make_transform(dataset_name: str, train: bool) -> transforms.Compose:
    if dataset_name in {"mnist", "fashion_mnist"}:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    if dataset_name == "cifar10":
        augmentations = []
        if train:
            augmentations.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        return transforms.Compose(
            [
                *augmentations,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def make_dataset(
        dataset_name: str,
        data_dir: Path,
        train: bool,
        download: bool,
) -> Dataset:
    transform = make_transform(dataset_name, train=train)
    if dataset_name == "mnist":
        return datasets.MNIST(
            root=str(data_dir),
            train=train,
            download=download,
            transform=transform,
        )
    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(
            root=str(data_dir),
            train=train,
            download=download,
            transform=transform,
        )
    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=str(data_dir),
            train=train,
            download=download,
            transform=transform,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def maybe_subset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None:
        return dataset

    limit = min(limit, len(dataset))
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def get_dataloaders(
        dataset_name: str,
        data_dir: Path,
        batch_size: int,
        train_limit: int | None,
        test_limit: int | None,
        num_workers: int,
        seed: int,
        download: bool,
) -> DataBundle:
    dataset_name = canonical_dataset_name(dataset_name)
    train_dataset = make_dataset(dataset_name, data_dir, train=True, download=download)
    test_dataset = make_dataset(dataset_name, data_dir, train=False, download=download)

    sample_image, _ = train_dataset[0]
    input_dim = int(sample_image.numel())
    image_shape = tuple(int(size) for size in sample_image.shape)
    num_classes = len(getattr(train_dataset, "classes", range(10)))
    display_name = {
        "mnist": "MNIST",
        "fashion_mnist": "FashionMNIST",
        "cifar10": "CIFAR-10",
    }[dataset_name]

    train_dataset = maybe_subset(train_dataset, train_limit, seed=seed)
    test_dataset = maybe_subset(test_dataset, test_limit, seed=seed + 10_000)

    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return DataBundle(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        input_dim=input_dim,
        image_shape=image_shape,
        num_classes=num_classes,
        display_name=display_name,
    )


def dense_total_parameter_count(
        input_dim: int,
        num_classes: int,
        model_dim: int,
        depth: int,
        expansion: int,
) -> int:
    hidden_dim = model_dim * expansion
    stem = input_dim * model_dim + model_dim
    dense_ffn = model_dim * hidden_dim + hidden_dim + hidden_dim * model_dim + model_dim
    block = 2 * model_dim + dense_ffn
    final_norm = 2 * model_dim
    head = model_dim * num_classes + num_classes
    return stem + depth * block + final_norm + head


def adaln_total_parameter_count(
        input_dim: int,
        num_classes: int,
        model_dim: int,
        depth: int,
        expansion: int,
) -> int:
    hidden_dim = model_dim * expansion
    stem = input_dim * model_dim + model_dim
    conditioning = 3 * (model_dim * hidden_dim + hidden_dim)
    projection_down = hidden_dim * model_dim + model_dim
    block = 2 * model_dim + conditioning + projection_down
    final_norm = 2 * model_dim
    head = model_dim * num_classes + num_classes
    return stem + depth * block + final_norm + head


def cnn_dense_total_parameter_count(
        image_shape: tuple[int, int, int],
        num_classes: int,
        model_dim: int,
        depth: int,
        expansion: int,
) -> int:
    in_channels, _, _ = image_shape
    hidden_channels = model_dim * expansion
    stem = in_channels * model_dim * 3 * 3 + model_dim
    spatial = model_dim * 3 * 3 + model_dim
    norms = 4 * model_dim
    dense_ffn = (
            model_dim * hidden_channels
            + hidden_channels
            + hidden_channels * model_dim
            + model_dim
    )
    block = norms + spatial + dense_ffn
    final_norm = 2 * model_dim
    head = model_dim * num_classes + num_classes
    return stem + depth * block + final_norm + head


def cnn_adaln_total_parameter_count(
        image_shape: tuple[int, int, int],
        num_classes: int,
        model_dim: int,
        depth: int,
        expansion: int,
) -> int:
    in_channels, _, _ = image_shape
    hidden_channels = model_dim * expansion
    stem = in_channels * model_dim * 3 * 3 + model_dim
    spatial = model_dim * 3 * 3 + model_dim
    norms = 4 * model_dim
    conditioning = 3 * (model_dim * hidden_channels + hidden_channels)
    projection_down = hidden_channels * model_dim + model_dim
    block = norms + spatial + conditioning + projection_down
    final_norm = 2 * model_dim
    head = model_dim * num_classes + num_classes
    return stem + depth * block + final_norm + head


def infer_wide_dense_model_dim(
        base_model_dim: int,
        target_params: int,
        parameter_counter,
) -> int:
    max_dim = max(base_model_dim * 3, 2)
    while parameter_counter(max_dim) < target_params and max_dim < base_model_dim * 32:
        max_dim *= 2

    best_dim = 1
    best_delta = None

    for candidate_dim in range(1, max_dim + 1):
        candidate_params = parameter_counter(candidate_dim)
        delta = abs(candidate_params - target_params)
        if best_delta is None or delta < best_delta:
            best_dim = candidate_dim
            best_delta = delta

    return best_dim


def make_classifier(
        architecture: str,
        input_dim: int,
        image_shape: tuple[int, int, int],
        num_classes: int,
        model_dim: int,
        depth: int,
        ffn_kind: str,
        expansion: int,
        conditioning_init_std: float,
        residual_init_std: float,
        conditioning_mode: str,
        num_experts: int,
        router_noise_std: float,
) -> nn.Module:
    if architecture == "mlp":
        return ResidualMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            model_dim=model_dim,
            depth=depth,
            ffn_kind=ffn_kind,
            expansion=expansion,
            conditioning_init_std=conditioning_init_std,
            residual_init_std=residual_init_std,
            conditioning_mode=conditioning_mode,
            num_experts=num_experts,
            router_noise_std=router_noise_std,
        )
    if architecture == "cnn":
        return ResidualCNNClassifier(
            image_shape=image_shape,
            num_classes=num_classes,
            model_dim=model_dim,
            depth=depth,
            ffn_kind=ffn_kind,
            expansion=expansion,
            conditioning_init_std=conditioning_init_std,
            residual_init_std=residual_init_std,
            conditioning_mode=conditioning_mode,
            num_experts=num_experts,
            router_noise_std=router_noise_std,
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def make_models(
        architecture: str,
        input_dim: int,
        image_shape: tuple[int, int, int],
        num_classes: int,
        model_dim: int,
        depth: int,
        base_expansion: int,
        matched_expansion: int,
        conditioning_init_std: float,
        residual_init_std: float,
        include_wide_dense: bool,
        include_moe: bool,
        include_adaln_ablations: bool,
        num_experts: int,
        router_noise_std: float,
) -> dict[str, nn.Module]:
    if matched_expansion == base_expansion:
        raise ValueError("matched_expansion must differ from base_expansion.")

    name_prefix = "" if architecture == "mlp" else f"{architecture}_"
    models = {
        f"{name_prefix}dense_{base_expansion}x": make_classifier(
            architecture=architecture,
            input_dim=input_dim,
            image_shape=image_shape,
            num_classes=num_classes,
            model_dim=model_dim,
            depth=depth,
            ffn_kind="dense",
            expansion=base_expansion,
            conditioning_init_std=conditioning_init_std,
            residual_init_std=residual_init_std,
            conditioning_mode="gamma_beta",
            num_experts=num_experts,
            router_noise_std=router_noise_std,
        ),
        f"{name_prefix}adaln_{base_expansion}x": make_classifier(
            architecture=architecture,
            input_dim=input_dim,
            image_shape=image_shape,
            num_classes=num_classes,
            model_dim=model_dim,
            depth=depth,
            ffn_kind="adaln",
            expansion=base_expansion,
            conditioning_init_std=conditioning_init_std,
            residual_init_std=residual_init_std,
            conditioning_mode="gamma_beta",
            num_experts=num_experts,
            router_noise_std=router_noise_std,
        ),
        f"{name_prefix}dense_{matched_expansion}x": make_classifier(
            architecture=architecture,
            input_dim=input_dim,
            image_shape=image_shape,
            num_classes=num_classes,
            model_dim=model_dim,
            depth=depth,
            ffn_kind="dense",
            expansion=matched_expansion,
            conditioning_init_std=conditioning_init_std,
            residual_init_std=residual_init_std,
            conditioning_mode="gamma_beta",
            num_experts=num_experts,
            router_noise_std=router_noise_std,
        ),
    }

    if include_adaln_ablations:
        for conditioning_mode in ("beta", "gamma"):
            models[f"{name_prefix}adaln_{conditioning_mode}_{base_expansion}x"] = (
                make_classifier(
                    architecture=architecture,
                    input_dim=input_dim,
                    image_shape=image_shape,
                    num_classes=num_classes,
                    model_dim=model_dim,
                    depth=depth,
                    ffn_kind="adaln",
                    expansion=base_expansion,
                    conditioning_init_std=conditioning_init_std,
                    residual_init_std=residual_init_std,
                    conditioning_mode=conditioning_mode,
                    num_experts=num_experts,
                    router_noise_std=router_noise_std,
                )
            )

    if include_wide_dense:
        if architecture == "mlp":
            target_params = adaln_total_parameter_count(
                input_dim=input_dim,
                num_classes=num_classes,
                model_dim=model_dim,
                depth=depth,
                expansion=base_expansion,
            )

            def parameter_counter(candidate_dim: int) -> int:
                return dense_total_parameter_count(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    model_dim=candidate_dim,
                    depth=depth,
                    expansion=base_expansion,
                )

        elif architecture == "cnn":
            target_params = cnn_adaln_total_parameter_count(
                image_shape=image_shape,
                num_classes=num_classes,
                model_dim=model_dim,
                depth=depth,
                expansion=base_expansion,
            )

            def parameter_counter(candidate_dim: int) -> int:
                return cnn_dense_total_parameter_count(
                    image_shape=image_shape,
                    num_classes=num_classes,
                    model_dim=candidate_dim,
                    depth=depth,
                    expansion=base_expansion,
                )

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        wide_model_dim = infer_wide_dense_model_dim(
            base_model_dim=model_dim,
            target_params=target_params,
            parameter_counter=parameter_counter,
        )
        models[f"{name_prefix}dense_{base_expansion}x_wide_{wide_model_dim}d"] = (
            make_classifier(
                architecture=architecture,
                input_dim=input_dim,
                image_shape=image_shape,
                num_classes=num_classes,
                model_dim=wide_model_dim,
                depth=depth,
                ffn_kind="dense",
                expansion=base_expansion,
                conditioning_init_std=conditioning_init_std,
                residual_init_std=residual_init_std,
                conditioning_mode="gamma_beta",
                num_experts=num_experts,
                router_noise_std=router_noise_std,
            )
        )

    if include_moe:
        models[f"{name_prefix}moe_{num_experts}e_{base_expansion}x"] = make_classifier(
            architecture=architecture,
            input_dim=input_dim,
            image_shape=image_shape,
            num_classes=num_classes,
            model_dim=model_dim,
            depth=depth,
            ffn_kind="moe",
            expansion=base_expansion,
            conditioning_init_std=conditioning_init_std,
            residual_init_std=residual_init_std,
            conditioning_mode="gamma_beta",
            num_experts=num_experts,
            router_noise_std=router_noise_std,
        )

    return models


def make_optimizer(
        model: nn.Module,
        lr: float,
        weight_decay: float,
        optimizer_name: str,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def train_one_epoch(
        models: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        dataloader: DataLoader,
        device: torch.device,
        epoch_index: int,
        grad_clip: float | None,
        moe_load_balance_weight: float,
        moe_router_z_loss_weight: float,
) -> dict[str, ClassificationStats]:
    totals = {
        name: {
            "loss": 0.0,
            "auxiliary_loss": 0.0,
            "correct": 0.0,
            "count": 0,
            "expert_usage": None,
        }
        for name in models
    }

    for model in models.values():
        model.train()

    progress = tqdm(dataloader, desc=f"TRAIN E{epoch_index}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)
        postfix = {}

        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            task_loss = F.cross_entropy(logits, targets)
            auxiliary_loss = model.auxiliary_loss(
                moe_load_balance_weight=moe_load_balance_weight,
                moe_router_z_loss_weight=moe_router_z_loss_weight,
            )
            loss = task_loss + auxiliary_loss
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = predictions.eq(targets).sum().item()
                totals[name]["loss"] += float(task_loss.detach().cpu()) * batch_size
                totals[name]["auxiliary_loss"] += (
                        float(auxiliary_loss.detach().cpu()) * batch_size
                )
                totals[name]["correct"] += correct
                totals[name]["count"] += batch_size
                expert_usage = model.moe_expert_usage()
                if expert_usage is not None:
                    expert_usage = expert_usage.detach().cpu()
                    if totals[name]["expert_usage"] is None:
                        totals[name]["expert_usage"] = torch.zeros_like(expert_usage)
                    totals[name]["expert_usage"] += expert_usage * batch_size
                postfix[f"{name}_acc"] = f"{correct / batch_size:.3f}"

        progress.set_postfix(postfix)

    stats = {}
    for name, values in totals.items():
        expert_usage = values["expert_usage"]
        if expert_usage is not None:
            expert_usage = tuple((expert_usage / values["count"]).tolist())
        stats[name] = ClassificationStats(
            loss=values["loss"] / values["count"],
            accuracy=values["correct"] / values["count"],
            auxiliary_loss=values["auxiliary_loss"] / values["count"],
            expert_usage=expert_usage,
        )
    return stats


@torch.no_grad()
def evaluate(
        models: dict[str, nn.Module],
        dataloader: DataLoader,
        device: torch.device,
        epoch_index: int,
) -> dict[str, ClassificationStats]:
    totals = {
        name: {"loss": 0.0, "correct": 0.0, "count": 0, "expert_usage": None}
        for name in models
    }

    for model in models.values():
        model.eval()

    progress = tqdm(dataloader, desc=f"TEST  E{epoch_index}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = targets.size(0)
        postfix = {}

        for name, model in models.items():
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(targets).sum().item()

            totals[name]["loss"] += float(loss.cpu()) * batch_size
            totals[name]["correct"] += correct
            totals[name]["count"] += batch_size
            expert_usage = model.moe_expert_usage()
            if expert_usage is not None:
                expert_usage = expert_usage.detach().cpu()
                if totals[name]["expert_usage"] is None:
                    totals[name]["expert_usage"] = torch.zeros_like(expert_usage)
                totals[name]["expert_usage"] += expert_usage * batch_size
            postfix[name] = f"{correct / batch_size:.3f}"

        progress.set_postfix(postfix)

    stats = {}
    for name, values in totals.items():
        expert_usage = values["expert_usage"]
        if expert_usage is not None:
            expert_usage = tuple((expert_usage / values["count"]).tolist())
        stats[name] = ClassificationStats(
            loss=values["loss"] / values["count"],
            accuracy=values["correct"] / values["count"],
            expert_usage=expert_usage,
        )
    return stats


def make_history(model_names: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }
        for name in model_names
    }


def append_history(
        history: dict[str, dict[str, list[float]]],
        train_stats: dict[str, ClassificationStats],
        test_stats: dict[str, ClassificationStats],
):
    for name in history:
        history[name]["train_loss"].append(train_stats[name].loss)
        history[name]["test_loss"].append(test_stats[name].loss)
        history[name]["train_accuracy"].append(train_stats[name].accuracy)
        history[name]["test_accuracy"].append(test_stats[name].accuracy)


def brief_epoch_summary(
        epoch: int,
        total_epochs: int,
        test_stats: dict[str, ClassificationStats],
) -> str:
    parts = []
    for name, stats in test_stats.items():
        usage = ""
        if stats.expert_usage is not None:
            usage_values = ",".join(f"{value:.2f}" for value in stats.expert_usage)
            usage = f", use [{usage_values}]"
        parts.append(f"{name}: loss {stats.loss:.4f}, acc {stats.accuracy:.4f}{usage}")
    return f"Epoch {epoch:02d}/{total_epochs:02d} | " + " | ".join(parts)


def print_parameter_table(models: dict[str, nn.Module]):
    print("Model parameter counts:")
    for name, model in models.items():
        block_params = count_parameters(model.blocks)
        total_params = count_parameters(model)
        print(
            f"  {name:>10} | blocks {block_params:>10,} | "
            f"total {total_params:>10,}"
        )


def plot_history(
        history: dict[str, dict[str, list[float]]],
        title: str,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(next(iter(history.values()))["train_loss"]) + 1)
    panels = [
        (axes[0], "loss", "Cross-entropy loss"),
        (axes[1], "accuracy", "Classification accuracy"),
    ]

    for ax, metric, label in panels:
        for name, values in history.items():
            ax.plot(epochs, values[f"train_{metric}"], marker="o", label=f"{name} train")
            ax.plot(
                epochs,
                values[f"test_{metric}"],
                marker="o",
                linestyle="--",
                label=f"{name} test",
            )
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()


def show_plots():
    import matplotlib.pyplot as plt

    if plt.get_backend().lower() == "agg":
        for figure_number in plt.get_fignums():
            plt.figure(figure_number).canvas.draw()
    else:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Residual MLP/CNN comparison for self-conditioned AdaLN FFN blocks "
            "versus dense and MoE FFNs with similar parameter count."
        )
    )
    parser.add_argument("--architecture", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist", "fashion-mnist", "cifar10", "cifar-10"],
        default="cifar10",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--base-expansion", type=int, default=4)
    parser.add_argument(
        "--matched-expansion",
        type=int,
        default=None,
        help="Dense comparison expansion. Defaults to 2 * --base-expansion.",
    )
    parser.add_argument("--conditioning-init-std", type=float, default=1e-2)
    parser.add_argument("--residual-init-std", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--moe-load-balance-weight", type=float, default=1e-2)
    parser.add_argument("--moe-router-z-loss-weight", type=float, default=1e-3)
    parser.add_argument("--moe-router-noise-std", type=float, default=0.0)
    parser.add_argument("--adaln-ablations", action="store_true")
    parser.add_argument("--no-wide-dense", action="store_true")
    parser.add_argument("--no-moe", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace):
    matched_expansion = args.matched_expansion
    if matched_expansion is None:
        matched_expansion = args.base_expansion * 2

    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    data = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        num_workers=args.num_workers,
        seed=args.seed,
        download=not args.no_download,
    )

    models = make_models(
        architecture=args.architecture,
        input_dim=data.input_dim,
        image_shape=data.image_shape,
        num_classes=data.num_classes,
        model_dim=args.model_dim,
        depth=args.depth,
        base_expansion=args.base_expansion,
        matched_expansion=matched_expansion,
        conditioning_init_std=args.conditioning_init_std,
        residual_init_std=args.residual_init_std,
        include_wide_dense=not args.no_wide_dense,
        include_moe=not args.no_moe,
        include_adaln_ablations=args.adaln_ablations,
        num_experts=args.num_experts,
        router_noise_std=args.moe_router_noise_std,
    )
    models = {name: model.to(device) for name, model in models.items()}
    optimizers = {
        name: make_optimizer(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
        )
        for name, model in models.items()
    }

    print(f"Device: {device}")
    print(
        f"Dataset: {data.display_name} | input_dim={data.input_dim} | "
        f"image_shape={data.image_shape} | classes={data.num_classes}"
    )
    print(
        f"Architecture: {args.architecture} | dim={args.model_dim}, "
        f"depth={args.depth}, "
        f"base_expansion={args.base_expansion}, "
        f"matched_expansion={matched_expansion}"
    )
    if not args.no_moe:
        print(
            f"MoE: experts={args.num_experts}, "
            f"load_balance_weight={args.moe_load_balance_weight:g}, "
            f"router_z_loss_weight={args.moe_router_z_loss_weight:g}, "
            f"router_noise_std={args.moe_router_noise_std:g}"
        )
    print_parameter_table(models)

    history = make_history(list(models))
    final_test_stats = None

    for epoch in range(1, args.num_epochs + 1):
        train_stats = train_one_epoch(
            models=models,
            optimizers=optimizers,
            dataloader=data.train_dataloader,
            device=device,
            epoch_index=epoch,
            grad_clip=args.grad_clip,
            moe_load_balance_weight=args.moe_load_balance_weight,
            moe_router_z_loss_weight=args.moe_router_z_loss_weight,
        )
        test_stats = evaluate(
            models=models,
            dataloader=data.test_dataloader,
            device=device,
            epoch_index=epoch,
        )
        final_test_stats = test_stats
        append_history(history, train_stats, test_stats)
        print(brief_epoch_summary(epoch, args.num_epochs, test_stats))

    if final_test_stats is not None:
        best_name = max(final_test_stats, key=lambda name: final_test_stats[name].accuracy)
        best_stats = final_test_stats[best_name]
        print(
            f"Best final test accuracy: {best_name} "
            f"({best_stats.accuracy:.4f}, loss {best_stats.loss:.4f})"
        )

    if not args.no_plots:
        title = (
            f"{data.display_name} residual {args.architecture.upper()} | "
            f"dim={args.model_dim}, "
            f"depth={args.depth}, base={args.base_expansion}x, "
            f"matched={matched_expansion}x"
        )
        plot_history(history, title)
        show_plots()


if __name__ == "__main__":
    main(get_args())
