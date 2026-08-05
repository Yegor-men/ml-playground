import copy
import hashlib
import math
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

D_MODEL = 512
NUM_HEADS = (1, 2, 4, 8, D_MODEL)
NUM_BLOCKS = 5
FFN_DIM = 4 * D_MODEL
DROPOUT = 0.1

# WikiText-2 next-byte language modelling. A byte vocabulary keeps the output
# head compact and makes tokenization deterministic and dependency-free.
VOCAB_SIZE = 256
SEQUENCE_LENGTH = 128
BATCH_SIZE = 8
TRAIN_STEPS_PER_EPOCH = 500
VALIDATION_BATCHES = 100
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EMA_DECAY = 0.999
SEED = 0

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "wikitext-2"
WIKITEXT_BASE_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/"
    "d5678bc8ac0cdd79dbd5e44d4130271018bcec4e/"
    "word_language_model/data/wikitext-2"
)
WIKITEXT_FILES = {
    "train": {
        "filename": "train.txt",
        "url": f"{WIKITEXT_BASE_URL}/train.txt",
        "sha256": (
            "9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f"
        ),
    },
    "validation": {
        "filename": "valid.txt",
        "url": f"{WIKITEXT_BASE_URL}/valid.txt",
        "sha256": (
            "f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8"
        ),
    },
    "test": {
        "filename": "test.txt",
        "url": f"{WIKITEXT_BASE_URL}/test.txt",
        "sha256": (
            "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0"
        ),
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_wikitext(data_directory: Path = DATA_DIRECTORY):
    data_directory.mkdir(parents=True, exist_ok=True)

    for metadata in WIKITEXT_FILES.values():
        path = data_directory / metadata["filename"]
        if path.exists() and file_sha256(path) == metadata["sha256"]:
            continue

        temporary_path = path.with_suffix(path.suffix + ".download")
        print(f"Downloading WikiText-2: {metadata['filename']}")
        try:
            urllib.request.urlretrieve(metadata["url"], temporary_path)
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {metadata['url']}"
            ) from error

        actual_sha256 = file_sha256(temporary_path)
        if actual_sha256 != metadata["sha256"]:
            temporary_path.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {metadata['filename']}: "
                f"expected {metadata['sha256']}, got {actual_sha256}"
            )
        temporary_path.replace(path)


def load_byte_tokens(path: Path) -> torch.Tensor:
    data = bytearray(path.read_bytes())
    return torch.frombuffer(data, dtype=torch.uint8).long()


def load_wikitext(data_directory: Path = DATA_DIRECTORY):
    download_wikitext(data_directory)
    return {
        split: load_byte_tokens(
            data_directory / metadata["filename"]
        )
        for split, metadata in WIKITEXT_FILES.items()
    }


def sample_text_batch(
        tokens: torch.Tensor,
        generator: torch.Generator,
        batch_size: int = BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - SEQUENCE_LENGTH - 1
    if max_start < 0:
        raise ValueError(
            f"Corpus has {tokens.numel()} bytes, but a sequence needs "
            f"{SEQUENCE_LENGTH + 1}"
        )

    starts = torch.randint(
        max_start + 1,
        (batch_size,),
        generator=generator,
    )
    offsets = torch.arange(SEQUENCE_LENGTH + 1)
    windows = tokens[starts[:, None] + offsets[None, :]]
    return windows[:, :-1], windows[:, 1:]


def make_evaluation_batches(tokens: torch.Tensor, seed: int):
    generator = torch.Generator().manual_seed(seed)
    batches = [
        sample_text_batch(tokens, generator)
        for _ in range(VALIDATION_BATCHES)
    ]
    return tuple(zip(*batches))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads "
                f"({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.attention_dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        x = x.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            d_model,
        )
        return self.output_dropout(self.output(x))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(d_model, ffn_dim)
        self.activation = nn.SiLU()
        self.hidden_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(ffn_dim, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.activation(x)
        x = self.hidden_dropout(x)
        x = self.output(x)
        return self.output_dropout(x)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            ffn_dim: int,
            dropout: float,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ByteLanguageModel(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, D_MODEL)
        self.embedding_dropout = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=D_MODEL,
                num_heads=num_heads,
                ffn_dim=FFN_DIM,
                dropout=DROPOUT,
            )
            for _ in range(NUM_BLOCKS)
        ])
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight

        # Start every block as an exact identity map. On the first update only
        # these output projections learn; gradients then begin flowing into the
        # attention and FFN internals as the residual branches open up.
        for block in self.blocks:
            nn.init.zeros_(block.attention.output.weight)
            nn.init.zeros_(block.ffn.output.weight)

    @staticmethod
    def _initialize(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = input_ids.size(1)
        if sequence_length > SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"the configured maximum of {SEQUENCE_LENGTH}"
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        x = (
                self.token_embedding(input_ids)
                + self.position_embedding(positions)[None, :, :]
        )
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def make_models():
    # num_heads only changes tensor reshaping, not parameter shapes. Cloning one
    # state dict therefore gives every experiment bit-identical initialization.
    template = ByteLanguageModel(num_heads=NUM_HEADS[0])
    initial_state = copy.deepcopy(template.state_dict())

    models = {}
    for num_heads in NUM_HEADS:
        model = ByteLanguageModel(num_heads=num_heads)
        model.load_state_dict(initial_state)
        models[num_heads] = model.to(device)
    return models


def make_ema_models(models):
    ema_models = {
        num_heads: copy.deepcopy(model).eval()
        for num_heads, model in models.items()
    }
    for model in ema_models.values():
        model.requires_grad_(False)
    return ema_models


@torch.no_grad()
def update_ema_model(ema_model, model):
    model_parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        ema_parameter.lerp_(model_parameters[name], 1.0 - EMA_DECAY)

    # LayerNorm has no running statistics, but copying buffers keeps this helper
    # correct if a buffered layer is added later.
    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        ema_buffer.copy_(model_buffers[name])


@torch.no_grad()
def evaluate(models, batches):
    totals = {
        num_heads: {"loss": 0.0, "correct": 0}
        for num_heads in models
    }

    for model in models.values():
        model.eval()

    input_batches, target_batches = batches
    num_tokens = sum(targets.numel() for targets in target_batches)

    for input_ids, targets in zip(input_batches, target_batches):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        for num_heads, model in models.items():
            logits = model(input_ids)
            totals[num_heads]["loss"] += F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            ).item()
            totals[num_heads]["correct"] += (
                    logits.argmax(dim=-1) == targets
            ).sum().item()

    return {
        num_heads: {
            "loss": total["loss"] / num_tokens,
            "perplexity": math.exp(min(total["loss"] / num_tokens, 20.0)),
            "bits_per_byte": total["loss"] / num_tokens / math.log(2.0),
            "accuracy": total["correct"] / num_tokens,
        }
        for num_heads, total in totals.items()
    }


def plot_history(history: dict[int, dict[str, list[float]]]):
    epochs = range(1, NUM_EPOCHS + 1)
    _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for num_heads, values in history.items():
        axes[0].plot(
            epochs,
            values["train_loss"],
            marker="o",
            label=f"{num_heads} heads train",
        )
        axes[0].plot(
            epochs,
            values["validation_loss"],
            marker="o",
            linestyle="--",
            label=f"{num_heads} heads validation",
        )
        axes[1].plot(
            epochs,
            values["validation_accuracy"],
            marker="o",
            label=f"{num_heads} heads",
        )

    axes[0].set_title("Next-byte cross-entropy")
    axes[1].set_title("Validation next-byte accuracy")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    plt.tight_layout()


def printable_bytes(tokens: torch.Tensor, width: int = 96) -> str:
    raw = bytes(tokens[:width].tolist()).decode("utf-8", errors="replace")
    return "".join(character if character.isprintable() else "·" for character in raw)


@torch.no_grad()
def plot_test_predictions(models: dict[int, nn.Module], test_tokens: torch.Tensor):
    input_ids = test_tokens[:SEQUENCE_LENGTH].unsqueeze(0).to(device)
    target_ids = test_tokens[1: SEQUENCE_LENGTH + 1]
    rows = len(models) + 1
    figure, axes = plt.subplots(rows, 1, figsize=(15, 1.7 * rows))

    axes[0].text(
        0.0,
        0.5,
        printable_bytes(target_ids),
        family="monospace",
        fontsize=9,
        va="center",
    )
    axes[0].set_title("Actual next bytes")
    axes[0].axis("off")

    for axis, (num_heads, model) in zip(axes[1:], models.items()):
        model.eval()
        predicted_ids = model(input_ids).argmax(dim=-1).squeeze(0).cpu()
        accuracy = predicted_ids.eq(target_ids).float().mean().item()
        axis.text(
            0.0,
            0.5,
            printable_bytes(predicted_ids),
            family="monospace",
            fontsize=9,
            va="center",
        )
        axis.set_title(f"{num_heads} heads | next-byte accuracy {accuracy:.1%}")
        axis.axis("off")

    figure.suptitle("WikiText-2 test excerpt and model predictions")
    figure.tight_layout()


def main():
    print(f"Device: {device}")
    print(
        f"Model width: {D_MODEL}; heads: {NUM_HEADS}; "
        f"head dimensions: {tuple(D_MODEL // h for h in NUM_HEADS)}"
    )
    print(
        f"Task: WikiText-2 next-byte prediction; context "
        f"{SEQUENCE_LENGTH}; batch size {BATCH_SIZE}"
    )

    corpus = load_wikitext()
    validation_batches = make_evaluation_batches(
        corpus["validation"],
        seed=SEED + 2,
    )
    test_batches = make_evaluation_batches(
        corpus["test"],
        seed=SEED + 3,
    )
    print(
        f"Corpus bytes: train {corpus['train'].numel():,}; "
        f"validation {corpus['validation'].numel():,}; "
        f"test {corpus['test'].numel():,}"
    )

    models = make_models()
    ema_models = make_ema_models(models)
    optimizers = {
        num_heads: torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        for num_heads, model in models.items()
    }
    history = {
        num_heads: {
            "train_loss": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }
        for num_heads in models
    }

    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    }
    assert len(parameter_counts) == 1
    print(f"Parameters per model: {parameter_counts.pop():,}")

    train_generator = torch.Generator().manual_seed(SEED + 1)
    for epoch in range(1, NUM_EPOCHS + 1):
        for model in models.values():
            model.train()

        progress = tqdm(
            range(TRAIN_STEPS_PER_EPOCH),
            total=TRAIN_STEPS_PER_EPOCH,
            desc=f"TRAIN - E{epoch}",
        )
        epoch_loss_totals = {num_heads: 0.0 for num_heads in models}
        for _ in progress:
            input_ids, targets = sample_text_batch(
                corpus["train"],
                train_generator,
            )
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            for num_heads, model in models.items():
                optimizer = optimizers[num_heads]
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    targets.flatten(),
                )
                loss.backward()
                optimizer.step()
                update_ema_model(ema_models[num_heads], model)
                epoch_loss_totals[num_heads] += loss.item()

        validation_stats = evaluate(ema_models, validation_batches)
        for num_heads in models:
            history[num_heads]["train_loss"].append(
                epoch_loss_totals[num_heads] / TRAIN_STEPS_PER_EPOCH
            )
            history[num_heads]["validation_loss"].append(
                validation_stats[num_heads]["loss"]
            )
            history[num_heads]["validation_accuracy"].append(
                validation_stats[num_heads]["accuracy"]
            )
        stats = " | ".join(
            f"{num_heads:>3} heads: "
            f"loss {validation_stats[num_heads]['loss']:.4f}, "
            f"byte-ppl {validation_stats[num_heads]['perplexity']:.2f}, "
            f"bpb {validation_stats[num_heads]['bits_per_byte']:.3f}"
            for num_heads in models
        )
        print(f"EMA VALIDATION - E{epoch} | {stats}")

    test_stats = evaluate(ema_models, test_batches)
    stats = " | ".join(
        f"{num_heads:>3} heads: "
        f"loss {test_stats[num_heads]['loss']:.4f}, "
        f"byte-ppl {test_stats[num_heads]['perplexity']:.2f}, "
        f"bpb {test_stats[num_heads]['bits_per_byte']:.3f}"
        for num_heads in models
    )
    print(f"EMA TEST | {stats}")
    plot_history(history)
    plot_test_predictions(ema_models, corpus["test"])
    plt.show()


if __name__ == "__main__":
    main()
