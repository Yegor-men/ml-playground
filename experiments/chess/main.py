"""Learn a legal-move chess policy from rated human games. Edit Config and Run.

Dependencies: torch, numpy, chess, matplotlib, tqdm.
Independent transformer blocks refine square states using 2D rotary attention.
The policy scores start/end vector dot products without tree search.
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from urllib.request import urlopen

import chess
import chess.pgn
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data/chess"
SQUARES = 64
MIN_RATING = 100
MAX_RATING = 4000


@dataclass
class Config:
    mode: str = "train"  # "train", "inspect", "test"
    device: str = "auto"
    seed: int = 42

    dim: int = 1024
    heads: int = 16
    blocks: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    checkpoint_passes: bool = True  # Recompute block activations in backward to limit memory.
    rules: str = "standard"  # Legacy king-capture checkpoints remain playable.
    temperature: float = 1.0
    batch_size: int = 512
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # The run name suffix separates this experiment from the original checkpoint.
    pretrain_checkpoint_path: str = ""
    pretrain_run_name: str = "elo_weighted"
    resume: bool = True
    show_plots: bool = True
    save_every_positions: int = 100_000  # At the next completed optimizer batch; also saves at epoch end.
    preview_positions: int = 2_048  # Fixed held-out prefix for fast intermediate validation/plots.
    pretrain_epochs: int | None = None  # Until Ctrl+C, or N additional full passes.
    pretrain_months: tuple[str, ...] = tuple(f"2013-{month:02d}" for month in range(1, 13))
    pretrain_pgn: str = ""  # Optional local PGN override, under data/.
    pretrain_url: str = ""  # Optional download URL for that override.
    pretrain_max_games: int | None = None  # Across all files; None uses all.
    pretrain_max_positions: int | None = None  # Optional total dataset limit.
    pretrain_shard_positions: int = 50_000  # RAM bound, not a dataset limit.
    pretrain_min_rating: int = MIN_RATING
    pretrain_validation_fraction: float = 0.1
    pretrain_learning_rate: float = 1e-4
    pretrain_imitation_weight: float = 1.0
    pretrain_advantage_weight: float = 1.0  # 0 gives behaviour cloning.
    pretrain_negative_log_cap: float = 8.0
    value_loss_weight: float = 1.0
    ema_decay: float = 0.999
    inspect_fen: str = chess.STARTING_FEN
    top_moves: int = 8


CONFIG = Config()
MODEL_VERSION = 10
LEGACY_MODEL_VERSION = 6
DATASET_VERSION = 3  # Store the final White score for position value training.
META_SIZE = 14
SIDE_TO_MOVE_INDEX = 13
NUM_ACTIONS = 64 * 64
ACTION_ENCODING = "from_square * 64 + to_square; automatic queen promotion"
ARCHITECTURE_KEYS = ("dim", "heads", "blocks", "mlp_ratio", "dropout")
MEAN_METRICS = ("loss", "nll", "accuracy", "value_bce")
OUTCOME_METRICS = (("winner_match", "winner_positions"), ("loser_disagree", "loser_positions"))


def base_checkpoint_path(cfg):
    return DATA_DIRECTORY / f"chess_rope_value_d{cfg.dim}_b{cfg.blocks}_h{cfg.heads}.pt"


def checkpoint_path(cfg):
    if cfg.pretrain_checkpoint_path:
        return Path(cfg.pretrain_checkpoint_path)
    path = base_checkpoint_path(cfg)
    return path.with_stem(f"{path.stem}_{cfg.pretrain_run_name}") if cfg.pretrain_run_name else path


def action_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def action_move(board: chess.Board, action: int) -> chess.Move:
    origin, destination = divmod(action, 64)
    promotion = (chess.QUEEN if board.piece_type_at(origin) == chess.PAWN
                                and chess.square_rank(destination) in (0, 7) else None)
    return chess.Move(origin, destination, promotion=promotion)


def encode_board(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    pieces = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        pieces[square] = piece.piece_type + (0 if piece.color else 6)
    metadata = np.zeros(META_SIZE, dtype=np.float32)
    metadata[:4] = (board.has_kingside_castling_rights(chess.WHITE),
                    board.has_queenside_castling_rights(chess.WHITE),
                    board.has_kingside_castling_rights(chess.BLACK),
                    board.has_queenside_castling_rights(chess.BLACK))
    if board.ep_square is not None:
        metadata[4 + chess.square_file(board.ep_square)] = 1
    metadata[12] = min(board.halfmove_clock, 100) / 100
    metadata[SIDE_TO_MOVE_INDEX] = 1 if board.turn else -1
    return pieces, metadata


def legal_moves(board: chess.Board, rules: str) -> list[chess.Move]:
    moves = board.legal_moves if rules == "standard" else board.pseudo_legal_moves
    return [move for move in moves if move.promotion in (None, chess.QUEEN)]


def game_result(board: chess.Board, rules: str) -> tuple[bool, bool | None]:
    if rules == "standard":
        outcome = board.outcome(claim_draw=True)
        return (False, None) if outcome is None else (True, outcome.winner)
    if board.king(chess.WHITE) is None:
        return True, chess.BLACK
    if board.king(chess.BLACK) is None:
        return True, chess.WHITE
    if board.halfmove_clock >= 100 or board.is_repetition(3) or not any(board.pseudo_legal_moves):
        return True, None
    return False, None


def two_dimensional_sinusoid(dim: int) -> torch.Tensor:
    """Deterministic position vectors, one half for file and one for rank."""
    half = dim // 2
    frequency = torch.exp(-math.log(10_000) * torch.arange(0, half, 2) / half)
    file = (torch.arange(64) % 8).float()
    rank = (torch.arange(64) // 8).float()

    def encode(coordinates: torch.Tensor) -> torch.Tensor:
        phases = coordinates[:, None] * frequency[None, :]
        return torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(-2)

    return torch.cat((encode(file), encode(rank)), dim=-1)  # [64, dim]


def rope_tables(head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """2D rotary angles: first half files, second half ranks."""
    half = head_dim // 2
    frequency = torch.exp(-math.log(10_000) * torch.arange(0, half, 2) / half)
    file = (torch.arange(64) % 8).float()
    rank = (torch.arange(64) // 8).float()
    phases = torch.cat((file[:, None] * frequency, rank[:, None] * frequency), dim=-1)
    phases = phases.repeat_interleave(2, dim=-1)
    return phases.cos()[None, None], phases.sin()[None, None]


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


# ============================================================================
# 64 ATTENDED SQUARES -> START/END VECTOR PROJECTIONS -> 64x64 MOVE LOGITS
# ============================================================================

class BoardAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        if self.head_dim % 4:
            raise ValueError("2D rotary attention requires a head dimension divisible by 4")
        cos, sin = rope_tables(self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, squares, dim = x.shape
        qkv = self.qkv(x).reshape(batch, squares, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [batch, heads, 64, head_dim]
        q = q * self.cos + rotate_pairs(q) * self.sin
        k = k * self.cos + rotate_pairs(k) * self.sin
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, squares, dim))


class BoardBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, dropout = cfg.dim, cfg.dropout
        self.norm1 = nn.LayerNorm(dim)
        self.attention = BoardAttention(dim, cfg.heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * cfg.mlp_ratio)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feedforward(self.norm2(x))


class ChessPolicy(nn.Module):
    """One shared policy scoring every origin/destination square pair."""

    def __init__(self, cfg: Config, legacy_blocks=None):
        super().__init__()
        if cfg.heads <= 0 or cfg.dim <= 0 or cfg.dim % cfg.heads:
            raise ValueError("dim must be positive and divisible by heads")
        self.checkpoint_passes = cfg.checkpoint_passes
        self.piece_type = nn.Embedding(7, cfg.dim)
        self.white_projection = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.black_projection = nn.Linear(cfg.dim, cfg.dim, bias=False)
        nn.init.eye_(self.white_projection.weight)
        nn.init.eye_(self.black_projection.weight)
        self.has_value = legacy_blocks is None
        self.register_buffer("positions", two_dimensional_sinusoid(cfg.dim), persistent=False)
        self.extra_state = nn.Sequential(nn.Linear(META_SIZE, cfg.dim), nn.Tanh())
        blocks = cfg.blocks if self.has_value else legacy_blocks
        if blocks < 1:
            raise ValueError("blocks must be positive")
        self.layers = nn.ModuleList(BoardBlock(cfg) for _ in range(blocks))
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.start_cell = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.end_cell = nn.Linear(cfg.dim, cfg.dim, bias=False)
        for head in (self.start_cell, self.end_cell):
            nn.init.normal_(head.weight, std=0.01)
        if self.has_value:
            self.white_king = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.black_king = nn.Linear(cfg.dim, cfg.dim, bias=False)
            for head in (self.white_king, self.black_king):
                nn.init.normal_(head.weight, std=0.01)

    def forward(self, pieces: torch.Tensor, metadata: torch.Tensor, *, with_value=False):
        kinds = torch.where(pieces <= 6, pieces, pieces - 6).long()
        base = self.piece_type(kinds)
        coloured = torch.where((pieces > 6)[..., None], self.black_projection(base),
                               self.white_projection(base))
        coloured = torch.where((pieces == 0)[..., None], base, coloured)
        x = coloured + self.positions[None] + self.extra_state(metadata)[:, None]
        for layer in self.layers:
            if self.has_value and self.training and self.checkpoint_passes and torch.is_grad_enabled():
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.final_norm(x)
        start = self.start_cell(x)  # [batch, 64, dim]
        end = self.end_cell(x)  # [batch, 64, dim]
        # Each row is an origin, each column a destination. Scaling keeps
        # logit magnitudes stable as the embedding dimension changes.
        logits = start @ end.transpose(-2, -1) / math.sqrt(start.shape[-1])
        logits = logits.flatten(1)
        if not with_value:
            return logits
        rows = torch.arange(len(pieces), device=pieces.device)
        white = self.white_king(x[rows, (pieces == chess.KING).long().argmax(dim=1)])
        black = self.black_king(x[rows, (pieces == chess.KING + 6).long().argmax(dim=1)])
        value_logit = (white * black).sum(dim=-1) / math.sqrt(white.shape[-1])
        return logits, value_logit


def encode_batch(boards: list[chess.Board], device: torch.device):
    encoded = [encode_board(b) for b in boards]
    return (torch.as_tensor(np.stack([x[0] for x in encoded]), dtype=torch.long, device=device),
            torch.as_tensor(np.stack([x[1] for x in encoded]), dtype=torch.float32, device=device))


def masked_policy(logits: torch.Tensor, masks: torch.Tensor, temperature: float):
    """Legal-move distribution for training and play."""
    scaled = (logits / temperature).masked_fill(~masks, torch.finfo(logits.dtype).min)
    log_probs = F.log_softmax(scaled, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return probs, log_probs, entropy


@dataclass
class Step:
    pieces: np.ndarray
    metadata: np.ndarray
    action: int
    legal: np.ndarray
    target: float = 0.0  # Game-level outcome minus rating-based expectation.
    rating: int = 100
    white_score: float = .5


def open_pgn(path, raw=None):
    """Read compressed PGNs without expanding a monthly archive on disk."""
    stream = raw if raw is not None else path
    if path.suffix == ".zst":
        try:
            from compression import zstd  # Python 3.14+
            return zstd.open(stream, "rt", encoding="utf-8")
        except ImportError:
            try:
                import zstandard
            except ImportError as exc:
                raise RuntimeError("For .zst PGNs use Python 3.14+, install zstandard, "
                                   "or decompress to .pgn and change pretrain_pgn.") from exc
            return io.TextIOWrapper(zstandard.open(stream, "rb"), encoding="utf-8")
    if path.suffix == ".gz":
        return gzip.open(stream, "rt", encoding="utf-8")
    if path.suffix == ".bz2":
        return bz2.open(stream, "rt", encoding="utf-8")
    return io.TextIOWrapper(raw, encoding="utf-8") if raw is not None else path.open("rt", encoding="utf-8")


def download_pgn(path, url):
    data_root = DATA_DIRECTORY.parent.resolve()
    if not path.resolve().is_relative_to(data_root):
        raise ValueError("Keep all PGNs inside the repository's shared data/ directory.")
    if path.is_file():
        return
    if not url:
        raise FileNotFoundError(f"Place a rated PGN at {path}, or configure its URL.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0)) or None
            with tqdm(total=total, desc=f"Download {path.name}", unit="B", unit_scale=True) as progress:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    progress.update(len(chunk))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def pgn_sources(cfg):
    if cfg.pretrain_pgn:
        return [(Path(cfg.pretrain_pgn), cfg.pretrain_url)]
    sources = []
    for month in cfg.pretrain_months:
        # Parse instead of interpolating arbitrary strings into paths/URLs.
        from datetime import datetime
        if datetime.strptime(month, "%Y-%m").strftime("%Y-%m") != month:
            raise ValueError("Use YYYY-MM months in pretrain_months.")
        name = f"lichess_db_standard_rated_{month}.pgn.zst"
        sources.append((DATA_DIRECTORY / name, f"https://database.lichess.org/standard/{name}"))
    if not sources:
        raise ValueError("Configure pretrain_months or a local pretrain_pgn.")
    return sources


def game_samples(game, cfg):
    """Return a whole supported game, or None. Both colours supply examples."""
    try:
        white, black = int(game.headers["WhiteElo"]), int(game.headers["BlackElo"])
        score = {"1-0": 1., "0-1": 0., "1/2-1/2": .5}[game.headers["Result"]]
        board = game.board()
        moves = list(game.mainline_moves())
        if (game.errors or game.headers.get("Variant", "Standard") != "Standard"
                or not board.is_valid() or board.chess960 or min(white, black) < cfg.pretrain_min_rating
                or max(white, black) > MAX_RATING or not moves
                or any(m.promotion not in (None, chess.QUEEN) for m in moves)):
            return None
    except (ValueError, KeyError):
        return None
    identity = board.fen() + " " + " ".join(m.uci() for m in moves)
    digest = hashlib.sha256(identity.encode()).digest()
    # PGN ratings lack Glicko-2 RD and volatility; this logistic baseline is approximate.
    expected = 1 / (1 + 10 ** ((black - white) / 400))
    advantage = 2 * (score - expected)
    samples, example = [], None
    for i, move in enumerate(moves):
        allowed = legal_moves(board, "standard")
        if move not in allowed:
            return None
        if i == len(moves) // 2:
            example = (board.fen(), move.uci())
        pieces, metadata = encode_board(board)
        samples.append(Step(pieces, metadata, action_index(move),
                            np.array([action_index(m) for m in allowed], dtype=np.uint16),
                            advantage if board.turn else -advantage, white if board.turn else black, score))
        board.push(move)
    return digest, samples, example


@dataclass
class PGNDataset:
    directory: Path
    manifest: dict

    @property
    def examples(self):
        return [(chess.Board(fen), chess.Move.from_uci(move)) for fen, move in self.manifest["examples"]]

    def batches(self, split, batch_size, shuffle=False, seed=None, start=0, limit=None):
        # Only one shard and one batch at a time are materialized in RAM.
        shards = list(self.manifest["shards"])
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(shards)
        remaining_skip, emitted = start, 0
        for name in shards:
            with np.load(self.directory / name, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            indices = np.flatnonzero(arrays["validation"] == (split == "validation"))
            if shuffle:
                rng.shuffle(indices)
            if remaining_skip >= len(indices):
                remaining_skip -= len(indices)
                continue
            indices = indices[remaining_skip:]
            remaining_skip = 0
            if limit is not None:
                indices = indices[:max(0, limit - emitted)]
            for start in range(0, len(indices), batch_size):
                rows = indices[start:start + batch_size]
                offsets, ids = arrays["offsets"], arrays["legal"]
                yield (arrays["pieces"][rows], arrays["metadata"][rows], arrays["actions"][rows],
                       arrays["targets"][rows], [ids[offsets[i]:offsets[i + 1]] for i in rows],
                       arrays["ratings"][rows], arrays["white_scores"][rows])
                emitted += len(rows)
            if limit is not None and emitted >= limit:
                return


def prepare_pretraining(cfg):
    """Import PGNs once into compact disk shards, deduplicating across all files."""
    if cfg.rules != "standard":
        raise ValueError("Human PGN training requires standard rules.")
    sources = pgn_sources(cfg)
    for path, url in sources:
        download_pgn(path, url)
    signature = dict(version=DATASET_VERSION,
                     files=[(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p, _ in sources],
                     settings={key: getattr(cfg, key) for key in (
                         "pretrain_max_games", "pretrain_max_positions", "pretrain_min_rating",
                         "pretrain_validation_fraction", "pretrain_shard_positions")})
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    directory = DATA_DIRECTORY / "pgn_cache" / fingerprint
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if all((directory / name).is_file() for name in manifest["shards"]):
            print(f"Cached PGNs: {manifest['counts']} | positions {manifest['positions']} | {directory}")
            return PGNDataset(directory, manifest)
    directory.mkdir(parents=True, exist_ok=True)
    counts = dict(scanned=0, accepted=0, skipped=0, duplicates=0)
    positions = dict(train=0, validation=0)
    buffer, held_out_flags, shards, examples, seen = [], [], [], [], set()

    def flush():
        if not buffer:
            return
        name = f"shard_{len(shards):05d}.npz"
        destination = directory / name
        temporary = destination.with_suffix(".tmp.npz")
        sizes = np.array([len(s.legal) for s in buffer], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        np.savez_compressed(temporary,
                            pieces=np.stack([s.pieces for s in buffer]),
                            metadata=np.stack([s.metadata for s in buffer]),
                            actions=np.array([s.action for s in buffer], dtype=np.uint16),
                            targets=np.array([s.target for s in buffer], dtype=np.float32),
                            ratings=np.array([s.rating for s in buffer], dtype=np.uint16),
                            white_scores=np.array([s.white_score for s in buffer], dtype=np.float32),
                            legal=np.concatenate([s.legal for s in buffer]), offsets=offsets,
                            validation=np.array(held_out_flags, dtype=np.bool_))
        temporary.replace(destination)
        shards.append(name)
        buffer.clear()
        held_out_flags.clear()

    if cfg.pretrain_max_games is not None:
        total, unit = cfg.pretrain_max_games, "game"
    elif cfg.pretrain_max_positions is not None:
        total, unit = cfg.pretrain_max_positions, "position"
    else:
        total, unit = sum(size for _, size, _ in signature["files"]), "B"

    stop = False
    previous_bytes = 0
    last_pending_byte = max(0, total - 1)
    with tqdm(total=total, desc="Cache rated games", unit=unit, unit_scale=unit == "B") as progress:
        for archive, ((path, _), (_, archive_size, _)) in enumerate(zip(sources, signature["files"]), 1):
            with path.open("rb") as raw, open_pgn(path, raw) as source:
                while not stop:
                    progress.set_postfix_str(f"archive {archive}/{len(sources)}, "
                                             f"{counts['scanned']:,} scanned, {counts['accepted']:,} kept",
                                             refresh=False)
                    if unit == "game":
                        completed = counts["accepted"]
                    elif unit == "position":
                        completed = sum(positions.values())
                    else:
                        # Buffered decompression can read ahead. Finish the bar after the last shard is written.
                        completed = min(previous_bytes + raw.tell(), last_pending_byte)
                    progress.update(completed - progress.n)
                    if cfg.pretrain_max_games is not None and counts["accepted"] >= cfg.pretrain_max_games:
                        stop = True
                        break
                    game = chess.pgn.read_game(source)
                    if game is None:
                        break
                    counts["scanned"] += 1
                    result = game_samples(game, cfg)
                    if result is None:
                        counts["skipped"] += 1
                        continue
                    digest, samples, example = result
                    if digest in seen:
                        counts["duplicates"] += 1
                        continue
                    size = len(samples)
                    if (cfg.pretrain_max_positions is not None
                            and sum(positions.values()) + size > cfg.pretrain_max_positions):
                        stop = True
                        break
                    # Duplicate games always have the same split, even across archives.
                    held_out = int.from_bytes(digest[:8], "big") / 2 ** 64 < cfg.pretrain_validation_fraction
                    buffer.extend(samples)
                    held_out_flags.extend([held_out] * size)
                    positions["validation" if held_out else "train"] += size
                    if held_out and len(examples) < 4:
                        examples.append(example)
                    seen.add(digest)
                    counts["accepted"] += 1
                    if len(buffer) >= cfg.pretrain_shard_positions:
                        flush()
            if stop:
                break
            previous_bytes += archive_size
        flush()
        progress.set_postfix_str(f"{counts['scanned']:,} scanned, {counts['accepted']:,} kept", refresh=False)
        if unit == "B":
            progress.update(total - progress.n)
        else:
            progress.total = progress.n
    if not positions["train"] or not positions["validation"]:
        raise ValueError("Need nonempty training and validation game splits; increase dataset limits.")
    manifest = dict(signature=signature, counts=counts, positions=positions, examples=examples, shards=shards)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(f"Cached PGNs: {counts} | positions {positions} | {directory}")
    return PGNDataset(directory, manifest)


def pretraining_loss(nll, advantages, cfg):
    # Negative-weight log likelihood alone is unbounded below. Cap only its
    # negative component and retain an ordinary imitation anchor on every move.
    surprise = torch.where(advantages >= 0, advantages * nll,
                           advantages * nll.clamp(max=cfg.pretrain_negative_log_cap))
    return cfg.pretrain_imitation_weight * nll + cfg.pretrain_advantage_weight * surprise


def pretraining_metrics(sums, positions):
    metrics = {key: sums[key] / max(1, positions) for key in MEAN_METRICS}
    for key, total in OUTCOME_METRICS:
        metrics[key] = sums[key] / sums[total] if sums.get(total, 0) else None
    return metrics


def outcome_rates(metrics):
    def percent(key):
        rate = metrics[key]
        return f"{rate:.1%}" if rate is not None else "n/a"

    return f"winner match {percent('winner_match')}, loser disagreement {percent('loser_disagree')}"


def pretraining_epoch(network, optimizer, dataset, device, cfg, state=None, on_batch=None, limit=None,
                      ema=None):
    training = optimizer is not None
    network.train(training)
    split = "train" if training else "validation"
    state = state if state is not None else dict(positions=0, sums=dict.fromkeys(MEAN_METRICS, 0.))
    sums = state["sums"]
    for keys in OUTCOME_METRICS:
        for key in keys:
            sums.setdefault(key, 0)
    total = dataset.manifest["positions"][split]
    if limit is not None:
        total = min(total, limit)
    start = state["positions"]
    with torch.set_grad_enabled(training), tqdm(total=total, initial=start, desc=split,
                                                unit="position", leave=False) as progress:
        batches = dataset.batches(split, cfg.batch_size, shuffle=training,
                                  seed=state.get("seed", cfg.seed), start=start,
                                  limit=None if limit is None else total - start)
        for pieces, metadata, actions, targets, legal, ratings, white_scores in batches:
            p = torch.as_tensor(pieces, dtype=torch.long, device=device)
            m = torch.as_tensor(metadata, dtype=torch.float32, device=device)
            a = torch.as_tensor(actions.astype(np.int64), device=device)
            advantages = torch.as_tensor(targets, device=device)
            scores = torch.as_tensor(white_scores, device=device)
            logits, value_logits = network(p, m, with_value=True)
            masks = torch.zeros_like(logits, dtype=torch.bool)
            for j, ids in enumerate(legal):
                masks[j, ids.astype(np.int64)] = True
            _, log_probs, _ = masked_policy(logits, masks, cfg.temperature)
            nll = -log_probs.gather(1, a[:, None]).squeeze(1)
            value_losses = F.binary_cross_entropy_with_logits(value_logits, scores, reduction="none")
            weights = torch.as_tensor(ratings, dtype=torch.float32, device=device) / 100
            position_losses = pretraining_loss(nll, advantages, cfg) + cfg.value_loss_weight * value_losses
            loss = (weights * position_losses).sum() / weights.sum().clamp_min(1e-8)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip)
                optimizer.step()
                if ema is not None:
                    with torch.no_grad():
                        for average, current in zip(ema.parameters(), network.parameters()):
                            average.lerp_(current, 1 - cfg.ema_decay)
            mover_scores = torch.where(m[:, SIDE_TO_MOVE_INDEX] > 0, scores, 1 - scores)
            won, lost = mover_scores == 1, mover_scores == 0
            matches = log_probs.argmax(-1) == a
            totals = torch.stack((loss.detach() * len(actions), nll.detach().sum(), matches.sum(),
                                  value_losses.detach().sum(), (matches & won).sum(), won.sum(),
                                  (~matches & lost).sum(), lost.sum())).tolist()
            for key, value in zip((*MEAN_METRICS, *OUTCOME_METRICS[0], *OUTCOME_METRICS[1]), totals):
                sums[key] += value
            state["positions"] += len(actions)
            progress.update(len(actions))
            if on_batch is not None:
                on_batch(state)
                network.train(training)  # A preview temporarily puts the model into eval mode.
    return pretraining_metrics(sums, state["positions"])


def plot_history(history, cfg=None, positions=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    if positions:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), num="Training progress", clear=True)
    else:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = [r["positions"] for r in history] if positions else [r["epoch"] for r in history]
    xlabel = "Training positions" if positions else "Epoch"
    for ax, key, title in zip(axes.flat, ("nll", "value_bce", "winner_match", "loser_disagree"),
                              ("Human move NLL", "Value BCE", "Winner move agreement", "Loser move disagreement")):
        for split, label in (("train", "Train"), ("validation", "EMA validation")):
            values = np.array([row[split].get(key) for row in history], dtype=float)
            if np.isfinite(values).any():
                ax.plot(x, values, marker=".", label=label)
        ax.set(title=title, xlabel=xlabel)
        if key in ("winner_match", "loser_disagree"):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if ax.lines:
            ax.legend()
        ax.grid(alpha=.25)
    figure.tight_layout()
    if positions:
        plt.show(block=False)
        plt.pause(.001)
    else:
        plt.show()
        plt.close(figure)
    return figure


def plot_examples(network, dataset, device, cfg):
    import matplotlib.pyplot as plt
    network.eval()
    examples = dataset.examples
    figure, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 6), squeeze=False)
    for ax, (board, human_move) in zip(axes.flat, examples):
        with torch.no_grad():
            logits = network(*encode_batch([board], device))[0]
        predicted = max(legal_moves(board, cfg.rules), key=lambda m: float(logits[action_index(m)]))
        plot_board(ax, board)
        ax.set_title(f"Held-out game: {'White' if board.turn else 'Black'} to move\n"
                     f"Human: {board.san(human_move)} | Model: {board.san(predicted)}")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def pretrain(network, device, cfg, optimizer, ema):
    dataset = prepare_pretraining(cfg)
    for group in optimizer.param_groups:
        group.update(lr=cfg.pretrain_learning_rate, weight_decay=cfg.weight_decay)
    summary = deepcopy(getattr(network, "pretraining_summary", None) or {})
    rows = summary.setdefault("history", [])
    previews = summary.setdefault("previews", [])
    summary.setdefault("trained_positions", 0)
    summary.update(source=dataset.manifest["signature"]["files"], counts=dataset.manifest["counts"],
                   positions=dataset.manifest["positions"], dataset=dataset.directory.name,
                   settings={k: v for k, v in asdict(cfg).items() if k.startswith("pretrain_")})
    network.pretraining_summary = summary
    progress = getattr(network, "training_progress", None)
    if progress and progress["dataset"] != dataset.directory.name:
        raise ValueError("An intermediate checkpoint requires the same dataset to resume its cursor. "
                         "Restore the saved dataset/filter settings before resuming.")
    completed = rows[-1]["epoch"] if rows else 0
    first_epoch = progress["epoch"] if progress else completed + 1
    epochs = count(first_epoch) if cfg.pretrain_epochs is None else range(first_epoch,
                                                                          first_epoch + cfg.pretrain_epochs)
    next_save = (summary["trained_positions"] // cfg.save_every_positions + 1) * cfg.save_every_positions
    live_figure = None
    print(f"Rated human games: move policy + White outcome prediction | EMA decay {cfg.ema_decay}")
    try:
        for epoch in epochs:
            state = progress or dict(epoch=epoch, dataset=dataset.directory.name, seed=cfg.seed + epoch,
                                     positions=0, sums=dict.fromkeys(MEAN_METRICS, 0.))
            network.training_progress = state
            start_positions = state["positions"]
            base_positions = summary["trained_positions"]
            report_positions = start_positions
            report_sums = dict(state["sums"])

            def intermediate(current):
                nonlocal next_save, report_positions, report_sums, live_figure
                if cfg.show_plots and live_figure is not None:
                    import matplotlib.pyplot as plt
                    if plt.fignum_exists(live_figure.number):
                        live_figure.canvas.flush_events()
                summary["trained_positions"] = base_positions + current["positions"] - start_positions
                if summary["trained_positions"] < next_save:
                    return
                count_positions = current["positions"] - report_positions
                window_sums = {key: value - report_sums.get(key, 0)
                               for key, value in current["sums"].items()}
                window = pretraining_metrics(window_sums, count_positions)
                # A fixed validation prefix keeps these updates fast; full validation stays at epoch end.
                preview = pretraining_epoch(ema, None, dataset, device, cfg, limit=cfg.preview_positions)
                previews.append(dict(positions=summary["trained_positions"], epoch=epoch,
                                     epoch_positions=current["positions"], train=window, validation=preview))
                save_checkpoint(network, optimizer, cfg, ema)
                tqdm.write(f"Saved {summary['trained_positions']:,} training positions | epoch {epoch}, "
                           f"position {current['positions']:,} | train NLL {window['nll']:.4f} | "
                           f"EMA preview NLL {preview['nll']:.4f}, value BCE {preview['value_bce']:.4f}, "
                           f"{outcome_rates(preview)}")
                while next_save <= summary["trained_positions"]:
                    next_save += cfg.save_every_positions
                report_positions, report_sums = current["positions"], dict(current["sums"])
                if cfg.show_plots:
                    live_figure = plot_history(previews, positions=True)

            training = pretraining_epoch(network, optimizer, dataset, device, cfg, state=state,
                                         on_batch=intermediate, ema=ema)
            evaluation = pretraining_epoch(ema, None, dataset, device, cfg)
            rows.append(dict(epoch=epoch, train=training, validation=evaluation, dataset=dataset.directory.name,
                             positions=summary["trained_positions"]))
            network.training_progress = None
            progress = None
            print(f"epoch {epoch} | train NLL {training['nll']:.4f}, value BCE {training['value_bce']:.4f}, "
                  f"{outcome_rates(training)} | EMA held-out NLL {evaluation['nll']:.4f}, "
                  f"value BCE {evaluation['value_bce']:.4f}, {outcome_rates(evaluation)}")
            save_checkpoint(network, optimizer, cfg, ema)
            print(f"Checkpoint (epoch {epoch}): {checkpoint_path(cfg)}")
            if cfg.show_plots:
                plot_history(rows)
    except KeyboardInterrupt:
        if checkpoint_path(cfg).is_file():
            load_checkpoint(network, optimizer, device, cfg, ema)
            print("\nStopped; resume from the most recent saved batch:", checkpoint_path(cfg))
        else:
            print("\nStopped before the first checkpoint; no model was saved.")
    if cfg.show_plots:
        plot_examples(ema, dataset, device, cfg)


def save_checkpoint(network, optimizer, cfg, ema):
    path = checkpoint_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(model_version=MODEL_VERSION, model=network.state_dict(),
                    ema_model=ema.state_dict(), ema_decay=cfg.ema_decay,
                    optimizer=optimizer.state_dict(),
                    architecture={key: getattr(cfg, key) for key in ARCHITECTURE_KEYS},
                    rules=cfg.rules, action_encoding=ACTION_ENCODING,
                    policy_temperature=cfg.temperature,
                    training_progress=getattr(network, "training_progress", None),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=(torch.cuda.get_rng_state(next(network.parameters()).device)
                              if next(network.parameters()).device.type == "cuda" else None),
                    pretraining=getattr(network, "pretraining_summary", None)), temporary)
    temporary.replace(path)


def read_checkpoint(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    if (saved.get("model_version") not in (MODEL_VERSION, LEGACY_MODEL_VERSION)
            or saved.get("action_encoding") != ACTION_ENCODING):
        raise ValueError("Checkpoint architecture/action encoding is incompatible with this policy.")
    return saved


def read_training_checkpoint(path, cfg):
    saved = read_checkpoint(path, "cpu")
    if saved["model_version"] != MODEL_VERSION:
        raise ValueError(
            "This checkpoint uses a legacy architecture. Select a new path for policy/value training.")
    architecture = {key: saved["architecture"][key] for key in ARCHITECTURE_KEYS}
    for key, expected in {**architecture, "rules": saved["rules"]}.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"Checkpoint {key}={expected}; match Config before resuming. "
                             "A wider model requires a fresh checkpoint path.")
    if saved.get("pool") is not None:
        raise ValueError("Use a pretrained checkpoint to resume historical-game training.")
    return saved


def load_checkpoint(network, optimizer, device, cfg, ema=None):
    saved = read_training_checkpoint(checkpoint_path(cfg), cfg)
    network.load_state_dict(saved["model"])
    if ema is not None:
        ema.load_state_dict(saved["ema_model"])
    network.pretraining_summary = saved.get("pretraining")
    network.training_progress = saved.get("training_progress")
    if optimizer is not None:
        optimizer.load_state_dict(saved["optimizer"])
    if saved.get("torch_rng") is not None:
        torch.set_rng_state(saved["torch_rng"].cpu())
    if saved.get("cuda_rng") is not None and next(network.parameters()).device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), next(network.parameters()).device)


@torch.no_grad()
def print_top_moves(network, board, device, cfg):
    if game_result(board, cfg.rules)[0]:
        print("Game over:", board.result(claim_draw=True))
        return
    logits = network(*encode_batch([board], device))
    moves = legal_moves(board, cfg.rules)
    ids = [action_index(move) for move in moves]
    scores = logits[0, ids]
    probs = F.softmax(scores / cfg.temperature, dim=0)
    print("Move       Actor logit   P(select)")
    for i in scores.argsort(descending=True).tolist()[:cfg.top_moves]:
        print(f"{moves[i].uci():10} {scores[i].item():+9.3f}   {probs[i].item():8.2%}")


def plot_board(ax, board):
    from matplotlib.patches import Rectangle
    for square in chess.SQUARES:
        x, y = chess.square_file(square), chess.square_rank(square)
        ax.add_patch(Rectangle((x, y), 1, 1, color="#e1dbc8" if (x + y) % 2 else "#698b7c"))
        piece = board.piece_at(square)
        if piece:
            ax.text(x + .5, y + .5, piece.unicode_symbol(), ha="center", va="center",
                    fontsize=28, fontfamily="DejaVu Sans")
    ax.set(xlim=(0, 8), ylim=(0, 8), xticks=np.arange(8) + .5, yticks=np.arange(8) + .5)
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticklabels(range(1, 9))
    ax.set_aspect("equal")


def quick_test(network, device, cfg):
    network.eval()
    board = chess.Board()
    logits = network(*encode_batch([board], device))
    assert logits.shape == (1, NUM_ACTIONS)
    assert network.start_cell.bias is None and network.end_cell.bias is None
    masks = torch.zeros_like(logits, dtype=torch.bool)
    ids = [action_index(move) for move in legal_moves(board, "standard")]
    masks[0, ids] = True
    probs, log_probs, _ = masked_policy(logits, masks, cfg.temperature)
    assert torch.count_nonzero(probs) == 20
    torch.testing.assert_close(probs.sum(), probs.new_tensor(1.))
    (-log_probs[0, ids[0]]).backward()
    for head in (network.start_cell, network.end_cell):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    print("TEST PASSED | bias-free projections, 4096 actions, legal masks, policy gradients")


def main():
    cfg = CONFIG
    if cfg.mode not in ("train", "inspect", "test") or cfg.rules not in ("standard", "king-capture"):
        raise ValueError("Check mode and rules in Config.")
    if min(cfg.temperature, cfg.batch_size, cfg.grad_clip, cfg.save_every_positions, cfg.preview_positions) <= 0:
        raise ValueError("Need positive temperature, batch_size and grad_clip.")
    if ((cfg.pretrain_epochs is not None and cfg.pretrain_epochs < 0)
            or any(v is not None and v < 1 for v in (cfg.pretrain_max_games, cfg.pretrain_max_positions))
            or cfg.pretrain_shard_positions < 1 or not 0 < cfg.pretrain_validation_fraction < 1
            or cfg.pretrain_imitation_weight <= 0 or cfg.pretrain_advantage_weight < 0
            or cfg.pretrain_negative_log_cap <= 0 or cfg.pretrain_learning_rate <= 0):
        raise ValueError("Check pretraining sizes, split, learning rate and loss weights in Config.")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    name = cfg.device
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    network = ChessPolicy(cfg).to(device)
    if not 0 <= cfg.ema_decay < 1 or cfg.value_loss_weight <= 0:
        raise ValueError("Use an EMA decay in [0, 1) and a positive value loss weight")
    ema = deepcopy(network).eval().requires_grad_(False) if cfg.mode != "test" else None
    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg.pretrain_learning_rate,
                                  weight_decay=cfg.weight_decay)
    destination = checkpoint_path(cfg)
    source = base_checkpoint_path(cfg)
    if cfg.resume and destination.is_file():
        load_checkpoint(network, optimizer, device, cfg, ema)
        print("Resumed:", destination)
    elif (cfg.resume and ema is not None and cfg.pretrain_run_name
          and not cfg.pretrain_checkpoint_path and source.is_file()):
        saved = read_training_checkpoint(source, cfg)
        network.load_state_dict(saved["ema_model"])
        ema.load_state_dict(saved["ema_model"])
        del saved
        print(f"Initialized from EMA weights in {source}; new run writes to {destination}")
    elif cfg.resume:
        print("No checkpoint at", destination, "— starting fresh; existing models are preserved.")
    print(f"Device: {device} | {sum(p.numel() for p in network.parameters()):,} parameters")
    if cfg.mode == "test":
        quick_test(network, device, cfg)
    elif cfg.mode == "inspect":
        network.eval()
        print_top_moves(ema, chess.Board(cfg.inspect_fen), device, cfg)
    else:
        try:
            pretrain(network, device, cfg, optimizer, ema)
        except KeyboardInterrupt:
            print("\nStopped. Existing checkpoints are unchanged.")


if __name__ == "__main__":
    main()
