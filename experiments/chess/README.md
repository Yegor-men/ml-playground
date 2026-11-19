# Square-pair chess policy

Run `main.py` in PyCharm. Settings live in `Config`; there are no command-line
arguments. Dependencies: PyTorch, NumPy, `chess`, matplotlib, tqdm. Training uses
rated historical games for both White and Black. There is no self-play training,
historical opponent pool, or synthetic rating system.

## Model size and move scoring

The default uses width 1024 and eight independent rotary transformer blocks.

The initial square states contain piece and colour embeddings plus fixed 2D
sinusoidal square embeddings and a learned projection of board metadata. Each
head applies 2D RoPE to its queries and keys. Training recomputes block activations during
backpropagation by default (`checkpoint_passes = True`) to save memory.

Separate bias-free start and end projections score the final square states via
`logits = start @ end.transpose(-2, -1) / sqrt(dim)`. The 4096 actions use
`from_square * 64 + to_square`. The game ratings provide the expected score
used by the policy loss. The policy uses temperature 1 by default.
Legal moves are masked before softmax; this includes castling and en passant. Promotion is always to a queen;
games containing underpromotions are excluded. The value head projects the
contextual White and Black king vectors independently and takes their scaled
dot product. Its sigmoid predicts White's score. Each position uses the actual
final score (1, 0.5, or 0) in binary cross-entropy.
Repetition history is used for game termination in the player, but is not encoded
in the network input.

## More historical games, bounded RAM

`pretrain_months` defaults to all twelve months of 2013. The
[Lichess monthly archives](https://database.lichess.org/#standard_games) contain
about 3.39 million games before filtering and total about 512 MB compressed. They are CC0.
`WhiteElo`/`BlackElo` contain Lichess Glicko-2 ratings, not FIDE Elo. These old
archives can be supplemented with any available `YYYY-MM` month. Recent months
can be much larger downloads.

`pretrain_max_games` caps the number of accepted, unique games across all input
files combined. It defaults to `None`, as does `pretrain_max_positions`.
Both players must meet `pretrain_min_rating`
(default 100, so the dataset includes a broad range of playing strengths).
The importer scans the configured files in order when a cap is set.

Downloaded files stay compressed under `data/chess/`. On first use, the importer
encodes games into compressed NumPy shards under `data/chess/pgn_cache/`.
`pretrain_shard_positions = 50_000` controls the approximate number of positions
in each shard; whole games are kept together, so a shard can exceed that number
by one game. One shard at a time is loaded into RAM, and training shuffles shards
and positions within each shard. Increasing the total dataset size therefore
does not require holding all positions in RAM. Parsing and encoding happen once;
subsequent runs reuse the cache. Changing sources or filters creates a new cache.
Cache version 3 adds the actual White result. The first run builds a fresh
cache from the locally stored PGNs.
Caches and downloaded PGNs can be deleted when no run is using them; they will
be rebuilt or downloaded as needed.

For your own PGN, set `pretrain_pgn` to a path under the repository's shared
`data/` directory. This overrides `pretrain_months`. `pretrain_url` optionally
supplies a URL to download that file if missing. Readers support `.pgn`, `.gz`,
`.bz2` and `.zst`; Zstandard needs Python 3.14's `compression.zstd` or the
`zstandard` package on earlier Python versions.

The importer excludes unknown ratings/results, malformed games and variants.
Recorded resignation and time-forfeit results remain valid even when the final
board is nonterminal. Exact starting-position/move sequences are deduplicated
across all input archives, irrespective of their headers. A hash of each full
game assigns roughly 10% of games to validation, consistently across files.
Positions from one game cannot cross splits, though different games can share
openings and transpositions. Validation includes every held-out position in the
selected dataset, rather than a changing sample each epoch.

## Training and resuming

`pretrain_epochs = None` trains until Ctrl+C. An integer means that many passes,
counting completion of a resumed partial epoch as the first pass.
`save_every_positions = 100_000` saves an intermediate checkpoint at the next
completed optimizer batch after each threshold. Each completed epoch also saves.
Saves replace the selected checkpoint atomically, rather than keeping many copies.
At 600 positions/second, 100,000 training positions take about 2.8 minutes,
plus preview, plotting and disk-write overhead.

Each intermediate update evaluates a fixed held-out prefix of at most
`preview_positions = 2_048` positions. The plots show unweighted policy NLL and
value BCE, plus legal top-1 agreement on winners' plies and disagreement on
losers' plies. Draws only contribute to the losses in these plots.
It reuses one nonblocking window so training continues. Each epoch still performs
full held-out validation and shows its epoch curves. Close epoch/endgame figures
to continue if your backend blocks. Final plots compare human and model moves in
held-out positions. Use `show_plots = False` for headless runs.
The agreement rates describe recorded moves and game outcomes. In particular,
loser disagreement counts every alternative legal move equally.

`resume = True` restores weights, optimizer moments, RNG state and the shuffled
data cursor, continuing inside the epoch from the saved batch. Only work since
the latest checkpoint is discarded after an interruption. Resuming a partial
epoch requires the same dataset and filter settings. Otherwise the program stops
instead of silently applying the old cursor to different data. Training starts
fresh when no matching checkpoint exists. The default run writes
`data/chess/chess_rope_value_d1024_b8_h16_elo_weighted.pt`. On its first run, if
the unsuffixed checkpoint exists, its EMA weights initialize the network and the
new EMA copy. The new run starts with fresh optimizer state, metric history and
data cursor. `pretrain_checkpoint_path` overrides the output path.
The default batch size is 512 and AdamW uses a learning rate of `1e-4`.
The live weights train with the optimizer. A separate EMA copy
updates after each step using decay `ema_decay = 0.999`. Checkpoints keep both
copies, and validation and play use the EMA weights. `play.py` also loads
the original rotary policy checkpoints.

Each saved metric row records a dataset cache ID. Increasing the dataset or
changing filters keeps past history when resuming, but measurements across that
change describe different data and should not be compared as a fixed benchmark.
The game-hash split remains stable.

## Rating-based outcome weighting

The outcome weighting uses the masked move likelihood. Both heads use the
mover's rating as a per-position weight:

```
E_white = 1 / (1 + 10 ** ((black_rating - white_rating) / 400))
A_white = 2 * (actual_white_score - E_white)  # win=1, draw=0.5, loss=0
A_black = -A_white
logits = policy(board) / temperature
nll = -log softmax(legal_mask(logits))[human_move]
surprise_loss = A * nll                     if A >= 0
                A * min(nll, log_cap)       if A < 0
position_loss = imitation_weight * nll + advantage_weight * surprise_loss + value_loss_weight * value_bce
weight = mover_rating / 100
batch_loss = sum(weight * position_loss) / sum(weight)
```

This is offline imitation with a bounded signed game-outcome term. Human action
probabilities are unknown, so it is not an unbiased REINFORCE update. PGNs lack
Glicko-2 RD/volatility and colour calibration; the expectation is approximate.
Every move from one player receives the same game-level advantage, which cannot
identify individual blunders. Negative outcomes reduce the imitation signal;
sufficiently large upsets reverse it, while the negative term saturates at
`pretrain_negative_log_cap = 8` to keep the loss bounded below. Set
`pretrain_advantage_weight = 0` for ordinary legal-move behaviour cloning.

## Play and evaluation

Run `play.py`, then open <http://127.0.0.1:8000>. It selects the configured
run's checkpoint if present, then tries the unsuffixed and older checkpoint
paths. Set `CHECKPOINT_PATH` explicitly to compare models.
The browser shows the saved epoch or intermediate position and held-out accuracy.
`MODE = "training"` shows the training curves without starting the web server.

The default **Control both sides** lets you make any legal move for the colour on
turn. **Make model move** asks the model to move for that side. Alternatively,
choose White or Black and enable automatic replies; switching control preserves
the board. **Undo ply** always removes exactly one move and never triggers an
automatic replacement move.

The policy/value model receives the board and metadata. The board's evaluation
bar reports White's expected score, with the exact final score for terminal boards.

FEN loading supports arbitrary valid standard-chess positions and the side to
move. Endgame presets cover king + queen, king + rook, and king + rook + knight
against a king. Loading a FEN resets move/repetition history. These let you test
endgame behaviour without assuming the model has learned a reliable mating plan.

The page also provides board flipping and PGN export, using local piece images
and no web framework or CDN. The model samples a legal move from the current
board's policy. Temperature defaults to 1.0, with 0 selecting the highest logit.
Min-p defaults to 0.1, keeping moves at least one tenth as probable as the
most probable move after temperature scaling. The browser lets you edit both
settings. The value head feeds the evaluation bar. Training uses the full
legal-move softmax.
No repetition penalty has been added; existing standard draw claims still apply.
Human games do not update the weights. All browser tabs share one local game.

Optional `MODE = "benchmark"` plays paired games against a separately installed
Stockfish executable. Configure `STOCKFISH_PATH`, an even `GAMES_PER_ELO`,
`STOCKFISH_ELOS` inside the engine's UCI_Elo range, and `STOCKFISH_SECONDS`.
Results and PGNs go to `data/chess/evaluation/`. Capped games remain unfinished,
not draws. Any engine-relative estimate is not an official human rating.

## Files and checks

Commit `main.py`, `play.py`, this README, and `test_pretraining.py`. The obsolete
self-play `ratings.py` and `test_ratings.py` have been removed. Do not commit
`data/`, generated caches/checkpoints, or `__pycache__/`.

Run `python -m unittest discover -s experiments/chess -p 'test_*.py'` in the
project interpreter. Tests cover multi-file deduplication, stable game splits,
cache reuse, compression, filters, shard roundtrips, value and policy sampling,
intermediate saves, exact CPU dropout resuming, both-side control and endgames.
`mode = "test"` runs a small model/masking/gradient check; `"inspect"` prints top
moves for `inspect_fen`.
