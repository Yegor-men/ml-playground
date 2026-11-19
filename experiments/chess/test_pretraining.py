"""PGN caching, legal moves, offline gradients, resumable training and playback."""
import bz2
import gzip
import math
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np
import torch

import main as training
import play


class PretrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        root = Path(__file__).resolve().parents[2] / "data"
        self.temporary = tempfile.TemporaryDirectory(dir=root, prefix="chess_test_")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.data_patch = patch.object(training, "DATA_DIRECTORY", self.path / "chess")
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.cfg = training.Config(dim=32, heads=4, blocks=2, batch_size=32, dropout=0.,
                                   device="cpu", resume=False, pretrain_epochs=1,
                                   pretrain_max_games=30, pretrain_max_positions=1000,
                                   pretrain_shard_positions=40, pretrain_validation_fraction=.3,
                                   pretrain_pgn=str(self.path / "games.pgn"),
                                   pretrain_checkpoint_path=str(self.path / "pretrained.pt"),
                                   show_plots=False)
        rng = random.Random(42)
        games = []
        for i in range(30):
            board = chess.Board()
            for _ in range(8):
                board.push(rng.choice(list(board.legal_moves)))
            game = chess.pgn.Game.from_board(board)
            game.headers.update(WhiteElo="2000", BlackElo="1800", Result="1-0")
            games.append(str(game))
        self.pgn = "\n\n".join(games) + "\n\n"
        Path(self.cfg.pretrain_pgn).write_text(self.pgn)

    def test_targets_and_legal_masks_survive_shard_roundtrip(self):
        dataset = training.prepare_pretraining(self.cfg)
        self.assertEqual(dataset.manifest["counts"]["accepted"], 30)
        self.assertGreater(len(dataset.manifest["shards"]), 1)
        self.assertEqual(sum(dataset.manifest["positions"].values()), 240)
        advantage = 2 * (1 - 1 / (1 + 10 ** (-.5)))
        for split in ("train", "validation"):
            count = 0
            for pieces, metadata, actions, targets, legal, ratings, white_scores in dataset.batches(split, 17):
                self.assertEqual(pieces.shape[1], 64)
                np.testing.assert_allclose(targets, advantage * metadata[:, 13], rtol=1e-6)
                np.testing.assert_array_equal(ratings, np.where(metadata[:, 13] == 1, 2000, 1800))
                self.assertTrue(np.all(white_scores == 1))
                for action, ids in zip(actions, legal):
                    self.assertIn(action, ids)
                count += len(actions)
            self.assertEqual(count, dataset.manifest["positions"][split])
        self.assertTrue(dataset.examples)

    def test_multiple_files_dedup_and_stable_game_splits(self):
        second = self.path / "other.pgn.gz"
        second.write_bytes(gzip.compress(self.pgn.encode()))
        sources = [(Path(self.cfg.pretrain_pgn), ""), (second, "")]
        cfg = replace(self.cfg, pretrain_max_games=None, pretrain_max_positions=None)
        tqdm = training.tqdm
        bars = []

        def track_progress(**kwargs):
            bar = tqdm(**kwargs)
            bars.append(bar)
            return bar

        with patch.object(training, "pgn_sources", return_value=sources), \
                patch.object(training, "tqdm", side_effect=track_progress):
            dataset = training.prepare_pretraining(cfg)
        archive_bytes = sum(path.stat().st_size for path, _ in sources)
        self.assertEqual((bars[0].n, bars[0].total, bars[0].unit), (archive_bytes, archive_bytes, "B"))
        self.assertIn("60 scanned, 30 kept", bars[0].postfix)
        self.assertEqual(dataset.manifest["counts"]["duplicates"], 30)
        self.assertEqual(dataset.manifest["counts"]["accepted"], 30)
        # Reversing archive order doesn't change split membership.
        with patch.object(training, "pgn_sources", return_value=list(reversed(sources))):
            reordered = training.prepare_pretraining(cfg)
        for split in ("train", "validation"):
            def signatures(data):
                return sorted((p.tobytes(), m.tobytes(), int(a))
                              for ps, ms, actions, _, _, _, _ in data.batches(split, 32)
                              for p, m, a in zip(ps, ms, actions))
            self.assertEqual(signatures(dataset), signatures(reordered))

    def test_cache_reuse_and_filter_invalidation(self):
        dataset = training.prepare_pretraining(self.cfg)
        with patch.object(training, "game_samples", side_effect=AssertionError("reparsed cached PGNs")):
            cached = training.prepare_pretraining(self.cfg)
        self.assertEqual(dataset.directory, cached.directory)
        changed = training.prepare_pretraining(replace(self.cfg, pretrain_min_rating=1700))
        self.assertNotEqual(dataset.directory, changed.directory)

    def test_filters_and_position_limit(self):
        board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
        board.push_uci("a7a8n")
        underpromotion = chess.pgn.Game.from_board(board)
        underpromotion.headers.update(WhiteElo="2000", BlackElo="1800", Result="1/2-1/2")
        Path(self.cfg.pretrain_pgn).write_text(
            self.pgn.replace('[WhiteElo "2000"]', '[WhiteElo "?"]')
            + self.pgn.replace('1-0', '*') + str(underpromotion) + "\n\n" + self.pgn)
        dataset = training.prepare_pretraining(self.cfg)
        self.assertEqual(dataset.manifest["counts"]["skipped"], 61)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            training.prepare_pretraining(replace(self.cfg, pretrain_max_positions=1))

    def test_compressed_pgn_readers(self):
        for suffix, compress in ((".gz", gzip.compress), (".bz2", bz2.compress)):
            path = self.path / ("games.pgn" + suffix)
            path.write_bytes(compress(self.pgn.encode()))
            with training.open_pgn(path) as source:
                self.assertEqual(source.read(), self.pgn)
        try:
            from compression import zstd
        except ImportError:
            return
        path = self.path / "games.pgn.zst"
        path.write_bytes(zstd.compress(self.pgn.encode()))
        with training.open_pgn(path) as source:
            self.assertEqual(source.read(), self.pgn)
        with path.open("rb") as raw, training.open_pgn(path, raw) as source:
            self.assertEqual(source.read(), self.pgn)
            self.assertEqual(raw.tell(), path.stat().st_size)

    def test_signed_gradient_and_bounded_negative_term(self):
        nll = torch.tensor([1., 1., 20.], requires_grad=True)
        targets = torch.tensor([1.5, -1.5, -1.5])
        training.pretraining_loss(nll, targets, self.cfg).sum().backward()
        torch.testing.assert_close(nll.grad, torch.tensor([2.5, -.5, 1.]))
        imitation = replace(self.cfg, pretrain_advantage_weight=0.)
        torch.testing.assert_close(training.pretraining_loss(nll, targets, imitation), nll)

    def test_shuffled_data_cursor_resumes_without_replaying_positions(self):
        dataset = training.prepare_pretraining(self.cfg)

        def sequence(start, batch_size):
            return [(p.tobytes(), int(a)) for ps, _, actions, _, _, _, _ in dataset.batches(
                "train", batch_size, shuffle=True, seed=123, start=start) for p, a in zip(ps, actions)]

        self.assertEqual(sequence(47, 17), sequence(0, 32)[47:])

    def test_rating_weights_both_heads_and_metrics_use_the_movers_result(self):
        cfg = replace(self.cfg, pretrain_advantage_weight=0.)
        device = torch.device("cpu")
        network = training.ChessPolicy(cfg).eval()
        logits = torch.zeros(5, training.NUM_ACTIONS)
        logits[:, 0] = torch.tensor([1., -1., -1., 1., 1.])
        values = torch.tensor([.5, -.3, .1, .6, -.4])
        metadata = np.zeros((5, training.META_SIZE), dtype=np.float32)
        metadata[:, training.SIDE_TO_MOVE_INDEX] = [1, -1, 1, -1, 1]
        ratings = np.array([1000, 2000, 1500, 1200, 1800], dtype=np.uint16)
        scores = np.array([1., 0., 0., 1., .5], dtype=np.float32)
        batch = (np.zeros((5, 64), dtype=np.uint8), metadata, np.zeros(5, dtype=np.uint16),
                 np.zeros(5, dtype=np.float32), [np.array([0, 1])] * 5, ratings, scores)
        dataset = training.PGNDataset(self.path, {"positions": {"validation": 5}})
        with patch.object(dataset, "batches", return_value=[batch]), \
                patch.object(network, "forward", return_value=(logits, values)):
            metrics = training.pretraining_epoch(network, None, dataset, device, cfg)
        nll = -logits[:, :2].log_softmax(-1)[:, 0]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(values, torch.from_numpy(scores), reduction="none")
        weights = ratings / ratings.sum()
        self.assertAlmostEqual(metrics["loss"], np.dot(weights, (nll + bce).numpy()), places=6)
        self.assertAlmostEqual(metrics["value_bce"], bce.mean().item(), places=6)
        self.assertEqual(metrics["winner_match"], .5)
        self.assertEqual(metrics["loser_disagree"], .5)

    def test_intermediate_save_and_exact_resume_with_dropout(self):
        cfg = replace(self.cfg, dropout=.1, save_every_positions=64, preview_positions=16)
        save = training.save_checkpoint

        def stop_after_save(*args):
            save(*args)
            raise KeyboardInterrupt

        with patch.object(training, "CONFIG", cfg), patch.object(training, "save_checkpoint", side_effect=stop_after_save):
            training.main()
        partial = training.read_checkpoint(training.checkpoint_path(cfg), "cpu")
        cursor = partial["training_progress"]["positions"]
        self.assertGreaterEqual(cursor, 64)
        self.assertLess(cursor, partial["pretraining"]["positions"]["train"])
        self.assertEqual(partial["model_version"], training.MODEL_VERSION)
        with patch.object(training, "CONFIG", replace(cfg, resume=True, pretrain_min_rating=1700)):
            with self.assertRaisesRegex(ValueError, "same dataset"):
                training.main()
        with patch.object(training, "CONFIG", replace(cfg, resume=True)):
            training.main()
        resumed = training.read_checkpoint(training.checkpoint_path(cfg), "cpu")
        self.assertIsNone(resumed["training_progress"])
        self.assertEqual(resumed["pretraining"]["trained_positions"], 160)
        self.assertEqual(len(resumed["pretraining"]["history"]), 1)
        uninterrupted = replace(cfg, pretrain_checkpoint_path=str(self.path / "uninterrupted.pt"))
        with patch.object(training, "CONFIG", uninterrupted):
            training.main()
        reference = training.read_checkpoint(training.checkpoint_path(uninterrupted), "cpu")
        for key in resumed["model"]:
            torch.testing.assert_close(resumed["model"][key], reference["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(resumed["ema_model"][key], reference["ema_model"][key], rtol=0, atol=0)

    def test_both_side_control_single_ply_undo_and_endgame_positions(self):
        network = training.ChessPolicy(self.cfg).eval()
        player = play.Player(network, self.cfg, torch.device("cpu"))
        player.action("/move", dict(move="e2e4"))
        state = player.action("/move", dict(move="e7e5"))
        self.assertEqual(state["history"], [dict(number=1, white="e4", black="e5")])
        self.assertEqual(state["movetext"], "1. e4 e5 *")
        state = player.action("/undo", {})
        self.assertEqual(state["plies"], 1)
        self.assertEqual(state["history"], [dict(number=1, white="e4", black=None)])
        self.assertEqual(player.board.turn, chess.BLACK)
        player.action("/control", dict(side="white"))
        with self.assertRaisesRegex(ValueError, "model's turn"):
            player.action("/move", dict(move="c7c5"))
        player.action("/control", dict(side="both"))
        player.action("/move", dict(move="c7c5"))
        for fen in play.ENDGAMES.values():
            state = player.action("/position", dict(fen=fen))
            self.assertFalse(state["ended"])
            self.assertEqual(state["plies"], 0)
            self.assertEqual(state["history"], [])
            self.assertTrue(state["moves"])
        black_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 17"
        player.action("/position", dict(fen=black_fen))
        state = player.action("/move", dict(move="e7e5"))
        self.assertEqual(state["history"], [dict(number=17, white=None, black="e5")])
        self.assertEqual(state["movetext"], "17... e5 *")
        fen = "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"
        player.action("/position", dict(fen=fen))
        state = player.action("/move", dict(move="g6g7"))
        self.assertTrue(state["ended"])
        self.assertEqual(state["movetext"], "1. Qg7# 1-0")
        self.assertTrue(player.board.is_checkmate())
        player.action("/undo", {})
        self.assertEqual(player.board.fen(), fen)
        with self.assertRaises(ValueError):
            player.action("/position", dict(fen="8/8/8/8/8/8/8/8 w - - 0 1"))

    def test_training_resume_optimizer_and_play_checkpoint(self):
        torch.manual_seed(self.cfg.seed)
        initial = training.ChessPolicy(self.cfg)
        with patch.object(training, "CONFIG", self.cfg):
            training.main()
        first = training.read_checkpoint(training.checkpoint_path(self.cfg), "cpu")
        for head in ("start_cell.weight", "end_cell.weight"):
            self.assertFalse(torch.equal(initial.state_dict()[head], first["model"][head]))
        with patch.object(training, "CONFIG", replace(self.cfg, resume=True)):
            training.main()
        second = training.read_checkpoint(training.checkpoint_path(self.cfg), "cpu")
        self.assertEqual([r["epoch"] for r in second["pretraining"]["history"]], [1, 2])
        for key, state in first["optimizer"]["state"].items():
            self.assertEqual(second["optimizer"]["state"][key]["step"], state["step"] * 2)
        with patch.object(play, "CHECKPOINT_PATH", training.checkpoint_path(self.cfg)), patch.object(play, "DEVICE", "cpu"):
            model, cfg, device = play.load_model()
        moves, scores = play.Player(model, cfg, device).evaluate()
        self.assertEqual(len(moves), 20)
        self.assertTrue(all(math.isfinite(x) for x in scores))
        self.assertIn("epoch 2", play.training_summary(model))
        self.assertNotIn("pool", second)

    def test_unlimited_training_interrupt_keeps_last_epoch(self):
        epoch = training.pretraining_epoch
        calls = 0

        def interrupt_next_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt
            return epoch(*args, **kwargs)

        with patch.object(training, "CONFIG", replace(self.cfg, pretrain_epochs=None)), \
                patch.object(training, "pretraining_epoch", side_effect=interrupt_next_epoch):
            training.main()
        saved = training.read_checkpoint(training.checkpoint_path(self.cfg), "cpu")
        self.assertEqual(saved["pretraining"]["history"][-1]["epoch"], 1)

    def test_named_run_warm_starts_from_ema_and_preserves_original(self):
        base = replace(self.cfg, pretrain_checkpoint_path="", pretrain_run_name="")
        device = torch.device("cpu")
        source_model = training.ChessPolicy(base)
        source_ema = training.ChessPolicy(base)
        optimizer = torch.optim.AdamW(source_model.parameters())
        training.save_checkpoint(source_model, optimizer, base, source_ema)
        source = training.checkpoint_path(base)
        original = source.read_bytes()
        cfg = replace(base, pretrain_run_name="elo_weighted", resume=True)
        pretrain = training.pretrain

        def check_initialization(network, device, cfg, optimizer, ema):
            for key, weights in source_ema.state_dict().items():
                torch.testing.assert_close(network.state_dict()[key], weights, rtol=0, atol=0)
                torch.testing.assert_close(ema.state_dict()[key], weights, rtol=0, atol=0)
            self.assertEqual(len(optimizer.state), 0)
            self.assertIsNone(getattr(network, "training_progress", None))
            pretrain(network, device, cfg, optimizer, ema)

        with patch.object(training, "CONFIG", cfg), \
                patch.object(training, "pretrain", side_effect=check_initialization):
            training.main()
        self.assertEqual(source.read_bytes(), original)
        saved = training.read_checkpoint(training.checkpoint_path(cfg), device)
        self.assertEqual(saved["pretraining"]["history"][0]["epoch"], 1)

    def test_projection_masks_and_special_moves(self):
        model = training.ChessPolicy(self.cfg)
        training.quick_test(model, torch.device("cpu"), self.cfg)
        self.assertEqual(tuple(model.positions.shape), (64, self.cfg.dim))
        _, value = model(*training.encode_batch([chess.Board()], torch.device("cpu")), with_value=True)
        torch.nn.functional.binary_cross_entropy_with_logits(value, torch.tensor([.5])).backward()
        for king in (model.white_king, model.black_king):
            self.assertTrue(torch.isfinite(king.weight.grad).all())
        for fen, ucis in (
                ("1r2k3/P6p/8/8/8/8/8/4K3 w - - 0 1", ("a7a8q", "a7b8q")),
                ("4k3/8/8/8/8/8/p6P/1R2K3 b - - 0 1", ("a2a1q", "a2b1q")),
                ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ("e1g1", "e1c1")),
                ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ("e5d6",))):
            board = chess.Board(fen)
            allowed = training.legal_moves(board, "standard")
            for uci in ucis:
                move = chess.Move.from_uci(uci)
                self.assertIn(move, allowed)
                self.assertEqual(training.action_move(board, training.action_index(move)), move)
        values = np.array([3., 2., 1.])
        np.testing.assert_array_equal(play.probabilities(values, dict(temperature=0.)), [1., 0., 0.])
        probs = play.probabilities(values, dict(temperature=1., min_p=.2))
        self.assertEqual(probs[2], 0.)
        self.assertAlmostEqual(probs.sum(), 1.)

    def test_min_p_adapts_to_confidence_and_temperature(self):
        options = play.sampling_options({})
        self.assertEqual(options["min_p"], .1)
        self.assertNotIn("top_k", options)
        self.assertNotIn("top_p", options)
        sharp = np.log([.90, .04, .03, .02, .01])
        flat = np.log([.25, .24, .23, .18, .10])
        np.testing.assert_array_equal(play.probabilities(sharp, options), [1, 0, 0, 0, 0])
        np.testing.assert_allclose(play.probabilities(flat, options), np.exp(flat))
        hotter = play.probabilities(sharp, {**options, "temperature": 2.})
        self.assertEqual(np.count_nonzero(hotter), 5)
        self.assertAlmostEqual(hotter.sum(), 1.)
        np.testing.assert_allclose(play.probabilities(flat + 50, options), play.probabilities(flat, options))
        np.testing.assert_allclose(play.probabilities(sharp, {**options, "min_p": 0.}), np.exp(sharp))
        np.testing.assert_array_equal(play.probabilities([1., 1., 0.], {**options, "min_p": 1.}), [.5, .5, 0])
        np.testing.assert_array_equal(play.probabilities(sharp, {**options, "temperature": 0.}), [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(play.probabilities([1000., 0., -1000.], options), [1, 0, 0])
        self.assertEqual(len(play.probabilities([], options)), 0)
        for invalid in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                play.sampling_options(dict(min_p=invalid))

    def test_model_samples_current_policy_with_value_head(self):
        player = play.Player(training.ChessPolicy(self.cfg).eval(), self.cfg, torch.device("cpu"))
        moves = [chess.Move.from_uci(uci) for uci in ("e2e4", "d2d4", "a2a4")]
        logits = np.full(training.NUM_ACTIONS, -10.)
        logits[[training.action_index(move) for move in moves]] = np.log([.64, .35, .01])
        logits[training.action_index(chess.Move.from_uci("a1a8"))] = 10.
        logits[0] = 9.  # Same-square pair, a1 to a1.
        evaluated = []

        def evaluate(board):
            evaluated.append(board.fen())
            return moves, logits, .99

        def choose(count, p):
            self.assertEqual(count, 3)
            np.testing.assert_allclose(p, np.array([.64, .35, 0.]) / .99)
            return 1

        with patch.object(player, "evaluate_position", side_effect=evaluate), \
                patch.object(player, "rng") as rng:
            state = player.action("/state", {"sampling": {"temperature": 2., "min_p": 0.}})
            self.assertEqual(len(state["raw_rows"]), 3 * play.RAW_LOGIT_ROWS)
            self.assertEqual(state["raw_rows"][:3], [
                dict(pair="a1 → a8", logit=10., legal=False, origin_type="white"),
                dict(pair="a1 → a1", logit=9., legal=False, origin_type="white"),
                dict(pair="e2 → e4", logit=np.log(.64), legal=True, origin_type="white"),
            ])
            for origin_type, first_square in (("black", "a7"), ("empty", "a3")):
                rows = [row for row in state["raw_rows"] if row["origin_type"] == origin_type]
                self.assertEqual(len(rows), play.RAW_LOGIT_ROWS)
                self.assertEqual(rows[0]["pair"], f"{first_square} → a1")
            self.assertEqual([row["move"] for row in state["rows"]], ["e4", "d4", "a4"])
            rng.choice.side_effect = choose
            player.model_move(play.sampling_options({}))
            rng.choice.assert_called_once()
        self.assertEqual(evaluated, [chess.STARTING_FEN])
        self.assertEqual(player.board.move_stack, [moves[1]])

    def test_move_scores_batch_nonterminal_successors_and_use_exact_results(self):
        network = training.ChessPolicy(self.cfg).eval()
        player = play.Player(network, self.cfg, torch.device("cpu"))
        fen = "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1"
        player.board = chess.Board(fen)
        moves = [chess.Move.from_uci(uci) for uci in ("g5g6", "g5g7", "g5g4", "g5e5")]
        logits = np.zeros(training.NUM_ACTIONS)
        logits[[training.action_index(move) for move in moves]] = [4., 3., 2., 1.]
        value_logits = torch.logit(torch.tensor([.75, .25]))
        with patch.object(player, "evaluate_position", return_value=(moves, logits, .6)), \
                patch.object(network, "forward", return_value=(None, value_logits)) as forward:
            state = player.action("/state", {})
            refreshed = player.action("/state", {"sampling": {"temperature": 0.}})
        forward.assert_called_once()
        positions = [chess.Board(fen) for _ in moves[2:]]
        for board, move in zip(positions, moves[2:]):
            board.push(move)
        for actual, expected in zip(forward.call_args.args, training.encode_batch(positions, player.device)):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(state["white_score"], .6)
        self.assertEqual([row["move"] for row in state["rows"]], ["Qg6", "Qg7#", "Qg4", "Qe5+"])
        scores = [row["white_score"] for row in state["rows"]]
        self.assertEqual(scores[:2], [.5, 1.])
        np.testing.assert_allclose(scores[2:], [.75, .25], atol=1e-7)
        self.assertEqual(scores, [row["white_score"] for row in refreshed["rows"]])
        self.assertEqual(player.board.fen(), fen)
        self.assertFalse(player.board.move_stack)

    def test_exact_terminal_scores(self):
        network = training.ChessPolicy(self.cfg).eval()
        player = play.Player(network, self.cfg, torch.device("cpu"))
        with patch.object(network, "forward", side_effect=AssertionError("terminal network evaluation")):
            for fen, expected in (
                ("7k/5KQ1/8/8/8/8/8/8 b - - 0 1", 1.0),
                ("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", .5),
            ):
                state = player.action("/position", {"fen": fen})
                self.assertTrue(state["ended"])
                self.assertEqual(state["white_score"], expected)
                self.assertEqual(state["raw_rows"], [])

    def test_width_paths_and_legacy_pretrained_checkpoint(self):
        small = replace(self.cfg, pretrain_checkpoint_path="", dim=512)
        large = replace(small, dim=1024)
        self.assertNotEqual(training.checkpoint_path(small), training.checkpoint_path(large))
        with patch.object(training, "CONFIG", self.cfg):
            training.main()
        path = training.checkpoint_path(self.cfg)
        saved = training.read_checkpoint(path, "cpu")
        saved.update(pool=None, history=[], iteration=0, training_version=3)
        torch.save(saved, path)
        model = training.ChessPolicy(self.cfg)
        optimizer = torch.optim.AdamW(model.parameters())
        training.load_checkpoint(model, optimizer, "cpu", self.cfg)
        with self.assertRaisesRegex(ValueError, "wider model"):
            training.load_checkpoint(training.ChessPolicy(replace(self.cfg, dim=64)), None,
                                     "cpu", replace(self.cfg, dim=64))
        legacy = training.ChessPolicy(self.cfg, legacy_blocks=1).eval()
        saved.update(model_version=training.LEGACY_MODEL_VERSION, model=legacy.state_dict(),
                     architecture=dict(dim=self.cfg.dim, heads=self.cfg.heads, blocks=1,
                                       mlp_ratio=self.cfg.mlp_ratio, dropout=self.cfg.dropout))
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, "legacy architecture"):
            training.load_checkpoint(model, None, "cpu", self.cfg)
        with patch.object(play, "CHECKPOINT_PATH", path), patch.object(play, "DEVICE", "cpu"):
            loaded, cfg, device = play.load_model()
        player = play.Player(loaded, cfg, device)
        inputs = training.encode_batch([player.board], device)
        torch.testing.assert_close(loaded(*inputs), legacy(*inputs))


if __name__ == "__main__":
    unittest.main()
