"""Training additions: batches that keep sibling groups together, the auxiliary and ranking losses, and that a model with
auxiliary heads still exports only the value logits.

Skipped when torch, lightning or onnxruntime are not installed.
Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import json
import os
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

try:
    import numpy as np
    import onnxruntime as ort
    import torch
    from export_onnx import export_model, output_difference
    from state_value_net import StateValueModule, masked_kl, policy_cross_entropy, sibling_ranking
    from train import GroupBatchSampler, MixedDataset, MixSampler, StateDataset, policy_dirs, widen_state_dict
except ImportError as error:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


def write_source(directory, rows, features, label_dtype, self_play):
    """A prepared data directory with random states and labels, plus root_q.npy and winners.npy for self-play data."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(rows)
    np.save(os.path.join(directory, "states.npy"), rng.random((rows, features)).astype(np.float16))
    np.save(os.path.join(directory, "win_rates.npy"), rng.dirichlet(np.ones(4), rows).astype(label_dtype))
    if self_play:
        np.save(os.path.join(directory, "root_q.npy"), rng.dirichlet(np.ones(4), rows).astype(np.float32))
        np.save(os.path.join(directory, "winners.npy"), np.array([3, 2, 1] * (rows // 3 + 1), dtype=np.int8)[:rows])


def random_moves(rng, counts, max_tokens=3):
    """Moves per row as (tokens, weights, share): a move type, then board or resource tokens (-1 after the used ones),
    and visit shares that add up to 1 per row"""
    rows = []
    for count in counts:
        shares = rng.dirichlet(np.ones(count)) if count else []
        moves = []
        for m in range(count):
            used = int(rng.integers(1, max_tokens + 1))
            tokens = [int(rng.integers(0, 19))] + [int(rng.integers(19, 213)) for _ in range(used - 1)] + [-1] * (max_tokens - used)
            weights = [1] + [int(rng.integers(1, 10)) for _ in range(used - 1)] + [0] * (max_tokens - used)
            moves.append((tokens, weights, float(shares[m])))
        rows.append(moves)
    return rows


def write_policy(directory, rows, max_tokens):
    """The policy arrays of a dataset (as from selfplay_to_dataset.py or a reanalysis) with the given moves per row"""
    os.makedirs(directory, exist_ok=True)
    moves = [move for row in rows for move in row]
    np.save(os.path.join(directory, "policy_offsets.npy"), np.concatenate([[0], np.cumsum([len(row) for row in rows])]).astype(np.int64))
    np.save(os.path.join(directory, "policy_tokens.npy"), np.array([m[0] for m in moves], dtype=np.int16).reshape(-1, max_tokens))
    np.save(os.path.join(directory, "policy_weights.npy"), np.array([m[1] for m in moves], dtype=np.uint8).reshape(-1, max_tokens))
    np.save(os.path.join(directory, "policy_targets.npy"), np.array([m[2] for m in moves], dtype=np.float32))
    with open(os.path.join(directory, "policy_layout.json"), "w") as file:
        json.dump(POLICY_LAYOUT | {"max_tokens": [max_tokens]}, file)


def flat_batch(rows, max_tokens, order):
    """A policy batch as StateDataset builds it, with the moves in the given order instead of row by row (the rows and
    slots say where each move belongs, so the order must not matter)"""
    moves = [(r, s, move) for r, row in enumerate(rows) for s, move in enumerate(row)]
    moves = [moves[i] for i in order]
    width = max([len(row) for row in rows] + [1])
    targets, mask = torch.zeros((len(rows), width)), torch.zeros((len(rows), width), dtype=torch.bool)
    for r, s, (_, _, share) in moves:
        targets[r, s], mask[r, s] = share, True
    return {"policy_tokens": torch.tensor([m[2][0] for m in moves], dtype=torch.int64).reshape(-1, max_tokens),
            "policy_weights": torch.tensor([m[2][1] for m in moves], dtype=torch.float32).reshape(-1, max_tokens),
            "policy_rows": torch.tensor([m[0] for m in moves], dtype=torch.int64),
            "policy_slots": torch.tensor([m[1] for m in moves], dtype=torch.int64),
            "policy_targets": targets, "policy_mask": mask}


def reference_policy_loss(embeddings, query, rows, temperature=1.0):
    """The policy loss row by row in plain loops, independent of the batch layout: the logit of a move is the query of its
    type times the weighted sum of its token embeddings, then a softmax over the row's moves against its visit shares
    (to the power 1/T, normalized); rows without moves don't count. Also the share of rows whose most visited move has the
    highest logit."""
    losses, hits = [], []
    for r, moves in enumerate(rows):
        if not moves:
            continue
        logits = []
        for tokens, weights, _ in moves:
            embedded = sum(weight * embeddings[token] for token, weight in zip(tokens, weights) if token >= 0)
            logits.append((query[r, tokens[0]] * embedded).sum())
        logits = torch.stack(logits)
        shares = torch.tensor([share for _, _, share in moves]) ** (1.0 / temperature)
        shares = shares / shares.sum()
        losses.append(-(shares * torch.log_softmax(logits, dim=0)).sum())
        hits.append(float(int(logits.argmax()) == int(shares.argmax())))
    return torch.stack(losses).mean(), sum(hits) / len(hits)


def moves_of(batch, row):
    """The moves of one batch row, found through their rows and slots, in slot order: tokens, weights and targets"""
    pick = torch.nonzero(batch["policy_rows"] == row).flatten()
    pick = pick[torch.argsort(batch["policy_slots"][pick])]
    slots = batch["policy_slots"][pick]
    return batch["policy_tokens"][pick], batch["policy_weights"][pick], batch["policy_targets"][row, slots]


def check_layout(test, batch):
    """Every move has its own cell, the cells of a row are its first slots, and the mask marks exactly those cells"""
    rows, slots, mask = batch["policy_rows"], batch["policy_slots"], batch["policy_mask"]
    test.assertEqual(len(rows), len(batch["policy_tokens"]))
    test.assertEqual(len(rows), len(batch["policy_weights"]))
    cells = set(zip(rows.tolist(), slots.tolist()))
    test.assertEqual(len(rows), len(cells), "two moves in one cell")
    test.assertEqual(cells, set(map(tuple, torch.nonzero(mask).tolist())), "the mask doesn't mark the moves' cells")
    for r in range(mask.shape[0]):
        test.assertEqual(sorted(s for row, s in cells if row == r), list(range(int(mask[r].sum()))), f"row {r} has gaps")
    test.assertTrue((batch["policy_targets"][~mask] == 0).all())


@unittest.skipIf(IMPORT_ERROR is not None, f"training dependencies missing: {IMPORT_ERROR}")
class TrainingTest(unittest.TestCase):
    def test_a_widened_checkpoint_computes_the_same_on_the_old_features(self):
        # The view layout appends features: the old net's weights go on, the new columns start at 0
        old = StateValueModule(in_dim=8, hidden=16, normalize_inputs=True, placement_weight=0.1)
        old.model.input_mean.copy_(torch.arange(8, dtype=torch.float32))
        old.model.input_scale.copy_(torch.full((8,), 0.5))
        wide = StateValueModule(in_dim=11, hidden=16, normalize_inputs=True, placement_weight=0.1)
        widened = widen_state_dict(old.state_dict(), 11, new_mean=np.array([1.0, 2.0, 3.0]), new_scale=np.array([0.1, 0.2, 0.3]))
        wide.load_state_dict(widened, strict=True)
        np.testing.assert_allclose(wide.model.input_mean[8:].numpy(), [1, 2, 3])
        np.testing.assert_allclose(wide.model.input_scale[8:].numpy(), [0.1, 0.2, 0.3], rtol=1e-6)

        old.eval()
        wide.eval()
        x = torch.rand(5, 11) * 4
        with torch.no_grad():
            torch.testing.assert_close(wide.model(x), old.model(x[:, :8]))

        # A checkpoint of the same width stays as it is, a wider one can't be narrowed
        self.assertIs(old.state_dict()["model.net.0.weight"].shape[1], 8)
        same = widen_state_dict(old.state_dict(), 8)
        torch.testing.assert_close(same["model.net.0.weight"], old.state_dict()["model.net.0.weight"])
        with self.assertRaises(ValueError):
            widen_state_dict(wide.state_dict(), 8)

    def test_a_source_can_give_another_feature_file(self):
        with tempfile.TemporaryDirectory() as directory:
            write_source(directory, 6, 5, np.float32, self_play=False)
            np.save(os.path.join(directory, "views.npy"), np.ones((6, 7), dtype=np.float16))
            dataset = StateDataset(os.path.join(directory, "views.npy"), os.path.join(directory, "win_rates.npy"), directory)
            self.assertEqual((6, 7), dataset.states.shape)
            # Windows keeps memory-mapped files open until the arrays are gone
            del dataset
            import gc
            gc.collect()

    def test_batches_keep_sibling_groups_together(self):
        groups = np.array([-1, 0, 0, 0, -1, 1, 1, -1, 2, 2, 2, 2, -1, -1, 3, 3])
        indices = np.arange(1, 16)  # sample 0 is not in this split
        np.random.seed(0)

        for shuffle in (True, False):
            batches = list(GroupBatchSampler(indices, groups, batch_size=4, shuffle=shuffle))
            self.assertEqual(indices.tolist(), sorted(i for batch in batches for i in batch))
            for group in range(4):
                self.assertEqual(1, sum(any(groups[i] == group for i in batch) for batch in batches), f"group {group} split")

    def test_losses_skip_missing_labels(self):
        uniform = torch.full((2, 4), 0.25)
        target = torch.tensor([[0.25] * 4, [float("nan")] * 4])
        self.assertAlmostEqual(0.0, masked_kl(torch.zeros(2, 4), target).item(), places=6)
        self.assertEqual(0.0, masked_kl(torch.zeros(1, 4), torch.full((1, 4), float("nan"))).item())
        self.assertGreater(masked_kl(torch.tensor([[5.0, 0, 0, 0]]), uniform[:1]).item(), 0)

    def test_ranking_compares_the_actors_value_within_groups(self):
        labels = torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.1, 0.4, 0.4, 0.1], [0.7, 0.1, 0.1, 0.1], [0.25] * 4])
        groups, actors = torch.tensor([5, 5, -1, -1]), torch.tensor([1, 1, 0, 0])

        # Logits that rank the actor's (player 1's) value the same way as the labels: right order, smaller loss
        right = torch.tensor([[0, 2.0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        loss_right, accuracy_right, pairs = sibling_ranking(right, labels, groups, actors)
        loss_wrong, accuracy_wrong, _ = sibling_ranking(right[[1, 0, 2, 3]], labels, groups, actors)

        self.assertEqual(1, pairs)
        self.assertEqual(1.0, accuracy_right.item())
        self.assertEqual(0.0, accuracy_wrong.item())
        self.assertLess(loss_right.item(), loss_wrong.item())

        # Without groups in the batch there is nothing to compare
        self.assertEqual(0, sibling_ranking(right, labels, torch.full((4,), -1), actors)[2])

    def test_the_placement_score_lets_the_ranking_learn_far_behind(self):
        # The actor (player 1) never wins in either sibling, but finishes second more often in the first
        win_rates = torch.tensor([[0.5, 0.0, 0.3, 0.2], [0.5, 0.0, 0.3, 0.2]])
        placement_labels = torch.tensor([[[0.5, 0.2, 0.2, 0.1], [0.0, 0.6, 0.3, 0.1], [0.3, 0.1, 0.4, 0.2], [0.2, 0.1, 0.1, 0.6]],
                                         [[0.5, 0.2, 0.2, 0.1], [0.0, 0.2, 0.3, 0.5], [0.3, 0.4, 0.2, 0.1], [0.2, 0.2, 0.3, 0.3]]])
        groups, actors = torch.tensor([3, 3]), torch.tensor([1, 1])
        logits = torch.zeros(2, 4)

        # On the win rates alone there is nothing to rank, with the placement score the first sibling has to come out ahead
        self.assertEqual(0.0, sibling_ranking(logits, win_rates, groups, actors)[0].item())
        right = torch.zeros(2, 4, 4)
        right[0, 1, 1] = 3.0  # the net puts player 1 in second place in the first sibling
        loss_right = sibling_ranking(logits, win_rates, groups, actors, right, placement_labels, 0.5)[0]
        loss_flat = sibling_ranking(logits, win_rates, groups, actors, torch.zeros(2, 4, 4), placement_labels, 0.5)[0]
        self.assertGreater(loss_flat.item(), 0.0)
        self.assertLess(loss_right.item(), loss_flat.item())

    def test_the_export_parity_catches_small_values_off_next_to_large_ones(self):
        tolerance = 1e-4  # export_model's for float32 exports
        logits = np.array([[480.0, 0.5, -0.25, 3.0]], dtype=np.float32)
        # float32 rounding of logits in the hundreds passes
        self.assertLess(output_difference(logits, logits + np.array([[3e-3, 1e-6, 0, 2e-5]], dtype=np.float32)), tolerance)
        # A small value 2% off next to a large one does not, though it is far below the largest value's rounding
        off = logits.copy()
        off[0, 1] += 0.01
        self.assertGreater(output_difference(logits, off), tolerance)
        # float16 queries: their rounding passes, a query 10% off does not
        queries = np.array([[1.5, -0.002, 0.75]], dtype=np.float16)
        self.assertLess(output_difference(queries, queries.astype(np.float32) * np.float32(1.001)), tolerance)
        self.assertGreater(output_difference(queries, queries.astype(np.float32) * np.array([[1, 1, 1.1]], dtype=np.float32)), tolerance)

    def test_a_model_with_auxiliary_heads_exports_the_value_and_the_placements(self):
        torch.manual_seed(0)
        module = StateValueModule(in_dim=32, placement_weight=1.0, vp_share_weight=1.0, rounds_weight=1.0, ranking_weight=1.0)
        model = module.model.eval()
        outputs = model.forward_all(torch.randn(3, 32))
        self.assertEqual((3, 4), outputs["logits"].shape)
        self.assertEqual((3, 4, 4), outputs["placements"].shape)
        self.assertEqual((3,), outputs["rounds_left"].shape)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state_value_net.onnx")
            self.assertLess(export_model(model, path, 32), 1e-4)
            # The C# side reads "logits" by name and the placement distribution logits from "placements"
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.assertEqual(["logits", "placements"], [output.name for output in session.get_outputs()])
            logits, placements = session.run(None, {"state": np.zeros((2, 32), dtype=np.float32)})
            self.assertEqual((2, 4), logits.shape)
            self.assertEqual((2, 4, 4), placements.shape)

        # Input normalization is part of the model, so the export applies it too
        normalized = StateValueModule(in_dim=32, normalize_inputs=True)
        states = np.random.default_rng(1).integers(0, 2, (500, 32)).astype(np.float32)
        states[:, :4] = np.random.default_rng(2).normal(5, 3, (500, 4))
        self.assertEqual(4, normalized.set_input_normalization(states))
        standardized = normalized.model._normalize(torch.from_numpy(states)).numpy()
        np.testing.assert_allclose(standardized[:, :4].mean(axis=0), 0, atol=1e-4)
        np.testing.assert_allclose(standardized[:, :4].std(axis=0), 1, atol=1e-3)
        np.testing.assert_array_equal(standardized[:, 4:], states[:, 4:])
        with tempfile.TemporaryDirectory() as directory:
            self.assertLess(export_model(normalized.model.eval(), os.path.join(directory, "normalized.onnx"), 32), 1e-4)

        # Aux heads for other losses only: the placements head is untrained, so the export leaves it out
        rounds_only = StateValueModule(in_dim=32, rounds_weight=1.0)
        self.assertIsNotNone(rounds_only.model.aux)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rounds_only.onnx")
            self.assertLess(export_model(rounds_only.model.eval(), path, 32), 1e-4)
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.assertEqual(["logits"], [output.name for output in session.get_outputs()])

        # Without auxiliary losses the model has exactly the parameters of earlier checkpoints
        plain = StateValueModule(in_dim=32)
        self.assertIsNone(plain.model.aux)
        self.assertTrue(all(name.startswith("model.net.") for name in plain.state_dict()))

    def test_self_play_targets_mix_the_result_and_the_root_values(self):
        # The datasets keep their arrays memory-mapped, which Windows doesn't let the cleanup delete
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            write_source(directory, rows=3, features=8, label_dtype=np.float32, self_play=True)
            dataset = StateDataset(os.path.join(directory, "states.npy"), os.path.join(directory, "win_rates.npy"), directory, target_lambda=0.25)
            batch = dataset.__getitems__([2, 0])
            root_q = np.load(os.path.join(directory, "root_q.npy"))
            expected = 0.75 * root_q[[2, 0]] + 0.25 * np.eye(4)[[1, 3]]
            np.testing.assert_allclose(batch["y"].numpy(), expected, rtol=1e-6)

            # Without λ, or without the self-play arrays, the stored labels stay
            stored = StateDataset(os.path.join(directory, "states.npy"), os.path.join(directory, "win_rates.npy"), directory)
            np.testing.assert_array_equal(stored.__getitems__([2])["y"].numpy(), np.load(os.path.join(directory, "win_rates.npy"))[[2]])

    def test_mixed_sources_are_drawn_by_their_shares(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            first, second = os.path.join(directory, "a"), os.path.join(directory, "b")
            write_source(first, rows=10, features=8, label_dtype=np.float32, self_play=True)
            write_source(second, rows=50, features=8, label_dtype=np.float16, self_play=False)
            datasets = [StateDataset(os.path.join(d, "states.npy"), os.path.join(d, "win_rates.npy"), d, target_lambda=0.5) for d in (first, second)]
            mixed = MixedDataset(datasets)
            self.assertEqual(60, len(mixed))

            # The first source's 10 rows at share 1, the second's 50 at share 3: 40 per epoch, 10 and 30
            sampler = MixSampler([np.arange(10), np.arange(50) + 10], [1, 3], epoch_samples=40)
            drawn = np.array(list(sampler))
            self.assertEqual(40, len(drawn))
            self.assertEqual(10, int((drawn < 10).sum()))
            self.assertEqual(10, len(set(drawn[drawn < 10].tolist())), "a source with enough rows is drawn without replacement")

            batch = mixed.__getitems__([0, 15, 59, 3])
            self.assertEqual(torch.float32, batch["y"].dtype)
            self.assertEqual((4, 8), tuple(batch["x"].shape))
            # Rows come back per source; the second source's rows are its stored states
            np.testing.assert_array_equal(batch["x"][2:].numpy(), np.load(os.path.join(second, "states.npy"))[[5, 49]])


    def test_board_symmetries_move_the_move_tokens_like_the_features(self):
        from augmentation import BOARD_INDEX, BOARD_SYMMETRY_COUNT, _EDGE_SIZE, _EDGE_START, move_token_symmetries
        from state_to_vector import _INTERSECTION_SIZE, _INTERSECTION_START, _TILE_SIZE, _TILE_START
        table = move_token_symmetries(POLICY_LAYOUT)
        for t in range(BOARD_SYMMETRY_COUNT):
            for name, start, size in (("intersections", _INTERSECTION_START, _INTERSECTION_SIZE), ("edges", _EDGE_START, _EDGE_SIZE),
                                      ("tiles", _TILE_START, _TILE_SIZE)):
                first, count = POLICY_LAYOUT[name]
                for old in range(count):
                    new = table[t, first + old] - first
                    # The feature block of the element moves to the same place as its token
                    self.assertEqual(start + old * size, BOARD_INDEX[t][start + new * size], f"{name} {old} under symmetry {t}")
            # Move types, resources and victims stay
            np.testing.assert_array_equal(table[t, :POLICY_LAYOUT["intersections"][0]], np.arange(POLICY_LAYOUT["intersections"][0]))
            victims = POLICY_LAYOUT["victims"][0]
            np.testing.assert_array_equal(table[t, victims:], np.arange(victims, POLICY_LAYOUT["vocabulary"][1]))

    def test_the_policy_loss_matches_a_row_by_row_reference_whatever_the_order_of_the_moves(self):
        # Rows with 3, 0, 1, 7, 300 and 2 moves: the loss, the accuracy and the gradients of the queries and the token
        # embeddings must equal the plain loop over the rows, for the moves row by row, reversed and shuffled
        torch.manual_seed(1)
        rng = np.random.default_rng(1)
        module = StateValueModule(in_dim=8, hidden=16, policy_weight=1.0, policy_types=19, policy_vocabulary=213, policy_dim=4)
        rows = random_moves(rng, [3, 0, 1, 7, 300, 2])
        query = module.model.forward_all(torch.randn(len(rows), 8))["policy_query"].detach().requires_grad_()
        embeddings = module.model.policy_embeddings.weight
        expected, expected_hits = reference_policy_loss(embeddings, query, rows)
        expected_query, expected_embeddings = torch.autograd.grad(expected, [query, embeddings])
        total = sum(len(row) for row in rows)
        for name, order in (("row by row", range(total)), ("reversed", range(total - 1, -1, -1)), ("shuffled", rng.permutation(total))):
            with self.subTest(name):
                batch = flat_batch(rows, 3, list(order))
                loss, hits = policy_cross_entropy(module.model, query, batch)
                self.assertAlmostEqual(expected.item(), loss.item(), places=5)
                self.assertAlmostEqual(expected_hits, hits.item(), places=6)
                gradient_query, gradient_embeddings = torch.autograd.grad(loss, [query, embeddings])
                torch.testing.assert_close(gradient_query, expected_query, rtol=1e-4, atol=1e-6)
                torch.testing.assert_close(gradient_embeddings, expected_embeddings, rtol=1e-4, atol=1e-6)
                # Rows without moves get no gradient
                self.assertTrue((gradient_query[1] == 0).all())

    def test_the_structured_net_scores_moves_like_the_mlp(self):
        from structured_net import StructuredValueNet
        rng = np.random.default_rng(2)
        module = StateValueModule(in_dim=8, hidden=16, policy_weight=1.0, policy_types=19, policy_vocabulary=213, policy_dim=4)
        batch = flat_batch(random_moves(rng, [4, 2, 5]), 3, list(rng.permutation(11)))
        query = torch.randn(3, 19, 4)
        args = (query, batch["policy_rows"], batch["policy_tokens"], batch["policy_weights"])
        torch.testing.assert_close(StructuredValueNet.policy_logits(module.model, *args), module.model.policy_logits(*args))

    def test_a_batch_without_any_moves_has_no_policy_loss(self):
        module = StateValueModule(in_dim=8, hidden=16, policy_weight=1.0, policy_types=19, policy_vocabulary=213, policy_dim=4)
        query = torch.randn(3, 19, 4, requires_grad=True)
        loss, hits = policy_cross_entropy(module.model, query, flat_batch([[], [], []], 3, []))
        self.assertEqual(0.0, loss.item())
        self.assertEqual(0.0, hits.item())
        loss.backward()
        self.assertTrue((query.grad == 0).all())

    def test_each_row_of_a_batch_gets_exactly_its_own_moves(self):
        # Requested out of order and with a row twice (sampling with replacement), at temperature 2
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            source = os.path.join(directory, "v2")
            write_source(source, rows=8, features=8, label_dtype=np.float16, self_play=False)
            stored = random_moves(np.random.default_rng(3), [2, 0, 5, 1, 40, 0, 3, 6])
            write_policy(source, stored, 3)
            dataset = StateDataset(os.path.join(source, "states.npy"), os.path.join(source, "win_rates.npy"), source, policy_temperature=2.0)
            indices = [6, 2, 4, 0, 5, 2, 7, 3, 1]
            batch = dataset.__getitems__(indices)
            check_layout(self, batch)
            for row, index in enumerate(indices):
                tokens, weights, targets = moves_of(batch, row)
                moves = stored[index]
                np.testing.assert_array_equal(tokens.numpy(), np.array([m[0] for m in moves], dtype=np.int64).reshape(-1, 3), f"row {row}")
                np.testing.assert_array_equal(weights.numpy(), np.array([m[1] for m in moves], dtype=np.float32).reshape(-1, 3), f"row {row}")
                shares = np.array([m[2] for m in moves]) ** 0.5
                np.testing.assert_allclose(targets.numpy(), shares / shares.sum() if len(moves) else [], rtol=1e-5, err_msg=f"row {row}")
            # The loss of the batch is the reference loss of the stored moves of the requested rows
            module = StateValueModule(in_dim=8, hidden=16, policy_weight=1.0, policy_types=19, policy_vocabulary=213, policy_dim=4)
            query = torch.randn(len(indices), 19, 4)
            loss, _ = policy_cross_entropy(module.model, query, batch)
            expected, _ = reference_policy_loss(module.model.policy_embeddings.weight, query, [stored[i] for i in indices], temperature=2.0)
            self.assertAlmostEqual(expected.item(), loss.item(), places=5)

    def test_each_row_turns_its_moves_with_its_own_board_symmetry(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            source = os.path.join(directory, "v2")
            write_source(source, rows=4, features=8, label_dtype=np.float16, self_play=False)
            # Board tokens only (intersections, edges, tiles), which every symmetry but the identity moves
            stored = [[([5, 19 + i, 73 + j], [1, 1, 1], 0.5) for i, j in ((0, 3), (20, 40))] for _ in range(4)]
            write_policy(source, stored, 3)
            dataset = StateDataset(os.path.join(source, "states.npy"), os.path.join(source, "win_rates.npy"), source)
            ts = np.array([3, 0, 7, 11])
            batch = dataset._policy_batch(np.array([2, 0, 3, 1]), ts)
            for row in range(4):
                tokens, _, _ = moves_of(batch, row)
                expected = dataset.token_symmetries[ts[row]][np.array([m[0] for m in stored[0]])]
                np.testing.assert_array_equal(tokens.numpy(), expected, f"row {row} under symmetry {ts[row]}")
            self.assertFalse((dataset.token_symmetries[3][[19, 73]] == [19, 73]).all(), "the symmetry must move the tokens")

    def test_mixed_sources_keep_each_rows_moves_with_its_row(self):
        # A source without policy targets (one token per move), then two with (3 and 2 tokens per move); the batch comes
        # back source by source, and the moves of the later sources must point at their rows behind the earlier ones
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            rng = np.random.default_rng(4)
            paths = [os.path.join(directory, name) for name in ("plain", "first", "second")]
            write_source(paths[0], rows=5, features=8, label_dtype=np.float16, self_play=False)
            write_source(paths[1], rows=6, features=8, label_dtype=np.float32, self_play=True)
            write_source(paths[2], rows=6, features=8, label_dtype=np.float16, self_play=False)
            stored = {1: random_moves(rng, [3, 1, 0, 4, 2, 6], 3), 2: random_moves(rng, [2, 5, 1, 0, 3, 2], 2)}
            write_policy(paths[1], stored[1], 3)
            write_policy(paths[2], stored[2], 2)
            mixed = MixedDataset([StateDataset(os.path.join(p, "states.npy"), os.path.join(p, "win_rates.npy"), p) for p in paths])
            indices = np.array([12, 3, 5, 16, 0, 8, 11, 13, 6])
            batch = mixed.__getitems__(indices)
            check_layout(self, batch)
            self.assertEqual(3, batch["policy_tokens"].shape[1])
            source = np.searchsorted(mixed.offsets, indices, side="right") - 1
            order = [(s, i - mixed.offsets[s]) for s in np.unique(source) for i in indices[source == s]]
            expected_rows = []
            for row, (s, index) in enumerate(order):
                np.testing.assert_array_equal(batch["x"][row].numpy(), np.load(os.path.join(paths[s], "states.npy"))[index])
                moves = stored[s][index] if s in stored else []
                expected_rows.append([(m[0] + [-1] * (3 - len(m[0])), m[1] + [0] * (3 - len(m[1])), m[2]) for m in moves])
                tokens, weights, targets = moves_of(batch, row)
                np.testing.assert_array_equal(tokens.numpy(), np.array([m[0] for m in expected_rows[-1]], dtype=np.int64).reshape(-1, 3), f"row {row}")
                np.testing.assert_array_equal(weights.numpy(), np.array([m[1] for m in expected_rows[-1]], dtype=np.float32).reshape(-1, 3), f"row {row}")
                np.testing.assert_allclose(targets.numpy(), [m[2] for m in moves], rtol=1e-6, err_msg=f"row {row}")
            module = StateValueModule(in_dim=8, hidden=16, policy_weight=1.0, policy_types=19, policy_vocabulary=213, policy_dim=4)
            query = torch.randn(len(indices), 19, 4)
            loss, _ = policy_cross_entropy(module.model, query, batch)
            expected, _ = reference_policy_loss(module.model.policy_embeddings.weight, query, expected_rows)
            self.assertAlmostEqual(expected.item(), loss.item(), places=5)

    def test_policy_targets_can_come_from_another_directory(self):
        # A reanalysis of a value-only source: one entry per state, entries without moves stay out of the policy loss
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            source, reanalysis = os.path.join(directory, "v2"), os.path.join(directory, "reanalysis")
            write_source(source, rows=3, features=8, label_dtype=np.float16, self_play=False)

            def write_reanalysis(path, offsets):
                os.makedirs(path)
                np.save(os.path.join(path, "policy_offsets.npy"), np.array(offsets))
                np.save(os.path.join(path, "policy_tokens.npy"), np.array([[1, -1], [2, -1], [3, 21]], dtype=np.int16))
                np.save(os.path.join(path, "policy_weights.npy"), np.ones((3, 2), dtype=np.uint8))
                np.save(os.path.join(path, "policy_targets.npy"), np.array([0.25, 0.75, 0], dtype=np.float32))
                with open(os.path.join(path, "policy_layout.json"), "w") as file:
                    json.dump(POLICY_LAYOUT | {"max_tokens": [2]}, file)
            write_reanalysis(reanalysis, [0, 0, 3, 3])

            # Targets of another dataset (a row more or less) or offsets that don't cover the moves stop the training
            for name, offsets in (("more", [0, 0, 3, 3, 3]), ("fewer", [0, 0, 3]), ("short", [0, 0, 2, 2]), ("start", [1, 1, 3, 3]),
                                  ("descending", [0, 2, 1, 3])):
                write_reanalysis(os.path.join(directory, name), offsets)
                with self.assertRaises(ValueError, msg=name):
                    StateDataset(os.path.join(source, "states.npy"), os.path.join(source, "win_rates.npy"), source,
                                 policy_dir=os.path.join(directory, name))

            plain = StateDataset(os.path.join(source, "states.npy"), os.path.join(source, "win_rates.npy"), source)
            self.assertEqual(plain.policy_paths, {})
            dataset = StateDataset(os.path.join(source, "states.npy"), os.path.join(source, "win_rates.npy"), source, policy_dir=reanalysis)
            batch = dataset.__getitems__([0, 1, 2])
            np.testing.assert_array_equal(batch["policy_mask"].sum(dim=1).numpy(), [0, 3, 0])
            np.testing.assert_allclose(batch["policy_targets"][1].numpy(), [0.25, 0.75, 0])
            np.testing.assert_array_equal(moves_of(batch, 1)[0][2].numpy(), [3, 21])
            # The value labels stay the source's
            np.testing.assert_allclose(batch["y"].numpy(), plain.__getitems__([0, 1, 2])["y"].numpy())

            with self.assertRaises(ValueError):
                StateDataset(os.path.join(source, "states.npy"), os.path.join(source, "win_rates.npy"), source, policy_dir=directory)
            for_source = policy_dirs([f"{source}={reanalysis}"], [source])
            self.assertEqual(for_source(source + os.sep), reanalysis)
            self.assertIsNone(for_source(directory))
            with self.assertRaises(SystemExit):
                policy_dirs([f"{directory}={reanalysis}"], [source])
            with self.assertRaises(SystemExit):
                policy_dirs([reanalysis], [source])

    def test_rows_with_more_moves_than_the_cap_get_no_policy_target(self):
        # Rows with 2, 3 (the cap), 4 and no moves: the one over the cap keeps its value label but no moves, and the batch is
        # only as wide as the largest row left
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            source, reanalysis = os.path.join(directory, "v2"), os.path.join(directory, "reanalysis")
            write_source(source, rows=4, features=8, label_dtype=np.float16, self_play=False)
            os.makedirs(reanalysis)
            np.save(os.path.join(reanalysis, "policy_offsets.npy"), np.array([0, 2, 5, 9, 9]))
            np.save(os.path.join(reanalysis, "policy_tokens.npy"), np.array([[token, -1] for token in range(1, 10)], dtype=np.int16))
            np.save(os.path.join(reanalysis, "policy_weights.npy"), np.ones((9, 2), dtype=np.uint8))
            np.save(os.path.join(reanalysis, "policy_targets.npy"), np.array([0.5, 0.5, 0.2, 0.3, 0.5, 0.1, 0.2, 0.3, 0.4], dtype=np.float32))
            with open(os.path.join(reanalysis, "policy_layout.json"), "w") as file:
                json.dump(POLICY_LAYOUT | {"max_tokens": [2]}, file)

            def batch(**cap):
                dataset = StateDataset(os.path.join(source, "states.npy"), os.path.join(source, "win_rates.npy"), source,
                                       policy_dir=reanalysis, **cap)
                return dataset.__getitems__([2, 1, 0, 3])

            capped, full = batch(policy_max_moves=3), batch()
            np.testing.assert_array_equal(full["policy_mask"].sum(dim=1).numpy(), [4, 3, 2, 0])
            np.testing.assert_array_equal(capped["policy_mask"].sum(dim=1).numpy(), [0, 3, 2, 0])
            self.assertEqual((4, 3), tuple(capped["policy_mask"].shape))
            self.assertEqual((5, 2), tuple(capped["policy_tokens"].shape))
            check_layout(self, capped)
            self.assertEqual(0, len(moves_of(capped, 0)[0]))
            np.testing.assert_array_equal(moves_of(capped, 1)[0][:, 0].numpy(), [3, 4, 5])
            np.testing.assert_allclose(capped["policy_targets"][1].numpy(), [0.2, 0.3, 0.5])
            np.testing.assert_allclose(capped["policy_targets"][2].numpy(), [0.5, 0.5, 0])
            np.testing.assert_allclose(capped["y"].numpy(), full["y"].numpy())

    def test_a_model_with_a_policy_head_exports_queries_and_embeddings(self):
        module = StateValueModule(in_dim=32, hidden=16, policy_weight=1.0, policy_types=19, policy_vocabulary=213, policy_dim=4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.onnx")
            self.assertLess(export_model(module.model.eval(), path, 32), 1e-4)
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.assertEqual(["logits", "policy_query", "policy_embeddings"], [output.name for output in session.get_outputs()])
            _, queries, embeddings = session.run(None, {"state": np.zeros((3, 32), dtype=np.float32)})
            self.assertEqual((3, 19 * 4), queries.shape)
            np.testing.assert_allclose(embeddings, module.model.policy_embeddings.weight.detach().numpy(), rtol=1e-6)


# The token layout of Agents/Inference/ActionEncoding.cs (as SelfPlayCollector writes it into policy_layout.json)
POLICY_LAYOUT = {"move_types": [0, 19], "intersections": [19, 54], "edges": [73, 72], "tiles": [145, 49], "victims": [194, 4],
                 "given_resources": [198, 5], "received_resources": [203, 5], "discarded_resources": [208, 5], "vocabulary": [0, 213],
                 "max_tokens": [6]}


if __name__ == "__main__":
    unittest.main()
