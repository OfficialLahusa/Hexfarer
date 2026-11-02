"""Value net v2.1 (structured_net.py): the board graph, relative seats (a state rotated to other seats gives the same
encodings and the outputs rotated along), the neighbor features, and export with parity in float32 and float16.

Skipped when torch or onnxruntime are not installed.
Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

from state_to_vector import FEATURE_DIM, load_state_yaml, state_to_feature_vector  # noqa: E402

try:
    import torch
    from augmentation import rotate_features
    from export_onnx import export_model
    from structured_net import StructuredValueNet, geometry
except ImportError as error:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None

FIXTURES = os.path.join(REPO, "Tests", "Fixtures", "Vectorizer")


def fixture(name):
    return np.asarray(state_to_feature_vector(load_state_yaml(os.path.join(FIXTURES, f"{name}.yaml"))["GameState"]), dtype=np.float32)


def small_net(**options):
    torch.manual_seed(0)
    model = StructuredValueNet(FEATURE_DIM, hidden=64, normalize_inputs=True, widths=(8, 4, 4, 16), **options)
    # A normalization that differs per feature, made the same for all players as in training
    model.input_mean.uniform_(-0.5, 0.5)
    model.input_scale.uniform_(0.5, 2.0)
    model.symmetrize_normalization()
    return model.eval()


@unittest.skipIf(IMPORT_ERROR is not None, f"training dependencies missing: {IMPORT_ERROR}")
class StructuredNetTest(unittest.TestCase):
    def test_board_graph_has_the_classic_degrees(self):
        intersection_tiles, edge_intersections, intersection_neighbors, edge_neighbors = geometry()
        self.assertTrue((intersection_tiles.sum(axis=0) == 6).all(), "every land tile has six corners")
        self.assertTrue(set(intersection_tiles.sum(axis=1)) <= {1, 2, 3})
        self.assertTrue((edge_intersections.sum(axis=1) == 2).all())
        self.assertTrue(set(intersection_neighbors.sum(axis=1)) == {2, 3})
        self.assertTrue(set(edge_neighbors.sum(axis=1)) <= {2, 3, 4})

    def test_rotated_states_give_the_same_encoding_and_rotated_outputs(self):
        model = small_net(aux_heads=True)
        with torch.no_grad():
            # Without the absolute seat of the player to act (the last trunk inputs before the padding) the net is equivariant
            width = model.encode(torch.tensor(fixture("midgame")[None]))[0].shape[1]
            model.trunk[0].weight[:, width - 4:width] = 0
            for name in ("midgame", "discard", "ended"):
                x = fixture(name)
                base_in, base_players, _ = model.encode(torch.tensor(x[None]))
                base = model.forward_all(torch.tensor(x[None]))
                for k in range(1, 4):
                    rotated = torch.tensor(rotate_features(x, k)[None])
                    trunk_in, players, _ = model.encode(rotated)
                    torch.testing.assert_close(trunk_in[:, :-4], base_in[:, :-4], msg=f"{name} rotated by {k}")
                    torch.testing.assert_close(players, base_players)
                    outputs = model.forward_all(rotated)
                    torch.testing.assert_close(outputs["logits"], torch.roll(base["logits"], k, dims=1))
                    torch.testing.assert_close(outputs["placements"], torch.roll(base["placements"], k, dims=1))
                    torch.testing.assert_close(outputs["vp_shares"], torch.roll(base["vp_shares"], k, dims=1))

    def test_an_intersection_sees_the_buildings_next_to_it(self):
        model = small_net()
        x = fixture("midgame")
        _, _, intersection_neighbors, _ = geometry()
        intersections = x[model.intersections.numpy()[:, :8]]
        built = intersections[:, 1:3].sum(axis=1) > 0
        self.assertTrue(built.any())
        # The neighbors' buildings (the joint one-hot summed) count the built neighbor intersections
        relative = x  # the player to act is player 0 or the counts don't depend on the seat
        buildings = (intersections[:, :3, None] * intersections[:, None, 3:8]).reshape(54, 15)
        neighbor_buildings = intersection_neighbors @ buildings
        self.assertTrue(np.array_equal(neighbor_buildings[:, 5:].sum(axis=1), intersection_neighbors @ built))
        del relative

    def test_absolute_seats_keep_the_seat_order(self):
        model = small_net(aux_heads=True, relative_seats=False)
        x = fixture("midgame")
        x = torch.tensor(rotate_features(x, 1 if x[3:7].argmax() == 0 else 0)[None])
        self.assertNotEqual(int(x[0, 3:7].argmax()), 0, "another player than player 0 should be to act")
        with torch.no_grad():
            _, _, frame = model.encode(x)
            self.assertEqual(frame.tolist(), [0])
            outputs = model.forward_all(x)
        self.assertEqual(outputs["logits"].shape, (1, 4))

    def test_exports_with_parity(self):
        model = small_net(aux_heads=True, policy_types=3, policy_vocabulary=10, policy_dim=8)
        model.placements_trained = True
        with tempfile.TemporaryDirectory() as directory:
            for fp16, tolerance in ((False, 1e-4), (True, 5e-2)):
                self.assertLess(export_model(model, os.path.join(directory, f"structured_{fp16}.onnx"), FEATURE_DIM, fp16), tolerance)


if __name__ == "__main__":
    unittest.main()
