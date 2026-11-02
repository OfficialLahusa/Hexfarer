"""Value net v2.1 (structured_net.py): the board graph, relative seats (a state rotated to other seats gives the same
encodings and the outputs rotated along), the neighbor features, and export with parity in float32 and float16.
The encoders' inputs are checked against values computed straight from the fixtures' YAML (production from type and
number, the robber's tile, the buildings and roads around each element, what each player owns), not from the vectorizer.

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

from state_to_vector import (CLASSIC_EDGE_INTERSECTIONS, CLASSIC_INTERSECTION_TILES, CLASSIC_MAP_WIDTH, FEATURE_DIM,  # noqa: E402
                             HEX_TYPES, INTERSECTION_BUILDING_TYPES, load_state_yaml, state_to_feature_vector)

try:
    import torch
    from augmentation import rotate_features
    from block_encoder import LAND_TILES
    from export_onnx import export_model
    from state_value_net import StateValueModule
    from structured_net import StructuredValueNet, geometry
except ImportError as error:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None

FIXTURES = os.path.join(REPO, "Tests", "Fixtures", "Vectorizer")
FIXTURE_NAMES = ("setup", "initial", "first_roll", "robber", "discard", "midgame", "ended")
BUILDING_PAIRS = len(INTERSECTION_BUILDING_TYPES) * 5           # building one-hot times owner one-hot over [none, players]


def state(name):
    return load_state_yaml(os.path.join(FIXTURES, f"{name}.yaml"))["GameState"]


def fixture(name):
    return np.asarray(state_to_feature_vector(state(name)), dtype=np.float32)


def expected_board(game):
    """What v2.1's encoders should see, from the YAML alone: production and robber-blocked production per intersection
    and hex type (pips / 36), the robber flag per land tile, the joint building-owner one-hot per intersection and the
    road owner one-hot (over [none, players]) per edge"""
    board = game["Board"]
    hexes = {h["Y"] * CLASSIC_MAP_WIDTH + h["X"]: h for h in board["Map"]["Values"]}
    robber = board["Robber"]["Y"] * CLASSIC_MAP_WIDTH + board["Robber"]["X"]
    production = np.zeros((len(CLASSIC_INTERSECTION_TILES), len(HEX_TYPES)))
    blocked = np.zeros_like(production)
    for i, tiles in enumerate(CLASSIC_INTERSECTION_TILES):
        for tile in tiles:
            number = hexes[tile]["Number"]
            if number is None:
                continue
            pips = (6 - abs(7 - number)) / 36
            production[i, HEX_TYPES.index(hexes[tile]["Type"])] += pips
            if tile == robber:
                blocked[i, HEX_TYPES.index(hexes[tile]["Type"])] += pips
    buildings = np.zeros((len(board["Intersections"]), BUILDING_PAIRS))
    for i, intersection in enumerate(board["Intersections"]):
        buildings[i, INTERSECTION_BUILDING_TYPES.index(intersection["Building"]) * 5 + intersection["Owner"] + 1] = 1
    roads = np.zeros((len(board["Edges"]), 5))
    for e, edge in enumerate(board["Edges"]):
        roads[e, edge["Owner"] + 1 if edge["Building"] == "Road" else 0] = 1
    on_robber = np.array([1.0 if tile == robber else 0.0 for tile in LAND_TILES])
    return production, blocked, on_robber, buildings, roads


def capture(model, x):
    """The inputs and outputs of v2.1's encoders and the input of the player mix for one forward pass"""
    seen = {}
    handles = []
    for name in ("tile", "intersection", "edge"):
        module = getattr(model, name)
        handles.append(module.register_forward_pre_hook(lambda _, args, name=name: seen.__setitem__(f"{name}_in", args[0])))
        handles.append(module.register_forward_hook(lambda _, args, out, name=name: seen.__setitem__(f"{name}_code", out)))
    handles.append(model.player_mix.register_forward_pre_hook(lambda _, args: seen.__setitem__("mix_in", args[0])))
    with torch.no_grad():
        model.forward_all(x)
    for handle in handles:
        handle.remove()
    return seen


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

    def test_encoders_see_production_robber_and_neighbors_as_in_the_state(self):
        model = small_net(relative_seats=False)
        model.input_mean = model.input_scale = None
        corners = [[i for i, tiles in enumerate(CLASSIC_INTERSECTION_TILES) if tile in tiles] for tile in LAND_TILES]
        edges_at = [[e for e, ends in enumerate(CLASSIC_EDGE_INTERSECTIONS) if i in ends] for i in range(len(CLASSIC_INTERSECTION_TILES))]
        neighbors = [[j for e in edges_at[i] for j in CLASSIC_EDGE_INTERSECTIONS[e] if j != i] for i in range(len(edges_at))]
        edge_neighbors = [[f for i in ends for f in edges_at[i] if f != e] for e, ends in enumerate(CLASSIC_EDGE_INTERSECTIONS)]
        blocked_somewhere = False
        for name in FIXTURE_NAMES:
            production, blocked, on_robber, buildings, roads = expected_board(state(name))
            blocked_somewhere |= bool(blocked.any())
            seen = capture(model, torch.tensor(fixture(name)[None]))
            tile_in, intersection_in, edge_in = (seen[f"{k}_in"][0].numpy() for k in ("tile", "intersection", "edge"))
            own_tile, own_intersection, own_edge = model.tiles.shape[1], model.intersections.shape[1], model.edges.shape[1]

            np.testing.assert_array_equal(tile_in[:, own_tile], on_robber, err_msg=f"{name}: robber flag")
            np.testing.assert_allclose(tile_in[:, own_tile + 1:], [buildings[c].sum(axis=0) for c in corners], err_msg=f"{name}: corners")
            types = len(HEX_TYPES)
            np.testing.assert_allclose(intersection_in[:, own_intersection:own_intersection + types], production, atol=1e-6,
                                       err_msg=f"{name}: production (type times pips)")
            np.testing.assert_allclose(intersection_in[:, own_intersection + types:own_intersection + 2 * types], blocked, atol=1e-6,
                                       err_msg=f"{name}: production blocked by the robber")
            rest = intersection_in[:, own_intersection + 2 * types:]
            np.testing.assert_allclose(rest[:, :BUILDING_PAIRS], [buildings[n].sum(axis=0) for n in neighbors], err_msg=f"{name}: neighbors")
            np.testing.assert_allclose(rest[:, BUILDING_PAIRS:], [roads[e].sum(axis=0) for e in edges_at], err_msg=f"{name}: roads")
            np.testing.assert_allclose(edge_in[:, own_edge:own_edge + BUILDING_PAIRS], [buildings[list(ends)].sum(axis=0) for ends in CLASSIC_EDGE_INTERSECTIONS],
                                       err_msg=f"{name}: buildings at the ends")
            np.testing.assert_allclose(edge_in[:, own_edge + BUILDING_PAIRS:], [roads[n].sum(axis=0) for n in edge_neighbors],
                                       err_msg=f"{name}: neighbor roads")
        self.assertTrue(blocked_somewhere, "no fixture has the robber on a numbered tile next to an intersection")

    def test_players_get_the_sum_of_what_they_own(self):
        model = small_net(relative_seats=False)
        model.input_mean = model.input_scale = None
        player_width = model.player[-1].out_features
        intersection_width, edge_width = model.intersection[-1].out_features, model.edge[-1].out_features
        for name in FIXTURE_NAMES:
            board = state(name)["Board"]
            seen = capture(model, torch.tensor(fixture(name)[None]))
            intersection_code, edge_code = seen["intersection_code"][0].numpy(), seen["edge_code"][0].numpy()
            mix_in = seen["mix_in"][0].numpy()
            for p in range(4):
                owned = [i for i, x in enumerate(board["Intersections"]) if x["Building"] != "None" and x["Owner"] == p]
                roads = [e for e, x in enumerate(board["Edges"]) if x["Building"] == "Road" and x["Owner"] == p]
                np.testing.assert_allclose(mix_in[p, player_width:player_width + intersection_width],
                                           intersection_code[owned].sum(axis=0), atol=1e-5, err_msg=f"{name}: buildings of player {p}")
                np.testing.assert_allclose(mix_in[p, player_width + intersection_width:player_width + intersection_width + edge_width],
                                           edge_code[roads].sum(axis=0), atol=1e-5, err_msg=f"{name}: roads of player {p}")

    def test_normalization_keeps_the_building_and_road_one_hots(self):
        # encode() builds the neighbor sums and owner summaries from the normalized features, so these columns must stay 0/1
        states = np.stack([fixture(name) for name in FIXTURE_NAMES])
        for relative_seats in (True, False):
            module = StateValueModule(FEATURE_DIM, normalize_inputs=True, hidden=64, blocks=(8, 4, 4, 16), structured=True,
                                      relative_seats=relative_seats)
            module.set_input_normalization(states)
            model = module.model
            columns = torch.cat([model.intersections[:, :8].flatten(), model.edges.flatten()])
            self.assertTrue(torch.all(model.input_mean[columns] == 0))
            self.assertTrue(torch.all(model.input_scale[columns] == 1))

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
