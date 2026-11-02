"""Block encoders of the value net (block_encoder.py): every feature lands in exactly one block, the blocks read the right
features of a real state, the robber flag marks its tile, and a net with blocks exports with parity in float32 and float16.

Skipped when torch or onnxruntime are not installed.
Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import copy
import os
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

from state_to_vector import (CLASSIC_MAP_WIDTH, FEATURE_DIM, PLAYER_BLOCK_SIZE, PLAYER_BLOCK_STARTS, VIEW_FEATURE_DIM,  # noqa: E402
                             _TILE_SIZE, _TILE_START, load_state_yaml, state_to_feature_vector)

try:
    import torch
    from block_encoder import LAND_TILES, BlockEncoder, block_indices
    from export_onnx import export_model
    from state_value_net import StateValueNet
except ImportError as error:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None

FIXTURES = os.path.join(REPO, "Tests", "Fixtures", "Vectorizer")


@unittest.skipIf(IMPORT_ERROR is not None, f"training dependencies missing: {IMPORT_ERROR}")
class BlockEncoderTest(unittest.TestCase):
    def test_every_feature_but_the_constant_tiles_lands_in_exactly_one_block(self):
        for in_dim in (FEATURE_DIM, VIEW_FEATURE_DIM):
            indices = block_indices(in_dim)
            used = [i for name in ("tiles", "intersections", "edges", "players") for element in indices[name] for i in element if i != in_dim]
            used += indices["global_features"]
            self.assertEqual(len(used), len(set(used)), "a feature is in two blocks")
            constant = {_TILE_START + t * _TILE_SIZE + j for t in range(49) if t not in LAND_TILES for j in range(_TILE_SIZE)}
            self.assertEqual(set(used) | constant, set(range(in_dim)))
            self.assertEqual(len(LAND_TILES), 19)

    def test_blocks_read_the_features_of_their_element_and_the_robber_flag_marks_its_tile(self):
        state = load_state_yaml(os.path.join(FIXTURES, "midgame.yaml"))["GameState"]
        x = torch.tensor(np.asarray([state_to_feature_vector(state)], dtype=np.float32))
        encoder = BlockEncoder(FEATURE_DIM)
        tiles, intersections, edges, players, global_features = encoder.inputs(x, x)

        for slot, tile in enumerate(LAND_TILES):
            start = _TILE_START + tile * _TILE_SIZE
            self.assertTrue(torch.equal(tiles[0, slot, :_TILE_SIZE], x[0, start:start + _TILE_SIZE]))
        robber = int(round(x[0, 19].item())) * CLASSIC_MAP_WIDTH + int(round(x[0, 18].item()))
        self.assertIn(robber, LAND_TILES)
        self.assertEqual(tiles[0, :, -1].tolist(), [1.0 if tile == robber else 0.0 for tile in LAND_TILES])

        for p in range(4):
            start = PLAYER_BLOCK_STARTS[p]
            self.assertTrue(torch.equal(players[0, p, :PLAYER_BLOCK_SIZE], x[0, start:start + PLAYER_BLOCK_SIZE]))
        self.assertEqual(intersections.shape[1:], (54, 14))
        self.assertEqual(edges.shape[1:], (72, 7))
        self.assertEqual(encoder(x, x).shape, (1, encoder.out_dim))

    def test_a_change_in_the_state_shows_up_in_the_block_of_its_element(self):
        # Change one thing in the YAML, vectorize again, and check where the features changed: the index lists in
        # block_indices must match what the vectorizer writes, also within the derived and endgame features
        base = load_state_yaml(os.path.join(FIXTURES, "midgame.yaml"))["GameState"]
        before = np.asarray(state_to_feature_vector(base))
        indices = block_indices(FEATURE_DIM)
        owner = {}
        for name in ("tiles", "intersections", "edges", "players"):
            for k, element in enumerate(indices[name]):
                for i in element:
                    owner[i] = (name, k)

        def changes(edit):
            state = copy.deepcopy(base)
            edit(state)
            changed = np.nonzero(np.asarray(state_to_feature_vector(state)) != before)[0]
            self.assertGreater(len(changed), 0)
            return {owner.get(int(i), ("global", 0)) for i in changed}

        robber = base["Board"]["Robber"]["Y"] * CLASSIC_MAP_WIDTH + base["Board"]["Robber"]["X"]
        board_blocks = {"tiles", "intersections", "edges", "global"}
        # Changes that concern one player only, chosen to reach every kind of derived and endgame feature: the hand (build
        # groups, race, victory points this turn), a knight card (knight lead), a port (trade rates), a city (production)
        for p in range(4):
            def more_lumber(state, p=p):
                state["Players"][p]["ResourceCards"]["Lumber"] += 1

            def knight_in_hand(state, p=p):
                state["Players"][p]["DevelopmentCards"]["Knight"] += 1

            def wool_port(state, p=p):
                state["Players"][p]["PortPrivileges"] = "WoolTwoToOne"
            for edit in (more_lumber, knight_in_hand, wool_port):
                self.assertEqual(changes(edit), {("players", p)}, f"{edit.__name__} of player {p}")

            settlement = next(i for i, x in enumerate(base["Board"]["Intersections"]) if x["Building"] == "Settlement" and x["Owner"] == p)

            def upgrade(state, i=settlement):
                state["Board"]["Intersections"][i]["Building"] = "City"
            self.assertEqual(changes(upgrade), {("intersections", settlement), ("players", p)}, f"city of player {p}")

        hexes = {h["Y"] * CLASSIC_MAP_WIDTH + h["X"]: h for h in base["Board"]["Map"]["Values"]}
        for slot, tile in enumerate(LAND_TILES):
            if tile == robber or hexes[tile]["Number"] is None:
                continue
            def other_number(state, tile=tile):
                h = next(h for h in state["Board"]["Map"]["Values"] if h["Y"] * CLASSIC_MAP_WIDTH + h["X"] == tile)
                h["Number"] = 5 if h["Number"] != 5 else 9
            board = {element for element in changes(other_number) if element[0] in board_blocks}
            self.assertEqual(board, {("tiles", slot)}, f"number of tile {tile}")

        empty = [i for i, x in enumerate(base["Board"]["Intersections"]) if x["Building"] == "None"][:6]
        for p, i in zip(range(4), empty):
            def settle(state, i=i, p=p):
                state["Board"]["Intersections"][i].update(Building="Settlement", Owner=p)
            board = {element for element in changes(settle) if element[0] in board_blocks}
            self.assertEqual(board, {("intersections", i)}, f"settlement of player {p} on intersection {i}")

        free = [e for e, x in enumerate(base["Board"]["Edges"]) if x["Building"] == "None"][:6]
        for p, e in zip(range(4), free):
            def build_road(state, e=e, p=p):
                state["Board"]["Edges"][e].update(Building="Road", Owner=p)
            board = {element for element in changes(build_road) if element[0] in board_blocks}
            self.assertEqual(board, {("edges", e)}, f"road of player {p} on edge {e}")

    def test_a_net_with_blocks_exports_with_parity(self):
        torch.manual_seed(0)
        model = StateValueNet(FEATURE_DIM, hidden=64, aux_heads=True, normalize_inputs=True, policy_types=3, policy_vocabulary=10,
                              policy_dim=8, blocks=(8, 4, 4, 16))
        model.placements_trained = True
        model.input_scale.uniform_(0.5, 2.0)
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            for fp16, tolerance in ((False, 1e-4), (True, 5e-2)):
                self.assertLess(export_model(model, os.path.join(directory, f"blocks_{fp16}.onnx"), FEATURE_DIM, fp16), tolerance)


if __name__ == "__main__":
    unittest.main()
