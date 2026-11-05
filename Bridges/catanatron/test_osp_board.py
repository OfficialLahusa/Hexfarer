"""
Catanatron's board in Open Seat Protocol keys (osp_board.py), tested against the protocol's reference shape of the
classic board (../osp/classic-board.json) and anchored to the meaning of the compass directions, so that a mirrored or
rotated mapping, which keeps the board's shape, can't pass. Run: .venv/Scripts/python -m unittest test_osp_board
"""

import json
import os
import random
import unittest
from collections import defaultdict

from catanatron.models.coordinate_system import UNIT_VECTORS, Direction
from catanatron.models.enums import BRICK, ORE, SHEEP, WHEAT, WOOD, EdgeRef, NodeRef
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, number_probability
from catanatron.models.tiles import LandTile, Port

import osp_board

REFERENCE = os.path.join(os.path.dirname(__file__), "..", "osp", "classic-board.json")

# The protocol's axial steps by compass direction (pointy-top hexes, q to the east, r to the south-east)
STEPS = {"EAST": (1, 0), "SOUTHEAST": (0, 1), "SOUTHWEST": (-1, 1), "WEST": (-1, 0), "NORTHWEST": (0, -1), "NORTHEAST": (1, -1)}
# A corner lies between the two neighbors on either side of it
CORNERS = {"NORTH": ("NORTHWEST", "NORTHEAST"), "NORTHEAST": ("NORTHEAST", "EAST"), "SOUTHEAST": ("EAST", "SOUTHEAST"),
           "SOUTH": ("SOUTHEAST", "SOUTHWEST"), "SOUTHWEST": ("SOUTHWEST", "WEST"), "NORTHWEST": ("WEST", "NORTHWEST")}
# SGN's codes by meaning: forest gives lumber, hills brick, pastures wool, fields grain, mountains ore
SGN_TILES = {"FO": WOOD, "H": BRICK, "P": SHEEP, "F": WHEAT, "M": ORE, "D": None}
SGN_HARBORS = {"HRL": WOOD, "HRB": BRICK, "HRW": SHEEP, "HRG": WHEAT, "HRO": ORE, "HR": None}


def step(hex_, direction):
    dq, dr = STEPS[direction]
    return (hex_[0] + dq, hex_[1] + dr)


def frozen(value):
    """Nested lists as nested tuples, for comparing as sets"""
    return tuple(frozen(v) for v in value) if isinstance(value, (list, tuple)) else value


def maps():
    """Random classic maps with both of Catanatron's number placements"""
    for seed in range(1, 6):
        for placement in ("official_spiral", "random"):
            yield CatanMap.from_template(BASE_MAP_TEMPLATE, number_placement=placement, rng=random.Random(seed))


class ShapeTests(unittest.TestCase):
    def test_every_map_has_the_reference_shape(self):
        with open(REFERENCE, encoding="utf-8") as file:
            reference = json.load(file)
        for catan_map in maps():
            shape = osp_board.shape(catan_map)
            for name in ("land", "sea", "intersections", "paths", "harbors"):
                self.assertEqual(set(frozen(reference[name])), set(frozen(shape[name])), name)
                self.assertEqual(len(reference[name]), len(shape[name]), name)

    def test_every_land_node_has_its_own_key_of_three_hexes(self):
        for catan_map in maps():
            keys = osp_board.node_keys(catan_map)
            self.assertEqual(54, len(keys))
            self.assertEqual(54, len(set(keys.values())))
            self.assertTrue(all(len(key) == 3 for key in keys.values()))


class CompassTests(unittest.TestCase):
    """The mapping keeps the compass: east in Catanatron is east in the protocol, and so on"""

    def test_directions(self):
        for direction in Direction:
            self.assertEqual(STEPS[direction.value], osp_board.axial(UNIT_VECTORS[direction]), direction)

    def test_corners_lie_between_the_neighbors_on_either_side(self):
        for catan_map in maps():
            keys = osp_board.node_keys(catan_map)
            for coordinate, tile in catan_map.land_tiles.items():
                center = osp_board.axial(coordinate)
                for ref in NodeRef:
                    left, right = CORNERS[ref.value]
                    self.assertEqual(tuple(sorted((center, step(center, left), step(center, right)))), keys[tile.nodes[ref]], (center, ref))

    def test_edges_lie_between_a_tile_and_its_neighbor_in_their_direction(self):
        for catan_map in maps():
            keys = osp_board.node_keys(catan_map)
            for coordinate, tile in catan_map.land_tiles.items():
                center = osp_board.axial(coordinate)
                for ref in EdgeRef:
                    a, b = osp_board.path_key(keys, tile.edges[ref])
                    self.assertEqual({center, step(center, ref.value)}, set(a) & set(b), (center, ref))

    def test_harbors_face_the_land_in_their_direction(self):
        for catan_map in maps():
            keys = osp_board.node_keys(catan_map)
            land = {osp_board.axial(c) for c in catan_map.land_tiles}
            harbors = dict(osp_board.harbors(catan_map, keys))
            served = defaultdict(set)
            for coordinate, tile in catan_map.tiles.items():
                if not isinstance(tile, Port):
                    continue
                water = osp_board.axial(coordinate)
                facing = step(water, tile.direction.value)
                self.assertIn(facing, land)
                paths = [path for path, code in harbors.items() if all(water in end and facing in end for end in path)]
                self.assertEqual(1, len(paths), water)
                self.assertEqual(tile.resource, SGN_HARBORS[harbors[paths[0]]], water)
                served[tile.resource].update(paths[0])
            # The same nodes Catanatron gives the harbors' privileges to
            for resource, nodes in catan_map.port_nodes.items():
                self.assertEqual({keys[node] for node in nodes}, served[resource], resource)


class BoardTests(unittest.TestCase):
    def test_tiles_robber_and_production(self):
        for catan_map in maps():
            robber = next(c for c, tile in catan_map.land_tiles.items() if tile.resource is None)
            board = osp_board.board(catan_map, robber)
            tiles = {tuple(entry["tile"]): entry for entry in board["tiles"]}
            self.assertEqual(37, len(tiles))
            self.assertEqual(list(osp_board.axial(robber)), board["robber"])
            for coordinate, tile in catan_map.tiles.items():
                entry = tiles[osp_board.axial(coordinate)]
                if isinstance(tile, LandTile):
                    self.assertEqual((tile.resource, tile.number), (SGN_TILES[entry["tCode"]], entry["rNumber"]))
                else:
                    self.assertEqual(("S", None), (entry["tCode"], entry["rNumber"]))

            # Each node's production from the protocol's board equals Catanatron's own
            for node_id, key in osp_board.node_keys(catan_map).items():
                production = defaultdict(float)
                for hex_ in key:
                    entry = tiles[hex_]
                    if entry["rNumber"] is not None:
                        production[SGN_TILES[entry["tCode"]]] += number_probability(entry["rNumber"])
                expected = catan_map.node_production[node_id]
                self.assertEqual(set(expected), set(production), node_id)
                for resource in expected:
                    self.assertAlmostEqual(expected[resource], production[resource], msg=node_id)


if __name__ == "__main__":
    unittest.main()
