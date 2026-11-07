"""
JSettlers' classic board in Open Seat Protocol keys (OspBoard.java), tested against the protocol's reference shape of the
classic board (../osp/classic-board.json) and anchored to the meaning of the compass directions (JSettlers numbers a
hex's corners clockwise from north), so that a mirrored or rotated mapping, which keeps the board's shape, can't pass;
the tiles and harbors keep their resources. Needs a JDK and JSettlers' full jar in JSETTLERS_JAR.
Run: python -m unittest test_jsettlers_board
"""

import json
import os
import unittest

import jsettlers

REFERENCE = os.path.join(os.path.dirname(__file__), "..", "osp", "classic-board.json")

# The protocol's axial steps by compass direction (pointy-top hexes, q to the east, r to the south-east)
STEPS = {"EAST": (1, 0), "SOUTHEAST": (0, 1), "SOUTHWEST": (-1, 1), "WEST": (-1, 0), "NORTHWEST": (0, -1), "NORTHEAST": (1, -1)}
# JSettlers' corners of a hex, clockwise from north, each between the two neighbors on either side of it
CORNERS = [("NORTHWEST", "NORTHEAST"), ("NORTHEAST", "EAST"), ("EAST", "SOUTHEAST"), ("SOUTHEAST", "SOUTHWEST"),
           ("SOUTHWEST", "WEST"), ("WEST", "NORTHWEST")]
# JSettlers' hex types and port types by meaning, as SGN's codes
HEX_CODES = {1: "H", 2: "M", 3: "P", 4: "F", 5: "FO", 6: "D"}
PORT_CODES = {0: "HR", 1: "HRB", 2: "HRO", 3: "HRW", 4: "HRG", 5: "HRL"}


def step(hex_, direction):
    dq, dr = STEPS[direction]
    return [hex_[0] + dq, hex_[1] + dr]


def frozen(value):
    """Nested lists as nested tuples, for comparing as sets"""
    return tuple(frozen(v) for v in value) if isinstance(value, (list, tuple)) else value


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class BoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jsettlers.build()
        with open(REFERENCE, encoding="utf-8") as file:
            cls.reference = json.load(file)
        # Several random classic boards
        cls.boards = [json.loads(jsettlers.run("osp.jsettlers.OspBoard", "describe", str(seed))) for seed in range(1, 6)]

    def test_every_board_has_the_reference_shape(self):
        for board in self.boards:
            for name in ("land", "sea", "intersections", "paths", "harbors"):
                self.assertEqual(set(frozen(self.reference[name])), set(frozen(board["shape"][name])), name)
                self.assertEqual(len(self.reference[name]), len(board["shape"][name]), name)

    def test_every_corner_lies_where_its_compass_direction_says(self):
        for board in self.boards:
            self.assertEqual(19, len(board["hexes"]))
            for hex_ in board["hexes"]:
                for direction, key in enumerate(hex_["corners"]):
                    a, b = CORNERS[direction]
                    expected = sorted([hex_["hex"], step(hex_["hex"], a), step(hex_["hex"], b)])
                    self.assertEqual(expected, key, f"corner {direction} of {hex_['hex']}")

    def test_tiles_and_harbors_keep_their_meaning(self):
        for board in self.boards:
            tiles = {tuple(t["tile"]): t for t in board["board"]["tiles"]}
            for hex_ in board["hexes"]:
                tile = tiles[tuple(hex_["hex"])]
                self.assertEqual(HEX_CODES[hex_["type"]], tile["tCode"])
                self.assertEqual(None if hex_["type"] == 6 else hex_["number"], tile["rNumber"])
            self.assertEqual(18, sum(1 for t in tiles.values() if t["tCode"] == "S"))
            harbors = sorted(h["hCode"] for h in board["board"]["harbors"])
            self.assertEqual(sorted(PORT_CODES[p["type"]] for p in board["ports"]), harbors)
            self.assertEqual(["HR"] * 4 + ["HRB", "HRG", "HRL", "HRO", "HRW"], harbors)
            # The robber starts on the desert
            desert = [h["hex"] for h in board["hexes"] if h["type"] == 6]
            self.assertEqual(desert, [board["board"]["robber"]])


if __name__ == "__main__":
    unittest.main()
