"""
StacSettlers' classic board in Open Seat Protocol keys (OSPBoard.java) and a fresh game's state (OSPGame.java), tested
against the protocol's reference shape of the classic board (../osp/classic-board.json) and anchored to the meaning of
the compass directions (StacSettlers, like JSettlers 1.1, numbers a hex's corners clockwise from north), so a mirrored or
rotated mapping can't pass; tiles, numbers and harbors keep their meaning under StacSettlers' own constants (JSettlers
1.1.13's: the desert is hex type 0 and the sea 6, the reverse of JSettlers 2). Needs a JDK and StacSettlers' jar in
STACSETTLERS_JAR. Run: python -m unittest test_stacsettlers_board
"""

import json
import os
import unittest

import stacsettlers

REFERENCE = os.path.join(os.path.dirname(__file__), "..", "osp", "classic-board.json")

# The protocol's axial steps by compass direction (pointy-top hexes, q to the east, r to the south-east)
STEPS = {"EAST": (1, 0), "SOUTHEAST": (0, 1), "SOUTHWEST": (-1, 1), "WEST": (-1, 0), "NORTHWEST": (0, -1), "NORTHEAST": (1, -1)}
# A hex's corners clockwise from north, each between the two neighbors on either side of it
CORNERS = [("NORTHWEST", "NORTHEAST"), ("NORTHEAST", "EAST"), ("EAST", "SOUTHEAST"), ("SOUTHEAST", "SOUTHWEST"),
           ("SOUTHWEST", "WEST"), ("WEST", "NORTHWEST")]
# StacSettlers' hex types (JSettlers 1.1.13's SOCBoard: DESERT_HEX 0, CLAY 1, ORE 2, SHEEP 3, WHEAT 4, WOOD 5, WATER 6)
# and port types by meaning, as SGN's codes
DESERT = 0
HEX_CODES = {0: "D", 1: "H", 2: "M", 3: "P", 4: "F", 5: "FO"}
PORT_CODES = {0: "HR", 1: "HRB", 2: "HRO", 3: "HRW", 4: "HRG", 5: "HRL"}
RESOURCES = ["brick", "grain", "lumber", "ore", "wool"]
CARDS = ["knight", "monopoly", "road_building", "victory_point", "year_of_plenty"]


def step(hex_, direction):
    dq, dr = STEPS[direction]
    return [hex_[0] + dq, hex_[1] + dr]


def frozen(value):
    """Nested lists as nested tuples, for comparing as sets"""
    return tuple(frozen(v) for v in value) if isinstance(value, (list, tuple)) else value


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class BoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stacsettlers.build()
        with open(REFERENCE, encoding="utf-8") as file:
            cls.reference = json.load(file)
        # Several fresh games, each with a new random board
        cls.boards = [json.loads(stacsettlers.run("osp.stacsettlers.OSPBoard", "describe", str(seed))) for seed in range(1, 7)]

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

    def test_tiles_keep_their_resource_and_number(self):
        for board in self.boards:
            tiles = {tuple(t["tile"]): t for t in board["board"]["tiles"]}
            self.assertEqual(37, len(tiles))
            for hex_ in board["hexes"]:
                tile = tiles[tuple(hex_["hex"])]
                self.assertEqual(HEX_CODES[hex_["type"]], tile["tCode"], hex_)
                self.assertEqual(None if hex_["type"] == DESERT else hex_["number"], tile["rNumber"], hex_)
            self.assertEqual(18, sum(1 for t in tiles.values() if t["tCode"] == "S" and t["rNumber"] is None))
            # The classic set of tiles and numbers
            land = [t["tCode"] for t in tiles.values() if t["tCode"] != "S"]
            self.assertEqual({"FO": 4, "P": 4, "F": 4, "H": 3, "M": 3, "D": 1}, {code: land.count(code) for code in set(land)})
            numbers = sorted(t["rNumber"] for t in tiles.values() if t["rNumber"] is not None)
            self.assertEqual([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12], numbers)

    def test_harbors_keep_their_kind_and_the_robber_starts_on_the_desert(self):
        for board in self.boards:
            harbors = sorted(h["hCode"] for h in board["board"]["harbors"])
            self.assertEqual(sorted(PORT_CODES[p["type"]] for p in board["ports"]), harbors)
            self.assertEqual(["HR"] * 4 + ["HRB", "HRG", "HRL", "HRO", "HRW"], harbors)
            # A harbor's two corners are the port's: each harbor path is the edge of a port of its kind
            by_path = {frozen(sorted(h["path"])): h["hCode"] for h in board["board"]["harbors"]}
            for port in board["ports"]:
                self.assertEqual(PORT_CODES[port["type"]], by_path[frozen(sorted(port["path"]))])
            desert = [h["hex"] for h in board["hexes"] if h["type"] == DESERT]
            self.assertEqual(desert, [board["board"]["robber"]])

    def test_a_fresh_games_state(self):
        for board in self.boards:
            state = board["state"]
            self.assertEqual(("setup_settlement", 0, 0, 0), (state["prompt"], state["current"], state["on_turn"], state["free_roads"]))
            self.assertEqual(board["board"]["robber"], state["robber"])
            self.assertEqual({"settlements": [], "cities": [], "roads": []}, state["buildings"])
            self.assertEqual({**{r: 19 for r in RESOURCES}, "development": 25}, state["bank"])
            self.assertIsNone(state["offer"])
            self.assertEqual([0, 1, 2, 3], [p["seat"] for p in state["players"]])
            for player in state["players"]:
                self.assertEqual({r: 0 for r in RESOURCES}, player["hand"])
                self.assertEqual({c: 0 for c in CARDS}, player["development"])
                self.assertEqual((0, 0, 0, 0), (player["hand_size"], player["development_size"], player["public_vp"], player["vp"]))
                self.assertEqual({"knight": 0}, player["played"])
                self.assertEqual((0, False, False), (player["longest_road"], player["has_longest_road"], player["has_largest_army"]))
                self.assertEqual({"roads": 15, "settlements": 5, "cities": 4}, player["pieces_left"])
            # Seat 0 is the player StacSettlers drew to start: every intersection is a place for the first settlement
            self.assertEqual(0, board["seat_of_first"])
            legal = board["legal"]
            self.assertEqual(54, len(legal))
            self.assertEqual({"build_settlement"}, {move[0] for move in legal})
            self.assertEqual(set(frozen(self.reference["intersections"])), {frozen(move[1]) for move in legal})
        # StacSettlers draws who starts, and seat 0 is always that player
        self.assertGreater(len({board["first"] for board in self.boards}), 1)

    def test_a_view_hides_the_other_seats_cards(self):
        for board in self.boards:
            for player in board["view"]["players"]:
                hidden = player["seat"] != 1
                for field in ("hand", "development", "vp"):
                    self.assertEqual(hidden, player[field] is None, (player["seat"], field))


if __name__ == "__main__":
    unittest.main()
