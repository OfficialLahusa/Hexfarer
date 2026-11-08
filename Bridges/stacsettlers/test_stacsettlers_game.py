"""
StacSettlers' games in the Open Seat Protocol (OSPGame.java) and the recorder (Recorder.java), which plays random games
by StacSettlers' own rules (JSettlers 1.1.16's SOCGame) and writes them as a referee's messages: every move played passes
the checks StacSettlers' server makes before it takes a request (a robber's victims are the ones its game finds, a
discard is one it takes, a card is one it lets the player play, ...); every decision lists distinct legal moves, the move
played is one of them, states keep the protocol's shape and add up (pieces, cards), every kind of decision and move comes
up, and a game ends with a winner at 10 points. Needs a JDK and StacSettlers' jar in STACSETTLERS_JAR.
Run: python -m unittest test_stacsettlers_game
"""

import gzip
import json
import os
import tempfile
import unittest
from collections import Counter

import stacsettlers

RESOURCES = ["brick", "lumber", "wool", "grain", "ore"]
CARDS = ["knight", "road_building", "year_of_plenty", "monopoly", "victory_point"]


def canonical(node):
    return json.dumps(node, sort_keys=True)


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class RecorderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stacsettlers.build()
        cls.directory = tempfile.mkdtemp()
        path = os.path.join(cls.directory, "games.jsonl.gz")
        stacsettlers.run("osp.stacsettlers.Recorder", "--games", "12", "--seed", "11", "--out", path)
        with gzip.open(path, "rt", encoding="utf-8") as file:
            cls.messages = [json.loads(line) for line in file]
        cls.games = []
        for message in cls.messages:
            if message["type"] == "before":
                cls.games.append([])
            cls.games[-1].append(message)

    def test_every_game_starts_ends_and_alternates_decisions_and_steps(self):
        self.assertEqual(12, len(self.games))
        for game in self.games:
            self.assertEqual("before", game[0]["type"])
            self.assertEqual("after", game[-1]["type"])
            self.assertEqual("standard", game[0]["ruleset"]["preset"])
            self.assertEqual(4, game[0]["players"])
            self.assertEqual(37, len(game[0]["board"]["tiles"]))
            body = game[1:-1]
            self.assertEqual(["decide", "step"] * (len(body) // 2), [m["type"] for m in body])
            for n, (decide, step) in enumerate(zip(body[::2], body[1::2])):
                self.assertEqual((n, n), (decide["n"], step["n"]))
                self.assertEqual(decide["seat"], step["seat"])
                self.assertEqual(decide["seat"], decide["state"]["current"])
                legal = [canonical(m) for m in decide["legal"]]
                self.assertTrue(legal, f"no legal move at {n}")
                self.assertEqual(len(legal), len(set(legal)), f"a move listed twice at {n}")
                self.assertIn(canonical(step["action"]), set(legal), f"move {n}")

    def test_the_ruleset_names_stacsettlers_rules(self):
        # The rulebook without trades for the seat, where StacSettlers' engine is narrower: no resource bank, robbing only
        # players with cards, a Road Building card's roads placed at once; bank trades at every rate the player has
        self.assertEqual({"preset": "standard", "player_trading": "none", "bank_shortage": "unlimited", "robber_victims": "with_cards",
                          "road_building": "forced"}, self.games[0][0]["ruleset"])

    def test_seats_are_numbered_in_turn_order_from_the_first_player(self):
        for game in self.games:
            setup = [m["seat"] for m in game if m["type"] == "decide" and m["state"]["prompt"] == "setup_settlement"]
            self.assertEqual([0, 1, 2, 3, 3, 2, 1, 0], setup)
            first_roll = next(m for m in game if m["type"] == "decide" and m["state"]["prompt"] == "roll")
            self.assertEqual(0, first_roll["seat"])
        self.assertTrue(any(game[0]["stacsettlers_first"] != 0 for game in self.games), "every game started with player 0")

    def test_a_game_ends_with_a_winner_at_ten_points(self):
        winners = 0
        for game in self.games:
            after = game[-1]
            if after["truncated"]:
                continue
            winners += 1
            self.assertGreaterEqual(after["vp"][after["winner"]], 10)
            self.assertEqual(after["vp"][after["winner"]], after["state"]["players"][after["winner"]]["vp"])
        self.assertGreater(winners, 8)

    def test_states_add_up(self):
        for game in self.games:
            # The knights bought so far, by the steps' results
            knights = 0
            for message in game:
                if message["type"] == "step" and message["action"][0] == "buy_development" and message["result"]["card"] == "knight":
                    knights += 1
                state = message.get("state")
                if state is None:
                    continue
                buildings = state["buildings"]
                for player in state["players"]:
                    seat = player["seat"]
                    settlements = sum(1 for s, _ in buildings["settlements"] if s == seat)
                    cities = sum(1 for s, _ in buildings["cities"] if s == seat)
                    roads = sum(1 for s, _ in buildings["roads"] if s == seat)
                    self.assertEqual(5, settlements + player["pieces_left"]["settlements"])
                    self.assertEqual(4, cities + player["pieces_left"]["cities"])
                    self.assertEqual(15, roads + player["pieces_left"]["roads"])
                    self.assertEqual(sorted(RESOURCES), sorted(player["hand"]))
                    self.assertEqual(sorted(CARDS), sorted(player["development"]))
                    self.assertEqual(player["hand_size"], sum(player["hand"].values()))
                    self.assertEqual(player["development_size"], sum(player["development"].values()))
                    # Points: buildings, Longest Road and Largest Army, and the victory point cards on top
                    public = settlements + 2 * cities + 2 * player["has_longest_road"] + 2 * player["has_largest_army"]
                    self.assertEqual(public, player["public_vp"], player)
                    self.assertEqual(public + player["development"]["victory_point"], player["vp"], player)
                # Development cards: the bank's, the hands' and the knights played come from the 25 of a game
                held = sum(p["development_size"] for p in state["players"])
                self.assertLessEqual(state["bank"]["development"] + held + sum(p["played"]["knight"] for p in state["players"]), 25)
                # Every knight bought is held or played, under the card's own name
                self.assertEqual(knights, sum(p["development"]["knight"] + p["played"]["knight"] for p in state["players"]), "knights")

    def test_a_bought_card_is_the_one_the_state_shows(self):
        for game in self.games:
            for decide, step, after in zip(game[1::2], game[2::2], game[3::2]):
                if step["type"] != "step" or step["action"][0] != "buy_development" or "state" not in after:
                    continue
                seat, card = step["seat"], step["result"]["card"]
                had = decide["state"]["players"][seat]["development"][card]
                self.assertEqual(had + 1, after["state"]["players"][seat]["development"][card], step)

    def test_a_discard_gives_up_half_the_hand(self):
        discards = 0
        for message in self.messages:
            if message["type"] == "decide" and message["state"]["prompt"] == "discard":
                hand = message["state"]["players"][message["seat"]]["hand_size"]
                self.assertGreater(hand, 7)
                for move in message["legal"]:
                    self.assertEqual(hand // 2, sum(move[1].values()), move)
                discards += 1
        self.assertGreater(discards, 0)

    def test_results_have_the_protocols_shape(self):
        for message in self.messages:
            if message["type"] != "step":
                continue
            kind, result = message["action"][0], message["result"]
            if kind == "roll":
                self.assertTrue(all(1 <= d <= 6 for d in result["dice"]) and len(result["dice"]) == 2)
            elif kind == "buy_development":
                self.assertIn(result["card"], CARDS)
            elif kind == "move_robber":
                self.assertTrue(result["stolen"] is None or result["stolen"] in RESOURCES)
                self.assertEqual(result["stolen"] is None, message["action"][1]["victim"] is None)
            elif kind == "play_monopoly":
                self.assertTrue(all(int(seat) != message["seat"] for seat in result["taken"]))
            else:
                self.assertIsNone(result)

    def test_a_robber_victim_holds_cards_and_a_building_at_the_hex(self):
        robbed = 0
        for game in self.games:
            for decide in (m for m in game if m["type"] == "decide" and m["state"]["prompt"] == "move_robber"):
                state = decide["state"]
                for move in decide["legal"]:
                    hex_, victim = move[1]["hex"], move[1]["victim"]
                    self.assertNotEqual(state["robber"], hex_)
                    if victim is None:
                        continue
                    robbed += 1
                    self.assertNotEqual(decide["seat"], victim)
                    self.assertGreater(state["players"][victim]["hand_size"], 0)
                    owned = state["buildings"]["settlements"] + state["buildings"]["cities"]
                    self.assertTrue(any(seat == victim and hex_ in place for seat, place in owned), move)
        self.assertGreater(robbed, 20)

    def test_every_kind_of_decision_and_move_comes_up(self):
        prompts = Counter(m["state"]["prompt"] for m in self.messages if m["type"] == "decide")
        moves = Counter(m["action"][0] for m in self.messages if m["type"] == "step")
        self.assertTrue({"setup_settlement", "setup_road", "roll", "main", "discard", "move_robber", "road_building"} <= set(prompts), prompts)
        self.assertTrue({"build_settlement", "build_road", "build_city", "buy_development", "play_knight", "play_road_building",
                         "play_year_of_plenty", "play_monopoly", "bank_trade", "end_turn", "roll", "discard", "move_robber"} <= set(moves), moves)
        # Bank trades at each rate: 4:1, and at harbors 3:1 and 2:1
        rates = {sum(m["action"][1]["give"].values()) for m in self.messages if m["type"] == "step" and m["action"][0] == "bank_trade"}
        self.assertEqual({2, 3, 4}, rates)


if __name__ == "__main__":
    unittest.main()
