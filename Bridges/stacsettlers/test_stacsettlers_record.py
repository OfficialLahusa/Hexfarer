"""
StacSettlers' games recorded while they run (MoveLog.java, -Dosp.record=DIR on OSPServer): the requests of every seat
(here four Stac agents) as a referee log of protocol steps, the input of Hexfarer's replay harness. Each game starts
with before and ends with after (the winner at 10 points); decide and step alternate, numbered from 0, the deciding seat
is the state's current one, and each move played is one of the legal moves listed (two-step moves joined, discards in
the protocol's order). A trade between players, which the Stac agents make in chat, is no protocol move: it shows as a
gap (counted in after), where the next state starts elsewhere than the last step ended. Needs a JDK and StacSettlers'
jar in STACSETTLERS_JAR. Run: python -m unittest test_stacsettlers_record
"""

import json
import os
import tempfile
import unittest
from collections import Counter

import stacsettlers

STAC = (4, "Stac", "stac", "TRY_N_BEST_BUILD_PLANS:0|FAVOUR_DEV_CARDS:-5|NP_MAX_NUMBER_OF_OFFERS_WITHOUT_BUILDING:3")


def canonical(node):
    return json.dumps(node, sort_keys=True)


def record(games, trades):
    """Stac-only games, recorded; the games' messages"""
    directory, records = tempfile.mkdtemp(), tempfile.mkdtemp()
    config = stacsettlers.config("OSPRecord", games, [STAC])
    if not trades:
        config = config.replace("Trades=true", "Trades=false")
    output = stacsettlers.simulate(directory, config, None, {"osp.record": records}, timeout=600)
    found = []
    for name in sorted(os.listdir(records)):
        with open(os.path.join(records, name), encoding="utf-8") as file:
            found.append([json.loads(line) for line in file])
    return found, output


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stacsettlers.build()
        cls.games, cls.output = record(32, trades=False)
        cls.trading, _ = record(2, trades=True)

    def test_every_game_starts_ends_and_alternates_decisions_and_steps(self):
        self.assertEqual(32, len(self.games), self.output[-3000:])
        for game in self.games + self.trading:
            self.assertEqual("before", game[0]["type"])
            self.assertEqual("after", game[-1]["type"])
            self.assertEqual(37, len(game[0]["board"]["tiles"]))
            body = game[1:-1]
            self.assertEqual(["decide", "step"] * (len(body) // 2), [m["type"] for m in body])
            for n, (decide, step) in enumerate(zip(body[::2], body[1::2])):
                self.assertEqual((n, n), (decide["n"], step["n"]))
                self.assertEqual(decide["seat"], step["seat"])
                self.assertEqual(decide["seat"], decide["state"]["current"], decide["n"])
                self.assertIn(canonical(step["action"]), {canonical(m) for m in decide["legal"]}, f"{game[0]['game_id']} move {n}")

    def test_a_game_ends_with_a_winner_at_ten_points(self):
        for game in self.games + self.trading:
            after = game[-1]
            self.assertFalse(after["truncated"], after)
            self.assertGreaterEqual(after["vp"][after["winner"]], 10)

    def test_a_card_that_leaves_a_hand_is_played_in_a_step(self):
        # A Monopoly that finds no cards changes nothing but StacSettlers' game state with its second step: still a step (it
        # comes up in about one game in ten, hence the 32 games)
        played = 0
        for game in self.games + self.trading:
            body = game[1:-1] + [game[-1]]
            for decide, step, following in zip(body[::2], body[1::2], body[2::2]):
                if "state" not in following:
                    continue
                for seat in range(4):
                    had, has = decide["state"]["players"][seat]["development"], following["state"]["players"][seat]["development"]
                    for card in ("knight", "road_building", "year_of_plenty", "monopoly"):
                        if has[card] < had[card]:
                            played += 1
                            self.assertEqual((seat, "play_" + card), (step["seat"], step["action"][0]), f"{game[0]['game_id']} move {step['n']}")
        self.assertGreater(played, 20)

    def test_without_trades_nothing_happens_between_the_steps(self):
        # Every change of the game is a protocol move: no gaps
        for game in self.games:
            self.assertEqual(0, game[-1]["gaps"], game[0]["game_id"])

    def test_trades_between_players_are_gaps(self):
        self.assertGreater(sum(game[-1]["gaps"] for game in self.trading), 0)

    def test_every_kind_of_move_comes_up(self):
        moves = Counter(m["action"][0] for game in self.games for m in game if m["type"] == "step")
        self.assertTrue({"build_settlement", "build_road", "build_city", "buy_development", "play_knight", "bank_trade", "end_turn",
                         "roll", "discard", "move_robber"} <= set(moves), moves)

    def test_the_setup_goes_round_from_seat_0(self):
        for game in self.games:
            setup = [m["seat"] for m in game if m["type"] == "decide" and m["state"]["prompt"] == "setup_settlement"]
            self.assertEqual([0, 1, 2, 3, 3, 2, 1, 0], setup)


if __name__ == "__main__":
    unittest.main()
