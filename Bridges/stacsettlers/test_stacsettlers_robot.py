"""
A StacSettlers robot whose decisions an Open Seat Agent makes (OSPRobotBrain.java), in StacSettlers' headless games
(OSPSimulation.java, the agent kind "osp") against three of StacSettlers' Stac agents, which negotiate their trades in
chat among themselves (ChatNegotiation=true, Trades=true) while the server ends no turn on its own (ForceEndTurns=false):
every game ends with a winner, the robot made its decisions through the agent (here one that plays the first legal move,
building and playing cards where it can) without a fallback, a stalled move or a forced turn end, and the agent got the
game, its decisions and its end. Needs a JDK and StacSettlers' jar in STACSETTLERS_JAR.
Run: python -m unittest test_stacsettlers_robot
"""

import json
import os
import sys
import tempfile
import unittest

import stacsettlers

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_AGENT = os.path.join(HERE, "..", "jsettlers", "first_move_agent.py")
# A move the server hasn't taken within this time counts as a stall (in the tests short, so a broken request shows fast)
STALL_MS = "3000"
STAC = (3, "Stac", "stac", "TRY_N_BEST_BUILD_PLANS:0|FAVOUR_DEV_CARDS:-5|NP_MAX_NUMBER_OF_OFFERS_WITHOUT_BUILDING:3")


def agent_command(*flags):
    return f'"{sys.executable}" "{os.path.abspath(TEST_AGENT)}" ' + " ".join(flags)


def play(games, agent, logs, name="OSPTest", opponents=STAC, fully_observable=False, properties=None):
    """Headless games with one Open Seat Agent robot and three of StacSettlers' agents (and further system properties);
    the simulation's output"""
    directory = tempfile.mkdtemp()
    config = stacsettlers.config(name, games, [(1, "Hexfarer", "osp"), opponents], fully_observable=fully_observable)
    output = stacsettlers.simulate(directory, config, agent, {"osp.log": logs, "osp.debug": "true", "osp.stall_ms": STALL_MS, **(properties or {})},
                                   timeout=int(os.environ.get("STACSETTLERS_ROBOT_TIMEOUT", "600")))
    return output, directory


def results(output):
    """The games' result records (osp-result lines), and in each the line of the seat an Open Seat Agent plays"""
    lines = [json.loads(line[line.index("{"):]) for line in output.splitlines() if '"osp-result"' in line]
    return lines, [next(seat for seat in line["result"]["seats"] if seat["agent"] is not None) for line in lines]


def messages(logs):
    """What the agent got in each game, by its log file (the test agent writes each message to standard error)"""
    found = []
    for name in sorted(os.listdir(logs)):
        with open(os.path.join(logs, name), encoding="utf-8") as file:
            found.append([json.loads(line) for line in file if line.startswith("{")])
    return found


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class RobotTests(unittest.TestCase):
    GAMES = 8

    @classmethod
    def setUpClass(cls):
        stacsettlers.build()
        cls.logs = tempfile.mkdtemp()
        cls.output, cls.directory = play(cls.GAMES, agent_command("--weighted", "--full"), cls.logs)
        cls.lines, cls.stats = results(cls.output)

    def test_every_game_ends_with_a_winner_and_its_result_line(self):
        self.assertEqual(self.GAMES, len(self.lines), self.output[-3000:])
        for line in self.lines:
            self.assertIsNotNone(line["winner"], line)
            self.assertGreaterEqual(line["vp"][line["winner"]], 10, line)
            self.assertIsInstance(line["result"]["seed"], int)
            self.assertEqual(4, len(line["players"]))
            self.assertIn("Hexfarer", line["players"])

    def test_stacsettlers_counts_the_same_winner(self):
        # StacSettlers' own results table (results/<run>/results.txt): one row per game, the winner flagged
        tables = [os.path.join(root, "results.txt") for root, _, names in os.walk(os.path.join(self.directory, "results")) if "results.txt" in names]
        self.assertEqual(1, len(tables), self.output[-3000:])
        with open(tables[0], encoding="utf-8") as file:
            rows = [line.rstrip("\n").split("\t") for line in file][1:]
        winners = {}
        for row in rows:
            for i in range(2, len(row) - 7, 8):
                if row[i + 1] == "1":
                    winners[row[0]] = row[i]
        self.assertEqual({line["game_id"]: line["players"][line["winner"]] for line in self.lines}, winners)

    def test_the_robot_decides_through_its_agent_without_fallbacks(self):
        self.assertEqual(self.GAMES, len(self.stats), self.output[-3000:])
        for seat in self.stats:
            self.assertEqual(("first-move", "truth", False), (seat["agent"], seat["needs"], seat["observe"]))
            self.assertGreater(seat["robot_decisions"], 20, seat)
            self.assertGreater(seat["decisions"], 5, seat)
            self.assertLessEqual(seat["max_ms"], seat["time_ms"], seat)
            self.assertEqual((0, [], 0), (seat["fallbacks"], seat["failures"], seat["stalls"]), seat)

    def test_the_robot_declines_the_offers_made_to_it(self):
        # Stac agents offer to every player (they wait only for those that announced they participate, which the robot
        # never does); the robot takes no part in trades between players and answers each offer with a reject
        self.assertGreater(sum(seat["declined"] for seat in self.stats), 0, self.stats)
        self.assertTrue(any(line.startswith("[osp Hexfarer") and "] declined " in line for line in self.output.splitlines()))

    def test_the_server_never_ends_a_turn_on_its_own(self):
        self.assertNotIn("force end turn", self.output)

    def test_the_robot_plays_cards_and_builds(self):
        # The test agent builds and plays development cards where it can: in eight games every kind of move comes up but the rarer cards
        played = [json.loads(line[line.index("{"):]) for line in self.output.splitlines() if line.startswith("[osp ") and "] move " in line]
        kinds = {move["move"][0] for move in played}
        self.assertTrue({"build_settlement", "build_road", "roll", "end_turn", "move_robber", "buy_development", "play_knight"} <= kinds, kinds)

    def test_the_agent_got_the_game_its_decisions_and_its_end(self):
        games = messages(self.logs)
        self.assertEqual(self.GAMES, len(games))
        for got in games:
            kinds = [m["type"] for m in got]
            self.assertEqual(["handshake", "before"], kinds[:2])
            self.assertEqual("after", kinds[-1])
            self.assertTrue(all(k == "decide" for k in kinds[2:-1]))
            before = got[1]
            self.assertEqual({"preset": "standard", "player_trading": "first_accept", "bank_shortage": "unlimited", "robber_victims": "with_cards",
                              "road_building": "forced"}, before["ruleset"])
            self.assertEqual({"decide_ms": 120000, "game_ms": None, "threads": None, "gpu": True}, before["budget"])
            self.assertEqual(37, len(before["board"]["tiles"]))
            # Every decision carries the true state (the agent doesn't observe), numbered in order, its seat the one that acts
            decides = got[2:-1]
            numbers = [m["n"] for m in decides]
            self.assertEqual(numbers, sorted(numbers))
            self.assertEqual(len(numbers), len(set(numbers)))
            for decide in decides:
                self.assertEqual(before["seat"], decide["seat"])
                self.assertEqual(decide["seat"], decide["state"]["current"])
                self.assertTrue(all(p["hand"] is not None for p in decide["state"]["players"]))
            self.assertEqual(got[-1]["winner"] is None, got[-1]["truncated"])
            self.assertIsNotNone(got[-1]["winner"])


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class ViewRobotTests(unittest.TestCase):
    """A robot whose agent asks for its seat's view only (needs: view)"""

    def test_a_view_shows_the_seats_own_cards_and_no_others(self):
        stacsettlers.build()
        logs = tempfile.mkdtemp()
        output, _ = play(1, agent_command("--needs", "view"), logs, name="OSPView")
        lines, stats = results(output)
        self.assertEqual(1, len(lines), output[-3000:])
        self.assertEqual(("view", 0, []), (stats[0]["needs"], stats[0]["fallbacks"], stats[0]["failures"]))
        decides = [m for m in messages(logs)[0] if m["type"] == "decide"]
        self.assertGreater(len(decides), 5)
        for decide in decides:
            self.assertEqual(True, decide["own_shown"], decide)
            self.assertEqual(True, decide["others_hidden"], decide)


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class MCTSOpponentTests(unittest.TestCase):
    """Against Dobre's POMCP agents (cheap: 300 iterations on one thread), which offer trades to anyone and wait for the
    answer: the robot declines every offer made to it, so the game goes on to its end (about 3 minutes)"""

    def test_offers_to_the_robot_are_declined_and_the_game_ends(self):
        stacsettlers.build()
        logs = tempfile.mkdtemp()
        pomcp = (3, "POMCP", "mcts", "MCTS_FACTORED_BELIEF|MCTS_POMCP|MCTS_LIST_POMS|MCTS_PUCT|MCTS_Cp:4|MCTS_HUMAN_ACTION_TYPE_PDF_CONDITIONED_SEEDING"
                 "|MCTS_HUMAN_ACTION_TYPE_PDF_SEEDING_TEMPERATURE:7|MCTS_TYPED_ROLLOUTS|MCTS_HUMAN_ACTION_TYPE_CONDITIONED_DIST_ROLLOUTS"
                 "|MCTS_ITERATIONS:300|MCTS_THREADS:1|MCTS_OFFERS_LIMIT:3")
        output, _ = play(1, agent_command("--weighted"), logs, name="OSPMCTS", opponents=pomcp)
        lines, seats = results(output)
        self.assertEqual(1, len(lines), output[-3000:])
        self.assertIsNotNone(lines[0]["winner"], lines[0])
        self.assertEqual((0, [], 0), (seats[0]["fallbacks"], seats[0]["failures"], seats[0]["stalls"]), seats[0])
        self.assertGreater(seats[0]["declined"], 0, seats[0])

    def test_the_robot_trades_in_chat_with_them_and_the_game_ends(self):
        # The robot takes part (osp.trade_in_chat): its agent answers their offers, which they wait for, and makes its own
        stacsettlers.build()
        logs = tempfile.mkdtemp()
        pomcp = (3, "POMCP", "mcts", "MCTS_FACTORED_BELIEF|MCTS_POMCP|MCTS_LIST_POMS|MCTS_PUCT|MCTS_Cp:4|MCTS_HUMAN_ACTION_TYPE_PDF_CONDITIONED_SEEDING"
                 "|MCTS_HUMAN_ACTION_TYPE_PDF_SEEDING_TEMPERATURE:7|MCTS_TYPED_ROLLOUTS|MCTS_HUMAN_ACTION_TYPE_CONDITIONED_DIST_ROLLOUTS"
                 "|MCTS_ITERATIONS:300|MCTS_THREADS:1|MCTS_OFFERS_LIMIT:3")
        output, _ = play(1, agent_command("--weighted", "--trades"), logs, name="OSPMCTSTrade", opponents=pomcp, properties={"osp.trade_in_chat": "true"})
        lines, seats = results(output)
        self.assertEqual(1, len(lines), output[-3000:])
        self.assertIsNotNone(lines[0]["winner"], lines[0])
        self.assertEqual((0, [], 0), (seats[0]["fallbacks"], seats[0]["failures"], seats[0]["stalls"]), seats[0])
        self.assertGreater(seats[0]["answers"], 0, seats[0])
        self.assertGreater(seats[0]["offers"], 0, seats[0])


REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
# Hexfarer's own agent, as the repository builds it (dotnet build OpenSeat -c Release), and a tiny net of the tests
OPEN_SEAT = os.environ.get("OPEN_SEAT_DLL", os.path.join(REPO, "OpenSeat", "bin", "Release", "net10.0", "OpenSeat.dll"))
TINY_NET = os.path.join(REPO, "Tests", "Fixtures", "ValueNet", "policy_tiny.onnx")


@unittest.skipUnless(stacsettlers.available() and os.path.exists(OPEN_SEAT), "needs a JDK, StacSettlers' jar in STACSETTLERS_JAR and a built OpenSeat")
class HexfarerTradingTests(unittest.TestCase):
    """Hexfarer's own agent (OpenSeat agent, which keeps a copy of the game and checks it against each decision's state)
    trading in chat (osp.trade_in_chat), open-hand and fair, at a few iterations: its copy follows every decision about an
    offer, so it never gets lost, and it trades"""

    def check(self, flags):
        stacsettlers.build()
        agent = f'dotnet "{OPEN_SEAT}" agent --net "{TINY_NET}" --iterations 16 --leaves-per-batch 16 --provider cpu --threads 1 {flags}'.strip()
        output, _ = play(2, agent, tempfile.mkdtemp(), name="OSPHexfarerTrade", properties={"osp.trade_in_chat": "true"})
        lines, seats = results(output)
        self.assertEqual(2, len(lines), output[-3000:])
        for line, seat in zip(lines, seats):
            self.assertIsNotNone(line["winner"], line)
            self.assertEqual((0, [], 0), (seat["fallbacks"], seat["failures"], seat["stalls"]), seat)
        self.assertGreater(sum(seat["answers"] for seat in seats), 0, seats)
        self.assertGreater(sum(seat["trades"] for seat in seats), 0, seats)

    def test_open_hand(self):
        self.check("")

    def test_fair(self):
        self.check("--fair")


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class FailureTests(unittest.TestCase):
    """An agent that fails: the failure is told to it (if it still listens), recorded with its code in the game's result,
    and the robot plays on with the first legal move, so the game ends all the same"""

    def test_an_illegal_answer(self):
        stacsettlers.build()
        logs = tempfile.mkdtemp()
        output, _ = play(1, agent_command("--illegal", "--full"), logs, name="OSPIllegal")
        lines, seats = results(output)
        self.assertEqual(1, len(lines), output[-3000:])
        self.assertIsNotNone(lines[0]["winner"], lines[0])
        self.assertGreater(seats[0]["fallbacks"], 0, seats[0])
        self.assertEqual([("illegal", "referee")], [(f["code"], f["by"]) for f in seats[0]["failures"]])
        told = [m for m in messages(logs)[0] if m["type"] == "failure"]
        self.assertEqual(["illegal"], [m["code"] for m in told])

    def test_a_failed_agent_leaves_the_robots_trade_duties_kept(self):
        # Taking part in the chat trades, the robot owes the Stac agents its turn's NULL:ST and an answer to every offer
        # (they wait for it with no time limit, and the server ends no turn): once its agent failed it keeps them itself,
        # declining what is offered to it, so the games still end
        stacsettlers.build()
        logs = tempfile.mkdtemp()
        output, _ = play(2, agent_command("--illegal", "--trades"), logs, name="OSPIllegalTrade", properties={"osp.trade_in_chat": "true"})
        lines, seats = results(output)
        self.assertEqual(2, len(lines), output[-3000:])
        for line, seat in zip(lines, seats):
            self.assertIsNotNone(line["winner"], line)
            self.assertEqual(["illegal"], [f["code"] for f in seat["failures"]], seat)
            self.assertEqual(0, seat["stalls"], seat)
        # Offers came after the failure and were declined in chat
        self.assertGreater(sum(seat["declined"] for seat in seats), 0, seats)

    def test_a_run_stops_once_more_games_failed_than_allowed(self):
        # Every game fails (an illegal answer); one failed game is allowed, so the simulation stops at the second and
        # says why, without starting the third
        stacsettlers.build()
        output, _ = play(3, agent_command("--illegal"), tempfile.mkdtemp(), name="OSPStop", properties={"osp.max_failed_games": "1"})
        lines, seats = results(output)
        self.assertEqual(2, len(lines), output[-3000:])
        self.assertTrue(all(seat["failures"] for seat in seats), seats)
        stops = [line for line in output.splitlines() if line.startswith(stacsettlers.STOPPED)]
        self.assertEqual(1, len(stops), output[-3000:])
        self.assertIn("2 games", stops[0])


if __name__ == "__main__":
    unittest.main()
