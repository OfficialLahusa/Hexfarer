"""
The steps a StacSettlers robot sends an Open Seat Agent that observes (OSPRobotBrain.java with MoveLog.java): every move
of every seat from the game's start, in order and numbered, each decision numbered by the moves before it, the same
moves the referee's own log of the game records (-Dosp.record), and for a seat's view only what the seat may see: no
state, a bought card only for the buyer, a stolen card only for the thief and the victim. Without trades between players
(Trades=false) every change of the game is a step; with them (Trades=true) a trade the Stac agents make in chat is an
offer of the player on turn and the partner's accept (player_trading first_accept), so nothing happens between the
steps either. Needs a JDK and StacSettlers' jar in STACSETTLERS_JAR. Run: python -m unittest test_stacsettlers_steps
"""

import json
import os
import sys
import tempfile
import unittest

import stacsettlers

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_AGENT = os.path.join(HERE, "..", "jsettlers", "first_move_agent.py")
STAC = (3, "Stac", "stac", "TRY_N_BEST_BUILD_PLANS:0|FAVOUR_DEV_CARDS:-5|NP_MAX_NUMBER_OF_OFFERS_WITHOUT_BUILDING:3")


def play(games, flags, trades):
    """Games of one observing test agent against three Stac agents, recorded; the agent's messages and the records by game"""
    directory, logs, records = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    config = stacsettlers.config("OSPSteps", games, [(1, "Hexfarer", "osp"), STAC], trades=trades)
    agent = f'"{sys.executable}" "{os.path.abspath(TEST_AGENT)}" --observe --full --weighted ' + " ".join(flags)
    output = stacsettlers.simulate(directory, config, agent, {"osp.log": logs, "osp.record": records, "osp.stall_ms": "3000"}, timeout=900)
    got, recorded = {}, {}
    for name in os.listdir(logs):
        with open(os.path.join(logs, name), encoding="utf-8") as file:
            messages = [json.loads(line) for line in file if line.startswith("{")]
        got[messages[1]["game_id"]] = messages
    for name in os.listdir(records):
        with open(os.path.join(records, name), encoding="utf-8") as file:
            messages = [json.loads(line) for line in file]
        recorded[messages[0]["game_id"]] = messages
    lines = [json.loads(line[line.index("{"):]) for line in output.splitlines() if '"osp-result"' in line]
    return got, recorded, lines, output


class Checks:
    """What every observed game must show, for a seat's view or the true state"""

    def check_games(self, needs):
        self.assertEqual(self.GAMES, len(self.lines), self.output[-3000:])
        self.assertEqual(self.GAMES, len(self.got))
        for line in self.lines:
            seat = next(s for s in line["result"]["seats"] if s["agent"] is not None)
            self.assertEqual((needs, True, 0, [], 0), (seat["needs"], seat["observe"], seat["fallbacks"], seat["failures"], seat["stalls"]), seat)
            self.assertIsNotNone(line["winner"])

    def test_the_agent_follows_every_move_from_the_start(self):
        for game_id, messages in self.got.items():
            kinds = [m["type"] for m in messages]
            self.assertEqual(["handshake", "before"], kinds[:2])
            self.assertEqual("after", kinds[-1])
            # The game from its start: the setup's first settlement still to place
            self.assertEqual(("setup_settlement", 0), (messages[1]["state"]["prompt"], len(messages[1]["state"]["buildings"]["settlements"])))
            steps = 0
            for message in messages[2:-1]:
                if message["type"] == "step":
                    self.assertEqual(steps, message["n"], game_id)
                    steps += 1
                else:
                    self.assertEqual("decide", message["type"])
                    # A decision comes after every move before it, numbered by them
                    self.assertEqual(steps, message["n"], game_id)
            # Up to the end
            recorded = [m for m in self.recorded[game_id] if m["type"] == "step"]
            self.assertEqual(len(recorded), steps, game_id)

    def test_the_steps_are_the_referees_moves(self):
        for game_id, messages in self.got.items():
            seat = messages[1]["seat"]
            recorded = [m for m in self.recorded[game_id] if m["type"] == "step"]
            steps = [m for m in messages if m["type"] == "step"]
            for step, truth in zip(steps, recorded):
                self.assertEqual((truth["n"], truth["seat"], truth["action"]), (step["n"], step["seat"], step["action"]), game_id)
                self.assertEqual(self.shown(truth, seat), step["result"], f"{game_id} move {step['n']}")

    def test_each_decision_is_the_referees_state_as_the_seat_sees_it(self):
        # The state in the protocol's order (a discard waits for the lower seats' discards), as the seat may see it
        for game_id, messages in self.got.items():
            seat = messages[1]["seat"]
            states = {m["n"]: m["state"] for m in self.recorded[game_id] if m["type"] == "decide"}
            for decide in (m for m in messages if m["type"] == "decide"):
                self.assertEqual(self.seen(states[decide["n"]], seat), decide["state"], f"{game_id} move {decide['n']}")

    def test_the_game_ends_where_the_referees_does(self):
        for game_id, messages in self.got.items():
            after, recorded = messages[-1], self.recorded[game_id][-1]
            self.assertEqual((recorded["winner"], recorded["vp"]), (after["winner"], after["vp"]), game_id)
            # Nothing changed the game between the steps
            self.assertEqual(0, recorded["gaps"], game_id)


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class ViewStepTests(Checks, unittest.TestCase):
    """A seat's view, without trades between players"""
    GAMES = 3
    TRADES = False
    TRADING = "none"

    @classmethod
    def setUpClass(cls):
        stacsettlers.build()
        cls.got, cls.recorded, cls.lines, cls.output = play(cls.GAMES, ["--needs", "view"], cls.TRADES)

    @staticmethod
    def seen(state, seat):
        """A state as the seat may see it: the others' hands, cards and points hidden"""
        view = json.loads(json.dumps(state))
        for player in view["players"]:
            if player["seat"] != seat:
                player.update(hand=None, development=None, vp=None)
        return view

    @staticmethod
    def shown(truth, seat):
        """A step's result as the seat may see it"""
        kind, result = truth["action"][0], truth["result"]
        if kind == "buy_development" and truth["seat"] != seat:
            return {"card": None}
        if kind == "move_robber" and result is not None and seat not in (truth["seat"], truth["action"][1]["victim"]):
            return {"stolen": "unknown"}
        return result

    def test_the_robot_observes_and_never_fails(self):
        self.check_games("view")

    def test_a_view_hides_what_the_seat_cant_see(self):
        hidden = shown = 0
        for messages in self.got.values():
            seat = messages[1]["seat"]
            self.assertTrue(all(p["hand"] is None for p in messages[1]["state"]["players"] if p["seat"] != seat))
            for step in (m for m in messages if m["type"] == "step"):
                self.assertNotIn("state", step)
                if step["action"][0] == "buy_development":
                    if step["seat"] == seat:
                        shown += 1
                        self.assertIsNotNone(step["result"]["card"])
                    else:
                        hidden += 1
                        self.assertIsNone(step["result"]["card"])
        self.assertGreater(hidden, 0)

    def test_the_ruleset_names_the_trades_between_players(self):
        for messages in self.got.values():
            self.assertEqual(self.TRADING, messages[1]["ruleset"]["player_trading"])


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class TradeStepTests(ViewStepTests):
    """A seat's view, with the Stac agents trading among themselves in chat"""
    GAMES = 3
    TRADES = True
    TRADING = "first_accept"

    def test_a_trade_is_an_offer_and_its_accept(self):
        trades = 0
        for game_id, messages in self.got.items():
            seat = messages[1]["seat"]
            steps = [m for m in messages if m["type"] == "step"]
            for offer, answer in zip(steps, steps[1:]):
                if offer["action"][0] != "offer":
                    continue
                trades += 1
                terms = offer["action"][1]
                # Between two other seats, the player on turn offering one partner exactly what changes hands
                self.assertEqual(1, len(terms["to"]), offer)
                self.assertEqual((terms["to"][0], ["respond", {"accept": True}]), (answer["seat"], answer["action"]), game_id)
                self.assertNotIn(seat, (offer["seat"], answer["seat"]))
                self.assertTrue(terms["give"] and terms["get"] and not set(terms["give"]) & set(terms["get"]), offer)
        self.assertGreater(trades, 0)


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class TruthStepTests(Checks, unittest.TestCase):
    """The true state: each step carries the state before its move, the one the referee's log records for it"""
    GAMES = 1

    @classmethod
    def setUpClass(cls):
        stacsettlers.build()
        cls.got, cls.recorded, cls.lines, cls.output = play(cls.GAMES, ["--needs", "truth"], True)

    @staticmethod
    def seen(state, seat):
        return state

    @staticmethod
    def shown(truth, seat):
        return truth["result"]

    def test_the_robot_observes_and_never_fails(self):
        self.check_games("truth")

    def test_each_step_carries_the_state_before_it(self):
        for game_id, messages in self.got.items():
            states = {m["n"]: m["state"] for m in self.recorded[game_id] if m["type"] == "decide"}
            for step in (m for m in messages if m["type"] == "step"):
                self.assertEqual(states[step["n"]], step["state"], f"{game_id} move {step['n']}")


if __name__ == "__main__":
    unittest.main()
