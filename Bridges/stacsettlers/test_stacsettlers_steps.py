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


def play(games, flags, trades, properties=None):
    """Games of one observing test agent against three Stac agents, recorded (with further system properties); the
    agent's messages and the records by game"""
    directory, logs, records = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
    config = stacsettlers.config("OSPSteps", games, [(1, "Lanzenbrecher", "osp"), STAC], trades=trades)
    agent = f'"{sys.executable}" "{os.path.abspath(TEST_AGENT)}" --observe --full --weighted ' + " ".join(flags)
    output = stacsettlers.simulate(directory, config, agent, {"osp.log": logs, "osp.record": records, "osp.stall_ms": "3000", **(properties or {})},
                                   timeout=int(os.environ.get("STACSETTLERS_STEPS_TIMEOUT", "900")))
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
class ChatTradeStepTests(TradeStepTests):
    """A seat's view, with the robot taking part in the Stac agents' chat trades (osp.trade_in_chat): it announces that it
    participates, its agent answers the offers made to it (accept, decline or a counteroffer) and makes its own (here
    about once a turn, taking the first counteroffer to its own), and a trade that happens, with the robot's seat or
    between two others, is an offer of the player
    on turn and the partner's accept, on exactly the terms that changed hands"""
    GAMES = 3

    @classmethod
    def setUpClass(cls):
        stacsettlers.build()
        cls.got, cls.recorded, cls.lines, cls.output = play(cls.GAMES, ["--needs", "view", "--trades", "--take-counters"], True, {"osp.trade_in_chat": "true"})

    # A decision about an offer (answering one made to the seat, picking among the counteroffers to its own) shows the
    # offer, which the referee's log records no step for: the rest of its state is the referee's
    TRADE_FIELDS = ("offer", "prompt", "current")

    def test_each_decision_is_the_referees_state_as_the_seat_sees_it(self):
        about_offers = 0
        for game_id, messages in self.got.items():
            seat = messages[1]["seat"]
            states = {m["n"]: m["state"] for m in self.recorded[game_id] if m["type"] == "decide"}
            for decide in (m for m in messages if m["type"] == "decide"):
                expected, got = self.seen(states[decide["n"]], seat), dict(decide["state"])
                if got["prompt"] in ("answer_offer", "pick_partner"):
                    about_offers += 1
                    for field in self.TRADE_FIELDS:
                        expected.pop(field)
                        got.pop(field)
                self.assertEqual(expected, got, f"{game_id} move {decide['n']}")
        self.assertGreater(about_offers, 0)

    def test_a_trade_is_an_offer_and_its_accept(self):
        ours = {"offerer": 0, "partner": 0}
        for game_id, messages in self.got.items():
            seat = messages[1]["seat"]
            steps = [m for m in messages if m["type"] == "step"]
            for offer, answer in zip(steps, steps[1:]):
                if offer["action"][0] != "offer":
                    continue
                terms = offer["action"][1]
                self.assertEqual(1, len(terms["to"]), offer)
                self.assertEqual((terms["to"][0], ["respond", {"accept": True}]), (answer["seat"], answer["action"]), game_id)
                self.assertTrue(terms["give"] and terms["get"] and not set(terms["give"]) & set(terms["get"]), offer)
                if offer["seat"] == seat:
                    ours["offerer"] += 1
                elif answer["seat"] == seat:
                    ours["partner"] += 1
        # The robot's seat traded both ways: its own offers taken, and offers to it it took (or its counteroffers)
        self.assertGreater(ours["offerer"], 0, ours)
        self.assertGreater(ours["partner"], 0, ours)

    def test_the_agent_answers_offers_and_may_make_its_own(self):
        answers = offers = 0
        for game_id, messages in self.got.items():
            seat = messages[1]["seat"]
            for decide in (m for m in messages if m["type"] == "decide"):
                state, named = decide["state"], decide.get("trades") or {}
                if state["prompt"] == "answer_offer":
                    answers += 1
                    offer = state["offer"]
                    # An offer of another seat to this one, its answers and the counteroffer it may make instead
                    self.assertNotEqual(seat, offer["from"], decide)
                    self.assertIn(seat, offer["to"], decide)
                    # The next to answer as the rules name it, the first addressee from the offerer's left (the seat may
                    # answer out of that order under first_accept)
                    first = next(s for s in ((offer["from"] + k) % 4 for k in range(1, 4)) if s in offer["to"])
                    self.assertEqual(first, state["current"], decide)
                    self.assertTrue(offer["give"] and offer["get"], decide)
                    self.assertTrue(decide["legal"] and all(m[0] == "respond" for m in decide["legal"]), decide)
                    self.assertIn(["respond", {"accept": False}], decide["legal"])
                    # An accept only with the cards the offerer asks for
                    hand = state["players"][seat]["hand"]
                    holds = all(hand.get(name, 0) >= count for name, count in offer["get"].items())
                    self.assertEqual(holds, ["respond", {"accept": True}] in decide["legal"], decide)
                    self.assertEqual({"counter": {}}, named, decide)
                elif "offer" in named:
                    offers += 1
                    # Only on the seat's own turn, after the roll, to the other seats
                    self.assertEqual((seat, "main"), (state["on_turn"], state["prompt"]), decide)
                    self.assertEqual(sorted(s for s in range(4) if s != seat), sorted(named["offer"]["to"]), decide)
        self.assertGreater(answers, 0)
        self.assertGreater(offers, 0)

    @staticmethod
    def agents_offer(state, seat):
        """The test agent's offer at a decision naming offers (first_move_agent.py --trades): one of what it holds most
        for one of what it holds least, or None if it holds nothing"""
        hand = state["players"][seat]["hand"]
        held = sorted((count, name) for name, count in hand.items() if count > 0)
        if not held:
            return None
        most = held[-1][1]
        least = min((hand.get(r, 0), r) for r in ("brick", "lumber", "ore", "grain", "wool") if r != most)[1]
        return {"give": {most: 1}, "get": {least: 1}}

    @staticmethod
    def agents_counter(state, seat):
        """The test agent's counteroffer to the offer it answers, in the offerer's terms: the offer's give for one card of
        its own it isn't offered"""
        offer = state["offer"]
        held = sorted((count, name) for name, count in state["players"][seat]["hand"].items() if count > 0)
        mine = [name for _, name in held if name not in offer["give"]]
        return {"give": offer["give"], "get": {mine[-1]: 1}} if mine else None

    def test_the_robots_trades_are_on_terms_its_agent_agreed_to(self):
        # A trade with the robot's seat is on the terms of its agent's own offer, a counteroffer its agent took, an offer
        # its agent accepted, or its agent's counteroffer: never other cards than those its agent agreed to
        checked = countered = 0
        for game_id, messages in self.got.items():
            seat = messages[1]["seat"]
            decisions = {}
            for m in messages:
                if m["type"] == "decide":
                    decisions.setdefault(m["n"], []).append(m)
            steps = [m for m in messages if m["type"] == "step"]
            for offer, answer in zip(steps, steps[1:]):
                if offer["action"][0] != "offer" or seat not in (offer["seat"], answer["seat"]):
                    continue
                terms = {"give": offer["action"][1]["give"], "get": offer["action"][1]["get"]}
                agreed = []
                for decide in decisions.get(offer["n"], []):
                    state = decide["state"]
                    if offer["seat"] == seat and "offer" in (decide.get("trades") or {}):
                        agreed.append(self.agents_offer(state, seat))
                    elif offer["seat"] == seat and state["prompt"] == "pick_partner":
                        agreed.extend(state["offer"]["counters"].values())
                        countered += terms in state["offer"]["counters"].values()
                    elif answer["seat"] == seat and state["prompt"] == "answer_offer":
                        agreed.append({"give": state["offer"]["give"], "get": state["offer"]["get"]})
                        agreed.append(self.agents_counter(state, seat))
                self.assertIn(terms, agreed, f"{game_id} move {offer['n']}")
                checked += 1
        self.assertGreater(checked, 0)
        return countered

    def test_the_record_counts_the_robots_trades(self):
        for line in self.lines:
            stats = next(s for s in line["result"]["seats"] if s["agent"] is not None)
            game_id = line["game_id"]
            messages = self.got[game_id]
            seat = messages[1]["seat"]
            steps = [m for m in messages if m["type"] == "step"]
            traded = sum(1 for offer, answer in zip(steps, steps[1:]) if offer["action"][0] == "offer" and seat in (offer["seat"], answer["seat"]))
            answered = sum(1 for m in messages if m["type"] == "decide" and m["state"]["prompt"] == "answer_offer")
            self.assertEqual((traded, answered), (stats["trades"], stats["answers"]), stats)
            # It declines in chat only what its agent declined
            self.assertGreaterEqual(stats["answers"], stats["declined"], stats)
        self.assertGreater(sum(next(s for s in line["result"]["seats"] if s["agent"] is not None)["offers"] for line in self.lines), 0)


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class ChatCounterTradeStepTests(ChatTradeStepTests):
    """The same over more games, enough for counteroffers to the agent's own offers that it takes: the Stac agents mostly
    accept or decline its offers, a counteroffer it can take came about once in six games"""
    GAMES = 36

    def test_a_counteroffer_the_agent_took_traded_on_its_terms(self):
        # A trade of the agent's offer on the terms of a counteroffer its pick decision showed
        self.assertGreater(self.test_the_robots_trades_are_on_terms_its_agent_agreed_to(), 0)


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
