"""
Trades between players at JSettlers (player_trading "first_accept"): with the game option NT off, JSettlers' robots offer,
decline, counter and accept, and so does an Open Seat Agent's robot where a decision names the trades. Every trade
request becomes a protocol step (MoveLog.java): the player on turn offers, an addressee answers in any order (decline, a
counteroffer in the offerer's terms, or an accept that trades at once), the offerer takes a counteroffer or withdraws.
Here each step must explain the change to the next state, hands and open offer included, the robot of the Open Seat
Agent must trade both ways without a fallback, and no trade may leave a change the steps don't explain (a gap).
Needs a JDK and JSettlers' full jar in JSETTLERS_JAR. Run: python -m unittest test_jsettlers_trading
"""

import copy
import json
import os
import sys
import tempfile
import unittest

import jsettlers
from test_jsettlers_robot import play, results
from test_jsettlers_steps import expected_after, key, projection, read

HERE = os.path.dirname(os.path.abspath(__file__))
TRADES = ("offer", "respond", "counter", "confirm", "cancel")


def trading_projection(state):
    """The projection with the open offer"""
    after = projection(state)
    after["offer"] = key(state["offer"])
    return after


def add(hand, cards, sign=1):
    for name, count in cards.items():
        hand[name] = hand.get(name, 0) + sign * count


def expected_trade_after(board, state, step):
    """The trading projection of the state after the step, by the rules of player_trading first_accept"""
    seat, move = step["seat"], step["action"]
    kind, value = move[0], (move[1] if len(move) > 1 else None)
    if kind not in TRADES:
        # Other moves need the offer closed, and leave it so
        after = expected_after(board, state, step)
        after["offer"] = key(None)
        return after
    after = projection(state)
    hands = after["hands"]
    offer = copy.deepcopy(state["offer"])
    if kind == "offer":
        offer = {"from": seat, "give": value["give"], "get": value["get"], "to": sorted(value["to"]), "answers": {}}
    elif kind == "respond" and value["accept"]:
        # The first accept trades at once on the offer's terms and closes it
        add(hands[offer["from"]], offer["give"], -1)
        add(hands[offer["from"]], offer["get"])
        add(hands[seat], offer["give"])
        add(hands[seat], offer["get"], -1)
        offer = None
    elif kind == "respond":
        offer["answers"][str(seat)] = "decline"
        # Everybody declined: the offer closes
        if all(offer["answers"].get(str(s)) == "decline" for s in offer["to"]):
            offer = None
    elif kind == "counter":
        offer["answers"][str(seat)] = "counter"
        offer.setdefault("counters", {})[str(seat)] = value
    elif kind == "confirm":
        terms = offer["counters"][str(value["partner"])]
        add(hands[seat], terms["give"], -1)
        add(hands[seat], terms["get"])
        add(hands[value["partner"]], terms["give"])
        add(hands[value["partner"]], terms["get"], -1)
        offer = None
    elif kind == "cancel":
        offer = None
    after["offer"] = key(offer)
    return after


def zero_free(projected):
    # Hands written with or without the resources of count 0 compare alike
    projected = copy.deepcopy(projected)
    projected["hands"] = [{name: count for name, count in hand.items() if count} for hand in projected["hands"]]
    return projected


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class TradeLogTests(unittest.TestCase):
    def test_trade_requests_become_first_accept_steps(self):
        """JSettlers' trade requests handed to the move log as its server handles them (TradeLogCheck.java): offers,
        declines, counteroffers, accepts, picks and withdrawals, out of seat order, and what JSettlers keeps that the
        protocol has closed (a counteroffer to an earlier offer, an offer its maker went on from, one before the roll)"""
        jsettlers.build()
        output = jsettlers.run("osp.jsettlers.TradeLogCheck")
        self.assertIn("trade steps checked", output)


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class TradingTests(unittest.TestCase):
    """Two robot-only games with trading on and an observing agent's robot that trades"""

    @classmethod
    def setUpClass(cls):
        jsettlers.build()
        cls.logs, cls.records = tempfile.mkdtemp(), tempfile.mkdtemp()
        agent = f'"{sys.executable}" "{os.path.join(HERE, "first_move_agent.py")}" --observe --full --trades'
        cls.output = play(2, agent, cls.logs, record=cls.records, trading=True)
        cls.lines, cls.stats = results(cls.output)
        cls.games = [read(os.path.join(cls.logs, name)) for name in sorted(os.listdir(cls.logs))]
        cls.steps = [[m for m in messages if m["type"] == "step"] for messages in cls.games]

    def test_every_game_ends_with_a_winner_and_no_gap(self):
        self.assertEqual(2, len(self.lines), self.output[-3000:])
        for line in self.lines:
            self.assertIsNotNone(line["winner"], line)
            # A trade request that made no step but changed the game would be a gap
            self.assertEqual(0, line["gaps"], line)

    def test_the_ruleset_names_jsettlers_trading(self):
        for messages in self.games:
            ruleset = messages[1]["ruleset"]
            self.assertEqual("first_accept", ruleset["player_trading"])
            self.assertIn("offer_limit", ruleset)
            self.assertIsNone(ruleset["offer_limit"])

    def test_each_step_explains_the_change_to_the_next_state(self):
        for messages, steps in zip(self.games, self.steps):
            board = messages[1]["board"]
            for step, following in zip(steps, steps[1:]):
                self.assertEqual(zero_free(expected_trade_after(board, step["state"], step)), zero_free(trading_projection(following["state"])),
                                 f"move {step['n']} {step['action']} {step.get('result')}")

    def test_the_robots_and_the_agents_robot_trade_both_ways(self):
        agent_seats = [next(seat["seat"] for seat in line["result"]["seats"] if seat["agent"] is not None) for line in self.lines]
        kinds = {s["action"][0] for steps in self.steps for s in steps}
        for kind in ("offer", "respond", "counter", "cancel"):
            self.assertIn(kind, kinds)
        ours = [(s["action"][0], s["action"][1] if len(s["action"]) > 1 else None) for steps, seat in zip(self.steps, agent_seats) for s in steps if s["seat"] == seat]
        others = [s["action"][0] for steps, seat in zip(self.steps, agent_seats) for s in steps if s["seat"] != seat]
        self.assertIn("offer", [k for k, _ in ours], "the agent's robot made no offer")
        self.assertIn("respond", [k for k, _ in ours], "the agent's robot answered no offer")
        self.assertIn("offer", others, "JSettlers' robots made no offer")
        # A trade happened: an accept that traded at once, or a counteroffer taken
        self.assertTrue(any(s["action"][0] == "confirm" or s["action"] == ["respond", {"accept": True}] for steps in self.steps for s in steps))

    def test_answers_come_in_any_order(self):
        # JSettlers' answers are concurrent: some come before the next addressee in seat order has answered
        out_of_order = 0
        for steps in self.steps:
            for step in steps:
                if step["action"][0] in ("respond", "counter") and step["state"]["current"] != step["seat"]:
                    out_of_order += 1
        self.assertGreater(out_of_order, 0)

    def test_each_recorded_move_is_among_its_decisions_legal_moves(self):
        # The record's decide line lists the moves of the seat that moved (an addressee answering out of seat order, the
        # offerer withdrawing or taking a counteroffer), so a replay can check them; offers and counteroffers can be any cards
        # and no list holds them, and a list left out (null) is unknown
        checked = out_of_order = offers = 0
        for name in sorted(os.listdir(self.records)):
            decisions = {}
            for message in read(os.path.join(self.records, name)):
                if message["type"] == "decide":
                    decisions[message["n"]] = message
                elif message["type"] == "step" and message["action"][0] == "offer":
                    # An offer is made with the offer closed: its list holds no answers or picks of a standing one
                    legal = decisions[message["n"]]["legal"]
                    if legal is not None:
                        self.assertFalse([m for m in legal if m[0] in ("respond", "confirm", "cancel")], f"{name} move {message['n']}: {legal}")
                        offers += 1
                elif message["type"] == "step" and message["action"][0] != "counter":
                    decide = decisions[message["n"]]
                    self.assertEqual(decide["seat"], message["seat"])
                    if decide["legal"] is None:
                        continue
                    listed = [json.dumps(m, sort_keys=True) for m in decide["legal"]]
                    self.assertIn(json.dumps(message["action"], sort_keys=True), listed, f"{name} move {message['n']} seat {message['seat']}: {decide['legal']}")
                    checked += 1
                    out_of_order += decide["state"]["current"] != message["seat"]
        self.assertGreater(checked, 100)
        self.assertGreater(offers, 0)
        self.assertGreater(out_of_order, 0)

    def test_the_agents_robot_trades_without_a_fallback(self):
        self.assertEqual(2, len(self.stats), self.output[-3000:])
        for seat in self.stats:
            self.assertEqual(0, seat["fallbacks"], seat)
            self.assertEqual([], seat["failures"], seat)
        self.assertNotIn("no JSettlers request for", self.output)
