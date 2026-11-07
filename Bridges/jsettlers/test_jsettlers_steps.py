"""
Every move of a JSettlers game as a protocol step (OSPServer.java, MoveLog.java): JSettlers' server hands a robot's agent
that observes every move by every seat, with its result and the state before it, numbered by the game's moves, so an
agent can follow the game (Hexfarer's fair agent does) instead of taking it from each decision's state. The server also
writes each game as a referee log (-Dosp.record=DIR), which Hexfarer's replay harness and relay read, while it is played.
Here each step must explain the change to the next state by the rules (payouts, costs, cards, the robber), the steps
must be numbered without a gap, and each decision's state must be the state its move then starts from.
Needs a JDK and JSettlers' full jar in JSETTLERS_JAR. Run: python -m unittest test_jsettlers_steps
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

import jsettlers
from test_jsettlers_robot import play

HERE = os.path.dirname(os.path.abspath(__file__))
RESOURCES = ("brick", "lumber", "ore", "grain", "wool")
TILE_RESOURCES = {"H": "brick", "FO": "lumber", "M": "ore", "F": "grain", "P": "wool"}
COSTS = {"build_road": {"brick": 1, "lumber": 1}, "build_settlement": {"brick": 1, "lumber": 1, "grain": 1, "wool": 1},
         "build_city": {"grain": 2, "ore": 3}, "buy_development": {"ore": 1, "grain": 1, "wool": 1}}


def key(value):
    return json.dumps(value, sort_keys=True)


def projection(state):
    """What the rules let a move change, comparable as a whole: hands, cards, buildings, the robber, whose turn it is"""
    return {
        "hands": [dict(p["hand"]) for p in state["players"]],
        "development": [dict(p["development"]) for p in state["players"]],
        "knights": [p["played"]["knight"] for p in state["players"]],
        "settlements": sorted(key(s) for s in state["buildings"]["settlements"]),
        "cities": sorted(key(c) for c in state["buildings"]["cities"]),
        "roads": sorted(key(r) for r in state["buildings"]["roads"]),
        "robber": state["robber"],
        "on_turn": state["on_turn"],
    }


def add(hand, cards, sign=1):
    for name, count in cards.items():
        hand[name] += sign * count


def produced(board, intersection, robber=None):
    """The resources of the land hexes at an intersection, one per hex (robber's hex left out)"""
    tiles = {tuple(t["tile"]): t for t in board["tiles"]}
    for hex in intersection:
        tile = tiles.get(tuple(hex))
        if tile and tile["tCode"] in TILE_RESOURCES and list(hex) != robber:
            yield tile, TILE_RESOURCES[tile["tCode"]]


def expected_after(board, state, step):
    """The projection of the state after the step, by the rules JSettlers plays (no resource bank)"""
    after = projection(state)
    seat, move = step["seat"], step["action"]
    kind, value = move[0], (move[1] if len(move) > 1 else None)
    hands, cards = after["hands"], after["development"]
    paid = state["prompt"] == "main"
    if kind == "roll":
        total = sum(step["result"]["dice"])
        if total != 7:
            for buildings, amount in (("settlements", 1), ("cities", 2)):
                for owner, intersection in state["buildings"][buildings]:
                    for tile, resource in produced(board, intersection, state["robber"]):
                        if tile.get("rNumber") == total:
                            hands[owner][resource] += amount
    elif kind == "build_settlement":
        after["settlements"] = sorted(after["settlements"] + [key([seat, value])])
        if paid:
            add(hands[seat], COSTS[kind], -1)
        elif any(owner == seat for owner, _ in state["buildings"]["settlements"]):
            # The second setup settlement brings a card of each land hex next to it
            for _, resource in produced(board, value):
                hands[seat][resource] += 1
    elif kind == "build_road":
        after["roads"] = sorted(after["roads"] + [key([seat, value])])
        if paid:
            add(hands[seat], COSTS[kind], -1)
        if state["prompt"] == "setup_road":
            # The setup goes up the seats and back down: the last seat places twice, seat 0 then starts the game
            last = len(state["players"]) - 1
            first_round = not any(owner == seat for owner, _ in state["buildings"]["roads"])
            after["on_turn"] = (seat + 1 if seat < last else seat) if first_round else (seat - 1 if seat > 0 else 0)
    elif kind == "build_city":
        after["settlements"] = sorted(s for s in after["settlements"] if s != key([seat, value]))
        after["cities"] = sorted(after["cities"] + [key([seat, value])])
        add(hands[seat], COSTS[kind], -1)
    elif kind == "buy_development":
        add(hands[seat], COSTS[kind], -1)
        cards[seat][step["result"]["card"]] += 1
    elif kind == "play_knight":
        cards[seat]["knight"] -= 1
        after["knights"][seat] += 1
    elif kind == "play_road_building":
        cards[seat]["road_building"] -= 1
    elif kind == "play_year_of_plenty":
        cards[seat]["year_of_plenty"] -= 1
        for name in value:
            hands[seat][name] += 1
    elif kind == "play_monopoly":
        cards[seat]["monopoly"] -= 1
        for other, taken in step["result"]["taken"].items():
            hands[int(other)][value] -= taken
            hands[seat][value] += taken
    elif kind == "move_robber":
        after["robber"] = value["hex"]
        if value["victim"] is not None:
            stolen = step["result"]["stolen"]
            hands[value["victim"]][stolen] -= 1
            hands[seat][stolen] += 1
    elif kind == "discard":
        add(hands[seat], value, -1)
    elif kind == "bank_trade":
        add(hands[seat], value["give"], -1)
        add(hands[seat], value["get"])
    elif kind == "end_turn":
        after["on_turn"] = (state["on_turn"] + 1) % len(state["players"])
    else:
        raise AssertionError(f"no rule for {kind}")
    return after


def read(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.startswith("{")]


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class StepTests(unittest.TestCase):
    """Two robot-only games with an observing agent's robot, recorded by the server"""

    @classmethod
    def setUpClass(cls):
        jsettlers.build()
        cls.logs, cls.records = tempfile.mkdtemp(), tempfile.mkdtemp()
        agent = f'"{sys.executable}" "{os.path.join(HERE, "first_move_agent.py")}" --builds --observe --full'
        cls.output = play(2, agent, cls.logs, record=cls.records)
        cls.games = [read(os.path.join(cls.logs, name)) for name in sorted(os.listdir(cls.logs))]
        cls.recorded = [read(os.path.join(cls.records, name)) for name in sorted(os.listdir(cls.records))]

    def assertStepsExplainTheGame(self, before, steps, final=None):
        # Each step's move, applied to the state it carries, gives the next step's state
        board = before["board"]
        self.assertEqual(projection(before["state"]), projection(steps[0]["state"]), "the game starts where the before said")
        for step, following in zip(steps, steps[1:] + ([final] if final else [])):
            # The seat that moves is the one the state asks
            self.assertEqual(step["state"]["current"], step["seat"], step)
            self.assertEqual(expected_after(board, step["state"], step), projection(following["state"]),
                             f"move {step['n']} {step['action']} {step.get('result')}")

    def test_the_agent_gets_every_move_numbered_from_the_games_start(self):
        self.assertEqual(2, len(self.games), self.output[-3000:])
        for messages in self.games:
            kinds = [m["type"] for m in messages]
            self.assertEqual(["hello", "before"], kinds[:2])
            self.assertEqual("after", kinds[-1])
            steps = [m for m in messages if m["type"] == "step"]
            self.assertEqual(list(range(len(steps))), [s["n"] for s in steps])
            # The before is the empty board: the first move is seat 0's first settlement
            self.assertEqual([], messages[1]["state"]["buildings"]["settlements"])
            self.assertEqual((0, "build_settlement"), (steps[0]["seat"], steps[0]["action"][0]))
            # Before each decision the agent has every move so far: the decision's n is the next move's
            seen = 0
            for message in messages:
                if message["type"] == "step":
                    seen += 1
                elif message["type"] == "decide":
                    self.assertEqual(seen, message["n"], message["n"])

    def test_each_step_explains_the_change_to_the_next_state(self):
        for messages in self.games:
            steps = [m for m in messages if m["type"] == "step"]
            self.assertGreater(len(steps), 150)
            self.assertStepsExplainTheGame(messages[1], steps)

    def test_a_decisions_state_is_where_its_move_starts(self):
        for messages in self.games:
            steps = {m["n"]: m for m in messages if m["type"] == "step"}
            decides = [m for m in messages if m["type"] == "decide"]
            self.assertGreater(len(decides), 5)
            for decide in decides:
                self.assertIn(decide["n"], steps, decide["n"])
                move = steps[decide["n"]]
                self.assertEqual(decide["seat"], move["seat"])
                self.assertEqual(key(decide["state"]), key(move["state"]), f"move {decide['n']}")
                self.assertIn(key(move["action"]), [key(m) for m in decide["legal"]])

    def test_the_moves_cover_the_rules(self):
        kinds = {s["action"][0] for messages in self.games for s in messages if s["type"] == "step"}
        for kind in ("build_settlement", "build_road", "build_city", "roll", "end_turn", "move_robber", "buy_development",
                     "play_knight", "discard", "bank_trade"):
            self.assertIn(kind, kinds)
        robbed = [s for messages in self.games for s in messages if s["type"] == "step" and s["action"][0] == "move_robber"]
        self.assertTrue(any(s["action"][1]["victim"] is not None for s in robbed))

    def test_discards_come_from_the_lowest_seat_up(self):
        # JSettlers' players discard at the same time; the protocol has them discard one after another, lowest seat first
        for messages in self.games:
            steps = [m for m in messages if m["type"] == "step"]
            for i, step in enumerate(steps):
                if step["action"][0] == "discard" and i + 1 < len(steps) and steps[i + 1]["action"][0] == "discard":
                    self.assertLess(step["seat"], steps[i + 1]["seat"])
        self.assertTrue(any(m["type"] == "step" and m["action"][0] == "discard" for messages in self.games for m in messages))

    def test_results_show_what_the_referee_knows(self):
        for messages in self.games:
            for step in (m for m in messages if m["type"] == "step"):
                kind = step["action"][0]
                if kind == "roll":
                    self.assertTrue(all(1 <= d <= 6 for d in step["result"]["dice"]) and len(step["result"]["dice"]) == 2, step)
                elif kind == "buy_development":
                    self.assertIn(step["result"]["card"], ("knight", "road_building", "year_of_plenty", "monopoly", "victory_point"))
                elif kind == "move_robber":
                    victim = step["action"][1]["victim"]
                    self.assertEqual(victim is None, step["result"] is None or step["result"]["stolen"] is None, step)

    def test_the_server_records_each_game_as_a_referee_log_while_it_runs(self):
        self.assertEqual(2, len(self.recorded), os.listdir(self.records))
        for messages in self.recorded:
            kinds = [m["type"] for m in messages]
            self.assertEqual("before", kinds[0])
            self.assertEqual("after", kinds[-1])
            # A decision and its step for every move, numbered without a gap
            self.assertEqual(["decide", "step"] * ((len(kinds) - 2) // 2), kinds[1:-1])
            pairs = list(zip(messages[1:-1:2], messages[2:-1:2]))
            self.assertEqual(list(range(len(pairs))), [d["n"] for d, _ in pairs])
            for decide, step in pairs:
                self.assertEqual((decide["n"], decide["seat"]), (step["n"], step["seat"]))
                self.assertIn(key(step["action"]), [key(m) for m in decide["legal"]], f"move {step['n']}")
            # The steps explain the game up to its end, which names the winner
            steps = [dict(s, state=d["state"]) for d, s in pairs]
            self.assertStepsExplainTheGame(messages[0], steps, final=messages[-1])
            self.assertIsNotNone(messages[-1]["winner"])
            self.assertGreaterEqual(messages[-1]["vp"][messages[-1]["winner"]], 10)

    def test_a_recorded_games_line_counts_the_decisions_with_a_choice(self):
        lines = [json.loads(line[line.index("{"):]) for line in self.output.splitlines() if '"osp-game"' in line]
        self.assertEqual(2, len(lines), self.output[-3000:])
        for line in lines:
            self.assertTrue(all(0 < choices <= moves for choices, moves in zip(line["choices"], line["moves"])), line)

    def test_a_recorded_game_is_the_one_the_agent_saw(self):
        agent = {m["game_id"]: [s["action"] for s in self.games[i] if s["type"] == "step"] for i, m in enumerate(g[1] for g in self.games)}
        for messages in self.recorded:
            recorded = [m["action"] for m in messages if m["type"] == "step"]
            self.assertEqual(agent[messages[0]["game_id"]], recorded)


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class DiscardTests(unittest.TestCase):
    """Discards JSettlers' players make at the same time, written in the protocol's order (MoveLogCheck.java)"""

    def test_discards_sent_from_the_highest_seat_come_from_the_lowest_seat_up(self):
        jsettlers.build()
        output = jsettlers.run("osp.jsettlers.MoveLogCheck")
        self.assertIn("discards in seat order", output)


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class LockTests(unittest.TestCase):
    """The robot reads the server's game under the server's own lock for it (LockCheck.java)"""

    def test_a_read_waits_while_the_server_holds_the_game(self):
        jsettlers.build()
        output = jsettlers.run("osp.jsettlers.LockCheck")
        self.assertIn("waited for the game's monitor", output)
        self.assertIn("waited for the move log", output)
        self.assertIn("destroyed beside a waiting read", output)


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class RecordTests(unittest.TestCase):
    """Server runs recording into the same directory keep each other's records (RecordCheck.java)"""

    def test_a_second_run_keeps_the_first_runs_records(self):
        jsettlers.build()
        self.assertIn("both runs' records kept", jsettlers.run("osp.jsettlers.RecordCheck"))


if __name__ == "__main__":
    unittest.main()
