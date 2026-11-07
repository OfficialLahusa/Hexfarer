"""
A JSettlers robot whose decisions an Open Seat Agent makes (OSPRobotBrain.java), in robot-only games on JSettlers' own
server (OSPServer.java) against JSettlers' robots: every game ends with a winner, the robot made its decisions through
the agent (here one that plays the first legal move) without a single fallback, and the agent got the game and its end.
Needs a JDK and JSettlers' full jar in JSETTLERS_JAR. Run: python -m unittest test_jsettlers_robot
"""

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import unittest

import jsettlers

HERE = os.path.dirname(os.path.abspath(__file__))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def play(games, agent_command, logs, record=None, turn_seconds=60):
    """Robot-only games with one Open Seat Agent robot and three of JSettlers' (the server's referee logs into the
    directory record, if given); the server's standard error"""
    properties = {
        "jsettlers.startrobots": "4",
        "jsettlers.bots.start3p": "1,osp.jsettlers.OSPRobotClient",
        "jsettlers.bots.percent3p": "100",
        "jsettlers.bots.botgames.total": str(games),
        "jsettlers.bots.botgames.parallel": "1",
        "jsettlers.bots.botgames.shutdown": "Y",
        "jsettlers.bots.botgames.wait_sec": "3",
        "jsettlers.bots.fast_pause_percent": "1",
        "jsettlers.bots.timeout.turn": str(turn_seconds),
        "jsettlers.gameopt.NT": "t",
        "jsettlers.allow.debug": "N",
    }
    arguments = ["java", f"-Dosp.agent={agent_command}", f"-Dosp.log={logs}", "-Dosp.debug=true", "-cp", jsettlers.classpath(), "osp.jsettlers.OSPServer"]
    if record:
        arguments.insert(3, f"-Dosp.record={record}")
    properties["jsettlers.port"] = str(free_port())
    arguments += [f"-D{name}={value}" for name, value in properties.items()]
    result = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=int(os.environ.get("JSETTLERS_ROBOT_TIMEOUT", "900")))
    return result.stdout + result.stderr


def results(output):
    """The games' result records (osp-result lines), and in each the line of the seat an Open Seat Agent plays"""
    lines = [json.loads(line[line.index("{"):]) for line in output.splitlines() if '"osp-result"' in line]
    return lines, [next(seat for seat in line["result"]["seats"] if seat["agent"] is not None) for line in lines]


def agent_command(*flags):
    return f'"{sys.executable}" "{os.path.join(HERE, "first_move_agent.py")}" ' + " ".join(flags)


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class RobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jsettlers.build()
        cls.logs = tempfile.mkdtemp()
        agent = f'"{sys.executable}" "{os.path.join(HERE, "first_move_agent.py")}" --builds'
        cls.output = play(2, agent, cls.logs)
        cls.lines, cls.stats = results(cls.output)

    def test_every_game_ends_with_a_winner_and_its_result_line(self):
        self.assertEqual(2, len(self.lines), self.output[-3000:])
        for line in self.lines:
            # The line names the winner, the seat with the points, and the seed the agents got
            self.assertIsNotNone(line["winner"], line)
            self.assertGreaterEqual(line["vp"][line["winner"]], 10, line)
            self.assertIsInstance(line["result"]["seed"], int)
            self.assertEqual(sum(seat["moves"] for seat in line["result"]["seats"]), line["result"]["moves"])

    def test_an_unrecorded_games_line_counts_no_choices(self):
        # Listing the legal moves for every request would cost the server's one message thread: only a recorded game has them
        for line in self.lines:
            self.assertTrue(all(seat["choices"] is None and seat["moves"] > 20 for seat in line["result"]["seats"]), line)

    def test_each_decision_of_the_robot_is_one_move_of_its_seat(self):
        # The brain moves again only once the move log counted its last move: a decision more than its seat's moves is a
        # move sent twice, which JSettlers mostly ignores, so nothing else would show it
        self.assertEqual(2, len(self.stats), self.output[-3000:])
        for seat in self.stats:
            self.assertEqual(seat["moves"], seat["robot_decisions"], seat)

    def test_the_server_never_has_to_end_the_robots_turn(self):
        # JSettlers ends a robot's turn when it stops answering, e.g. after a bought piece it doesn't place (its own robots
        # sometimes get stuck too, so only ours count)
        self.assertNotIn("For robot extrabot", self.output)

    def test_no_move_starts_while_the_server_waits_for_a_piece(self):
        # In states 30 to 32 JSettlers waits for the piece the robot asked to build; a new move then (a second build request,
        # say) leaves the server waiting for a piece the robot can't place, until it ends the robot's turn
        started = [line for line in self.output.splitlines() if re.match(r"\[osp extrabot .*\] state 3[0-2]: ", line)]
        self.assertEqual([], started)
        # The debug lines are there to check, if the robot paid for a piece at all (the test agent builds only 2 to 5
        # times in two games, now and then not once)
        if any(re.match(r'\[osp extrabot .*\] state 20: \["build_', line) for line in self.output.splitlines()):
            self.assertTrue(any(re.match(r"\[osp extrabot .*\] state 3[0-2], pending ", line) for line in self.output.splitlines()), self.output[-3000:])

    def test_jsettlers_pinger_wakes_the_brain(self):
        # Once a second JSettlers' pinger sends the brain a message, so it looks at the game again even when no game
        # message comes (after reading the game while the server changed it, say)
        pinged = [line for line in self.output.splitlines() if line.startswith("[osp extrabot") and line.endswith("] pinged")]
        self.assertEqual(2, len(pinged), self.output[-3000:])

    def test_the_robot_decides_through_its_agent_without_fallbacks(self):
        for seat in self.stats:
            self.assertEqual(("first-move", "truth"), (seat["agent"], seat["needs"]))
            self.assertGreater(seat["robot_decisions"], 20, seat)
            self.assertGreater(seat["decisions"], 5, seat)
            self.assertLessEqual(seat["max_ms"], seat["time_ms"], seat)
            self.assertEqual((0, []), (seat["fallbacks"], seat["failures"]), seat)

    def test_the_agent_got_the_game_its_decisions_and_its_end(self):
        logs = [os.path.join(self.logs, name) for name in os.listdir(self.logs)]
        self.assertEqual(2, len(logs))
        for log in logs:
            with open(log, encoding="utf-8") as file:
                messages = [json.loads(line) for line in file if line.startswith("{")]
            kinds = [m["type"] for m in messages]
            self.assertEqual(["handshake", "before"], kinds[:2])
            self.assertEqual("after", kinds[-1])
            self.assertTrue(all(k == "decide" for k in kinds[2:-1]))
            # The decisions come in order, numbered by the robot's moves
            numbers = [m["n"] for m in messages if m["type"] == "decide"]
            self.assertEqual(numbers, sorted(numbers))



@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class FailureTests(unittest.TestCase):
    """An agent that fails: the failure is told to it (if it still listens), recorded with its code in the game's result,
    and the robot plays on with the first legal move, so the game ends all the same"""

    @classmethod
    def setUpClass(cls):
        jsettlers.build()

    def failed(self, *flags):
        logs = tempfile.mkdtemp()
        output = play(1, agent_command(*flags), logs)
        lines, seats = results(output)
        self.assertEqual(1, len(lines), output[-3000:])
        self.assertIsNotNone(lines[0]["winner"], lines[0])
        seat = seats[0]
        self.assertGreater(seat["fallbacks"], 0, seat)
        told = []
        for name in os.listdir(logs):
            with open(os.path.join(logs, name), encoding="utf-8") as file:
                told += [json.loads(line) for line in file if line.startswith("{") and '"failure"' in line]
        return [(f["code"], f["by"]) for f in seat["failures"]], told

    def test_an_illegal_answer(self):
        failures, told = self.failed("--illegal", "--full")
        self.assertEqual([("illegal", "referee")], failures)
        self.assertEqual(["illegal"], [m["code"] for m in told])

    def test_a_world_is_refused(self):
        failures, told = self.failed("--needs", "world", "--full")
        self.assertEqual([("unsupported", "referee")], failures)
        self.assertEqual(["unsupported"], [m["code"] for m in told])

    def test_a_game_turned_down(self):
        failures, told = self.failed("--refuse", "--full")
        self.assertEqual([("unsupported", "agent")], failures)
        self.assertEqual([], told)


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class LeanTests(unittest.TestCase):
    """A lean agent gets each decision with the state's hash only, and the state when it asks; the hash is the one Python
    computes from that state"""

    def test_a_lean_agent_gets_the_hash_and_the_state_when_it_asks(self):
        jsettlers.build()
        logs = tempfile.mkdtemp()
        output = play(1, agent_command("--lean", "--observe", "--full"), logs)
        lines, seats = results(output)
        self.assertEqual(1, len(lines), output[-3000:])
        with open(os.path.join(logs, os.listdir(logs)[0]), encoding="utf-8") as file:
            messages = [json.loads(line) for line in file if line.startswith("{")]
        # The robot's budget: 120 s per decision, any threads, a GPU allowed
        self.assertEqual({"decide_ms": 120000, "game_ms": None, "threads": None, "gpu": True}, next(m for m in messages if m["type"] == "before")["budget"])
        decides = [m for m in messages if m["type"] == "decide"]
        lean = [d for d in decides if "state" not in d]
        full = [d for d in decides if "state" in d]
        self.assertGreater(len(lean), 5)
        for decide in full:
            canonical = json.dumps(decide["state"], sort_keys=True, separators=(",", ":"))
            self.assertEqual(hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16], decide["hash"])
        # Every lean decision was asked again, and the result counts the resyncs
        self.assertEqual(len(lean), seats[0]["resyncs"])
        self.assertEqual((True, []), (seats[0]["lean"], seats[0]["failures"]))


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class StallTests(unittest.TestCase):
    """A lean agent that stalls past the server's turn limit: the server ends the robot's turn, a change no step explains,
    so the agent's next decision comes with the full state although it didn't ask, and the result counts that turn too"""

    def test_a_turn_the_server_ends_sends_the_state_and_counts(self):
        jsettlers.build()
        logs, records = tempfile.mkdtemp(), tempfile.mkdtemp()
        # A turn limit under 8 s leaves the server's own: it ends a robot's turn after 8 s without an action, checked every
        # 4.4 s, so at most 12.4 s into the stall
        output = play(1, agent_command("--lean", "--observe", "--full", "--stall", "20"), logs, record=records, turn_seconds=5)
        lines, seats = results(output)
        self.assertEqual(1, len(lines), output[-3000:])
        self.assertGreater(lines[0]["gaps"], 0, "the server should have ended the stalled turn")
        with open(os.path.join(logs, os.listdir(logs)[0]), encoding="utf-8") as file:
            decides = [m for m in (json.loads(line) for line in file if line.startswith("{")) if m["type"] == "decide"]
        asked = {d["n"] for d in decides if "state" not in d}
        self.assertTrue([d for d in decides if "state" in d and d["n"] not in asked], "a decision after the gap should bring the state unasked")
        # The turns after the setup, also the one the server ended: each moves the player on turn on by one seat. The record
        # has only decisions that made a move, none of a turn the server ended before the seat moved, so the seats passed
        # between two decisions count (the setup's last placement is the first player's, whose turn comes next)
        with open(os.path.join(records, os.listdir(records)[0]), encoding="utf-8") as file:
            states = [m["state"] for m in (json.loads(line) for line in file) if m["type"] == "decide"]
        turns = sum((b["on_turn"] - a["on_turn"]) % 4 for a, b in zip(states, states[1:]) if b["prompt"] not in ("setup_settlement", "setup_road"))
        self.assertEqual(turns, lines[0]["result"]["turns"])


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class ViewRobotTests(unittest.TestCase):
    """A robot whose agent asks for its seat's view only (needs: view)"""

    @classmethod
    def setUpClass(cls):
        jsettlers.build()
        cls.logs = tempfile.mkdtemp()
        agent = f'"{sys.executable}" "{os.path.join(HERE, "first_move_agent.py")}" --needs view'
        cls.output = play(1, agent, cls.logs)

    def test_a_view_shows_the_seats_own_cards_and_no_others(self):
        logs = [os.path.join(self.logs, name) for name in os.listdir(self.logs)]
        self.assertEqual(1, len(logs))
        with open(logs[0], encoding="utf-8") as file:
            decides = [m for m in (json.loads(line) for line in file if line.startswith("{")) if m["type"] == "decide"]
        self.assertGreater(len(decides), 5)
        for decide in decides:
            self.assertEqual(True, decide["own_shown"], decide)
            self.assertEqual(True, decide["others_hidden"], decide)


if __name__ == "__main__":
    unittest.main()
