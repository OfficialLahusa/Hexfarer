"""
A JSettlers robot whose decisions an Open Seat Agent makes (OspRobotBrain.java), in robot-only games on JSettlers' own
server (OspServer.java) against JSettlers' robots: every game ends with a winner, the robot made its decisions through
the agent (here one that plays the first legal move) without a single fallback, and the agent got the game and its end.
Needs a JDK and JSettlers' full jar in JSETTLERS_JAR. Run: python -m unittest test_jsettlers_robot
"""

import json
import os
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


def play(games, agent_command, logs):
    """Robot-only games with one Open Seat Agent robot and three of JSettlers'; the server's standard error"""
    properties = {
        "jsettlers.startrobots": "4",
        "jsettlers.bots.start3p": "1,osp.jsettlers.OspRobotClient",
        "jsettlers.bots.percent3p": "100",
        "jsettlers.bots.botgames.total": str(games),
        "jsettlers.bots.botgames.parallel": "1",
        "jsettlers.bots.botgames.shutdown": "Y",
        "jsettlers.bots.botgames.wait_sec": "3",
        "jsettlers.bots.fast_pause_percent": "1",
        "jsettlers.bots.timeout.turn": "60",
        "jsettlers.gameopt.NT": "t",
        "jsettlers.allow.debug": "N",
    }
    arguments = ["java", f"-Dosp.agent={agent_command}", f"-Dosp.log={logs}", "-Dosp.debug=true", "-cp", jsettlers.classpath(), "osp.jsettlers.OspServer"]
    properties["jsettlers.port"] = str(free_port())
    arguments += [f"-D{name}={value}" for name, value in properties.items()]
    result = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=int(os.environ.get("JSETTLERS_ROBOT_TIMEOUT", "900")))
    return result.stdout + result.stderr


@unittest.skipUnless(jsettlers.available(), "needs a JDK and JSettlers' jar in JSETTLERS_JAR")
class RobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jsettlers.build()
        cls.logs = tempfile.mkdtemp()
        agent = f'"{sys.executable}" "{os.path.join(HERE, "first_move_agent.py")}"'
        cls.output = play(2, agent, cls.logs)
        cls.stats = [json.loads(line[line.index("{"):]) for line in cls.output.splitlines() if '"osp-robot"' in line]

    def test_every_game_ends_with_a_winner_and_the_robots_line(self):
        self.assertEqual(2, len(self.stats), self.output[-3000:])
        for stats in self.stats:
            self.assertIsNotNone(stats["winner"], stats)
            self.assertGreaterEqual(max(stats["vp"]), 10, stats)

    def test_the_server_never_has_to_end_the_robots_turn(self):
        # JSettlers ends a robot's turn when it stops answering, e.g. after a bought piece it doesn't place (its own robots
        # sometimes get stuck too, so only ours count)
        self.assertNotIn("For robot extrabot", self.output)

    def test_jsettlers_pinger_wakes_the_brain(self):
        # Once a second JSettlers' pinger sends the brain a message, so it looks at the game again even when no game
        # message comes (after reading the game while the server changed it, say)
        pinged = [line for line in self.output.splitlines() if line.startswith("[osp extrabot") and line.endswith("] pinged")]
        self.assertEqual(2, len(pinged), self.output[-3000:])

    def test_the_robot_decides_through_its_agent_without_fallbacks(self):
        for stats in self.stats:
            self.assertEqual("first-move", stats["agent"])
            self.assertGreater(stats["decisions"], 20, stats)
            self.assertGreater(stats["asked"], 5, stats)
            self.assertEqual(0, stats["fallbacks"], stats)

    def test_the_agent_got_the_game_its_decisions_and_its_end(self):
        logs = [os.path.join(self.logs, name) for name in os.listdir(self.logs)]
        self.assertEqual(2, len(logs))
        for log in logs:
            with open(log, encoding="utf-8") as file:
                messages = [json.loads(line) for line in file if line.startswith("{")]
            kinds = [m["type"] for m in messages]
            self.assertEqual(["hello", "before"], kinds[:2])
            self.assertEqual("after", kinds[-1])
            self.assertTrue(all(k == "decide" for k in kinds[2:-1]))
            # The decisions come in order, numbered by the robot's moves
            numbers = [m["n"] for m in messages if m["type"] == "decide"]
            self.assertEqual(numbers, sorted(numbers))



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
        self.assertGreater(len(decides), 10)
        for decide in decides:
            self.assertEqual(True, decide["own_shown"], decide)
            self.assertEqual(True, decide["others_hidden"], decide)


if __name__ == "__main__":
    unittest.main()
