"""
Tests of summary.py, the strength of an Open Seat Agent at JSettlers from the server's osp-result lines: games where the
agent's seat failed stay out of the win rate and are counted beside it, a setting's batches count together, robot-only
games don't count, and the trade steps per game come from the referee logs. Needs nothing but Python.
Run: python -m unittest test_jsettlers_summary
"""

import json
import os
import tempfile
import unittest

import summary


def result_line(game, seat, winner, failures=(), fallbacks=0, gaps=0, rounds=12, agent="lanzenbrecher-test"):
    """An osp-result line as MoveLog writes it, with the agent in the seat (None: a robot-only game)"""
    seats = []
    for s in range(4):
        ours = agent is not None and s == seat
        seats.append({"seat": s, "agent": agent if ours else None, "decisions": 50 if ours else 0, "time_ms": 5000 if ours else 0,
                      "max_ms": 300 if ours else 0, "resyncs": 0, "fallbacks": fallbacks if ours else 0,
                      "failures": [{"n": 80, "code": code, "by": "agent", "message": "lost"} for code in failures] if ours else [],
                      "robot_decisions": 60 if ours else 0, "expired_trades": 1 if ours else 0})
    return "osp: " + json.dumps({"type": "osp-result", "game_id": game, "players": ["robot", "droid", "extrabot", "robot 2"], "winner": winner,
                                  "vp": [10 if s == winner else 4 for s in range(4)], "rounds": rounds, "gaps": gaps, "result": {"seats": seats}})


def write(directory, name, lines):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as file:
        file.write("some server line\n" + "\n".join(lines) + "\n")
    return path


class SummaryTests(unittest.TestCase):
    def test_failed_games_stay_out_of_the_win_rate_and_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            # Won, won with the agent's failure, won with the referee's fallbacks, lost, and a robot-only game
            log = write(directory, "trade-fair.log", [result_line("~botsOnly~1", 0, 0), result_line("~botsOnly~2", 1, 1, failures=["internal"], fallbacks=40),
                                                     result_line("~botsOnly~3", 2, 2, fallbacks=3, gaps=1), result_line("~botsOnly~4", 3, 0, gaps=2),
                                                     result_line("~botsOnly~5", 0, 2, agent=None)])
            [setting] = summary.settings([log])
        self.assertEqual(("trade-fair", 4, 2, 2), (setting["setting"], setting["games"], setting["failed_games"], setting["counted_games"]))
        self.assertEqual((1, 0.5), (setting["wins"], setting["win"]))
        self.assertEqual({"internal": 1}, setting["failures"])
        self.assertEqual((43, 3), (setting["fallbacks"], setting["gaps"]))
        # Points and rounds of the counted games only
        self.assertEqual((7.0, 12.0), (setting["vp"], setting["rounds"]))

    def test_a_settings_batches_count_together(self):
        with tempfile.TemporaryDirectory() as directory:
            first = write(directory, "trade-open.log", [result_line("~botsOnly~1", 0, 0), result_line("~botsOnly~2", 1, 0)])
            second = write(directory, "trade-open-b.log", [result_line("~botsOnly~1", 2, 2)])
            other = write(directory, "no-trade-open.log", [result_line("~botsOnly~1", 3, 1)])
            found = {s["setting"]: s for s in summary.settings([first, second, other])}
        self.assertEqual({"trade-open", "no-trade-open"}, set(found))
        self.assertEqual((3, 2), (found["trade-open"]["games"], found["trade-open"]["wins"]))
        self.assertEqual((1, 0), (found["no-trade-open"]["games"], found["no-trade-open"]["wins"]))

    def test_the_interval_is_wilsons(self):
        win, lo, hi = summary.wilson(323, 400)
        self.assertEqual((0.8075, 0.766, 0.843), (round(win, 4), round(lo, 3), round(hi, 3)))
        self.assertTrue(all(x != x for x in summary.wilson(0, 0)))

    def test_trade_steps_per_game_come_from_the_records(self):
        with tempfile.TemporaryDirectory() as directory:
            log = write(directory, "trade-open.log", [result_line("~botsOnly~7", 1, 1)])
            os.mkdir(os.path.join(directory, "trade-open-records"))
            steps = [{"type": "step", "seat": 1, "action": ["offer", {"give": {"ore": 1}, "get": {"wool": 1}, "to": [0, 2, 3]}]},
                     {"type": "step", "seat": 0, "action": ["respond", {"accept": False}]},
                     {"type": "step", "seat": 2, "action": ["respond", {"accept": True}]},
                     {"type": "step", "seat": 1, "action": ["roll"]}]
            with open(os.path.join(directory, "trade-open-records", "_botsOnly_7.jsonl"), "w", encoding="utf-8") as file:
                file.write("\n".join(json.dumps(s) for s in steps) + "\n")
            [setting] = summary.settings([log])
        self.assertEqual(1, setting["recorded_games"])
        self.assertEqual({"offer": 1.0}, setting["ours_per_game"])
        self.assertEqual({"respond:accept": 1.0, "respond:decline": 1.0}, setting["theirs_per_game"])


if __name__ == "__main__":
    unittest.main()
