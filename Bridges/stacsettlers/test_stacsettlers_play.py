"""
Tests of play.py's summary and of a simulation that runs out of time (stacsettlers.simulate): the summary says what each
seat asked for (needs, observe, lean), and a run that hangs keeps the games it finished. The summary tests need nothing
else; the timeout test needs a JDK and StacSettlers' jar in STACSETTLERS_JAR.
Run: python -m unittest test_stacsettlers_play
"""

import json
import os
import tempfile
import unittest

import play
import stacsettlers


def result_line(game, seat, winner, needs="truth", observe=True, lean=False):
    """An osp-result line as GameRecord writes it, with the agent in the seat"""
    seats = []
    for s in range(4):
        ours = s == seat
        seats.append({"seat": s, "agent": "hexfarer-test" if ours else None, "needs": needs if ours else None, "observe": observe if ours else None,
                      "lean": lean if ours else None, "decisions": 10 if ours else 0, "time_ms": 100 if ours else 0, "max_ms": 20 if ours else 0,
                      "robot_decisions": 12 if ours else 0, "fallbacks": 0, "failures": [], "stalls": 0, "declined": 0})
    return json.dumps({"type": "osp-result", "game_id": game, "players": ["Hexfarer" if s == seat else f"Stac_{s}" for s in range(4)],
                       "winner": winner, "vp": [10 if s == winner else 5 for s in range(4)], "result": {"turns": 60, "seats": seats}})


class SummaryTests(unittest.TestCase):
    def test_it_says_what_each_seat_asked_for(self):
        output = "\n".join([result_line("g0", 1, 1, needs="truth", observe=True), result_line("g1", 2, 0, needs="view", observe=False, lean=False)])
        result = play.summary(output)
        self.assertEqual(2, result["games"])
        self.assertEqual(1, result["wins"])
        self.assertEqual({"needs truth, observes": 1, "needs view, no steps": 1}, result["asked"])
        self.assertEqual(("truth", True, False), (result["per_game"][0]["needs"], result["per_game"][0]["observe"], result["per_game"][0]["lean"]))
        self.assertEqual(("view", False, False), (result["per_game"][1]["needs"], result["per_game"][1]["observe"], result["per_game"][1]["lean"]))
        self.assertFalse(result["timed_out"])

    def test_a_run_that_timed_out_says_so(self):
        output = result_line("g0", 0, 0) + "\n" + stacsettlers.TIMED_OUT.format(seconds=5)
        result = play.summary(output)
        self.assertEqual(1, result["games"])
        self.assertTrue(result["timed_out"])


@unittest.skipUnless(stacsettlers.available(), "needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
class TimeoutTests(unittest.TestCase):
    def test_a_simulation_out_of_time_returns_what_it_wrote(self):
        stacsettlers.build()
        config = stacsettlers.config("OSPTimeout", 50, [(4, "Stac", "stac")])
        output = stacsettlers.simulate(os.path.join(tempfile.mkdtemp(), "simulation"), config, timeout=3)
        # No exception: the output so far, ending with the line that says the run timed out
        self.assertIn(stacsettlers.TIMED_OUT.format(seconds=3), output)


if __name__ == "__main__":
    unittest.main()
