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


def result_line(game, seat, winner, needs="truth", observe=True, lean=False, failures=(), fallbacks=0):
    """An osp-result line as GameRecord writes it, with the agent in the seat (and its failures and the referee's
    fallbacks for it)"""
    seats = []
    for s in range(4):
        ours = s == seat
        seats.append({"seat": s, "agent": "hexfarer-test" if ours else None, "needs": needs if ours else None, "observe": observe if ours else None,
                      "lean": lean if ours else None, "decisions": 10 if ours else 0, "time_ms": 100 if ours else 0, "max_ms": 20 if ours else 0,
                      "robot_decisions": 12 if ours else 0, "fallbacks": fallbacks if ours else 0,
                      "failures": [{"n": 40, "code": code, "by": "agent", "message": "lost"} for code in failures] if ours else [],
                      "stalls": 0, "declined": 0})
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

    def test_failed_games_stay_out_of_the_win_rate_and_are_counted(self):
        # A win and a loss without trouble, a win with the agent's failure, a loss the referee played on with fallbacks
        output = "\n".join([result_line("g0", 0, 0), result_line("g1", 1, 1, failures=["internal"], fallbacks=30),
                            result_line("g2", 2, 0, fallbacks=4), result_line("g3", 3, 0)])
        result = play.summary(output)
        self.assertEqual((4, 2, 2), (result["games"], result["failed_games"], result["counted_games"]))
        self.assertEqual((1, 0.5), (result["wins"], result["win_rate"]))
        self.assertEqual([False, True, True, False], [g["failed"] for g in result["per_game"]])
        # The points of the counted games only (10 and 5)
        self.assertEqual(7.5, result["mean_vp"])
        self.assertEqual(1, len(result["failures"]))

    def test_it_sums_the_seats_trades_in_chat(self):
        # Records of a robot trading in chat carry its offers, answered offers and trades; older ones don't (zero)
        first, second = json.loads(result_line("g0", 0, 0)), json.loads(result_line("g1", 1, 1))
        first["result"]["seats"][0].update(offers=4, answers=7, trades=2)
        second["result"]["seats"][1].update(offers=1, answers=0, trades=1)
        older = result_line("g2", 2, 0)
        result = play.summary("\n".join([json.dumps(first), json.dumps(second), older]))
        self.assertEqual((5, 7, 3), (result["offers"], result["answers"], result["trades"]))
        self.assertEqual([(4, 7, 2), (1, 0, 1), (0, 0, 0)], [(g["offers"], g["answers"], g["trades"]) for g in result["per_game"]])


def output_of(*games):
    """A simulation's output with one result line per (won, failed) pair, our seat 0"""
    return "\n".join(result_line(f"g{i}", 0, 0 if won else 1, failures=["internal"] if failed else ()) for i, (won, failed) in enumerate(games))


class RunGamesTests(unittest.TestCase):
    """play.run_games: the missing games played in further simulations until enough count, and the stop rule"""

    def test_failed_games_are_replaced_until_enough_count(self):
        asked = []

        def simulation(round_number, games, allowance):
            asked.append((round_number, games, allowance))
            return output_of((True, False), (False, True), (True, False), (False, False)) if round_number == 1 else output_of((True, False))

        outputs, stopped = play.run_games(simulation, games=4, max_failed=3)
        # The second simulation plays the one failed game again, with one failed game less allowed
        self.assertEqual([(1, 4, 3), (2, 1, 2)], asked)
        self.assertIsNone(stopped)
        result = play.summary("\n".join(outputs))
        self.assertEqual((5, 1, 4, 3), (result["games"], result["failed_games"], result["counted_games"], result["wins"]))

    def test_a_run_goes_on_with_as_many_failed_games_as_allowed(self):
        asked = []

        def simulation(round_number, games, allowance):
            asked.append((round_number, games, allowance))
            return output_of((True, True), (True, False)) if round_number == 1 else output_of((False, False))

        outputs, stopped = play.run_games(simulation, games=2, max_failed=1)
        # One failed game, one allowed: the run plays it again (with none left to fail) and ends with enough games
        self.assertEqual([(1, 2, 1), (2, 1, 0)], asked)
        self.assertIsNone(stopped)

    def test_a_negative_limit_allows_any_number_of_failed_games(self):
        asked = []

        def simulation(round_number, games, allowance):
            asked.append((round_number, games, allowance))
            return output_of((True, True), (False, True), (True, False)) if round_number == 1 else output_of((True, False), (False, False))

        outputs, stopped = play.run_games(simulation, games=3, max_failed=-1)
        # The simulation gets no limit either (a negative allowance)
        self.assertEqual(2, len(asked))
        self.assertTrue(all(allowance < 0 for _, _, allowance in asked), asked)
        self.assertIsNone(stopped)

    def test_a_negative_limit_ends_a_run_whose_games_all_failed(self):
        asked = []

        def simulation(round_number, games, allowance):
            asked.append((round_number, games, allowance))
            # A setup that fails every game; without the rule the run would play them again forever
            if round_number > 10:
                raise AssertionError("the run replays forever")
            return output_of(*[(True, True)] * games)

        outputs, stopped = play.run_games(simulation, games=3, max_failed=-1)
        self.assertEqual([(1, 3, -1)], asked)
        self.assertIn("every game of simulation 1 failed", stopped)
        self.assertIn("3 of 3 games failed", stopped)

    def test_a_negative_limit_ends_a_run_once_a_replay_failed_every_game(self):
        asked = []

        def simulation(round_number, games, allowance):
            asked.append((round_number, games, allowance))
            if round_number > 10:
                raise AssertionError("the run replays forever")
            # The first simulation counts two games, every replay fails
            return output_of((True, False), (True, True), (False, True), (False, False)) if round_number == 1 else output_of(*[(True, True)] * games)

        outputs, stopped = play.run_games(simulation, games=4, max_failed=-1)
        self.assertEqual([(1, 4, -1), (2, 2, -3)], asked)
        self.assertIn("every game of simulation 2 failed", stopped)
        self.assertIn("4 of 6 games failed", stopped)
        # The games that count stay counted
        self.assertEqual(2, play.summary("\n".join(outputs))["counted_games"])

    def test_a_limit_still_replays_after_a_simulation_whose_games_all_failed(self):
        asked = []

        def simulation(round_number, games, allowance):
            asked.append((round_number, games, allowance))
            # One failed game, its replay fails too, the next replay counts: within a limit of 3 the run goes on
            return {1: output_of((True, False), (True, True)), 2: output_of((False, True))}.get(round_number, output_of((False, False)))

        outputs, stopped = play.run_games(simulation, games=2, max_failed=3)
        self.assertEqual([(1, 2, 3), (2, 1, 2), (3, 1, 1)], asked)
        self.assertIsNone(stopped)

    def test_a_run_stops_once_more_games_failed_than_allowed(self):
        asked = []

        def simulation(round_number, games, allowance):
            asked.append((round_number, games, allowance))
            # The simulation stopped itself at its second failed game (allowance 1), as the bridge does
            return output_of((True, True), (False, True)) + "\n" + stacsettlers.STOPPED + " 2 games ended with a failed Open Seat seat"

        outputs, stopped = play.run_games(simulation, games=10, max_failed=1)
        self.assertEqual([(1, 10, 1)], asked)
        self.assertIn("2 of 2 games failed", stopped)
        self.assertIn("at most 1", stopped)

    def test_a_simulation_that_finishes_no_game_ends_the_run(self):
        calls = []

        def simulation(round_number, games, allowance):
            calls.append(round_number)
            return stacsettlers.TIMED_OUT.format(seconds=5)

        outputs, stopped = play.run_games(simulation, games=3, max_failed=3)
        self.assertEqual([1], calls)
        self.assertIn("no game", stopped)
        # With a negative limit too, and the reason stays that no game finished
        calls.clear()
        outputs, stopped = play.run_games(simulation, games=3, max_failed=-1)
        self.assertEqual([1], calls)
        self.assertIn("finished no game", stopped)

    def test_the_default_allows_three_failed_games_or_one_in_a_hundred(self):
        self.assertEqual((3, 3, 4), (play.default_max_failed(40), play.default_max_failed(300), play.default_max_failed(400)))


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
