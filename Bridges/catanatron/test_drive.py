"""
The driver (drive.py) and the board import (osp_board.catan_map): boards import into Catanatron exactly, seats come out
in the wanted order with the same dice and development cards in every rotation, and a match with a stand-in agent
gives the agent its seat's messages in order and records every game. Run: .venv/Scripts/python -m unittest test_drive
"""

import gzip
import json
import os
import random
import sys
import subprocess
import tempfile
import unittest

from catanatron import Color
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, number_probability
from catanatron.models.player import RandomPlayer

import contextlib
import io

import analyze
import drive
import osp_board
import resume

HERE = os.path.dirname(os.path.abspath(__file__))


class ResumeTests(unittest.TestCase):
    def test_a_stopped_run_keeps_whole_maps_and_names_the_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            boards = os.path.join(directory, "boards.jsonl")
            with open(boards, "w", encoding="utf-8") as file:
                file.writelines(json.dumps({"seed": 500 + i, "board": None}) + "\n" for i in range(10))
            # Maps 501 and 502 are whole, 503 has two rotations; 504-509 are unplayed
            done = [(501, r) for r in range(4)] + [(503, 0), (503, 1)] + [(502, r) for r in range(4)]
            with open(os.path.join(directory, "results-0.jsonl"), "w", encoding="utf-8") as file:
                file.writelines(json.dumps({"map_seed": s, "rotation": r}) + "\n" for s, r in done)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                resume.main([directory, "1", "10", "2", "--boards", boards])
            # The rest of maps 1-9: 3 and 4-9, in two parts
            self.assertEqual(["3:7", "7:10"], printed.getvalue().split())
            with open(os.path.join(directory, "results-0.jsonl"), encoding="utf-8") as file:
                kept = {json.loads(line)["map_seed"] for line in file}
            with open(os.path.join(directory, "partial.jsonl"), encoding="utf-8") as file:
                moved = [json.loads(line)["map_seed"] for line in file]
            self.assertEqual(({501, 502}, [503, 503]), (kept, moved))

            # An agent lost its copy in a game of map 502: with --redo-lost the map is played again
            with open(os.path.join(directory, "agent-0.log"), "w", encoding="utf-8") as file:
                file.write(json.dumps({"type": "stats", "game_id": "map502-r1", "lost": True}, separators=(",", ":")) + "\n")
                file.write(json.dumps({"type": "stats", "game_id": "map501-r1", "lost": False}, separators=(",", ":")) + "\n")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                resume.main([directory, "1", "10", "1", "--boards", boards, "--redo-lost"])
            self.assertEqual(["2:10"], printed.getvalue().split())
            with open(os.path.join(directory, "results-0.jsonl"), encoding="utf-8") as file:
                self.assertEqual({501}, {json.loads(line)["map_seed"] for line in file})

    def test_a_map_with_a_failed_game_is_played_again(self):
        with tempfile.TemporaryDirectory() as directory:
            boards = os.path.join(directory, "boards.jsonl")
            with open(boards, "w", encoding="utf-8") as file:
                file.writelines(json.dumps({"seed": 500 + i, "board": None}) + "\n" for i in range(4))
            # Maps 501 and 502 have all four rotations, but a game of 502 ended at a failure
            def line(seed, rotation, failures=()):
                return json.dumps({"map_seed": seed, "rotation": rotation, "result": {"seats": [{"seat": 0, "failures": list(failures)}]}}) + "\n"
            with open(os.path.join(directory, "results-0.jsonl"), "w", encoding="utf-8") as file:
                file.writelines([line(501, r) for r in range(4)] + [line(502, r, [{"code": "timeout"}] if r == 2 else []) for r in range(4)])
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                resume.main([directory, "1", "4", "1", "--boards", boards])
            self.assertEqual(["2:4"], printed.getvalue().split())


class AnalyzeTests(unittest.TestCase):
    def test_our_agent_counts_under_the_projects_current_and_former_name(self):
        # Runs from before the rename name the agent hexfarer-..., later ones lanzenbrecher-...
        self.assertTrue(analyze.ours("lanzenbrecher-ismcts-fair-5000it"))
        self.assertTrue(analyze.ours("hexfarer-mcts"))
        self.assertFalse(analyze.ours("AB:2"))
        self.assertFalse(analyze.ours("CHALLENGER"))

    def test_a_failed_game_leaves_its_map_out_of_the_win_rate_and_is_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            def game(seed, rotation, winner, failures=()):
                seats = ["lanzenbrecher-mcts", "AB:2", "AB:2", "AB:2"]
                return {"game_id": f"map{seed}-r{rotation}", "map_seed": seed, "rotation": rotation, "lineup": ["open-seat-agent", "AB:2", "AB:2", "AB:2"],
                        "seats": seats, "winner": winner, "vp": [10 if s == winner else 5 for s in range(4)], "truncated": winner is None,
                        "turns": 50, "seconds": [1.0] * 4, "decisions": [10] * 4,
                        "result": {"seats": [{"seat": s, "failures": list(failures) if s == 0 else []} for s in range(4)]}}
            # Map 1: our agent wins every rotation; map 2: it wins three, the fourth game ended at its failure
            games = [game(1, r, 0) for r in range(4)] + [game(2, r, 0) for r in range(3)] + [game(2, 3, None, [{"code": "timeout"}])]
            with open(os.path.join(directory, "results-0.jsonl"), "w", encoding="utf-8") as file:
                file.writelines(json.dumps(g) + "\n" for g in games)
            summary = analyze.summarize(directory)
            self.assertEqual((1, 4, 1.0, 1), (summary["maps"], summary["games"], summary["win_rate"], summary["failed_games"]))


def as_sets(board):
    """A board's tiles and harbors as sets (their order in the lists means nothing)"""
    return (sorted(json.dumps(t, sort_keys=True) for t in board["tiles"]), sorted(json.dumps(h, sort_keys=True) for h in board["harbors"]), board["robber"])


def random_board(seed):
    m = CatanMap.from_template(BASE_MAP_TEMPLATE, number_placement="random", rng=random.Random(seed))
    desert = next(c for c, t in m.land_tiles.items() if t.resource is None)
    return osp_board.board(m, desert), m


class ImportTests(unittest.TestCase):
    def test_boards_import_exactly(self):
        for seed in range(8):
            board, original = random_board(seed)
            imported = osp_board.catan_map(board)
            self.assertEqual(as_sets(board), as_sets(osp_board.board(imported, osp_board.cube(board["robber"]))))
            # The same production at every corner (by the protocol's keys, since node ids may differ)
            produced = lambda m: {osp_board.node_keys(m)[n]: dict(m.node_production[n]) for n in m.land_nodes}
            self.assertEqual(produced(original), produced(imported))

    def test_cube_and_axial_invert(self):
        for q in range(-3, 4):
            for r in range(-3, 4):
                self.assertEqual((q, r), osp_board.axial(osp_board.cube((q, r))))


class SeatingTests(unittest.TestCase):
    def test_seats_come_out_in_order_with_the_same_chance(self):
        board, _ = random_board(3)
        for seed in (1, 77, 4242):
            decks, rolls = [], []
            for rotation in range(4):
                colors = drive.COLORS[rotation:] + drive.COLORS[:rotation]
                players = [RandomPlayer(c) for c in colors]
                game = drive.seated_game(players, seed, osp_board.catan_map(board))
                self.assertEqual(colors, [p.color for p in game.state.players])
                decks.append(list(game.state.development_listdeck))
                rolls.append([game.state.random.randint(1, 6) for _ in range(10)])
            # Every rotation draws the same development cards and dice
            self.assertTrue(all(d == decks[0] for d in decks))
            self.assertTrue(all(r == rolls[0] for r in rolls))


def fake_agent(seen, needs="truth", mode=""):
    return f'"{sys.executable}" "{os.path.join(HERE, "fake_open_seat_agent.py").replace(os.sep, "/")}" "{seen.replace(os.sep, "/")}" {needs} {mode}'.strip()


def read_lines(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


# An outside bot for --bot: picks the first legal move and writes every call of its lifecycle to CALLS
COUNTING_BOT = '''
import json
from catanatron.models.player import Player


class CountingBot(Player):
    def write(self, kind, game):
        with open(CALLS, "a", encoding="utf-8") as file:
            file.write(json.dumps({"kind": kind, "game": game.id, "color": self.color.value}) + "\\n")

    def before(self, game):
        self.write("before", game)

    def decide(self, game, playable_actions):
        self.write("decide", game)
        return playable_actions[0]

    def after(self, game):
        self.write("after", game)
'''


class DriveTests(unittest.TestCase):
    def drive(self, needs="truth", mode="", maps="0:1", decide_ms=None, max_failed=None):
        """One map (4 rotations) with the stand-in agent in the lineup's first entry: what it got, the results, the log"""
        directory = self.enterContext(tempfile.TemporaryDirectory())
        boards = os.path.join(directory, "boards.jsonl")
        with open(boards, "w", encoding="utf-8") as file:
            for seed in (5, 6):
                file.write(json.dumps({"seed": 900 + seed, "board": random_board(seed)[0]}) + "\n")
        seen, out, log = (os.path.join(directory, name) for name in ("seen.jsonl", "results.jsonl", "games.jsonl.gz"))
        arguments = ["--boards", boards, "--lineup", "open-seat-agent,R,W,R", "--open-seat-agent", fake_agent(seen, needs, mode), "--maps", maps, "--out", out, "--log", log]
        if decide_ms:
            arguments += ["--decide-ms", str(decide_ms)]
        if max_failed is not None:
            arguments += ["--max-failed", str(max_failed)]
        drive.main(arguments)
        return read_lines(seen), read_lines(out), read_lines(log)

    def test_a_view_agent_sees_only_its_seat_and_a_world_agent_is_refused(self):
        messages, results, logged = self.drive(needs="view")
        truths = {(m["game_id"], m["n"]): m["state"] for m in logged if m["type"] == "decide"}
        for message in messages:
            if message["type"] in ("before", "decide"):
                for player in message["state"]["players"]:
                    if player["seat"] != message["seat"]:
                        self.assertEqual((None, None, None), (player["hand"], player["development"], player["vp"]))
            if message["type"] == "decide":
                # The counts of every hand stay, as the rules show them
                for player, true in zip(message["state"]["players"], truths[(message["game_id"], message["n"])]["players"]):
                    self.assertEqual((sum(true["hand"].values()), sum(true["development"].values())), (player["hand_size"], player["development_size"]))
            if message["type"] == "step":
                # No state with a step under view, and another's bought card stays unknown
                self.assertNotIn("state", message)
                seat = next(m["seat"] for m in messages if m["type"] == "before" and m["game_id"] == message["game_id"])
                if message["action"][0] == "buy_development" and message["seat"] != seat:
                    self.assertEqual({"card": None}, message["result"])
        self.assertTrue(all(r["result"]["seats"][r["rotation"]]["needs"] == "view" for r in results))
        # The budget allows any threads and a GPU unless the run says otherwise
        self.assertTrue(all(m["budget"] == {"decide_ms": drive.DEFAULT_DECIDE_MS, "game_ms": None, "threads": None, "gpu": True}
                            for m in messages if m["type"] == "before"))

        with tempfile.TemporaryDirectory() as directory:
            seen = os.path.join(directory, "seen.jsonl")
            with self.assertRaises(drive.AgentFailure) as refused:
                drive.OpenSeatAgent(fake_agent(seen, "world"), subprocess.DEVNULL)
            self.assertEqual("unsupported", refused.exception.code)
            self.assertEqual("unsupported", read_lines(seen)[-1]["code"])

    def test_an_agent_that_doesnt_observe_gets_no_steps(self):
        messages, results, logged = self.drive(mode="blind")
        self.assertEqual([], [m for m in messages if m["type"] == "step"])
        self.assertTrue(any(m["type"] == "decide" for m in messages))
        self.assertTrue(all(r["result"]["seats"][r["rotation"]]["observe"] is False for r in results))

    def test_lines_that_arent_protocol_messages_are_skipped(self):
        messages, results, logged = self.drive(mode="noisy")
        self.assertEqual(4, len(results))
        self.assertTrue(all(not seat["failures"] for r in results for seat in r["result"]["seats"]))
        self.assertEqual(0, sum(1 for m in logged if m["type"] == "failure"))

    def test_a_lean_agent_gets_the_hash_and_the_state_when_it_asks(self):
        messages, results, logged = self.drive(mode="lean")
        truths = {(m["game_id"], m["n"]): m["state"] for m in logged if m["type"] == "decide"}
        decides = [m for m in messages if m["type"] == "decide"]
        for decide in decides:
            self.assertEqual(drive.state_hash(truths[(decide["game_id"], decide["n"])]), decide["hash"])
            if "state" in decide:
                self.assertEqual(0, decide["n"] % 5)
        asked = sum(1 for d in decides if "state" in d)
        self.assertGreater(asked, 0)
        self.assertEqual(asked, sum(r["result"]["seats"][r["rotation"]]["resyncs"] for r in results))

    def test_a_failure_ends_the_game_is_recorded_and_the_run_goes_on(self):
        for mode, code, by, decide_ms in (("illegal", "illegal", "referee", None), ("slow", "timeout", "referee", 500), ("refuse", "unsupported", "agent", None)):
            # Four failed games allowed: the run plays the whole map
            messages, results, logged = self.drive(mode=mode, decide_ms=decide_ms, max_failed=4)
            # Every game of the map was played and cut short at the failure, which the result record names
            self.assertEqual(4, len(results), mode)
            for result in results:
                self.assertTrue(result["truncated"], mode)
                failure = result["result"]["seats"][result["rotation"]]["failures"]
                self.assertEqual([(code, by)], [(f["code"], f["by"]) for f in failure], mode)
            self.assertEqual(4, sum(1 for m in logged if m["type"] == "failure"), mode)
            # The referee told the agent, except for the agent's own failure
            told = [m for m in messages if m["type"] == "failure"]
            self.assertEqual(0 if by == "agent" else 4, len(told), mode)

    def test_a_run_stops_once_more_games_failed_than_allowed(self):
        # Every game fails (turned down); with two failed games allowed the run stops after the third, which it wrote, and
        # says why; by default it allows three, or one in a hundred of its games
        directory = self.enterContext(tempfile.TemporaryDirectory())
        boards = os.path.join(directory, "boards.jsonl")
        with open(boards, "w", encoding="utf-8") as file:
            for seed in (5, 6):
                file.write(json.dumps({"seed": 900 + seed, "board": random_board(seed)[0]}) + "\n")
        seen, out = os.path.join(directory, "seen.jsonl"), os.path.join(directory, "results.jsonl")
        with self.assertRaises(SystemExit) as stopped:
            drive.main(["--boards", boards, "--lineup", "open-seat-agent,R,W,R", "--open-seat-agent", fake_agent(seen, "truth", "refuse"),
                        "--maps", "0:2", "--out", out, "--max-failed", "2"])
        self.assertIn("3 of 3 games failed", str(stopped.exception))
        self.assertIn("at most 2", str(stopped.exception))
        self.assertEqual(3, len(read_lines(out)))
        self.assertEqual((3, 3, 4), (drive.default_max_failed(8), drive.default_max_failed(300), drive.default_max_failed(400)))

    def test_an_outside_bot_is_seated_by_its_name_and_sees_each_game_start_and_end(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        boards = os.path.join(directory, "boards.jsonl")
        with open(boards, "w", encoding="utf-8") as file:
            file.write(json.dumps({"seed": 905, "board": random_board(5)[0]}) + "\n")
        calls, bot = os.path.join(directory, "calls.jsonl"), os.path.join(directory, "counting_bot.py")
        with open(bot, "w", encoding="utf-8") as file:
            file.write(f"CALLS = {calls!r}\n" + COUNTING_BOT)
        seen, out = os.path.join(directory, "seen.jsonl"), os.path.join(directory, "results.jsonl")
        drive.main(["--boards", boards, "--lineup", "open-seat-agent,COUNTER,R,COUNTER", "--bot", f"COUNTER={bot}",
                    "--open-seat-agent", fake_agent(seen), "--maps", "0:1", "--out", out])

        # Its two entries sit where each rotation puts them, named by the bot's name
        results = read_lines(out)
        self.assertEqual(4, len(results))
        for result in results:
            self.assertEqual(["COUNTER", "R", "COUNTER"], [result["seats"][(i + result["rotation"]) % 4] for i in (1, 2, 3)])
        # Every seated bot sees its game start once, decides only after it, and sees it end once, after its last decision,
        # as under Catanatron's own game loop (the driver plays its own)
        by_seat = {}
        for call in read_lines(calls):
            by_seat.setdefault((call["game"], call["color"]), []).append(call["kind"])
        self.assertEqual(8, len(by_seat))
        for kinds in by_seat.values():
            self.assertEqual("before", kinds[0])
            self.assertEqual("after", kinds[-1])
            self.assertEqual(["decide"] * (len(kinds) - 2), kinds[1:-1])
            self.assertGreater(len(kinds), 2)

    def test_a_match_with_a_stand_in_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            boards = os.path.join(directory, "boards.jsonl")
            with open(boards, "w", encoding="utf-8") as file:
                for seed in (5, 6):
                    file.write(json.dumps({"seed": 900 + seed, "board": random_board(seed)[0]}) + "\n")
            seen, out, log = (os.path.join(directory, name) for name in ("seen.jsonl", "results.jsonl", "games.jsonl.gz"))
            agent = f'"{sys.executable}" "{os.path.join(HERE, "fake_open_seat_agent.py").replace(os.sep, "/")}" "{seen.replace(os.sep, "/")}"'
            drive.main(["--boards", boards, "--lineup", "open-seat-agent,R,W,R", "--open-seat-agent", agent, "--maps", "0:2", "--out", out, "--log", log])

            with open(out, encoding="utf-8") as file:
                results = [json.loads(line) for line in file]
            self.assertEqual(8, len(results))
            for result in results:
                # The agent's entry sits in the rotation's seat, every seat moved, the winner (if any) has the points
                self.assertEqual("first-legal", result["seats"][result["rotation"]])
                self.assertTrue(all(d > 0 for d in result["decisions"]))
                if result["winner"] is not None:
                    self.assertGreaterEqual(result["vp"][result["winner"]], 10)

            # The agent got its seat's before, every step and after, and decided only for its seat
            with open(seen, encoding="utf-8") as file:
                messages = [json.loads(line) for line in file]
            with gzip.open(log, "rt", encoding="utf-8") as file:
                logged = [json.loads(line) for line in file]
            # All rotations of a map play with the map's seed (the same dice and cards)
            for map_seed in (905, 906):
                seeds = {m["seed"] for m in logged if m["type"] == "before" and m["game_id"].startswith(f"map{map_seed}-")}
                self.assertEqual({map_seed}, seeds)
            for game_id in {r["game_id"] for r in results if r["map_seed"] == 905}:
                mine = [m for m in messages if m.get("game_id") == game_id]
                seat = mine[0]["seat"]
                self.assertEqual("before", mine[0]["type"])
                self.assertEqual("after", mine[-1]["type"])
                self.assertTrue(all(m["seat"] == seat for m in mine if m["type"] == "decide"))
                # Every step, with the state before its move (the agent needs the full state): the logged decide's
                steps = [m for m in logged if m.get("game_id") == game_id and m["type"] == "step"]
                before = {m["n"]: m["state"] for m in logged if m.get("game_id") == game_id and m["type"] == "decide"}
                self.assertEqual([{**s, "state": before[s["n"]]} for s in steps], [m for m in mine if m["type"] == "step"])
                self.assertEqual(sum(1 for m in logged if m.get("game_id") == game_id and m["type"] == "decide" and m["seat"] == seat),
                                 sum(1 for m in mine if m["type"] == "decide"))


if __name__ == "__main__":
    unittest.main()
