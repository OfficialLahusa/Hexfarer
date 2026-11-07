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
import tempfile
import unittest

from catanatron import Color
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, number_probability
from catanatron.models.player import RandomPlayer

import contextlib
import io

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


class DriveTests(unittest.TestCase):
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
