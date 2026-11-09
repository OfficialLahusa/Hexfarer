"""
The Challenger bot as a Catanatron player (challenger_player.py), with a stand-in for the research release's bridge
(its run.py), so the tests need neither Node.js nor the release: one bridge per game, started at the game's start with
the player's generation and a seed of the game and the seat, ended at the game's end; every decision goes through it.
Run: .venv/Scripts/python -m unittest test_challenger_player
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from catanatron import Color, Game
from catanatron.models.player import RandomPlayer
from catanatron.registry import REGISTRY

import challenger_player
from challenger_player import ChallengerPlayer

HERE = os.path.dirname(os.path.abspath(__file__))

# Stands in for run.py's Ours: writes what it is asked to LOG and picks the last legal move
STAND_IN = '''
import json


def write(entry):
    with open(LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\\n")


class Ours:
    started = 0

    def __init__(self, color, generation, seed, error_file):
        Ours.started += 1
        self.id = Ours.started
        write({"kind": "start", "id": self.id, "color": color.value, "generation": generation, "seed": seed, "error_file": error_file})

    def before(self, game):
        write({"kind": "before", "id": self.id, "game": game.id})

    def decide(self, game, actions):
        write({"kind": "decide", "id": self.id, "game": game.id})
        return actions[-1]

    def close(self):
        write({"kind": "close", "id": self.id})
'''


class ChallengerPlayerTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.log = os.path.join(self.directory, "bridge.jsonl")
        with open(os.path.join(self.directory, "run.py"), "w", encoding="utf-8") as file:
            file.write(f"LOG = {self.log!r}\n" + STAND_IN)
        self.enterContext(mock.patch.dict(os.environ, {"CHALLENGER_ROOT": self.directory, "CHALLENGER_ERRORS": self.directory}))
        # Each test loads its own stand-in
        challenger_player._bridges.clear()

    def entries(self):
        with open(self.log, encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    def play(self, players, seed):
        game = Game(players, seed=seed)
        game.play()
        return game

    def test_one_bridge_per_game_from_its_start_to_its_end(self):
        players = [ChallengerPlayer(Color.RED), RandomPlayer(Color.BLUE), ChallengerPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)]
        games = [self.play(players, seed) for seed in (11, 12)]
        entries = self.entries()
        starts = [e for e in entries if e["kind"] == "start"]
        # Two Challenger seats, two games: four bridges, each closed
        self.assertEqual(4, len(starts))
        self.assertEqual(sorted(e["id"] for e in starts), sorted(e["id"] for e in entries if e["kind"] == "close"))
        for start in starts:
            mine = [e for e in entries if e["id"] == start["id"]]
            # Started, told the game, decided in that one game only, then closed, in that order
            self.assertEqual("start", mine[0]["kind"])
            self.assertEqual("before", mine[1]["kind"])
            self.assertEqual("close", mine[-1]["kind"])
            decides = mine[2:-1]
            self.assertTrue(decides)
            self.assertEqual({"decide"}, {e["kind"] for e in decides})
            self.assertEqual({mine[1]["game"]}, {e["game"] for e in decides})
        # Each game's two seats had their own bridges, in that game
        for game in games:
            self.assertEqual(2, len({e["id"] for e in entries if e.get("game") == game.id}))

    def test_the_generation_and_a_seed_of_the_game_and_the_seat(self):
        players = [ChallengerPlayer(Color.RED, ChallengerPlayer.Params(generation="gen1")), RandomPlayer(Color.BLUE),
                   ChallengerPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)]
        for seed in (21, 21, 22):
            self.play(players, seed)
        starts = [e for e in self.entries() if e["kind"] == "start"]
        self.assertEqual(6, len(starts))
        self.assertEqual({("RED", "gen1"), ("ORANGE", "gen0")}, {(e["color"], e["generation"]) for e in starts})
        # Per game, the seats' seeds by color (the game seats its players in its own order)
        same, again, other = ([e["seed"] for e in sorted(starts[i:i + 2], key=lambda e: e["color"])] for i in (0, 2, 4))
        # The same game and seat repeat their seed; another seat or game gets another
        self.assertEqual(same, again)
        self.assertNotEqual(same[0], same[1])
        self.assertNotEqual(same[0], other[0])
        self.assertNotEqual(same[1], other[1])
        # Each bridge's error file lies in CHALLENGER_ERRORS and names its seat
        for start in starts:
            self.assertEqual(self.directory, os.path.dirname(start["error_file"]))
            self.assertIn(start["color"], os.path.basename(start["error_file"]))
        self.assertEqual(len(starts), len({e["error_file"] for e in starts}))

    def test_a_decision_is_the_bridges_and_needs_a_started_game(self):
        player = ChallengerPlayer(Color.RED)
        with self.assertRaises(RuntimeError):
            player.decide(None, ["a", "b"])
        game = Game([player, RandomPlayer(Color.BLUE), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)], seed=31)
        self.assertEqual("b", player.decide(game, ["a", "b"]))
        player.after(game)
        with self.assertRaises(RuntimeError):
            player.decide(game, ["a", "b"])

    def test_the_registry_builds_it_with_a_generation(self):
        key = REGISTRY.register_source(os.path.join(HERE, "challenger_player.py"), "CHALLENGER")
        self.assertEqual("gen2", REGISTRY.build(f"{key}:gen2", Color.RED).params.generation)
        self.assertEqual("gen0", REGISTRY.build(key, Color.RED).params.generation)


if __name__ == "__main__":
    unittest.main()
