"""
Catanatron's own speed, without search, on one thread: random games (moves applied, the bots choosing from the legal
moves), legal move lists and game copies per second, comparable with Lanzenbrecher's `OpenSeat bench`. One JSON line.

Usage: python bench.py [--games N] [--seed S]
"""

import argparse
import json
import random
import time

from catanatron import Color, Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.player import RandomPlayer

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    random.seed(args.seed)
    moves = lists = copies = ended = 0
    play_seconds = list_seconds = copy_seconds = 0.0
    for g in range(args.games):
        game = Game([RandomPlayer(c) for c in COLORS], seed=args.seed + g)
        start = time.perf_counter()
        winner = game.play()
        play_seconds += time.perf_counter() - start
        moves += len(game.state.action_records)
        ended += winner is not None
        # Legal move lists and copies of the final position, as a search needs them
        start = time.perf_counter()
        for _ in range(200):
            generate_playable_actions(game.state)
            lists += 1
        list_seconds += time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(200):
            game.copy()
            copies += 1
        copy_seconds += time.perf_counter() - start
    print(json.dumps({"engine": "catanatron", "games": args.games, "ended": ended, "moves": moves,
                      "games_per_s": round(args.games / play_seconds, 2), "moves_per_s": round(moves / play_seconds),
                      "legal_lists_per_s": round(lists / list_seconds), "copies_per_s": round(copies / copy_seconds)}))


if __name__ == "__main__":
    main()
