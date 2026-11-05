"""
Plays Catanatron games and records them as the referee's Open Seat Protocol messages (see ../PROTOCOL.md), one JSON
object per line, gzipped: per game a `before` (board, ruleset, bots, full state), a `decide` before every move (full
state and legal moves), a `step` after it (the move and its full result) and an `after`. Hexfarer's replay harness
plays these logs through its own rules and compares every state and every list of legal moves.

Usage: python record.py --bots R,R,R,R --games 100 --seed 1 --out games.jsonl.gz
"""

import argparse
import gzip
import json
import sys
from collections import Counter

import catanatron.players  # noqa: F401 (registers the bots)
from catanatron import Color, Game
from catanatron.game import TURNS_LIMIT
from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.registry import REGISTRY

import osp_board
from osp_game import Encoder, card_set

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def record_game(game, write, game_id, bots):
    """Plays the game to its end and writes its messages; returns the winner's seat or None"""
    encoder = Encoder(game)
    state = game.state
    write({"type": "before", "game_id": game_id, "seed": game.seed, "players": len(state.colors),
           "ruleset": {"preset": "catanatron"}, "bots": [bots[COLORS.index(c)] for c in state.colors],
           "board": osp_board.board(state.board.map, state.board.robber_coordinate), "state": encoder.state()})
    n = 0
    while game.winning_color() is None and state.num_turns < TURNS_LIMIT:
        seat = state.current_player_index
        if state.current_prompt == ActionPrompt.DISCARD:
            # One protocol move for the seat's whole discard, played card by card by its bot
            write({"type": "decide", "game_id": game_id, "n": n, "seat": seat, "state": encoder.state(),
                   "legal": encoder.discard_options(seat)})
            discarded = []
            while state.current_prompt == ActionPrompt.DISCARD and state.current_player_index == seat:
                discarded.append(game.play_tick().result)
            write({"type": "step", "game_id": game_id, "n": n, "seat": seat, "action": ["discard", card_set(discarded)], "result": None})
        else:
            write({"type": "decide", "game_id": game_id, "n": n, "seat": seat, "state": encoder.state(),
                   "legal": encoder.legal(game.playable_actions)})
            taken = None
            player = state.current_player()
            action = player.decide(game, game.playable_actions)
            if action.action_type == ActionType.PLAY_MONOPOLY:
                taken = encoder.monopoly_taken(action)
            record = game.execute(action)
            write({"type": "step", "game_id": game_id, "n": n, "seat": seat, "action": encoder.move(record.action),
                   "result": encoder.result(record, taken)})
        n += 1
    winner = game.winning_color()
    write({"type": "after", "game_id": game_id, "winner": None if winner is None else encoder.seats[winner],
           "vp": [p["vp"] for p in encoder.state()["players"]], "truncated": winner is None, "turns": state.num_turns,
           "state": encoder.state()})
    return None if winner is None else encoder.seats[winner]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record Catanatron games as Open Seat Protocol messages")
    parser.add_argument("--bots", default="R,R,R,R", help="four of Catanatron's bot codes, e.g. AB:2,R,W,R")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1, help="seed of the first game; game i uses seed + i")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    bots = args.bots.split(",")
    if len(bots) != 4:
        raise SystemExit("--bots needs four bot codes")

    results = Counter()
    with gzip.open(args.out, "wt", encoding="utf-8") as file:
        def write(message):
            file.write(json.dumps(message, separators=(",", ":")) + "\n")

        for i in range(args.games):
            players = [REGISTRY.build(code, color) for code, color in zip(bots, COLORS)]
            game = Game(players, seed=args.seed + i)
            winner = record_game(game, write, f"catanatron-{args.seed + i}", bots)
            results["truncated" if winner is None else "won"] += 1
            print(f"game {i + 1}/{args.games}: seed {args.seed + i}, winner seat {winner}", file=sys.stderr, flush=True)
    print(f"{args.games} games to {args.out}: {dict(results)}", file=sys.stderr)


if __name__ == "__main__":
    main()
