"""
How much Catanatron's AlphaBeta searches per decision, to compare it with Hexfarer's MCTS at equal work. Plays games of
four Catanatron bots (e.g. AB:2) on protocol boards and counts, for every decision an AlphaBeta bot searches (more than
one legal move), the positions its value function scores (leaves), the positions it generates (each outcome of each
move it expands is a game copy with the move applied: Catanatron expands all moves of a node before its alpha-beta
cutoffs) and the nodes it expands. Prints one JSON line: the distribution per decision, overall and per phase.

Usage: python ab_work.py --boards BOARDS [--lineup AB:2,AB:2,AB:2,AB:2] [--maps 0:8] [--jobs 4] [--per-decision FILE]
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from multiprocessing import Pool

import catanatron.players  # noqa: F401 (registers the bots)
from catanatron.players import minimax, tree_search_utils
from catanatron.registry import REGISTRY

import osp_board
from drive import COLORS, seated_game

# Work of the decision in progress (one game per process at a time)
_work = {"leaves": 0, "positions": 0, "expanded": 0}


def _instrument():
    """Wraps the functions AlphaBeta reaches through module globals, so Catanatron's code stays as it is"""
    get_value_fn = minimax.get_value_fn
    execute_spectrum = tree_search_utils.execute_spectrum
    expand_spectrum = minimax.expand_spectrum
    decide = minimax.AlphaBetaPlayer.decide

    def counted_get_value_fn(*args, **kwargs):
        value_fn = get_value_fn(*args, **kwargs)

        def counted(game, color):
            _work["leaves"] += 1
            return value_fn(game, color)
        return counted

    def counted_execute_spectrum(game, action):
        outcomes = execute_spectrum(game, action)
        _work["positions"] += len(outcomes)
        return outcomes

    def counted_expand_spectrum(game, actions):
        _work["expanded"] += 1
        return expand_spectrum(game, actions)

    def counted_decide(self, game, playable_actions):
        for key in _work:
            _work[key] = 0
        start = time.perf_counter()
        action = decide(self, game, playable_actions)
        # A decision with a single legal move returns before any search
        if _work["expanded"]:
            _decisions.append({"setup": game.state.is_initial_build_phase, "prompt": game.state.current_prompt.name,
                               "legal": len(playable_actions), "seconds": round(time.perf_counter() - start, 4), **_work})
        return action

    minimax.get_value_fn = counted_get_value_fn
    tree_search_utils.execute_spectrum = counted_execute_spectrum
    minimax.expand_spectrum = counted_expand_spectrum
    minimax.AlphaBetaPlayer.decide = counted_decide


_decisions = []


def _play(job):
    """The searched decisions of one game on one board (in a worker process)"""
    entry, lineup = job
    if minimax.AlphaBetaPlayer.decide.__name__ != "counted_decide":
        _instrument()
    _decisions.clear()
    players = [REGISTRY.build(code, COLORS[seat]) for seat, code in enumerate(lineup)]
    random.seed(entry["seed"] * 4)
    game = seated_game(players, entry["seed"], osp_board.catan_map(entry["board"]))
    game.play()
    return [{"map_seed": entry["seed"], **d} for d in _decisions]


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def at(q): return values[min(len(values) - 1, int(q * len(values)))]
    return {"n": len(values), "mean": round(statistics.fmean(values), 1), "median": at(0.5), "p10": at(0.1), "p90": at(0.9),
            "max": values[-1]}


def summarize(decisions):
    groups = {"all": decisions, "setup": [d for d in decisions if d["setup"]], "main": [d for d in decisions if not d["setup"]]}
    return {name: {key: distribution([d[key] for d in group]) for key in ("leaves", "positions", "expanded", "legal")}
            for name, group in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", required=True)
    parser.add_argument("--lineup", default="AB:2,AB:2,AB:2,AB:2")
    parser.add_argument("--maps", default="0:8", help="boards FIRST:END, one game per board")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--per-decision", help="every searched decision as a JSON line")
    args = parser.parse_args(argv)
    lineup = args.lineup.split(",")
    with open(args.boards, encoding="utf-8") as file:
        boards = [json.loads(line) for line in file][slice(*(int(x) for x in args.maps.split(":")))]
    with Pool(args.jobs) as pool:
        decisions = [d for game in pool.map(_play, [(entry, lineup) for entry in boards], chunksize=1) for d in game]
    if args.per_decision:
        with open(args.per_decision, "w", encoding="utf-8") as out:
            out.writelines(json.dumps(d) + "\n" for d in decisions)
    print(json.dumps({"lineup": lineup, "games": len(boards), "searched_decisions": len(decisions), **summarize(decisions)}))


if __name__ == "__main__":
    # Catanatron keeps legal moves in sets, whose order depends on the hash seed: with it fixed, a game repeats
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit(subprocess.call([sys.executable, *sys.argv], env={**os.environ, "PYTHONHASHSEED": "0"}))
    main()
