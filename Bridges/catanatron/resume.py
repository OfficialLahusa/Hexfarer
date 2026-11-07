"""
Prepares a stopped match run (drive.py) to go on: keeps the results of the maps whose four rotations are all played,
moves the results of maps left unfinished to partial.jsonl (a map counts only as a whole), and prints the unfinished
maps of the run's range split over PARALLEL drivers, one --maps value per line.

Usage: python resume.py RUN_DIR FIRST END PARALLEL [--boards boards.jsonl] [--redo-lost]
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def ranges(indices):
    """Sorted line numbers as FIRST:END ranges, comma-separated"""
    parts, start, previous = [], None, None
    for i in sorted(indices):
        if start is None:
            start = previous = i
        elif i == previous + 1:
            previous = i
        else:
            parts.append(f"{start}:{previous + 1}")
            start = previous = i
    if start is not None:
        parts.append(f"{start}:{previous + 1}")
    return ",".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("first", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("parallel", type=int)
    parser.add_argument("--boards", default=None)
    parser.add_argument("--redo-lost", action="store_true", help="also play again the maps where an agent's copy lost a game")
    args = parser.parse_args(argv)
    boards = args.boards or os.path.join(os.path.dirname(os.path.abspath(args.run_dir)), "boards-70000.jsonl")
    with open(boards, encoding="utf-8") as file:
        seeds = [json.loads(line)["seed"] for line in file]

    files = sorted(glob.glob(os.path.join(args.run_dir, "results-*.jsonl")))
    lines = {path: [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] for path in files}
    rotations, failed = defaultdict(set), set()
    for results in lines.values():
        for r in results:
            rotations[r["map_seed"]].add(r["rotation"])
            # A game that ended at a failure (its result record names it) leaves its map to be played again
            if any(seat.get("failures") for seat in (r.get("result") or {}).get("seats", [])):
                failed.add(r["map_seed"])
    complete = {seed for seed, done in rotations.items() if done == {0, 1, 2, 3} and seed not in failed}
    if args.redo_lost:
        # A map where an agent's copy lost the game (it then played the first legal move) is played again
        for path in glob.glob(os.path.join(args.run_dir, "agent-*.log")):
            with open(path, encoding="utf-8", errors="replace") as file:
                for line in file:
                    if line.startswith('{"type":"stats"') and json.loads(line)["lost"]:
                        complete.discard(int(json.loads(line)["game_id"].split("-")[0][3:]))

    # Unfinished maps leave the results (into partial.jsonl), so every map counts with all four rotations or not at all
    with open(os.path.join(args.run_dir, "partial.jsonl"), "a", encoding="utf-8") as partial:
        for path, results in lines.items():
            kept = [r for r in results if r["map_seed"] in complete]
            for r in results:
                if r["map_seed"] not in complete:
                    partial.write(json.dumps(r) + "\n")
            with open(path, "w", encoding="utf-8") as file:
                file.writelines(json.dumps(r) + "\n" for r in kept)

    remaining = [i for i in range(args.first, args.end) if seeds[i] not in complete]
    per = -(-len(remaining) // args.parallel) if remaining else 0
    for k in range(args.parallel):
        chunk = remaining[k * per:(k + 1) * per]
        if chunk:
            print(ranges(chunk))


if __name__ == "__main__":
    main()
