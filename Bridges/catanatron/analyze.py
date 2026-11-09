"""
Summarizes match runs of drive.py: per run the Open Seat Agents' share of wins against what equal strength would give
(their seats out of four), with a 95% interval bootstrapped over whole maps (each map counts with its four rotations),
points, game lengths, each side's time per decision, and the agents' stats (states taken over from the referee, moves
replaced, games where the agent's copy was lost).

Usage: python analyze.py RUN_DIR [RUN_DIR ...] [--json]
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict


def load(run_dir):
    results = []
    for path in sorted(glob.glob(os.path.join(run_dir, "results-*.jsonl"))):
        with open(path, encoding="utf-8") as file:
            results.extend(json.loads(line) for line in file if line.strip())
    stats = []
    for path in sorted(glob.glob(os.path.join(run_dir, "agent-*.log"))):
        with open(path, encoding="utf-8", errors="replace") as file:
            for line in file:
                if line.startswith('{"type":"stats"'):
                    stats.append(json.loads(line))
    return results, stats


def ours(code):
    """Whether a seat's code is our agent's, under the project's current name or the one older runs have (hexfarer-...)"""
    return code.startswith(("lanzenbrecher", "hexfarer"))


def failed(game):
    """Whether the game ended at a failure (drive.py's result record names it): it says nothing about strength"""
    return any(seat.get("failures") for seat in (game.get("result") or {}).get("seats", []))


def summarize(run_dir, reps=2000, seed=1):
    results, stats = load(run_dir)
    by_map = defaultdict(list)
    for r in results:
        by_map[r["map_seed"]].append(r)
    # A map counts with its four rotations, none of them ended at a failure (resume.py plays such a map again)
    maps = {m: games for m, games in by_map.items() if sorted(g["rotation"] for g in games) == [0, 1, 2, 3] and not any(failed(g) for g in games)}
    failed_games = sum(1 for r in results if failed(r))
    games = [g for gs in maps.values() for g in gs]
    if not games:
        return {"run": os.path.basename(run_dir), "games": 0, "failed_games": failed_games}
    seats_ours = sum(ours(c) for c in games[0]["seats"])
    expected = seats_ours / 4

    def share(sample):
        won = sum(1 for g in sample if g["winner"] is not None and ours(g["seats"][g["winner"]]))
        return won / len(sample)

    rate = share(games)
    rng = random.Random(seed)
    keys = list(maps)
    boot = sorted(share([g for k in (rng.choice(keys) for _ in keys) for g in maps[k]]) for _ in range(reps))
    low, high = boot[int(0.025 * reps)], boot[int(0.975 * reps) - 1]

    vp_ours = [v for g in games for c, v in zip(g["seats"], g["vp"]) if ours(c)]
    vp_theirs = [v for g in games for c, v in zip(g["seats"], g["vp"]) if not ours(c)]
    time_side = defaultdict(lambda: [0.0, 0])
    for g in games:
        for c, s, n in zip(g["seats"], g["seconds"], g["decisions"]):
            side = "ours" if ours(c) else c
            time_side[side][0] += s
            time_side[side][1] += n
    played = {g["game_id"] for g in games}
    kept_stats = [s for s in stats if s["game_id"] in played]
    return {
        "run": os.path.basename(run_dir),
        "lineup": games[0]["lineup"],
        "agent": next(c for c in games[0]["seats"] if ours(c)),
        "maps": len(maps), "games": len(games),
        "win_rate": round(rate, 4), "interval": [round(low, 4), round(high, 4)], "equal_strength": expected,
        "truncated": sum(g["truncated"] for g in games), "failed_games": failed_games,
        "vp_ours": round(sum(vp_ours) / len(vp_ours), 2), "vp_theirs": round(sum(vp_theirs) / len(vp_theirs), 2),
        "turns": round(sum(g["turns"] for g in games) / len(games), 1),
        "ms_per_decision": {k: round(v[0] / v[1] * 1000, 1) for k, v in time_side.items() if v[1]},
        "agent_games_with_patches": sum(1 for s in kept_stats if s["patches"] > 0),
        "agent_fallbacks": sum(s["fallbacks"] for s in kept_stats),
        "agent_lost": sum(1 for s in kept_stats if s["lost"]),
        "agent_stats_games": len(kept_stats),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for run in args.runs:
        s = summarize(run)
        if args.json or not s["games"]:
            print(json.dumps(s))
            continue
        # Runs before the bridge night wrote the Open Seat Agent's lineup entry as "osa"
        print(f"{s['run']}: {s['agent']} in {sum(seat in ('open-seat-agent', 'osa') for seat in s['lineup'])} of 4 seats, {s['games']} games on {s['maps']} maps")
        print(f"   our wins {s['win_rate']:.1%} [{s['interval'][0]:.1%}, {s['interval'][1]:.1%}] (equal strength {s['equal_strength']:.0%}), "
              f"VP ours {s['vp_ours']} theirs {s['vp_theirs']}, {s['turns']} turns, {s['truncated']} truncated, "
              f"{s['failed_games']} failed (their maps left out)")
        print(f"   ms per decision {s['ms_per_decision']}; agent: {s['agent_games_with_patches']} of {s['agent_stats_games']} agent-games took over "
              f"the referee's state, {s['agent_fallbacks']} moves replaced, {s['agent_lost']} lost")


if __name__ == "__main__":
    main()
