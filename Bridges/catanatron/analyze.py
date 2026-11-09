"""
Summarizes match runs of drive.py: per run the Open Seat Agents' share of wins against what equal strength would give
(their seats out of four), with a 95% interval bootstrapped over whole maps (each map counts with its four rotations),
points, game lengths, each side's time per decision, and the agents' stats (states taken over from the referee, moves
replaced, games where the agent's copy was lost). Every side's share comes with it from the same resamples, so a run
without our agent (e.g. two outside bots against each other) is summarized per side.

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
    stats, broken = read_stats(sorted(glob.glob(os.path.join(run_dir, "agent-*.log"))))
    return results, stats, broken


def read_stats(paths):
    """The agents' stats lines from their logs, and how many of them couldn't be read: where two agent processes wrote
    one log at once (runs before drive.py gave each agent its own), a line can hold pieces of two; it is skipped"""
    stats, broken = [], 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as file:
            for line in file:
                if line.startswith('{"type":"stats"'):
                    try:
                        stats.append(json.loads(line))
                    except json.JSONDecodeError:
                        broken += 1
    return stats, broken


def unreadable(broken):
    """The note for stats lines read_stats skipped"""
    return f"{broken} stats line{'' if broken == 1 else 's'} unreadable (agents wrote one log at once)"


def ours(code):
    """Whether a seat's code is our agent's, under the project's current name or the one older runs have (hexfarer-...)"""
    return code.startswith(("lanzenbrecher", "hexfarer"))


def side(code):
    """A seat's side in the summary: our agent's seats together as "ours", every other bot by its code"""
    return "ours" if ours(code) else code


def failed(game):
    """Whether the game ended at a failure (drive.py's result record names it): it says nothing about strength"""
    return any(seat.get("failures") for seat in (game.get("result") or {}).get("seats", []))


def summarize(run_dir, reps=2000, seed=1):
    results, stats, broken = load(run_dir)
    by_map = defaultdict(list)
    for r in results:
        by_map[r["map_seed"]].append(r)
    # A map counts with its four rotations, none of them ended at a failure (resume.py plays such a map again)
    maps = {m: games for m, games in by_map.items() if sorted(g["rotation"] for g in games) == [0, 1, 2, 3] and not any(failed(g) for g in games)}
    failed_games = sum(1 for r in results if failed(r))
    games = [g for gs in maps.values() for g in gs]
    if not games:
        return {"run": os.path.basename(run_dir), "games": 0, "failed_games": failed_games, "broken_stats_lines": broken}
    # Seats per side, the same in every rotation of the lineup
    seats = defaultdict(int)
    for code in games[0]["seats"]:
        seats[side(code)] += 1

    def wins(sample):
        counts = defaultdict(int)
        for g in sample:
            if g["winner"] is not None:
                counts[side(g["seats"][g["winner"]])] += 1
        return counts

    # Each resample draws whole maps (as many as the run has); every side's share is taken from the same resamples
    rng = random.Random(seed)
    keys = list(maps)
    boot = defaultdict(list)
    for _ in range(reps):
        sample = [g for k in (rng.choice(keys) for _ in keys) for g in maps[k]]
        counts = wins(sample)
        for s in seats:
            boot[s].append(counts[s] / len(sample))
    won = wins(games)

    vp, time_side = defaultdict(list), defaultdict(lambda: [0.0, 0])
    for g in games:
        for c, v, t, n in zip(g["seats"], g["vp"], g["seconds"], g["decisions"]):
            vp[side(c)].append(v)
            time_side[side(c)][0] += t
            time_side[side(c)][1] += n
    ms = {k: round(v[0] / v[1] * 1000, 1) for k, v in time_side.items() if v[1]}
    sides = {}
    for s in seats:
        shares = sorted(boot[s])
        sides[s] = {"seats": seats[s], "win_rate": round(won[s] / len(games), 4),
                    "interval": [round(shares[int(0.025 * reps)], 4), round(shares[int(0.975 * reps) - 1], 4)],
                    "equal_strength": seats[s] / 4, "vp": round(sum(vp[s]) / len(vp[s]), 2), "ms_per_decision": ms.get(s)}
    summary = {
        "run": os.path.basename(run_dir),
        "lineup": games[0]["lineup"],
        "maps": len(maps), "games": len(games),
        "truncated": sum(g["truncated"] for g in games), "failed_games": failed_games,
        "turns": round(sum(g["turns"] for g in games) / len(games), 1),
        "ms_per_decision": ms, "sides": sides, "broken_stats_lines": broken,
    }
    if "ours" not in seats:
        return summary
    # Our agent's fields, as before sides existed
    played = {g["game_id"] for g in games}
    kept_stats = [s for s in stats if s["game_id"] in played]
    theirs = [v for s, values in vp.items() if s != "ours" for v in values]
    summary.update({
        "agent": next(c for c in games[0]["seats"] if ours(c)),
        "win_rate": sides["ours"]["win_rate"], "interval": sides["ours"]["interval"], "equal_strength": sides["ours"]["equal_strength"],
        "vp_ours": sides["ours"]["vp"], "vp_theirs": round(sum(theirs) / len(theirs), 2) if theirs else None,
        "agent_games_with_patches": sum(1 for s in kept_stats if s["patches"] > 0),
        "agent_fallbacks": sum(s["fallbacks"] for s in kept_stats),
        "agent_lost": sum(1 for s in kept_stats if s["lost"]),
        "agent_stats_games": len(kept_stats),
    })
    return summary


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
        if "agent" not in s:
            # No seat of ours (e.g. two outside bots against each other): each side's share
            print(f"{s['run']}: no seat of ours; {s['games']} games on {s['maps']} maps, {s['turns']} turns, {s['truncated']} truncated, "
                  f"{s['failed_games']} failed (their maps left out)")
            for name, d in s["sides"].items():
                print(f"   {name} in {d['seats']} of 4 seats: wins {d['win_rate']:.1%} [{d['interval'][0]:.1%}, {d['interval'][1]:.1%}] "
                      f"(equal strength {d['equal_strength']:.0%}), VP {d['vp']}, {d['ms_per_decision']} ms per decision")
            continue
        # Runs before the bridge night wrote the Open Seat Agent's lineup entry as "osa"
        print(f"{s['run']}: {s['agent']} in {sum(seat in ('open-seat-agent', 'osa') for seat in s['lineup'])} of 4 seats, {s['games']} games on {s['maps']} maps")
        print(f"   our wins {s['win_rate']:.1%} [{s['interval'][0]:.1%}, {s['interval'][1]:.1%}] (equal strength {s['equal_strength']:.0%}), "
              f"VP ours {s['vp_ours']} theirs {s['vp_theirs']}, {s['turns']} turns, {s['truncated']} truncated, "
              f"{s['failed_games']} failed (their maps left out)")
        print(f"   ms per decision {s['ms_per_decision']}; agent: {s['agent_games_with_patches']} of {s['agent_stats_games']} agent-games took over "
              f"the referee's state, {s['agent_fallbacks']} moves replaced, {s['agent_lost']} lost"
              + (f"; {unreadable(s['broken_stats_lines'])}" if s["broken_stats_lines"] else ""))


if __name__ == "__main__":
    main()
