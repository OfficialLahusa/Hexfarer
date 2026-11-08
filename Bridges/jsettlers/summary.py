"""
The strength of an Open Seat Agent at JSettlers, from the server's output (the osp-result line MoveLog writes for every
game). Per setting (a log; batches named NAME-b, NAME-c, ... count with NAME): the games with the agent's seat, those
where the seat failed (a failure, or moves the referee made for it: fallbacks; they say nothing about strength, so they
stay out of the wins, the win rate, the points and the rounds and are counted beside them), the wins among the rest with
a 95% Wilson interval, the seat's points, the rounds, the gaps (changes of the game no step explains), fallbacks,
failures by code, trade decisions that expired, the decision time and, from the referee logs (NAME-records next to a
log, if the run recorded), the trade steps per game of the agent's seat and of the robots. Robot-only games without an
Open Seat seat don't count. A run the server's stop rule ended (osp.max_failed_games) is named as stopped.

Usage: python summary.py [--json OUT] LOG [LOG ...]
"""

import json
import math
import os
import re
import sys

import jsettlers

TRADES = ("offer", "respond", "counter", "confirm", "cancel")


def wilson(wins, n, z=1.96):
    """The share and its Wilson interval (NaN without games)"""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def setting_of(path):
    """A log's setting: its name without .log and a batch's -b, -c, ..."""
    return re.sub(r"-[b-z]$", "", os.path.basename(path)[:-4] if path.endswith(".log") else os.path.basename(path))


def failed(seat):
    """Whether the Open Seat Agent's seat failed in a game: a failure, or moves the referee made for it"""
    return bool(seat["failures"]) or seat["fallbacks"] > 0


def games_of(path):
    """A log's games with an Open Seat seat, each with that seat's line; and whether the server's stop rule ended the run"""
    games, stopped = [], None
    with open(path, encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.startswith(jsettlers.STOPPED):
                stopped = line.strip()
            if '"osp-result"' not in line:
                continue
            record = json.loads(line[line.index("{"):])
            seat = next((s for s in record["result"]["seats"] if s["agent"] is not None), None)
            # A robot-only game has none of our seats
            if seat is not None:
                games.append((record, seat))
    return games, stopped


def trade_steps(records, seats):
    """The trade steps per recorded game (offers, answers by kind, counteroffers, picks, withdrawals) of the agent's seat and
    of the robots, from the referee logs of the games in seats (game id: the agent's seat)"""
    ours, theirs, counted = {}, {}, 0
    for name in sorted(os.listdir(records)):
        seat = seats.get(name[:-6].replace("_", "~")) if name.endswith(".jsonl") else None
        if seat is None:
            continue
        counted += 1
        with open(os.path.join(records, name), encoding="utf-8", errors="replace") as file:
            for row in file:
                if '"step"' not in row:
                    continue
                message = json.loads(row)
                kind = message["action"][0]
                if message["type"] != "step" or kind not in TRADES:
                    continue
                if kind == "respond":
                    kind += ":" + ("accept" if message["action"][1]["accept"] else "decline")
                side = ours if message["seat"] == seat else theirs
                side[kind] = side.get(kind, 0) + 1
    return counted, ours, theirs


def settings(paths):
    """Each setting's numbers over its logs, in the order the settings first come"""
    grouped = {}
    for path in paths:
        grouped.setdefault(setting_of(path), []).append(path)
    found = []
    for name, logs in grouped.items():
        games, stopped, recorded, ours, theirs = [], [], 0, {}, {}
        for path in logs:
            these, stop = games_of(path)
            games += these
            if stop:
                stopped.append(f"{os.path.basename(path)}: {stop}")
            # The referee logs of this batch (game ids repeat between batches)
            records = path[:-4] + "-records" if path.endswith(".log") else path + "-records"
            if os.path.isdir(records):
                counted, mine, others = trade_steps(records, {record["game_id"]: seat["seat"] for record, seat in these})
                recorded += counted
                for total, part in ((ours, mine), (theirs, others)):
                    for kind, count in part.items():
                        total[kind] = total.get(kind, 0) + count
        counted = [(record, seat) for record, seat in games if not failed(seat)]
        wins = sum(record["winner"] == seat["seat"] for record, seat in counted)
        win, lo, hi = wilson(wins, len(counted))
        failures = {}
        for _, seat in games:
            for failure in seat["failures"]:
                failures[failure["code"]] = failures.get(failure["code"], 0) + 1
        decisions = sum(seat["decisions"] for _, seat in games)
        result = {
            "setting": name, "logs": [os.path.basename(p) for p in logs], "games": len(games), "failed_games": len(games) - len(counted),
            "counted_games": len(counted), "wins": wins, "win": win, "lo": lo, "hi": hi,
            "vp": sum(record["vp"][seat["seat"]] for record, seat in counted) / len(counted) if counted else float("nan"),
            "rounds": sum(record["rounds"] for record, _ in counted) / len(counted) if counted else float("nan"),
            "gaps": sum(record.get("gaps", 0) for record, _ in games), "fallbacks": sum(seat["fallbacks"] for _, seat in games),
            "failures": failures, "expired_trades": sum(seat.get("expired_trades", 0) for _, seat in games),
            "ms_per_decision": sum(seat["time_ms"] for _, seat in games) / decisions if decisions else float("nan"),
            "stopped": stopped,
        }
        if recorded:
            result["recorded_games"] = recorded
            result["ours_per_game"] = {k: v / recorded for k, v in sorted(ours.items())}
            result["theirs_per_game"] = {k: v / recorded for k, v in sorted(theirs.items())}
        found.append(result)
    return found


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    target = None
    if args[:1] == ["--json"]:
        target, args = args[1], args[2:]
    if not args:
        sys.exit(__doc__)
    results = settings(args)
    for s in results:
        print(f"{s['setting']:<36} {s['counted_games']:>4} games (+{s['failed_games']} failed)  win {s['win']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]"
              f"  vp {s['vp']:.2f}  rounds {s['rounds']:.1f}  gaps {s['gaps']}  fallbacks {s['fallbacks']}  failures {s['failures'] or 0}"
              f"  expired {s['expired_trades']}  {s['ms_per_decision']:.0f} ms/decision")
        for stop in s["stopped"]:
            print("    " + stop)
        if "ours_per_game" in s:
            print("    ours/game", json.dumps({k: round(v, 2) for k, v in s["ours_per_game"].items()}))
            print("    theirs/game", json.dumps({k: round(v, 2) for k, v in s["theirs_per_game"].items()}))
    if target:
        with open(target, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=1)


if __name__ == "__main__":
    main()
