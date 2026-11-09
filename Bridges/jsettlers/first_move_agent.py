"""
An Open Seat Agent for tests (see ../PROTOCOL.md, version 2): takes every game and answers every decision with the first
legal move (with --builds the first that builds or plays a card, if any, so the two-step moves come up; with --weighted
a random one, seeded by the game's seed and weighted like the recorders' random games, cities before settlements before
development cards before playing them before anything else, so every kind of move comes up), asks for the
true state (or what --needs says), and writes each message it got to standard error, for a decision with what its state
shows: whether the seat's own hand, cards and points are there, and whether every other seat's are hidden. With
--observe it asks for every move (step), and with --full it writes each message whole. To test the referee it can be
lean (--lean, with --observe: it asks for the state whenever a decision comes without it), answer its third decision
with a move that isn't listed (--illegal), turn every game down (--refuse), or stall SECONDS before its first decision
after the setup (--stall SECONDS), past a short turn limit of the server. With --trades it proposes where a decision names
trades: once per turn an offer of one card of the resource it holds most for one it holds least, to every seat named,
and a counteroffer (one card of its own for the offer's give) when it can't accept an offer; with --take-counters it
takes the first counteroffer to its own offer where it picks a partner.
Usage: python first_move_agent.py [--needs truth|view|world] [--builds] [--weighted] [--observe] [--full] [--lean] [--illegal]
       [--refuse] [--stall S] [--trades] [--take-counters]
"""

import json
import random
import sys
import time

# The weights of --weighted by move type; every other move weighs 1
WEIGHTS = {"build_city": 10000, "build_settlement": 1000, "buy_development": 100, "play_knight": 30, "play_road_building": 30,
           "play_year_of_plenty": 2, "play_monopoly": 6, "bank_trade": 0.2}


def weighted(legal, rng):
    """A legal move drawn by WEIGHTS"""
    return rng.choices(legal, weights=[WEIGHTS.get(move[0], 1) for move in legal])[0]


def main():
    needs = sys.argv[sys.argv.index("--needs") + 1] if "--needs" in sys.argv else "truth"
    builds = "--builds" in sys.argv
    rng = random.Random(0) if "--weighted" in sys.argv else None
    observe = "--observe" in sys.argv
    full = "--full" in sys.argv
    lean, illegal, refuse = "--lean" in sys.argv, "--illegal" in sys.argv, "--refuse" in sys.argv
    stall = float(sys.argv[sys.argv.index("--stall") + 1]) if "--stall" in sys.argv else 0
    trades = "--trades" in sys.argv
    take_counters = "--take-counters" in sys.argv
    decisions, stalled, offered_at = 0, False, -100
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        seen = {"type": message["type"], "n": message.get("n"), "prompt": (message.get("state") or {}).get("prompt")}
        if message["type"] == "decide" and message.get("state") is not None:
            players = message["state"]["players"]
            fields = ("hand", "development", "vp")
            seen["own_shown"] = all(players[message["seat"]][f] is not None for f in fields)
            seen["others_hidden"] = all(p[f] is None for p in players if p["seat"] != message["seat"] for f in fields)
        print(json.dumps(message if full else seen), file=sys.stderr, flush=True)
        if message["type"] == "handshake":
            answer = {"type": "handshake", "protocol": "osp", "version": 2, "name": "first-move", "rulesets": ["standard"],
                      "observe": observe, "needs": needs, "lean": lean}
        elif message["type"] == "before":
            if rng is not None:
                rng.seed(message.get("seed"))
            answer = ({"type": "failure", "game_id": message["game_id"], "n": None, "code": "unsupported",
                       "message": "this test agent turns every game down", "detail": {"preset": "standard"}}
                      if refuse else {"type": "ready", "game_id": message["game_id"]})
        elif message["type"] == "decide" and message.get("state") is None:
            # A lean decision: it keeps no game of its own, so it asks for the state
            answer = {"type": "resync", "game_id": message["game_id"], "n": message["n"]}
        elif message["type"] == "decide":
            decisions += 1
            # Every game has a decision after the setup (a short one may have few)
            if stall and not stalled and message["state"]["prompt"] not in ("setup_settlement", "setup_road"):
                stalled = True
                time.sleep(stall)
            legal = message["legal"]
            preferred = [m for m in legal if m[0].startswith(("build_", "play_"))] if builds else []
            move = ["no_such_move"] if illegal and decisions == 3 else weighted(legal, rng) if rng else (preferred or legal)[0]
            named = message.get("trades") or {}
            hand = message["state"]["players"][message["seat"]]["hand"] or {}
            held = sorted((count, name) for name, count in hand.items() if count > 0)
            if trades and "offer" in named and held and message["n"] - offered_at > 30:
                # About once a turn: one of what it holds most for one of what it holds least (or none of)
                offered_at = message["n"]
                most = held[-1][1]
                least = min((hand.get(r, 0), r) for r in ("brick", "lumber", "ore", "grain", "wool") if r != most)[1]
                move = ["offer", {"give": {most: 1}, "get": {least: 1}, "to": named["offer"]["to"]}]
            elif trades and "counter" in named and ["respond", {"accept": True}] not in legal:
                offer = message["state"]["offer"]
                mine = [name for _, name in held if name not in offer["give"]]
                if mine:
                    # In the offerer's terms: the offerer gives what it offered, gets one card of the counterer's
                    move = ["counter", {"give": offer["give"], "get": {mine[-1]: 1}}]
            confirms = [m for m in legal if m[0] == "confirm"]
            if take_counters and message["state"]["prompt"] == "pick_partner" and confirms:
                move = confirms[0]
            answer = {"type": "decide", "game_id": message["game_id"], "n": message["n"], "action": move}
        else:
            continue
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
