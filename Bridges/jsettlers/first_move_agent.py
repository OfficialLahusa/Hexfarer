"""
An Open Seat Agent for tests (see ../PROTOCOL.md): answers every decision with the first legal move (with --builds the
first that builds or plays a card, if any, so the two-step moves come up), asks for the true
state (or what --needs says), and writes each message it got to standard error, for a decision with what its state
shows: whether the seat's own hand, cards and points are there, and whether every other seat's are hidden.
Usage: python first_move_agent.py [--needs truth|view] [--builds]
"""

import json
import sys


def main():
    needs = sys.argv[sys.argv.index("--needs") + 1] if "--needs" in sys.argv else "truth"
    builds = "--builds" in sys.argv
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        seen = {"type": message["type"], "n": message.get("n"), "prompt": (message.get("state") or {}).get("prompt")}
        if message["type"] == "decide":
            players = message["state"]["players"]
            fields = ("hand", "development", "vp")
            seen["own_shown"] = all(players[message["seat"]][f] is not None for f in fields)
            seen["others_hidden"] = all(p[f] is None for p in players if p["seat"] != message["seat"] for f in fields)
        print(json.dumps(seen), file=sys.stderr, flush=True)
        if message["type"] == "hello":
            answer = {"type": "hello", "protocol": "osp", "version": 1, "name": "first-move", "rulesets": ["standard"], "observe": False, "needs": needs}
        elif message["type"] == "decide":
            legal = message["legal"]
            preferred = [m for m in legal if m[0].startswith(("build_", "play_"))] if builds else []
            answer = {"type": "decide", "game_id": message["game_id"], "n": message["n"], "action": (preferred or legal)[0]}
        else:
            continue
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
