"""
An Open Seat Agent for tests (see ../PROTOCOL.md, version 2): takes every game and answers every decision with the first
legal move (with --builds the first that builds or plays a card, if any, so the two-step moves come up), asks for the
true state (or what --needs says), and writes each message it got to standard error, for a decision with what its state
shows: whether the seat's own hand, cards and points are there, and whether every other seat's are hidden. With
--observe it asks for every move (step), and with --full it writes each message whole. To test the referee it can be
lean (--lean, with --observe: it asks for the state whenever a decision comes without it), answer its third decision
with a move that isn't listed (--illegal), or turn every game down (--refuse).
Usage: python first_move_agent.py [--needs truth|view|world] [--builds] [--observe] [--full] [--lean] [--illegal] [--refuse]
"""

import json
import sys


def main():
    needs = sys.argv[sys.argv.index("--needs") + 1] if "--needs" in sys.argv else "truth"
    builds = "--builds" in sys.argv
    observe = "--observe" in sys.argv
    full = "--full" in sys.argv
    lean, illegal, refuse = "--lean" in sys.argv, "--illegal" in sys.argv, "--refuse" in sys.argv
    decisions = 0
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
            answer = ({"type": "failure", "game_id": message["game_id"], "n": None, "code": "unsupported",
                       "message": "this test agent turns every game down", "detail": {"preset": "standard"}}
                      if refuse else {"type": "ready", "game_id": message["game_id"]})
        elif message["type"] == "decide" and message.get("state") is None:
            # A lean decision: it keeps no game of its own, so it asks for the state
            answer = {"type": "resync", "game_id": message["game_id"], "n": message["n"]}
        elif message["type"] == "decide":
            decisions += 1
            legal = message["legal"]
            preferred = [m for m in legal if m[0].startswith(("build_", "play_"))] if builds else []
            move = ["no_such_move"] if illegal and decisions == 3 else (preferred or legal)[0]
            answer = {"type": "decide", "game_id": message["game_id"], "n": message["n"], "action": move}
        else:
            continue
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
