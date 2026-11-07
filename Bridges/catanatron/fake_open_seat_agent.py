"""
A stand-in Open Seat Agent for tests: takes every game, answers every decision with the first legal move and writes every
message it gets to the file named by its first argument (one JSON object per line). The second argument, if given, is
what it needs (default truth); the third a way to misbehave: lean (decisions with only the hash; it asks for the state
at every fifth move of the game), illegal (answers its third decision of a game with a move that isn't listed), slow
(takes 2 s for its second), refuse (turns every game down), blind (doesn't observe).
"""

import json
import sys
import time

NEEDS = sys.argv[2] if len(sys.argv) > 2 else "truth"
MODE = sys.argv[3] if len(sys.argv) > 3 else ""


def answer(message):
    return json.dumps(message)


decisions = {}
with open(sys.argv[1], "a", encoding="utf-8") as seen:
    for line in sys.stdin:
        message = json.loads(line)
        seen.write(line if line.endswith("\n") else line + "\n")
        seen.flush()
        if message["type"] == "handshake":
            print(answer({"type": "handshake", "protocol": "osp", "version": 2, "name": "first-legal", "rulesets": ["catanatron"],
                          "observe": MODE != "blind", "needs": NEEDS, "lean": MODE == "lean"}), flush=True)
        elif message["type"] == "before":
            if MODE == "refuse":
                print(answer({"type": "failure", "game_id": message["game_id"], "n": None, "code": "unsupported",
                              "message": "this stand-in plays no games", "detail": {"preset": "catanatron"}}), flush=True)
            else:
                print(answer({"type": "ready", "game_id": message["game_id"]}), flush=True)
        elif message["type"] == "decide":
            n = message["n"]
            if MODE == "lean" and "state" not in message and n % 5 == 0:
                print(answer({"type": "resync", "game_id": message["game_id"], "n": n}), flush=True)
                continue
            mine = decisions[message["game_id"]] = decisions.get(message["game_id"], 0) + 1
            if MODE == "slow" and mine == 2:
                time.sleep(2)
            action = ["no_such_move"] if MODE == "illegal" and mine == 3 else message["legal"][0]
            print(answer({"type": "decide", "game_id": message["game_id"], "n": n, "action": action}), flush=True)
