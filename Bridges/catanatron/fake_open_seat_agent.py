"""
A stand-in Open Seat Agent for tests: answers every decision with the first legal move and writes every message it
gets to the file named by its first argument (one JSON object per line).
"""

import json
import sys

with open(sys.argv[1], "a", encoding="utf-8") as seen:
    for line in sys.stdin:
        message = json.loads(line)
        seen.write(line if line.endswith("\n") else line + "\n")
        seen.flush()
        if message["type"] == "hello":
            print(json.dumps({"type": "hello", "protocol": "osp", "version": 1, "name": "first-legal", "needs": "truth"}), flush=True)
        elif message["type"] == "decide":
            print(json.dumps({"type": "decide", "game_id": message["game_id"], "n": message["n"], "action": message["legal"][0]}), flush=True)
