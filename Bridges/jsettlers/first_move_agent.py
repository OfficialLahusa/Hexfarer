"""
An Open Seat Agent for tests (see ../PROTOCOL.md): answers every decision with the first legal move, asks for the true
state, and writes each message it got to standard error.
Usage: python first_move_agent.py [--needs truth|view]
"""

import json
import sys


def main():
    needs = sys.argv[sys.argv.index("--needs") + 1] if "--needs" in sys.argv else "truth"
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        print(json.dumps({"type": message["type"], "n": message.get("n"), "prompt": (message.get("state") or {}).get("prompt")}), file=sys.stderr, flush=True)
        if message["type"] == "hello":
            answer = {"type": "hello", "protocol": "osp", "version": 1, "name": "first-move", "rulesets": ["standard"], "observe": False, "needs": needs}
        elif message["type"] == "decide":
            answer = {"type": "decide", "game_id": message["game_id"], "n": message["n"], "action": message["legal"][0]}
        else:
            continue
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
