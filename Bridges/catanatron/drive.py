"""
Plays Catanatron games under Catanatron's referee with some seats played by Open Seat Agents (see ../PROTOCOL.md):
external processes that get every message of their seat (before, decide, step, after) over standard input and output
and answer the decisions. The other seats are Catanatron's bots. Every map is played once per rotation of the lineup
(4 rotations: each lineup entry sits in each seat once), all rotations of a map with the same Catanatron seed (the same
dice and development cards as long as the games run alike). One result line per game, with each side's time per decision.

Usage: python drive.py --boards boards.jsonl --lineup open-seat-agent,AB:2,AB:2,AB:2 --open-seat-agent "dotnet OpenSeat.dll agent --iterations 5000"
                       --maps 0:25 --out results.jsonl [--log games.jsonl.gz]
"""

import argparse
import gzip
import json
import os
import random
import shlex
import subprocess
import sys
import time

import catanatron.players  # noqa: F401 (registers the bots)
from catanatron import Color, Game, Player
from catanatron.game import TURNS_LIMIT
from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.registry import REGISTRY

import osp_board
from osp_game import Encoder, card_set

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


class OpenSeatAgentSeat(Player):
    """A seat whose decisions an Open Seat Agent makes (the driver asks it; Catanatron never calls decide)"""

    def decide(self, game, playable_actions):
        raise RuntimeError("The driver decides for Open Seat Agent seats")


class OpenSeatAgent:
    """An Open Seat Agent process: one JSON object per line each way"""

    def __init__(self, command, stderr):
        # POSIX splitting on every system (quotes group a path with spaces; write paths with forward slashes)
        self.process = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, text=True, encoding="utf-8", bufsize=1)
        hello = self.ask({"type": "hello", "protocol": "osp", "version": 1})
        if hello.get("protocol") != "osp" or hello.get("version") != 1:
            raise RuntimeError(f"The agent speaks {hello}")
        self.name = hello.get("name", "open-seat-agent")

    def send(self, message):
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def ask(self, message):
        self.send(message)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The agent ended")
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=60)


def seated_game(players, seed, catan_map):
    """A game with the players in the given turn order: Catanatron seats them by its seeded random sample, so they are
    handed over in the order that sample turns into the wanted one"""
    order = random.Random(seed).sample(range(len(players)), len(players))
    handed = [None] * len(players)
    for position, index in enumerate(order):
        handed[index] = players[position]
    game = Game(handed, seed=seed, catan_map=catan_map)
    assert [p.color for p in game.state.players] == [p.color for p in players], "the seating didn't come out as wanted"
    return game


def play(game, open_seat_agents, write, game_id, codes):
    """Plays the game; open_seat_agents maps seats to Open Seat Agents. Returns the result line."""
    encoder = Encoder(game)
    state = game.state
    times = {seat: [0.0, 0] for seat in range(4)}

    def tell(message, before=None):
        # The log has the state before each move in its decide; the agents (which need the full state) get it with the step
        write(message)
        for agent in open_seat_agents.values():
            agent.send(message if before is None else {**message, "state": before})

    board = osp_board.board(state.board.map, state.board.robber_coordinate)
    full = encoder.state()
    write({"type": "before", "game_id": game_id, "seed": game.seed, "players": 4, "ruleset": {"preset": "catanatron"},
           "bots": codes, "board": board, "state": full})
    for seat, agent in open_seat_agents.items():
        agent.send({"type": "before", "game_id": game_id, "seat": seat, "players": 4, "ruleset": {"preset": "catanatron"},
                  "board": board, "state": full})

    n = 0
    while game.winning_color() is None and state.num_turns < TURNS_LIMIT:
        seat = state.current_player_index
        discard = state.current_prompt == ActionPrompt.DISCARD
        legal = encoder.discard_options(seat) if discard else encoder.legal(game.playable_actions)
        decide = {"type": "decide", "game_id": game_id, "n": n, "seat": seat, "state": encoder.state(), "legal": legal}
        write(decide)
        start = time.perf_counter()
        if seat in open_seat_agents:
            move = open_seat_agents[seat].ask(decide)["action"]
            if move not in legal:
                raise RuntimeError(f"{game_id} move {n}: the agent's {move} is not legal")
        else:
            move = None
        if discard:
            if move is not None:
                for action in encoder.discard_actions(move, state.current_color()):
                    game.execute(action)
                discarded = move[1]
            else:
                cards = []
                while state.current_prompt == ActionPrompt.DISCARD and state.current_player_index == seat:
                    cards.append(game.play_tick().result)
                discarded = card_set(cards)
            times[seat][0] += time.perf_counter() - start
            times[seat][1] += 1
            tell({"type": "step", "game_id": game_id, "n": n, "seat": seat, "action": ["discard", discarded], "result": None}, decide["state"])
        else:
            action = encoder.decode(move, game.playable_actions) if move is not None else state.current_player().decide(game, game.playable_actions)
            times[seat][0] += time.perf_counter() - start
            times[seat][1] += 1
            taken = encoder.monopoly_taken(action) if action.action_type == ActionType.PLAY_MONOPOLY else None
            record = game.execute(action)
            tell({"type": "step", "game_id": game_id, "n": n, "seat": seat, "action": encoder.move(record.action), "result": encoder.result(record, taken)}, decide["state"])
        n += 1

    winner = game.winning_color()
    winner_seat = None if winner is None else encoder.seats[winner]
    final = encoder.state()
    tell({"type": "after", "game_id": game_id, "winner": winner_seat, "vp": [p["vp"] for p in final["players"]],
          "truncated": winner is None, "turns": state.num_turns, "state": final})
    return {"game_id": game_id, "seats": codes, "winner": winner_seat, "winner_code": None if winner_seat is None else codes[winner_seat],
            "vp": [p["vp"] for p in final["players"]], "truncated": winner is None, "turns": state.num_turns, "moves": n,
            "seconds": [round(times[s][0], 3) for s in range(4)], "decisions": [times[s][1] for s in range(4)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play Catanatron games with Open Seat Agents in some seats")
    parser.add_argument("--boards", required=True, help="protocol boards, one per line ({seed, board}), e.g. from OpenSeat boards")
    parser.add_argument("--lineup", default="open-seat-agent,AB:2,AB:2,AB:2", help="four seats: 'open-seat-agent' for an Open Seat Agent or a Catanatron bot code")
    parser.add_argument("--open-seat-agent", required=True, help="the command that starts an Open Seat Agent (one process per such entry of the lineup)")
    parser.add_argument("--maps", default="0:10", help="the boards to play, FIRST:END (line numbers, END exclusive), several comma-separated")
    parser.add_argument("--out", required=True, help="result lines, appended")
    parser.add_argument("--log", help="every game's referee messages, gzipped (for OpenSeat replay)")
    parser.add_argument("--agent-log", help="the agents' standard error (their stats lines)")
    args = parser.parse_args(argv)

    lineup = args.lineup.split(",")
    if len(lineup) != 4:
        raise SystemExit("--lineup needs four seats")
    with open(args.boards, encoding="utf-8") as file:
        all_boards = [json.loads(line) for line in file]
    # Ranges FIRST:END, comma-separated (e.g. the maps a stopped run didn't finish, see resume.py)
    boards = [board for part in args.maps.split(",") for board in all_boards[slice(*(int(x) for x in part.split(":")))]]
    agent_log = open(args.agent_log, "a", encoding="utf-8") if args.agent_log else subprocess.DEVNULL
    # One agent process per Open Seat Agent entry of the lineup, kept for all games (it follows the entry from seat to seat)
    agents = [OpenSeatAgent(args.open_seat_agent, agent_log) if code == "open-seat-agent" else None for code in lineup]
    log = gzip.open(args.log, "at", encoding="utf-8") if args.log else None
    write = (lambda m: log.write(json.dumps(m, separators=(",", ":")) + "\n")) if log else (lambda m: None)

    with open(args.out, "a", encoding="utf-8") as out:
        for entry in boards:
            for rotation in range(4):
                # Entry i of the lineup sits in seat (i + rotation) % 4
                codes, open_seat_agents, players = [None] * 4, {}, [None] * 4
                for i, code in enumerate(lineup):
                    seat = (i + rotation) % 4
                    codes[seat] = agents[i].name if agents[i] else code
                    players[seat] = OpenSeatAgentSeat(COLORS[seat]) if agents[i] else REGISTRY.build(code, COLORS[seat])
                    if agents[i]:
                        open_seat_agents[seat] = agents[i]
                random.seed(entry["seed"] * 4 + rotation)
                game = seated_game(players, entry["seed"], osp_board.catan_map(entry["board"]))
                game_id = f"map{entry['seed']}-r{rotation}"
                result = play(game, open_seat_agents, write, game_id, codes)
                result.update({"map_seed": entry["seed"], "rotation": rotation, "lineup": lineup})
                out.write(json.dumps(result) + "\n")
                out.flush()
                print(f"{game_id}: winner {result['winner_code']} in {result['turns']} turns", file=sys.stderr, flush=True)
    for agent in agents:
        if agent:
            agent.close()
    if log:
        log.close()


if __name__ == "__main__":
    # Catanatron keeps legal moves in sets, whose order depends on the hash seed: with it fixed, a game repeats
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit(subprocess.call([sys.executable, *sys.argv], env={**os.environ, "PYTHONHASHSEED": "0"}))
    main()
