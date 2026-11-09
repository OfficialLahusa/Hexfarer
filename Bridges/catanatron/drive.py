"""
Plays Catanatron games under Catanatron's referee with some seats played by Open Seat Agents (see ../PROTOCOL.md,
version 2): external processes that get their seat's messages over standard input and output and answer them. The
other seats are Catanatron's bots. The driver serves what each agent asks in its handshake or refuses it: its seat's
view (the other hands hidden) or every hand, steps only if it observes, decisions with only the state's hash if it is
lean; it can't draw worlds. Each game starts with before (the ruleset, the budget, the map's seed), which the agent takes
(ready) or turns down; a decision has the budget's time. A failure (the agent's own, an illegal, late or malformed answer,
a game it turns down) ends the game, recorded in its result, and the run goes on; an agent whose process ended stops it,
and so do more failed games than --max-failed allows (default 3, or one in a hundred of the run's games).
Every map is played once per rotation of the lineup (4 rotations: each lineup entry sits in each seat once), all
rotations of a map with the same Catanatron seed (the same dice and development cards as long as the games run alike).
One result line per game, with each side's time per decision and the game's result record.

Usage: python drive.py --boards boards.jsonl --lineup open-seat-agent,AB:2,AB:2,AB:2 --open-seat-agent "dotnet OpenSeat.dll agent --iterations 5000"
                       --maps 0:25 --out results.jsonl [--log games.jsonl.gz] [--decide-ms 300000] [--max-failed K]
                       [--bot NAME=path/to/bot.py ...] (outside Catanatron players the lineup names, e.g. challenger_player.py)
"""

import argparse
import copy
import gzip
import hashlib
import json
import os
import queue
import random
import shlex
import subprocess
import sys
import threading
import time

import catanatron.players  # noqa: F401 (registers the bots)
from catanatron import Color, Game, Player
from catanatron.game import TURNS_LIMIT
from catanatron.models.enums import ActionPrompt, ActionType
from catanatron.registry import REGISTRY, SpecError

import osp_board
from osp_game import Encoder, card_set

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

PROTOCOL, VERSION = "osp", 2
# How long an agent has for its handshake and its ready (starting a process, loading a net)
ANSWER_SECONDS = 60
# The driver's time per decision when the run sets none
DEFAULT_DECIDE_MS = 300_000


def default_max_failed(games):
    """The failed games a run of that many allows by default: three, or one in a hundred"""
    return max(3, games // 100)


def state_hash(state):
    """The hash decisions carry: the first 16 hex digits of the SHA-256 of the state's canonical JSON"""
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def view(state, seat):
    """The state as a seat sees it: the other seats' hands, development cards and points with them hidden"""
    seen = copy.deepcopy(state)
    for player in seen["players"]:
        if player["seat"] != seat:
            player["hand"] = player["development"] = player["vp"] = None
    return seen


def seen_result(action, result, mover, seat):
    """A move's result as a seat sees it: another's bought card is null, a steal between two others is "unknown" """
    if result is None or mover == seat:
        return result
    if action[0] == "buy_development":
        return {"card": None}
    if action[0] == "move_robber" and result.get("stolen") is not None and action[1].get("victim") != seat:
        return {"stolen": "unknown"}
    return result


class AgentFailure(Exception):
    """A failure of an agent's seat: its code (see PROTOCOL.md), whether the agent reported it itself"""

    def __init__(self, code, message, by_agent=False):
        super().__init__(message)
        self.code, self.by_agent = code, by_agent


class OpenSeatAgentSeat(Player):
    """A seat whose decisions an Open Seat Agent makes (the driver asks it; Catanatron never calls decide)"""

    def decide(self, game, playable_actions):
        raise RuntimeError("The driver decides for Open Seat Agent seats")


class OpenSeatAgent:
    """An Open Seat Agent process: one JSON object per line each way. The handshake says what it plays and wants; the
    driver serves view or truth, steps and lean decisions, and refuses the rest (a failure, unsupported)."""

    def __init__(self, command, stderr):
        # POSIX splitting on every system (quotes group a path with spaces; write paths with forward slashes)
        self.process = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, text=True, encoding="utf-8", bufsize=1)
        # Its lines arrive on a thread of their own, so an answer can be waited for with a time limit
        self.lines = queue.Queue()
        self.said = []
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self._handshake()
        except AgentFailure:
            # An agent this driver can't play with doesn't stay running
            self.close(wait=5)
            raise

    def _handshake(self):
        self.send({"type": "handshake", "protocol": PROTOCOL, "version": VERSION})
        handshake = self.receive(ANSWER_SECONDS)
        if handshake.get("type") == "failure":
            raise AgentFailure(handshake.get("code", "internal"), f"The agent refused the handshake: {handshake.get('message')}", by_agent=True)
        if handshake.get("type") != "handshake" or handshake.get("protocol") != PROTOCOL or handshake.get("version") != VERSION:
            self.refuse("version", f"The agent didn't answer the handshake in protocol {PROTOCOL} version {VERSION}: {handshake}")
        self.name = handshake.get("name", "open-seat-agent")
        self.needs = handshake.get("needs", "view")
        self.observe, self.lean = handshake.get("observe", False), handshake.get("lean", False)
        rulesets = handshake.get("rulesets", [])
        if not isinstance(self.observe, bool) or not isinstance(self.lean, bool) or not isinstance(rulesets, list):
            self.refuse("malformed", f"The agent's handshake has a field of the wrong type: {handshake}")
        # What this referee can't serve, it refuses instead of sending something else
        if self.needs not in ("view", "truth"):
            self.refuse("unsupported", f"Catanatron's driver serves view and truth, not {self.needs!r}: it draws no worlds")
        if self.lean and not self.observe:
            self.refuse("unsupported", "A lean agent has to observe: it keeps its own game from the steps")
        if "catanatron" not in rulesets:
            self.refuse("unsupported", f"Catanatron's driver plays the catanatron preset, the agent plays {rulesets}", {"preset": "catanatron"})

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, message):
        try:
            self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except OSError as error:
            raise AgentFailure("ended", f"The agent ended before it read its messages ({error})")

    def receive(self, seconds):
        """The agent's next message within the time; a timeout or an end of its output fails. Lines that aren't a JSON
        object (an agent's or a library's logging) are skipped, as Lanzenbrecher's and JSettlers' referees skip them, and the
        last few are kept for the failure's message."""
        deadline = time.perf_counter() + seconds
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.perf_counter()))
            except queue.Empty:
                raise AgentFailure("timeout", f"The agent gave no answer within {seconds:.1f} s{self._said()}")
            if line is None:
                self.lines.put(None)
                raise AgentFailure("ended", f"The agent ended without an answer{self._said()}")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                message = None
            if isinstance(message, dict):
                return message
            if line.strip():
                self.said = (self.said + [line.strip()[:200]])[-5:]

    def _said(self):
        return f"; it said last: {' | '.join(self.said)}" if self.said else ""

    def fail(self, game_id, n, code, message, detail=None):
        """Tells the agent what failed (if it still listens) and returns the failure"""
        failure = {"type": "failure", "game_id": game_id, "n": n, "code": code, "message": message}
        if detail is not None:
            failure["detail"] = detail
        try:
            self.send(failure)
        except AgentFailure:
            pass
        return AgentFailure(code, message)

    def refuse(self, code, message, detail=None):
        raise self.fail(None, None, code, message, detail)

    def close(self, wait=60):
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


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


def seat_record(seat, agent=None):
    """A seat's line in the game's result record (see PROTOCOL.md): null agent for Catanatron's own bots"""
    return {"seat": seat, "agent": agent.name if agent else None, "needs": agent.needs if agent else None,
            "observe": agent.observe if agent else None, "lean": agent.lean if agent else None,
            "decisions": 0, "time_ms": 0, "max_ms": 0, "resyncs": 0, "fallbacks": 0, "failures": []}


def ask(agent, record, decide, seat_state, budget_ms):
    """Asks the agent for a decision within the budget: a lean agent gets only the hash, and the state again when it
    asks (resync); answers to earlier decisions and games are skipped. Returns the chosen move."""
    started = time.perf_counter()
    message = {**decide, "hash": state_hash(seat_state)}
    if not agent.lean:
        message["state"] = seat_state
    agent.send(message)
    while True:
        left = budget_ms / 1000 - (time.perf_counter() - started)
        if left <= 0:
            raise agent.fail(decide["game_id"], decide["n"], "timeout", f"The agent gave no answer within {budget_ms} ms")
        try:
            answer = agent.receive(left)
        except AgentFailure as failure:
            if failure.code == "timeout":
                raise agent.fail(decide["game_id"], decide["n"], "timeout", f"The agent gave no answer within {budget_ms} ms")
            if failure.code == "malformed":
                raise agent.fail(decide["game_id"], decide["n"], "malformed", str(failure))
            raise
        kind = answer.get("type")
        if kind in ("ready", "decide") and answer.get("game_id") != decide["game_id"] or kind == "decide" and isinstance(answer.get("n"), int) and answer["n"] < decide["n"]:
            continue
        if kind == "failure":
            raise AgentFailure(answer.get("code", "internal"), f"The agent reported a failure: {answer.get('message')}", by_agent=True)
        if kind == "resync" and answer.get("n") == decide["n"]:
            record["resyncs"] += 1
            agent.send({**decide, "hash": message["hash"], "state": seat_state})
            continue
        if kind != "decide" or "action" not in answer:
            raise agent.fail(decide["game_id"], decide["n"], "malformed", f"The agent answered a decision with {answer}")
        if answer.get("n") != decide["n"]:
            raise agent.fail(decide["game_id"], decide["n"], "illegal", f"The agent answered decision {answer.get('n')} when asked for {decide['n']}")
        if answer["action"] not in decide["legal"]:
            raise agent.fail(decide["game_id"], decide["n"], "illegal", f"The agent's {answer['action']} is not one of the legal moves")
        spent = round((time.perf_counter() - started) * 1000)
        record["decisions"] += 1
        record["time_ms"] += spent
        record["max_ms"] = max(record["max_ms"], spent)
        return answer["action"]


def play(game, open_seat_agents, write, game_id, codes, budget_ms=DEFAULT_DECIDE_MS):
    """Plays the game; open_seat_agents maps seats to Open Seat Agents. A failure ends the game (cut short) and is
    recorded. Returns the result line, with the game's result record."""
    encoder = Encoder(game)
    state = game.state
    times = {seat: [0.0, 0] for seat in range(4)}
    records = [seat_record(seat, open_seat_agents.get(seat)) for seat in range(4)]
    playing = dict(open_seat_agents)
    # Any threads and a GPU allowed, as at every referee unless the run says otherwise
    budget = {"decide_ms": budget_ms, "game_ms": None, "threads": None, "gpu": True}

    def seat_state(seat, full):
        return full if playing[seat].needs == "truth" else view(full, seat)

    def tell(message, before=None, mover=None):
        # The log has the state before each move in its decide; observing agents get the step as their seat sees it,
        # with the state before the move under truth
        write(message)
        for seat, agent in list(playing.items()):
            if not agent.observe:
                continue
            seen = dict(message)
            if message["type"] == "step":
                if agent.needs == "truth":
                    seen["state"] = before
                else:
                    seen["result"] = seen_result(message["action"], message["result"], mover, seat)
            try:
                agent.send(seen)
            except AgentFailure as failure:
                lost(seat, failure, message.get("n"))

    def lost(seat, failure, n):
        records[seat]["failures"].append({"n": n, "code": failure.code, "by": "agent" if failure.by_agent else "referee", "message": str(failure)})
        playing.pop(seat, None)
        write({"type": "failure", "game_id": game_id, "seat": seat, "n": n, "code": failure.code, "by": "agent" if failure.by_agent else "referee", "message": str(failure)})

    board = osp_board.board(state.board.map, state.board.robber_coordinate)
    full = encoder.state()
    ruleset = {"preset": "catanatron"}
    write({"type": "before", "game_id": game_id, "seed": game.seed, "players": 4, "ruleset": ruleset, "budget": budget,
           "bots": codes, "board": board, "state": full})
    # The agents' seed is the map's: the same in a map's four rotations (the agent sits in another seat in each) and again
    # when the run is repeated, which with the fixed hash seed (see the end) replays a seeded agent's games
    for seat, agent in list(playing.items()):
        agent.send({"type": "before", "game_id": game_id, "seat": seat, "players": 4, "ruleset": ruleset, "budget": budget,
                    "seed": game.seed, "board": board, "state": seat_state(seat, full)})
    # Each agent takes the game or turns it down
    for seat, agent in list(playing.items()):
        try:
            while True:
                answer = agent.receive(ANSWER_SECONDS)
                if answer.get("type") == "failure":
                    raise AgentFailure(answer.get("code", "internal"), f"The agent turned the game down: {answer.get('message')} {answer.get('detail') or ''}".strip(), by_agent=True)
                if answer.get("type") == "ready" and answer.get("game_id") == game_id:
                    break
                if answer.get("type") in ("ready", "decide"):
                    continue
                raise agent.fail(game_id, None, "malformed", f"The agent answered before with {answer}")
        except AgentFailure as failure:
            lost(seat, failure, None)

    n = 0
    while not any(r["failures"] for r in records) and game.winning_color() is None and state.num_turns < TURNS_LIMIT:
        seat = state.current_player_index
        discard = state.current_prompt == ActionPrompt.DISCARD
        legal = encoder.discard_options(seat) if discard else encoder.legal(game.playable_actions)
        full = encoder.state()
        decide = {"type": "decide", "game_id": game_id, "n": n, "seat": seat, "state": full, "legal": legal}
        write(decide)
        start = time.perf_counter()
        if seat in open_seat_agents:
            try:
                move = ask(playing[seat], records[seat], {key: value for key, value in decide.items() if key != "state"}, seat_state(seat, full), budget_ms)
            except AgentFailure as failure:
                lost(seat, failure, n)
                break
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
            tell({"type": "step", "game_id": game_id, "n": n, "seat": seat, "action": ["discard", discarded], "result": None}, decide["state"], seat)
        else:
            action = encoder.decode(move, game.playable_actions) if move is not None else state.current_player().decide(game, game.playable_actions)
            times[seat][0] += time.perf_counter() - start
            times[seat][1] += 1
            taken = encoder.monopoly_taken(action) if action.action_type == ActionType.PLAY_MONOPOLY else None
            record = game.execute(action)
            tell({"type": "step", "game_id": game_id, "n": n, "seat": seat, "action": encoder.move(record.action), "result": encoder.result(record, taken)}, decide["state"], seat)
        n += 1

    failed = any(r["failures"] for r in records)
    winner = None if failed else game.winning_color()
    winner_seat = None if winner is None else encoder.seats[winner]
    final = encoder.state()
    result = {"seed": game.seed, "moves": n, "turns": state.num_turns, "seats": records}
    # Every agent still in the game hears its end, with the result record; the log keeps it as the game's result line
    after = {"type": "after", "game_id": game_id, "winner": winner_seat, "vp": [p["vp"] for p in final["players"]],
             "truncated": winner is None, "result": result}
    write({**after, "turns": state.num_turns, "state": final})
    for seat, agent in list(playing.items()):
        try:
            agent.send(after)
        except AgentFailure as failure:
            lost(seat, failure, None)
    # Catanatron's own loop (Game.play) ends a game by telling its observers, the players among them; this driver plays its
    # own loop, so it tells them here (a bot that runs a process per game, as the Challenger seat, ends it there)
    for observer in game.observers:
        observer.after(game)
    return {"game_id": game_id, "seats": codes, "winner": winner_seat, "winner_code": None if winner_seat is None else codes[winner_seat],
            "vp": [p["vp"] for p in final["players"]], "truncated": winner is None, "turns": state.num_turns, "moves": n,
            "seconds": [round(times[s][0], 3) for s in range(4)], "decisions": [times[s][1] for s in range(4)], "result": result}


def agent_log_paths(path, lineup):
    """Where each entry of the lineup writes its agent's standard error (None for bots, or without a path): a single Open
    Seat Agent the given path; several each their own file, NAME-entryI.EXT for entry I, since processes writing one file
    at once break each other's lines"""
    entries = [i for i, code in enumerate(lineup) if code == "open-seat-agent"]
    stem, ext = os.path.splitext(path or "")
    return [None if not path or i not in entries else path if len(entries) == 1 else f"{stem}-entry{i}{ext}" for i in range(len(lineup))]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play Catanatron games with Open Seat Agents in some seats")
    parser.add_argument("--boards", required=True, help="protocol boards, one per line ({seed, board}), e.g. from OpenSeat boards")
    parser.add_argument("--lineup", default="open-seat-agent,AB:2,AB:2,AB:2", help="four seats: 'open-seat-agent' for an Open Seat Agent or a Catanatron bot code")
    parser.add_argument("--open-seat-agent", required=True, help="the command that starts an Open Seat Agent (one process per such entry of the lineup)")
    parser.add_argument("--maps", default="0:10", help="the boards to play, FIRST:END (line numbers, END exclusive), several comma-separated")
    parser.add_argument("--out", required=True, help="result lines, appended")
    parser.add_argument("--log", help="every game's referee messages, gzipped (for OpenSeat replay)")
    parser.add_argument("--agent-log", help="the agents' standard error (their stats lines); with several Open Seat Agents in the lineup "
                                            "each writes its own file, NAME-entryI.EXT for lineup entry I")
    parser.add_argument("--decide-ms", type=int, default=DEFAULT_DECIDE_MS, help="the agents' time per decision (the game's budget)")
    parser.add_argument("--max-failed", type=int, help="failed games the run allows before it stops (default: 3, or 1 in 100 games; negative: any)")
    parser.add_argument("--bot", action="append", default=[], metavar="NAME=SOURCE",
                        help="an outside bot the lineup can name, as with catanatron-play's --bot: a Catanatron Player in a Python file "
                             "(path/to/bot.py, #Class to pick one); repeatable")
    args = parser.parse_args(argv)

    # Outside bots join Catanatron's registry under their names (the class name without NAME=), as its own bots are
    for spec in args.bot:
        name, given, source = spec.partition("=")
        try:
            REGISTRY.register_source(source if given else name, name if given else None)
        except SpecError as error:
            raise SystemExit(f"--bot {spec}: {error}")

    lineup = args.lineup.split(",")
    if len(lineup) != 4:
        raise SystemExit("--lineup needs four seats")
    with open(args.boards, encoding="utf-8") as file:
        all_boards = [json.loads(line) for line in file]
    # Ranges FIRST:END, comma-separated (e.g. the maps a stopped run didn't finish, see resume.py)
    boards = [board for part in args.maps.split(",") for board in all_boards[slice(*(int(x) for x in part.split(":")))]]
    max_failed = args.max_failed if args.max_failed is not None else default_max_failed(4 * len(boards))
    played = failed_games = 0
    # Each agent's standard error goes to its own file (see agent_log_paths), open for the whole run
    agent_logs = [open(p, "a", encoding="utf-8") if p else None for p in agent_log_paths(args.agent_log, lineup)]
    # One agent process per Open Seat Agent entry of the lineup, kept for all games (it follows the entry from seat to seat)
    agents = [OpenSeatAgent(args.open_seat_agent, agent_logs[i] or subprocess.DEVNULL) if code == "open-seat-agent" else None
              for i, code in enumerate(lineup)]
    log = gzip.open(args.log, "at", encoding="utf-8") if args.log else None
    write = (lambda m: log.write(json.dumps(m, separators=(",", ":")) + "\n")) if log else (lambda m: None)

    # The agents and the log close also when the run stops (an agent ended, too many failed games)
    try:
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
                    result = play(game, open_seat_agents, write, game_id, codes, args.decide_ms)
                    result.update({"map_seed": entry["seed"], "rotation": rotation, "lineup": lineup})
                    out.write(json.dumps(result) + "\n")
                    out.flush()
                    failures = [f for seat in result["result"]["seats"] for f in seat["failures"]]
                    print(f"{game_id}: winner {result['winner_code']} in {result['turns']} turns"
                          + "".join(f"; failure {f['code']}: {f['message']}" for f in failures), file=sys.stderr, flush=True)
                    # An agent whose process ended can't play the rest of the run
                    if any(f["code"] == "ended" for f in failures):
                        raise SystemExit(f"{game_id}: an Open Seat Agent ended; the run stops")
                    # Many failed games mean a broken setup, not bad luck: past the limit the run stops
                    played += 1
                    failed_games += bool(failures)
                    if 0 <= max_failed < failed_games:
                        raise SystemExit(f"{game_id}: stopped: {failed_games} of {played} games failed, at most {max_failed} allowed (--max-failed)")
    finally:
        for agent in agents:
            if agent:
                agent.close()
        # The agents' logs close after their processes ended (nothing more gets written)
        for file in agent_logs:
            if file:
                file.close()
        if log:
            log.close()


if __name__ == "__main__":
    # Catanatron keeps legal moves in sets, whose order depends on the hash seed: with it fixed, a game repeats
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit(subprocess.call([sys.executable, *sys.argv], env={**os.environ, "PYTHONHASHSEED": "0"}))
    main()
