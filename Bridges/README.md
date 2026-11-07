# Bridges

Adapters between Hexfarer and other CATAN engines, so that bots of either side can play under the other side's referee:
[Catanatron](https://github.com/bcollazo/catanatron) and [JSettlers](https://github.com/jdmonin/JSettlers2).

This folder is licensed under the GNU General Public License, version 3 or later (see `LICENSE`), since the adapters
build on those GPL-licensed engines; the rest of the repository is not (see `../LICENSE`). To keep the two apart:
- the adapters contain no Hexfarer code and Hexfarer contains no adapter code;
- they talk only across a process boundary, over the Open Seat Protocol (OSP, see `PROTOCOL.md`): JSON lines on
  standard input and output, the same for every engine, with the rules as an option of each game. Programs that play
  seats over it are Open Seat Agents;
- foreign code is only a dependency, never copied in.

Pinned versions:
- Catanatron: commit `ecf9311` (2026-09-08), the first with its protocol for bots in other languages.
- JSettlers: release 2.6.10 (its full jar, which JSETTLERS_JAR names; a JDK 8 or newer builds the adapter against it).

Contents:
- `PROTOCOL.md`: the Open Seat Protocol.
- `osp/classic-board.json`: the protocol's reference shape of the classic board (hexes, intersections, paths, harbors
  in protocol keys). Hexfarer's test `TheClassicBoardHasTheOpenSeatProtocolsReferenceShape` writes it
  (`HEXFARER_UPDATE_FIXTURES=1`) and checks against it; every adapter is tested against it too.
- `catanatron/`: Catanatron in protocol terms: its board (`osp_board.py`), its moves, results, legal moves and states
  (`osp_game.py`), and a recorder that plays Catanatron games and writes them as the referee's protocol messages
  (`record.py --bots AB:2,R,W,R --games 100 --out games.jsonl.gz`), the input of Hexfarer's replay harness. Setup and
  tests: `python -m venv .venv`, `.venv/Scripts/python -m pip install -r requirements.txt`,
  `.venv/Scripts/python -m unittest test_osp_board test_osp_game test_open_seat_agent test_drive`.
  - `open_seat_agent.py --bot AB:2 [--needs world|truth]`: Catanatron's bots as an Open Seat Agent under another
    referee (e.g. Hexfarer's Arena: `--baseline "AB2=command:python open_seat_agent.py --bot AB:2"`); each decision
    rebuilds a Catanatron game from the referee's state, the bot picks among the referee's legal moves.
  - `drive.py`: Hexfarer (or any Open Seat Agent) under Catanatron's referee.
- `jsettlers/`: JSettlers in protocol terms (Java, built by `jsettlers.py` against JSettlers' jar into `build/`):
  - `OSPBoard`: its classic board in protocol keys, from its own adjacency; `OSPGame`: states, legal moves by its own
    rules and moves played through its game logic.
  - `Recorder --games N --out games.jsonl.gz [--ruleset jsettlers]`: random games by JSettlers' rules as referee
    messages, for Hexfarer's replay harness (the differences found are in PROTOCOL.md's ruleset table).
  - `OSPServer` with `OSPRobotClient`: JSettlers' server whose robot seats include Open Seat Agents (-Dosp.agent=COMMAND,
    started through jsettlers.bots.start3p). The robot runs inside the server and reads the game under the server's
    locks; an agent gets the true state or its seat's view, as it asks, and if it observes, every move of the game as
    a step from the game's start (`MoveLog`, which turns JSettlers' requests into protocol moves). -Dosp.log=DIR keeps
    each agent's standard error, -Dosp.debug=true prints the robot's steps, -Dosp.record=DIR writes every game as a
    referee log while it runs (for the replay harness, the relay and the live relay), -Dosp.bots=smart|fast gives all of
    JSettlers' own robots one strategy (it starts 30% fast, the rest smart). Each game ends with an `osp-game` line on
    standard error: the players by seat, the winner, and per seat its moves, its decisions with a choice (only in a
    recorded game) and its robot brain's CPU time. A record never overwrites one of an earlier server run (JSettlers
    numbers its games afresh): the second `GAME.jsonl` is written as `GAME-2.jsonl`.
  - Tests: `python -m unittest test_jsettlers_board test_jsettlers_game test_jsettlers_robot test_jsettlers_steps` (with
    JSETTLERS_JAR set).

On Hexfarer's side, outside this folder: `OpenSeat replay LOG` plays a referee's log through Hexfarer's rules and lists
the differences, `OpenSeat relay LOG --save DIR` writes its games as save files for the client (a new part where the
rules part), `OpenSeat relay LOG|DIR --live` hosts them for the client's watchers while the referee writes them (the
client joins with the Watch role; a known rule difference such as JSettlers' unlimited bank comes in as an edit, so the
game stays one), and `OpenSeat agent` is Hexfarer's own Open Seat Agent (`--fair` for the agent that
plays from its seat's view). The Arena seats any agent as `NAME=command:COMMAND`,
and the client's local games seat those listed in `%AppData%\Hexfarer\settings.json` under `OpenSeatAgents`
(`[{"Name": "...", "Command": "..."}]`).
