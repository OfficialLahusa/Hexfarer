# Bridges

Adapters between Hexfarer and other CATAN engines, so that bots of either side can play under the other side's referee:
[Catanatron](https://github.com/bcollazo/catanatron), [JSettlers](https://github.com/jdmonin/JSettlers2) and
[StacSettlers](https://github.com/sorinMD/StacSettlers).

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
- StacSettlers: commit `8185c5e` of sorinMD/StacSettlers (a fork of JSettlers 1.1.16 with the Stac, SmartSettlers and
  Dobre's MCTS agents), built with its dependencies into `STACSettlers-1.0-bin.jar`, which STACSETTLERS_JAR names. Its
  Maven build needs two changes that aren't the adapter's: the compiler at release 8 (for a JDK 21) and the DeepCatan
  dependency left out (GitHub serves no copy of that jar), with its neural network seeder stubbed; nothing the
  adapter uses depends on either. Built and tested with JDK 21.

Contents:
- `PROTOCOL.md`: the Open Seat Protocol.
- `RULESET.md`: its ruleset options, their values and which engine plays what.
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
  - `drive.py`: Hexfarer (or any Open Seat Agent) under Catanatron's referee. It serves its seat's view or every hand
    (no worlds), steps and lean decisions, gives each decision `--decide-ms` (default 300 000), ends a game at a failure
    and records it in the game's result, which every result line carries.
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
    JSettlers' own robots one strategy (it starts 30% fast, the rest smart). Each game ends with its result record (see
    `PROTOCOL.md`) as an `osp-result` line on standard error: the players by seat, the winner, and per seat the agent and
    its decisions, their time, resyncs, fallbacks and failures, its moves, its decisions with a choice (only in a recorded
    game), its trade decisions that expired before they went out, and its robot brain's CPU time. An agent that fails is told why where it still listens, and the robot plays
    the first legal move for the rest of the game. A record never overwrites one of an earlier server run (JSettlers
    numbers its games afresh): the second `GAME.jsonl` is written as `GAME-2.jsonl`.
    Without the game option NT the players trade as JSettlers lets them, which the protocol names `player_trading`
    `first_accept`: JSettlers' robots offer, decline, counter and accept, the move log writes each trade request as a
    step (answers in the order they came, an accept trading at once), and the agent's robot answers every offer made to
    its seat as it stands, offers on its turn where its decision names `trades`, waits for the answers to its own offer
    (5 s, -Dosp.offerWait=MILLIS), then takes a counteroffer or withdraws. JSettlers' robots wait 5 s for an answer from a
    robot seat, so an answer has to come quickly: a decision whose offer was taken or withdrawn meanwhile is dropped.
  - Tests: `python -m unittest test_jsettlers_board test_jsettlers_game test_jsettlers_robot test_jsettlers_steps test_jsettlers_trading` (with
    JSETTLERS_JAR set).
- `stacsettlers/`: StacSettlers in protocol terms (Java, built by `stacsettlers.py` against its jar into `build/`), so
  Open Seat Agents play against its Stac agents, SmartSettlers and Dobre's MCTS agents in its headless simulation:
  - `OSPBoard` and `OSPGame` as for JSettlers, for the JSettlers 1.1.16 engine StacSettlers keeps (other hex type and
    development card numbers, no bank, its own checks for the legal moves).
  - `OSPSimulation CONFIG`: StacSettlers' own `Simulation` (its config format and results, unchanged; the config parsed
    by its own code) with one more agent kind, `osp`: `Agent=1,Hexfarer,osp` seats a robot whose decisions the Open Seat
    Agent of `-Dosp.agent=COMMAND` makes (`OSPRobotBrain`). The games run on `OSPServer`, StacSettlers' server through
    which the robot reads the true game under the lock the server handles requests with; the robot serves its agent
    the true state or its seat's view, as it asks, and if it observes, every move of the game as a step from the
    game's start (`MoveLog`, which turns the requests of all seats, StacSettlers' agents' too, into protocol moves), and
    plays the answer through StacSettlers' requests. It takes no part in the trades
    StacSettlers' agents negotiate in chat: it never announces that it participates (so Stac agents don't wait for its
    answers) and declines every offer made to it (Dobre's MCTS agents wait for the answer);
    with the bank and the harbors it trades as usual. Each game ends with its result record as an `osp-result` line on
    standard error (the seat's decisions, their time, fallbacks, failures, offers declined and stalls: moves the
    server didn't take within 20 s, after which the robot decides again without that move). -Dosp.log=DIR keeps each
    agent's standard error, -Dosp.debug=true prints the robot's moves. Run with `ChatNegotiation=true` (the Stac agents
    hang without it) and `ForceEndTurns=false` (else the server ends a turn after 30 s without a move);
    SmartSettlers and Dobre's typed MCTS agents need `FullyObservable=true`, which shows every hand to everyone.
  - `play.py`: games of one agent against three of StacSettlers' (presets `stac`, `ss`, `pomcp`, `jsettlers`, `random`, or
    any agent line) with a summary of its wins, decision times and failures, e.g.
    `python play.py --agent "OpenSeat agent --net NET --iterations 5000" --opponents stac --games 10 --out DIR`
    (`--opponents ss --fully-observable` for SmartSettlers).
  - `MoveLog`: the steps, from the requests of every seat: two-step moves joined, discards in the protocol's order, and a
    trade StacSettlers' agents make in chat (with `Trades=true`) as the offer of the player on turn and the partner's
    accept under `player_trading` `first_accept`, which the games then announce (`none` with `Trades=false`). Anything
    else that changes the game counts as a gap (in the log's `after`). With `-Dosp.record=DIR` (on `OSPSimulation`)
    every game is also written as a referee log while it runs, the input of `OpenSeat replay`; `record.py --games N
    --out DIR [--trades]` records Stac-only games that way.
  - `Recorder --games N --out games.jsonl.gz`: random games by StacSettlers' rules as referee messages, each move
    checked as its server would check the request.
  - Tests: `python -m unittest test_stacsettlers_board test_stacsettlers_game test_stacsettlers_robot
    test_stacsettlers_record test_stacsettlers_steps` (with STACSETTLERS_JAR set; the robot tests play against three Stac agents with the test
    agent of `jsettlers/`, one of them against three cheap POMCP agents for about 3 minutes; about 5 minutes in all).

On Hexfarer's side, outside this folder: `OpenSeat replay LOG` plays a referee's log through Hexfarer's rules and lists
the differences, `OpenSeat relay LOG --save DIR` writes its games as save files for the client (a new part where the
rules part), `OpenSeat relay LOG|DIR --live` hosts them for the client's watchers while the referee writes them (the
client joins with the Watch role; a known rule difference such as JSettlers' unlimited bank comes in as an edit, so the
game stays one), and `OpenSeat agent` is Hexfarer's own Open Seat Agent (`--fair` for the agent that
plays from its seat's view, `--lean` for decisions with only the state's hash, checked against its own copy of the game, which saves only under
Hexfarer's own referee; it offers and counters where a decision names `trades`). The Arena seats any agent as
`NAME=command:COMMAND`,
and the client's local games seat those listed in `%AppData%\Hexfarer\settings.json` under `OpenSeatAgents`
(`[{"Name": "...", "Command": "..."}]`).
