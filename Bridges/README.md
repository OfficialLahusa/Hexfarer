# Bridges

Adapters between Hexfarer and other CATAN engines, so that bots of either side can play under the other side's referee:
[Catanatron](https://github.com/bcollazo/catanatron) first, [JSettlers](https://github.com/jdmonin/JSettlers2) later.

This folder is licensed under the GNU General Public License, version 3 or later (see `LICENSE`), since the adapters
build on those GPL-licensed engines; the rest of the repository is not (see `../LICENSE`). To keep the two apart:
- the adapters contain no Hexfarer code and Hexfarer contains no adapter code;
- they talk only across a process boundary, over the Open Seat Protocol (OSP, see `PROTOCOL.md`): JSON lines on
  standard input and output, the same for every engine, with the rules as an option of each game. Programs that play
  seats over it are Open Seat Agents;
- foreign code is only a dependency, never copied in.

Pinned versions:
- Catanatron: commit `ecf9311` (2026-09-08), the first with its protocol for bots in other languages.

Contents:
- `PROTOCOL.md`: the Open Seat Protocol.
- `osp/classic-board.json`: the protocol's reference shape of the classic board (hexes, intersections, paths, harbors
  in protocol keys). Hexfarer's test `TheClassicBoardHasTheOpenSeatProtocolsReferenceShape` writes it
  (`HEXFARER_UPDATE_FIXTURES=1`) and checks against it; every adapter is tested against it too.
- `catanatron/`: Catanatron in protocol terms: its board (`osp_board.py`), its moves, results, legal moves and states
  (`osp_game.py`), and a recorder that plays Catanatron games and writes them as the referee's protocol messages
  (`record.py --bots AB:2,R,W,R --games 100 --out games.jsonl.gz`), the input of Hexfarer's replay harness. Setup and
  tests: `python -m venv .venv`, `.venv/Scripts/python -m pip install -r requirements.txt`,
  `.venv/Scripts/python -m unittest test_osp_board test_osp_game`.
