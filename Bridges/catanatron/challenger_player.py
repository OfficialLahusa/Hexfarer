"""
The Challenger bot of "Catan Challenger Research" (github.com/midori11111111/catan-challenger-research, GPL-3.0) as a
Catanatron player, so drive.py can seat it: --bot CHALLENGER=challenger_player.py, then CHALLENGER (generation gen0) or
CHALLENGER:gen1 in the lineup. It plays through the release's own bridge (run.py's Ours): the published JavaScript
policy in a Node.js process, each decision on a world drawn from public information only (its belief_game), one process
per game, started when the game starts and ended when it ends. The release pins Catanatron to ecf9311, as the driver
does. Settings come from the environment: CHALLENGER_ROOT, the release's folder (required); CHALLENGER_ERRORS, the
folder for the policy's error files (default: the system's temporary folder). Node.js 20 or later must be on PATH.
"""

import dataclasses
import importlib.util
import os
import sys
import tempfile
import zlib

from catanatron.models.player import Player

# The release's bridge module, loaded once per folder
_bridges = {}


def _bridge():
    """The release's run.py, from CHALLENGER_ROOT (its folder also on the import path, for its own modules)"""
    root = os.environ.get("CHALLENGER_ROOT")
    if not root or not os.path.isfile(os.path.join(root, "run.py")):
        raise RuntimeError("CHALLENGER_ROOT must name the catan-challenger-research folder (with its run.py)")
    root = os.path.abspath(root)
    if root not in _bridges:
        if root not in sys.path:
            sys.path.insert(0, root)
        spec = importlib.util.spec_from_file_location(f"challenger_run_{len(_bridges)}", os.path.join(root, "run.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _bridges[root] = module
    return _bridges[root]


class ChallengerPlayer(Player):
    """One seat played by the Challenger policy of the given generation (the release confirmed gen0)"""

    @dataclasses.dataclass(frozen=True)
    class Params:
        generation: str = "gen0"

    def __init__(self, color, params=None):
        super().__init__(color, params)
        self._ours = None

    def before(self, game):
        # A process per game: a seat reused for the next game ends the old one first
        self._close()
        # The game's seed and the seat's color give the bridge's seed (its worlds and the policy's own), so a game repeats
        seed = zlib.crc32(f"{game.seed}:{self.color.value}".encode("utf-8"))
        errors = os.environ.get("CHALLENGER_ERRORS") or tempfile.gettempdir()
        error_file = os.path.join(errors, f"challenger-{game.seed}-{self.color.value}-{game.id[:8]}.json")
        self._ours = _bridge().Ours(self.color, self.params.generation, seed, error_file)
        self._ours.before(game)

    def decide(self, game, playable_actions):
        if self._ours is None:
            raise RuntimeError("the Challenger seat decides only between its game's start and end")
        return self._ours.decide(game, playable_actions)

    def after(self, game):
        self._close()

    def _close(self):
        if self._ours is not None:
            self._ours.close()
            self._ours = None
