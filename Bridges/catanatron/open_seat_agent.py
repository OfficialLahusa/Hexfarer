"""Catanatron's bots as an Open Seat Agent (stub)"""

DEVELOPMENT_DECK = {}
CARD_NAMES = {}


def rebuild(board, state, legal, ruleset):
    raise NotImplementedError


def listed(move, legal):
    raise NotImplementedError


def run(lines, bot, needs="world"):
    raise NotImplementedError


class Agent:
    def __init__(self, bot):
        raise NotImplementedError

    def choose(self, board, state, legal, ruleset, seat):
        raise NotImplementedError
