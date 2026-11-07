"""
Catanatron's bots as an Open Seat Agent (open_seat_agent.py): a Catanatron game rebuilt from the referee's state writes
back as that state and lists the same moves, the bot answers with a listed move (a discard as one set of the right
size, offers declined, a robber move on the referee's hex), and the agent speaks the protocol line by line.
Run: .venv/Scripts/python -m unittest test_open_seat_agent
"""

import json
import random
import unittest

import catanatron.players  # noqa: F401 (registers the bots)
from catanatron import Game
from catanatron.models.enums import ActionPrompt
from catanatron.registry import REGISTRY

import open_seat_agent
import osp_board
import record
from osp_game import Encoder

CATANATRON = {"preset": "catanatron"}


def new_game(seed, bots=("R", "W", "R", "W")):
    players = [REGISTRY.build(code, color) for code, color in zip(bots, record.COLORS)]
    return Game(players, seed=seed)


def canonical(node):
    return json.dumps(node, sort_keys=True)


def positions(seeds=range(1, 5), every=3, max_turns=300):
    """Decisions of random Catanatron games, each as the referee would send it: board, full state and legal moves"""
    for seed in seeds:
        random.seed(seed)
        game = new_game(seed)
        encoder = Encoder(game)
        tick = 0
        while game.winning_color() is None and game.state.num_turns < max_turns:
            if tick % every == 0:
                state = game.state
                seat = state.current_player_index
                legal = encoder.discard_options(seat) if state.current_prompt == ActionPrompt.DISCARD else encoder.legal(game.playable_actions)
                yield game, osp_board.board(state.board.map, state.board.robber_coordinate), encoder.state(), legal
            game.play_tick()
            tick += 1


class RebuildTests(unittest.TestCase):
    def test_a_rebuilt_game_writes_back_as_the_referees_state_and_lists_the_same_moves(self):
        prompts = set()
        for game, board, state, legal in positions():
            rebuilt = open_seat_agent.rebuild(board, state, legal, CATANATRON)
            ours = Encoder(rebuilt).state()
            for field in ("prompt", "current", "on_turn", "free_roads", "robber", "buildings", "bank", "players"):
                self.assertEqual(canonical(state[field]), canonical(ours[field]), f"{field} at turn {state['turn']}")
            self.assertEqual(state["turn"], ours["turn"])
            # The board's own Longest Road holder too, which Catanatron's rules consult when roads are built: the referee's
            # (Catanatron's own board can disagree with its players' flags after a settlement cuts a road)
            holders = [record.COLORS[p["seat"]] for p in state["players"] if p["has_longest_road"]]
            self.assertEqual(holders[0] if holders else None, rebuilt.state.board.road_color)
            prompts.add(state["prompt"])
            if state["prompt"] == "discard":
                moves = Encoder(rebuilt).discard_options(state["current"])
            else:
                moves = Encoder(rebuilt).legal(rebuilt.playable_actions)
            self.assertEqual(sorted(map(canonical, legal)), sorted(map(canonical, moves)), f"legal moves at turn {state['turn']}")
        self.assertTrue({"setup_settlement", "setup_road", "roll", "main", "discard", "move_robber"} <= prompts, prompts)

    def test_the_deck_has_the_referees_size_and_no_card_more_than_remains(self):
        for game, board, state, legal in positions(seeds=[7], every=11):
            rebuilt = open_seat_agent.rebuild(board, state, legal, CATANATRON)
            deck = rebuilt.state.development_listdeck
            self.assertEqual(state["bank"]["development"], len(deck))
            # A card the hands and the played ones used up can't be in the deck
            for card, total in open_seat_agent.DEVELOPMENT_DECK.items():
                name = open_seat_agent.CARD_NAMES[card]
                held = sum(p["development"].get(name, 0) + p["played"].get(name, 0) for p in state["players"])
                self.assertLessEqual(deck.count(card), max(0, total - held))


class DecisionTests(unittest.TestCase):
    def test_the_bot_answers_with_a_listed_move(self):
        for bot in ("W", "AB:2"):
            agent = open_seat_agent.Agent(bot)
            checked = 0
            for game, board, state, legal in positions(seeds=[3], every=7):
                move = agent.choose(board, state, legal, CATANATRON, seat=state["current"])
                self.assertIn(canonical(move), set(map(canonical, legal)))
                checked += 1
            self.assertGreater(checked, 20)

    def test_a_discard_is_one_set_of_the_right_size(self):
        agent = open_seat_agent.Agent("AB:2")
        checked = 0
        for game, board, state, legal in positions(seeds=range(1, 6), every=1):
            if state["prompt"] != "discard":
                continue
            move = agent.choose(board, state, legal, CATANATRON, seat=state["current"])
            self.assertEqual("discard", move[0])
            self.assertIn(canonical(move), set(map(canonical, legal)))
            checked += 1
            if checked >= 5:
                break
        self.assertGreater(checked, 0)

    def test_offers_are_declined(self):
        game, board, state, legal = next(p for p in positions(seeds=[2], every=1) if p[2]["prompt"] == "main")
        state = dict(state, prompt="answer_offer")
        move = open_seat_agent.Agent("AB:2").choose(board, state, [["respond", {"accept": True}], ["respond", {"accept": False}]], {"preset": "standard"}, seat=state["current"])
        self.assertEqual(["respond", {"accept": False}], move)

    def test_a_robber_move_on_a_hex_without_victims_with_cards_takes_a_listed_victim(self):
        # The rulebook lets the robber take from a victim without cards; Catanatron then moves it without a victim
        moves = [["move_robber", {"hex": [0, 0], "victim": 2}], ["move_robber", {"hex": [1, -1], "victim": 1}]]
        self.assertEqual(moves[0], open_seat_agent.listed(["move_robber", {"hex": [0, 0], "victim": None}], moves))
        self.assertIsNone(open_seat_agent.listed(["move_robber", {"hex": [2, 0], "victim": None}], moves))
        self.assertEqual(moves[1], open_seat_agent.listed(moves[1], moves))


class ProtocolTests(unittest.TestCase):
    def test_the_agent_speaks_the_protocol(self):
        game, board, state, legal = next(p for p in positions(seeds=[4], every=1) if p[2]["prompt"] == "main")
        messages = [
            {"type": "hello", "protocol": "osp", "version": 1},
            {"type": "before", "game_id": "g1", "seat": state["current"], "players": 4, "ruleset": CATANATRON, "board": board, "state": state},
            {"type": "decide", "game_id": "g1", "n": 17, "seat": state["current"], "state": state, "legal": legal},
            {"type": "after", "game_id": "g1", "winner": 0, "vp": [10, 3, 4, 5], "truncated": False},
        ]
        answers = open_seat_agent.run([json.dumps(m) for m in messages], bot="W", needs="world")
        hello, decide = [json.loads(a) for a in answers]
        self.assertEqual({"type": "hello", "protocol": "osp", "version": 1, "name": "catanatron-W", "rulesets": ["standard", "catanatron"],
                          "observe": False, "needs": "world"}, hello)
        self.assertEqual(("decide", "g1", 17), (decide["type"], decide["game_id"], decide["n"]))
        self.assertIn(canonical(decide["action"]), set(map(canonical, legal)))

    def test_a_rebuild_failure_falls_back_to_a_listed_move(self):
        # A state the bot can't use (here no players at all) still gets a legal answer
        answers = open_seat_agent.run([json.dumps({"type": "hello", "protocol": "osp", "version": 1}),
                                       json.dumps({"type": "decide", "game_id": "g", "n": 0, "seat": 0, "state": {}, "legal": [["roll"]]})], bot="W")
        self.assertEqual(["roll"], json.loads(answers[1])["action"])


if __name__ == "__main__":
    unittest.main()
