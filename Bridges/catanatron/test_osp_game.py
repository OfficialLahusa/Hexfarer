"""
Catanatron's moves and states in the Open Seat Protocol (osp_game.py) and the recorder (record.py): every legal move has
its own protocol move, discards cover every card-by-card discard, and a recorded game replays in Catanatron from the log
alone (its moves decoded, its results forced) through the same states. Run: .venv/Scripts/python -m unittest test_osp_game
"""

import random
import unittest
from collections import Counter
from itertools import combinations

import catanatron.players  # noqa: F401 (registers the bots)
from catanatron import Color, Game
from catanatron.models.enums import RESOURCES, ActionPrompt, ActionRecord, ActionType
from catanatron.registry import REGISTRY

import osp_game
import record
from osp_game import CARD_NAMES, RESOURCES_BY_NAME, Encoder

CARDS_BY_NAME = {name: card for card, name in CARD_NAMES.items()}


def new_game(seed, bots=("R", "W", "R", "W")):
    players = [REGISTRY.build(code, color) for code, color in zip(bots, record.COLORS)]
    return Game(players, seed=seed)


def recorded(seed, bots=("R", "W", "R", "W")):
    """The messages of one recorded game"""
    messages = []
    record.record_game(new_game(seed, bots), messages.append, f"test-{seed}", list(bots))
    return messages


class EncodingTests(unittest.TestCase):
    def test_every_legal_move_has_its_own_protocol_move(self):
        for seed in range(1, 4):
            random.seed(seed)
            game = new_game(seed)
            encoder = Encoder(game)
            decisions = 0
            while game.winning_color() is None and game.state.num_turns < 300:
                if game.state.current_prompt != ActionPrompt.DISCARD:
                    moves = encoder.legal(game.playable_actions)
                    keys = [repr(m) for m in moves]
                    self.assertEqual(len(keys), len(set(keys)), keys)
                    for action, move in zip(game.playable_actions, moves):
                        self.assertEqual(action, encoder.decode(move, game.playable_actions))
                    decisions += 1
                game.play_tick()
            self.assertGreater(decisions, 100)

    def test_discard_options_are_every_card_by_card_discard(self):
        checked = 0
        for seed in range(1, 6):
            random.seed(seed)
            game = new_game(seed)
            encoder = Encoder(game)
            while game.winning_color() is None and game.state.num_turns < 400:
                state = game.state
                if state.current_prompt == ActionPrompt.DISCARD and checked < 40:
                    seat = state.current_player_index
                    hand = [r for r in RESOURCES for _ in range(state.player_state[f"P{seat}_{r}_IN_HAND"])]
                    count = state.discard_counts[seat]
                    # Every card-by-card discard ends in one of these sets of cards
                    expected = {tuple(sorted(Counter(c).items())) for c in combinations(hand, count)}
                    options = {tuple(sorted(Counter(RESOURCES_BY_NAME[n] for n, k in move[1].items() for _ in range(k)).items()))
                               for move in encoder.discard_options(seat)}
                    self.assertEqual(expected, options)
                    self.assertEqual(len(options), len(encoder.discard_options(seat)))
                    checked += 1
                game.play_tick()
        self.assertGreater(checked, 10)


class MeaningTests(unittest.TestCase):
    """The encoding means what Catanatron holds and does, by tables written here, not taken from osp_game"""

    RESOURCES_MEANT = {"lumber": "WOOD", "brick": "BRICK", "wool": "SHEEP", "grain": "WHEAT", "ore": "ORE"}
    CARDS_MEANT = {"knight": "KNIGHT", "road_building": "ROAD_BUILDING", "year_of_plenty": "YEAR_OF_PLENTY",
                   "monopoly": "MONOPOLY", "victory_point": "VICTORY_POINT"}

    def games(self):
        """Every decision of a few games, with the game and its encoder"""
        for seed in range(21, 25):
            random.seed(seed)
            game = new_game(seed)
            encoder = Encoder(game)
            while game.winning_color() is None and game.state.num_turns < 400:
                yield game, encoder
                game.play_tick()

    def test_states(self):
        for game, encoder in self.games():
            state, view = game.state, encoder.state()
            ps = state.player_state
            self.assertEqual(state.current_player_index, view["current"])
            for name, resource in self.RESOURCES_MEANT.items():
                self.assertEqual(state.resource_freqdeck[RESOURCES.index(resource)], view["bank"][name])
            for seat, player in enumerate(view["players"]):
                key = f"P{seat}"
                for name, resource in self.RESOURCES_MEANT.items():
                    self.assertEqual(ps[f"{key}_{resource}_IN_HAND"], player["hand"][name])
                for name, card in self.CARDS_MEANT.items():
                    self.assertEqual(ps[f"{key}_{card}_IN_HAND"], player["development"][name])
                self.assertEqual((ps[f"{key}_VICTORY_POINTS"], ps[f"{key}_ACTUAL_VICTORY_POINTS"]), (player["public_vp"], player["vp"]))
                # The pieces on the board are the ones that left the stock
                on_board = {kind: sum(1 for s, _ in view["buildings"][kind] if s == seat) for kind in ("roads", "settlements", "cities")}
                self.assertEqual({"roads": 15, "settlements": 5, "cities": 4},
                                 {kind: on_board[kind] + player["pieces_left"][kind] for kind in on_board})
            # The prompt to roll is exactly when rolling is legal (not during discards and robber moves)
            if state.current_prompt != ActionPrompt.DISCARD:
                self.assertEqual(view["prompt"] == "roll", ["roll"] in encoder.legal(game.playable_actions), view["prompt"])

    def test_moves(self):
        for game, encoder in self.games():
            if game.state.current_prompt == ActionPrompt.DISCARD:
                continue
            for action in game.playable_actions:
                move = encoder.move(action)
                if action.action_type == ActionType.MOVE_ROBBER:
                    victim = action.value[1]
                    self.assertEqual(None if victim is None else game.state.colors.index(victim), move[1]["victim"])
                elif action.action_type == ActionType.MARITIME_TRADE:
                    give, get = move[1]["give"], move[1]["get"]
                    given = [r for r in action.value[:-1] if r is not None]
                    self.assertEqual(1, len(give))
                    self.assertEqual(len(given), sum(give.values()))
                    self.assertEqual({next(n for n, r in self.RESOURCES_MEANT.items() if r == action.value[-1]): 1}, get)
                    self.assertFalse(set(give) & set(get))
                elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
                    self.assertEqual(sorted(action.value), sorted(self.RESOURCES_MEANT[n] for n in move[1]))
                elif action.action_type == ActionType.PLAY_MONOPOLY:
                    self.assertEqual(action.value, self.RESOURCES_MEANT[move[1]])

    def test_monopoly_takes_from_the_others_what_the_player_gains(self):
        played = 0
        for seed in range(30, 60):
            if played >= 3:
                break
            messages = recorded(seed)
            decides = {m["n"]: m for m in messages if m["type"] == "decide"}
            for step in (m for m in messages if m["type"] == "step" and m["action"][0] == "play_monopoly"):
                name, seat, taken = step["action"][1], step["seat"], step["result"]["taken"]
                self.assertNotIn(str(seat), taken)
                before, after = decides[step["n"]]["state"], decides[step["n"] + 1]["state"]
                self.assertEqual(sum(taken.values()), after["players"][seat]["hand"][name] - before["players"][seat]["hand"][name])
                for other, count in taken.items():
                    self.assertEqual(count, before["players"][int(other)]["hand"][name])
                played += 1
        self.assertGreater(played, 0)


class RecorderTests(unittest.TestCase):
    def replay(self, messages, seed, bots=("R", "W", "R", "W")):
        """Replays a recorded game in a new game from the log alone and checks every recorded state"""
        game = new_game(seed, bots)
        encoder = Encoder(game)
        self.assertEqual(messages[0]["state"], encoder.state())
        decides = {m["n"]: m for m in messages if m["type"] == "decide"}
        for step in (m for m in messages if m["type"] == "step"):
            decide = decides[step["n"]]
            self.assertEqual(decide["state"], encoder.state(), f"state before move {step['n']}")
            self.assertEqual(game.state.current_player_index, step["seat"])
            move, result, color = step["action"], step["result"], game.state.current_color()
            if move[0] == "discard":
                self.assertIn(move, decide["legal"])
                for action in encoder.discard_actions(move, color):
                    game.execute(action, action_record=ActionRecord(action, action.value))
                continue
            self.assertEqual(decide["legal"], encoder.legal(game.playable_actions), f"legal moves before move {step['n']}")
            action = encoder.decode(move, game.playable_actions)
            forced = None
            if action.action_type == ActionType.ROLL:
                forced = ActionRecord(action, tuple(result["dice"]))
            elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                forced = ActionRecord(action, CARDS_BY_NAME[result["card"]])
            elif action.action_type == ActionType.MOVE_ROBBER:
                forced = ActionRecord(action, None if result["stolen"] is None else RESOURCES_BY_NAME[result["stolen"]])
            elif action.action_type == ActionType.PLAY_MONOPOLY:
                self.assertEqual(result["taken"], encoder.monopoly_taken(action))
            game.execute(action, action_record=forced)
        self.assertEqual(messages[-1]["state"], encoder.state())

    def test_a_recorded_game_replays_from_its_log(self):
        for seed in (3, 7):
            random.seed(seed)
            messages = recorded(seed)
            self.assertEqual(["before"], [m["type"] for m in messages[:1]])
            self.assertEqual("after", messages[-1]["type"])
            # The bots' own randomness differs in the replay; only the log decides
            random.seed(seed + 1000)
            self.replay(messages, seed)

    def test_every_move_type_is_recorded_with_its_result(self):
        random.seed(11)
        messages = [m for seed in range(11, 15) for m in recorded(seed)]
        steps = [m for m in messages if m["type"] == "step"]
        kinds = Counter(m["action"][0] for m in steps)
        for kind in ("roll", "end_turn", "build_settlement", "build_road", "build_city", "buy_development", "play_knight",
                     "move_robber", "discard", "bank_trade"):
            self.assertIn(kind, kinds)
        for step in steps:
            kind, result = step["action"][0], step["result"]
            if kind == "roll":
                self.assertTrue(all(1 <= d <= 6 for d in result["dice"]))
            elif kind == "buy_development":
                self.assertIn(result["card"], CARDS_BY_NAME)
            elif kind == "move_robber":
                self.assertEqual(step["action"][1]["victim"] is None, result["stolen"] is None)
            # Every move was legal when it was made
            decide = next(m for m in messages if m["type"] == "decide" and m["game_id"] == step["game_id"] and m["n"] == step["n"])
            self.assertIn(step["action"], decide["legal"])


if __name__ == "__main__":
    unittest.main()
