"""
Block encoders in front of the value net's MLP (StateValueNet with blocks): the flat feature vector is split into its
board elements and players, each kind runs through one shared encoder, and the encodings go into the trunk in slot order
together with the global features.

- tiles: the 19 land tiles (type one-hot, normalized number, present flag) plus whether the robber stands on the tile
- intersections: building and owner one-hots plus the port type one-hot of a coastal intersection (zeros elsewhere)
- edges: road and owner one-hots
- players: the player block, its derived and endgame features, its awaited discard and whether it is to act (with the
  view layout also its belief block)
- global: every remaining feature (settings, turn, robber, banks, the hidden development cards of the view layout); the
  water and non-playable tile slots are constant and left out

A shared encoder sees one element at a time, so what it learns about a tile or a player holds for every slot (an MLP on
the one-hot vector has to learn it per slot); the trunk then mixes the encodings as before. Inputs and outputs of the
exported net stay the same, so the C# side needs no change.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DatasetProcessing"))
from state_to_vector import (BASE_FEATURE_DIM, CLASSIC_INTERSECTION_TILES, CLASSIC_MAP_WIDTH, COASTAL_INTERSECTIONS,  # noqa: E402
                             FEATURE_DIM, LEGACY_FEATURE_DIM, PLAYER_BLOCK_SIZE, PLAYER_BLOCK_STARTS, PLAYER_COUNT,
                             PORT_TYPES, VIEW_FEATURE_DIM, _INTERSECTION_SIZE, _INTERSECTION_START, _TILE_COUNT, _TILE_SIZE,
                             _TILE_START)

LAND_TILES = sorted({tile for tiles in CLASSIC_INTERSECTION_TILES for tile in tiles})
_TURN_PLAYER, _TURN_DISCARDS = 3, 12                # one-hot of the player to act, awaited discards per player
_ROBBER_X, _ROBBER_Y = 18, 19
_EDGE_START, _EDGES, _EDGE_SIZE = _INTERSECTION_START + 54 * _INTERSECTION_SIZE, 72, 7
_INTERSECTIONS = 54
_DERIVED_GROUPS = (5, 5, 1, 5, 4, 4, 4, 4, 7)       # per player in each group of derived features, as in augmentation.py
_ENDGAME_PER_PLAYER, _BELIEF_PER_PLAYER = 5, 5


def block_indices(in_dim: int) -> dict:
    """Source feature of every input of each block (in_dim stands for a constant 0), and the land tiles' map indices"""
    if in_dim not in (FEATURE_DIM, VIEW_FEATURE_DIM):
        raise ValueError(f"block encoders need the current layout ({FEATURE_DIM} or {VIEW_FEATURE_DIM} features), not {in_dim}")
    zero = in_dim

    tiles = [[_TILE_START + t * _TILE_SIZE + j for j in range(_TILE_SIZE)] for t in LAND_TILES]
    coastal = {intersection: c for c, intersection in enumerate(COASTAL_INTERSECTIONS)}
    ports = len(PORT_TYPES)
    intersections = [[_INTERSECTION_START + i * _INTERSECTION_SIZE + j for j in range(_INTERSECTION_SIZE)]
                     + ([LEGACY_FEATURE_DIM + coastal[i] * ports + j for j in range(ports)] if i in coastal else [zero] * ports)
                     for i in range(_INTERSECTIONS)]
    edges = [[_EDGE_START + e * _EDGE_SIZE + j for j in range(_EDGE_SIZE)] for e in range(_EDGES)]

    endgame_start = LEGACY_FEATURE_DIM + len(COASTAL_INTERSECTIONS) * ports
    players = []
    for p in range(PLAYER_COUNT):
        features = list(range(PLAYER_BLOCK_STARTS[p], PLAYER_BLOCK_STARTS[p] + PLAYER_BLOCK_SIZE))
        start = BASE_FEATURE_DIM
        for size in _DERIVED_GROUPS:
            features += [start + p * size + j for j in range(size)]
            start += PLAYER_COUNT * size
        features += [endgame_start + p * _ENDGAME_PER_PLAYER + j for j in range(_ENDGAME_PER_PLAYER)]
        features += [_TURN_DISCARDS + p, _TURN_PLAYER + p]
        if in_dim == VIEW_FEATURE_DIM:
            features += [FEATURE_DIM + p * _BELIEF_PER_PLAYER + j for j in range(_BELIEF_PER_PLAYER)]
        players.append(features)

    used = {i for block in (tiles, intersections, edges, players) for element in block for i in element}
    constant_tiles = {_TILE_START + t * _TILE_SIZE + j for t in range(_TILE_COUNT) if t not in LAND_TILES for j in range(_TILE_SIZE)}
    global_features = [i for i in range(in_dim) if i not in used and i not in constant_tiles]
    return dict(tiles=tiles, intersections=intersections, edges=edges, players=players, global_features=global_features)


def _encoder(inputs: int, outputs: int) -> nn.Module:
    return nn.Sequential(nn.Linear(inputs, 2 * outputs), nn.GELU(), nn.Linear(2 * outputs, outputs))


class BlockEncoder(nn.Module):
    def __init__(self, in_dim: int, tile: int = 32, intersection: int = 16, edge: int = 8, player: int = 128):
        super().__init__()
        indices = block_indices(in_dim)
        for name in ("tiles", "intersections", "edges", "players", "global_features"):
            self.register_buffer(name, torch.tensor(indices[name], dtype=torch.long), persistent=False)
        self.register_buffer("land_tiles", torch.tensor(LAND_TILES, dtype=torch.long), persistent=False)
        # One more input per tile: whether the robber stands on it
        self.tile = _encoder(self.tiles.shape[1] + 1, tile)
        self.intersection = _encoder(self.intersections.shape[1], intersection)
        self.edge = _encoder(self.edges.shape[1], edge)
        self.player = _encoder(self.players.shape[1], player)
        self.out_dim = (len(LAND_TILES) * tile + _INTERSECTIONS * intersection + _EDGES * edge + PLAYER_COUNT * player
                        + len(self.global_features))

    def inputs(self, x: torch.Tensor, raw: torch.Tensor):
        """The inputs of each block [batch, elements, features] and the global features [batch, features]; x: the
        (normalized) features [batch, in_dim], raw: the same unnormalized, for the robber's map coordinates"""
        padded = nn.functional.pad(x, (0, 1))
        robber = torch.round(raw[:, _ROBBER_Y].float()) * CLASSIC_MAP_WIDTH + torch.round(raw[:, _ROBBER_X].float())
        on_robber = (self.land_tiles[None, :].to(robber.dtype) == robber[:, None]).to(x.dtype)
        tiles = torch.cat([padded[:, self.tiles], on_robber[..., None]], dim=-1)
        return tiles, padded[:, self.intersections], padded[:, self.edges], padded[:, self.players], x[:, self.global_features]

    def forward(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        tiles, intersections, edges, players, global_features = self.inputs(x, raw)
        parts = [self.tile(tiles), self.intersection(intersections), self.edge(edges), self.player(players)]
        return torch.cat([part.flatten(1) for part in parts] + [global_features], dim=1)
