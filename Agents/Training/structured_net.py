"""
Value net v2.1: block encoders with the board's structure, in front of the v1 trunk (see block_encoder.py for v2).

1. Relative seats: the feature vector is rotated so the player to act is player 0 (owners of buildings and roads, the
   player blocks and their derived features, see augmentation.rotation_index), the outputs are rotated back. The
   player rotation is then a symmetry of the net instead of something learned from augmentation; the absolute seat of
   the player to act stays in the global features (the setup's snake order depends on it). The input normalization is
   made the same for all players (symmetrize_normalization), so normalizing and rotating commute.
2. Neighbor features, gathered along the fixed board graph: a tile also sees the buildings on its corners, an
   intersection the production of its tiles (resource times pips, and what the robber blocks), the buildings on its
   neighbor intersections and the owners of its roads, an edge the buildings on its ends and the owners of the roads
   next to it. So the shared encoders can learn local rules (distance rule, a spot's yield, roads reaching a
   settlement) once for every slot.
3. Per-player board summaries: each player's encoding also gets the sum of the encodings of the intersections and
   edges it owns.
4. Per-player heads: the value logit, the placements and the VP share of a player come from one head shared by all
   players, from the trunk's features and that player's encoding.

Inputs and outputs are those of the MLP, so the exported net is a drop-in for the C# search.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from augmentation import ROTATION_INDEX
from block_encoder import LAND_TILES, _ROBBER_X, _ROBBER_Y, _TURN_PLAYER, block_indices
from state_to_vector import CLASSIC_EDGE_INTERSECTIONS, CLASSIC_INTERSECTION_TILES, CLASSIC_MAP_WIDTH, PLAYER_COUNT

_TILE_TYPES = 8                                     # type one-hot (HEX_TYPES), then the normalized number and the present flag
_BUILDINGS, _OWNERS = 3, 5                          # building one-hot, owner one-hot over [none, players]
_HEAD_WIDTH = 256


def geometry():
    """Incidence matrices of the classic board: intersections x land tiles, edges x intersections, and the neighbor
    intersections and edges (sharing an edge or an end)"""
    tile_slot = {tile: slot for slot, tile in enumerate(LAND_TILES)}
    intersection_tiles = np.zeros((len(CLASSIC_INTERSECTION_TILES), len(LAND_TILES)), dtype=np.float32)
    for i, tiles in enumerate(CLASSIC_INTERSECTION_TILES):
        intersection_tiles[i, [tile_slot[t] for t in tiles]] = 1
    edge_intersections = np.zeros((len(CLASSIC_EDGE_INTERSECTIONS), len(CLASSIC_INTERSECTION_TILES)), dtype=np.float32)
    for e, ends in enumerate(CLASSIC_EDGE_INTERSECTIONS):
        edge_intersections[e, list(ends)] = 1
    intersection_neighbors = (edge_intersections.T @ edge_intersections > 0).astype(np.float32)
    np.fill_diagonal(intersection_neighbors, 0)
    edge_neighbors = (edge_intersections @ edge_intersections.T > 0).astype(np.float32)
    np.fill_diagonal(edge_neighbors, 0)
    return intersection_tiles, edge_intersections, intersection_neighbors, edge_neighbors


def _aggregate(values: torch.Tensor, matrix_t: torch.Tensor) -> torch.Tensor:
    """Sum over the neighbors given by matrix (targets x sources, passed transposed) of values [batch, sources, features],
    as one matrix product with the batch folded into the rows"""
    return (values.transpose(1, 2) @ matrix_t).transpose(1, 2)


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Module:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class StructuredValueNet(nn.Module):
    def __init__(self, in_dim, hidden=2048, aux_heads=False, normalize_inputs=False, policy_types=0, policy_vocabulary=0,
                 policy_dim=64, widths=(32, 16, 8, 128), relative_seats=True):
        super().__init__()
        # Without relative seats (an ablation) every state is encoded in absolute seat order like in the MLP
        self.relative_seats = relative_seats
        if normalize_inputs:
            self.register_buffer("input_mean", torch.zeros(in_dim))
            self.register_buffer("input_scale", torch.ones(in_dim))
        else:
            self.input_mean = None
            self.input_scale = None

        indices = block_indices(in_dim)
        for name in ("tiles", "intersections", "edges", "players", "global_features"):
            self.register_buffer(name, torch.tensor(indices[name], dtype=torch.long), persistent=False)
        self.register_buffer("land_tiles", torch.tensor(LAND_TILES, dtype=torch.long), persistent=False)
        # Source index per feature that makes player a player 0, one row per a (the rotation by -a)
        rotation = np.stack([ROTATION_INDEX[(PLAYER_COUNT - a) % PLAYER_COUNT, :in_dim] for a in range(PLAYER_COUNT)])
        self.register_buffer("rotation", torch.tensor(rotation, dtype=torch.long), persistent=False)
        intersection_tiles, edge_intersections, intersection_neighbors, edge_neighbors = geometry()
        for name, matrix in (("intersection_tiles", intersection_tiles), ("tile_intersections", intersection_tiles.T),
                             ("edge_intersections", edge_intersections), ("intersection_edges", edge_intersections.T),
                             ("intersection_neighbors", intersection_neighbors), ("edge_neighbors", edge_neighbors)):
            self.register_buffer(name, torch.tensor(np.ascontiguousarray(matrix)), persistent=False)

        tile, intersection, edge, player = widths
        pairs = _BUILDINGS * _OWNERS
        self.tile = _mlp(self.tiles.shape[1] + 1 + pairs, 4 * tile, tile)
        self.intersection = _mlp(self.intersections.shape[1] + 2 * _TILE_TYPES + pairs + _OWNERS, 4 * intersection, intersection)
        self.edge = _mlp(self.edges.shape[1] + pairs + _OWNERS, 4 * edge, edge)
        self.player = _mlp(self.players.shape[1], 2 * player, player)
        self.player_mix = _mlp(player + intersection + edge, 2 * player, player)

        trunk_dim = (len(LAND_TILES) * tile + self.intersections.shape[0] * intersection + self.edges.shape[0] * edge
                     + PLAYER_COUNT * player + len(self.global_features) + PLAYER_COUNT)
        # Zero padding to a multiple of 64: other widths fall back to much slower matrix kernels
        self.trunk_padding = -trunk_dim % 64
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim + self.trunk_padding, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(0.2),
        )
        # Per player: value logit, 4 placement logits, VP share logit
        self.head_trunk = nn.Linear(hidden // 2, _HEAD_WIDTH)
        self.head_player = nn.Linear(player, _HEAD_WIDTH)
        self.head_out = nn.Linear(_HEAD_WIDTH, 1 + PLAYER_COUNT + 1)
        self.aux = nn.ModuleDict({"rounds_left": nn.Linear(hidden // 2, 1)}) if aux_heads else None
        self.placements_trained = False

        self.policy_query = nn.Linear(hidden // 2, policy_types * policy_dim) if policy_types > 0 else None
        self.policy_embeddings = nn.Embedding(policy_vocabulary, policy_dim) if policy_types > 0 else None
        self.policy_types, self.policy_dim = policy_types, policy_dim

    def symmetrize_normalization(self):
        """Mean and scale averaged over the players' copies of each feature, so they are the same for every player"""
        if self.input_mean is not None and self.relative_seats:
            self.input_mean.copy_(self.input_mean[self.rotation].mean(dim=0))
            self.input_scale.copy_(self.input_scale[self.rotation].mean(dim=0))

    def _normalize(self, x):
        return x if self.input_mean is None else (x - self.input_mean) * self.input_scale

    def _acting(self, raw):
        """Index of the player to act [batch] (player 0 if none is set)"""
        return raw[:, _TURN_PLAYER:_TURN_PLAYER + PLAYER_COUNT].float().argmax(dim=1)

    def encode(self, x, raw=None):
        """The trunk's input, the players' encodings in relative order and the frame: the absolute seat of relative
        player 0 (the player to act, or 0 without relative seats)"""
        raw = x if raw is None else raw
        normalized = self._normalize(x)
        acting = self._acting(raw)
        frame = acting if self.relative_seats else torch.zeros_like(acting)
        relative = normalized.gather(1, self.rotation[frame])
        padded = F.pad(relative, (0, 1))
        tiles, intersections, edges = padded[:, self.tiles], padded[:, self.intersections], padded[:, self.edges]
        players = padded[:, self.players]

        # Production per tile and resource (type times pips), from the unnormalized numbers; tiles have no owner
        raw_tiles = raw[:, self.tiles].float()
        number = torch.round(raw_tiles[..., _TILE_TYPES] * 10 + 2)
        pips = (6 - (7 - number).abs()).clamp(min=0) * raw_tiles[..., _TILE_TYPES + 1] / 36
        production = raw_tiles[..., :_TILE_TYPES] * pips[..., None]
        robber = torch.round(raw[:, _ROBBER_Y].float()) * CLASSIC_MAP_WIDTH + torch.round(raw[:, _ROBBER_X].float())
        on_robber = (self.land_tiles[None, :].to(robber.dtype) == robber[:, None]).float()
        production, blocked, on_robber = (production.to(normalized.dtype), (production * on_robber[..., None]).to(normalized.dtype),
                                          on_robber.to(normalized.dtype))

        # Building and owner of each intersection as one joint one-hot, the owner of each edge's road
        buildings = (intersections[..., :_BUILDINGS, None] * intersections[..., None, _BUILDINGS:_BUILDINGS + _OWNERS]).flatten(2)
        roads = edges[..., 2:2 + _OWNERS]

        tile_in = torch.cat([tiles, on_robber[..., None], _aggregate(buildings, self.intersection_tiles)], dim=-1)
        intersection_in = torch.cat([intersections, _aggregate(production, self.tile_intersections), _aggregate(blocked, self.tile_intersections),
                                     _aggregate(buildings, self.intersection_neighbors), _aggregate(roads, self.edge_intersections)], dim=-1)
        edge_in = torch.cat([edges, _aggregate(buildings, self.intersection_edges), _aggregate(roads, self.edge_neighbors)], dim=-1)
        tile_code, intersection_code, edge_code = self.tile(tile_in), self.intersection(intersection_in), self.edge(edge_in)

        # Each player's encoding plus the sum of what it owns on the board
        owned_intersections = intersections[..., _BUILDINGS + 1:_BUILDINGS + _OWNERS].transpose(1, 2) @ intersection_code
        owned_edges = edges[..., 3:2 + _OWNERS].transpose(1, 2) @ edge_code
        player_code = self.player(players)
        player_code = player_code + self.player_mix(torch.cat([player_code, owned_intersections, owned_edges], dim=-1))

        trunk_in = torch.cat([tile_code.flatten(1), intersection_code.flatten(1), edge_code.flatten(1), player_code.flatten(1),
                              relative[:, self.global_features],
                              (acting[:, None] == torch.arange(PLAYER_COUNT, device=acting.device)[None, :]).to(relative.dtype)], dim=1)
        return trunk_in, player_code, frame

    def forward_all(self, x, raw=None):
        """Value logits and, if present, the outputs of the aux heads and the policy queries, in absolute seat order"""
        trunk_in, player_code, frame = self.encode(x, raw)
        features = self.trunk(F.pad(trunk_in, (0, self.trunk_padding)))
        per_player = self.head_out(F.gelu(self.head_trunk(features)[:, None, :] + self.head_player(player_code)))
        # Absolute player p is relative player p - frame
        source = (torch.arange(PLAYER_COUNT, device=frame.device)[None, :] - frame[:, None]) % PLAYER_COUNT

        def absolute(values):
            return values.gather(1, source.view(*source.shape, *[1] * (values.dim() - 2)).expand_as(values))

        outputs = {"logits": absolute(per_player[..., 0])}
        if self.aux is not None:
            outputs["placements"] = absolute(per_player[..., 1:1 + PLAYER_COUNT])
            outputs["vp_shares"] = absolute(per_player[..., 1 + PLAYER_COUNT])
            outputs["rounds_left"] = self.aux["rounds_left"](features).squeeze(-1)
        if self.policy_query is not None:
            outputs["policy_query"] = self.policy_query(features).view(-1, self.policy_types, self.policy_dim)
        return outputs

    def forward(self, x):
        return self.forward_all(x)["logits"]

    def policy_logits(self, query, rows, tokens, weights):
        """Logits of the moves, one after another, as in StateValueNet"""
        embedded = (self.policy_embeddings(tokens.clamp(min=0)) * (weights * (tokens >= 0))[..., None]).sum(dim=1)
        return (query[rows, tokens[:, 0].clamp(min=0)] * embedded).sum(dim=-1)
