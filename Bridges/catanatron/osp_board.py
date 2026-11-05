"""
Catanatron's board in the Open Seat Protocol's keys (see ../PROTOCOL.md): hexes in axial coordinates [q, r], an
intersection as its three hexes, a path as its two intersections, a harbor on a path.

Catanatron places tiles at cube coordinates (x, y, z) with east = (1, -1, 0) and south-east = (0, -1, 1)
(catanatron/models/coordinate_system.py); the protocol's q grows to the east and r to the south-east, so q = x and
r = z. Every tile, water and harbors included, carries its own node ids, so an intersection's three hexes are the tiles
that share its node id. Nothing here is copied from another engine: the keys follow from Catanatron's map alone.
"""

from collections import defaultdict

from catanatron.models.enums import BRICK, ORE, SHEEP, WHEAT, WOOD
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS
from catanatron.models.tiles import LandTile, Port

# SGN's tile codes by Catanatron's resource (None is the desert)
TILE_CODES = {WOOD: "FO", BRICK: "H", SHEEP: "P", WHEAT: "F", ORE: "M", None: "D"}
# SGN's harbor codes by the resource of a 2:1 harbor (None is a 3:1 harbor)
HARBOR_CODES = {None: "HR", WOOD: "HRL", BRICK: "HRB", SHEEP: "HRW", WHEAT: "HRG", ORE: "HRO"}


def axial(coordinate):
    """The protocol's hex [q, r] of a Catanatron cube coordinate (x, y, z)"""
    x, _, z = coordinate
    return (x, z)


def node_keys(catan_map):
    """Every land node's key: the sorted three hexes (land, water or harbor) that share its node id"""
    hexes_at = defaultdict(set)
    for coordinate, tile in catan_map.tiles.items():
        for node_id in tile.nodes.values():
            hexes_at[node_id].add(axial(coordinate))
    return {node_id: tuple(sorted(hexes_at[node_id])) for node_id in catan_map.land_nodes}


def path_key(keys, edge):
    """A Catanatron edge (two node ids) as the protocol's path: its two intersections, sorted"""
    a, b = edge
    return tuple(sorted((keys[a], keys[b])))


def path_keys(catan_map, keys):
    """The paths of every edge of a land tile"""
    return {path_key(keys, edge) for tile in catan_map.land_tiles.values() for edge in tile.edges.values()}


def harbors(catan_map, keys):
    """Each harbor's path and SGN code: the two nodes on the side of the harbor tile that its direction names"""
    result = []
    for tile in catan_map.tiles.values():
        if isinstance(tile, Port):
            a, b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            result.append((path_key(keys, (tile.nodes[a], tile.nodes[b])), HARBOR_CODES[tile.resource]))
    return sorted(result)


def shape(catan_map):
    """The board's shape as the protocol's reference file holds it (../osp/classic-board.json), every list sorted"""
    keys = node_keys(catan_map)
    land = sorted(axial(c) for c, tile in catan_map.tiles.items() if isinstance(tile, LandTile))
    sea = sorted(axial(c) for c, tile in catan_map.tiles.items() if not isinstance(tile, LandTile))
    return {
        "land": land,
        "sea": sea,
        "intersections": sorted(keys.values()),
        "paths": sorted(path_keys(catan_map, keys)),
        "harbors": sorted(path for path, _ in harbors(catan_map, keys)),
    }


def board(catan_map, robber_coordinate):
    """The protocol's board: every hex with its SGN tile code and number, the harbors, the robber"""
    keys = node_keys(catan_map)
    tiles = []
    for coordinate, tile in sorted(catan_map.tiles.items(), key=lambda item: axial(item[0])):
        if isinstance(tile, LandTile):
            tiles.append({"tile": list(axial(coordinate)), "tCode": TILE_CODES[tile.resource], "rNumber": tile.number})
        else:
            tiles.append({"tile": list(axial(coordinate)), "tCode": "S", "rNumber": None})
    return {
        "tiles": tiles,
        "harbors": [{"path": [[list(h) for h in end] for end in path], "hCode": code} for path, code in harbors(catan_map, keys)],
        "robber": list(axial(robber_coordinate)),
    }
