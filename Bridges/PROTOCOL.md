# Open Seat Protocol (OSP)

Draft, version 1. One protocol through which a bot of any CATAN engine plays a seat under the referee of any other
engine, with the rules as an option of each game.

## Not another standard

OSP invents as little as possible:
- **Lifecycle** from Catanatron's protocol for bots in other languages (version 1, `catanatron/protocol.py`): one JSON
  object per line, the messages `hello`, `before`, `decide`, `step` and `after`, only `hello` and `decide` answered, and
  an answer that must be one of the legal moves the referee listed.
- **Board coordinates** from the Settlers Game Notation (SGN 1.0, github.com/dnmfarrell/Settlers-Game-Notation): hexes in
  axial coordinates including the sea, an intersection as its three hexes, a path (edge) as its two intersections,
  harbors on paths, SGN's tile and harbor codes.
- **New in OSP:** the ruleset as explicit options, an engine-neutral move list and state view. Catanatron's version 1
  carries its own node numbers and rules.

The goal is one adapter per engine, not one per pair of engines. If Catanatron takes OSP up as its next protocol
version, its adapter disappears.

## Roles and transport

- **Referee:** holds the true game and runs it (a Hexfarer match or Arena duel, a Catanatron game, a JSettlers server
  through its adapter).
- **Agent:** a bot process that plays one or more seats. A program that plays seats over OSP is an **Open Seat Agent**:
  the adapters in this folder make Catanatron's and JSettlers' bots Open Seat Agents, and Hexfarer's agents play as one
  under other engines' referees.
- **Transport:** UTF-8 JSON objects, one per line, over the agent's standard input and output (or any line-based stream).
  The referee speaks first. Nothing but protocol lines on standard output; logs go to standard error.

## Messages

| Message | From | Answer | Content |
|---|---|---|---|
| `hello` | referee | `hello` | protocol name and version |
| `before` | referee | none | a new game: seat, ruleset, board, the seat's view of the start |
| `decide` | referee | `decide` | the seat's view now and its legal moves |
| `step` | referee | none | a move someone made, with its result as the seat may see it (only to agents that asked to observe) |
| `after` | referee | none | the end of a game |

```
→ {"type":"hello","protocol":"osp","version":1}
← {"type":"hello","protocol":"osp","version":1,"name":"hexfarer-ismcts","rulesets":["standard","catanatron"],
   "observe":true,"needs":"view"}
→ {"type":"before","game_id":"g17","seat":2,"players":4,"ruleset":{"preset":"standard","player_trading":"rulebook"},
   "board":{...},"state":{...}}
→ {"type":"step","game_id":"g17","n":143,"seat":1,"action":["move_robber",{"hex":[0,-1],"victim":2}],
   "result":{"stolen":"ore"}}
→ {"type":"decide","game_id":"g17","n":144,"seat":2,"state":{...},"legal":[["roll"],["play_knight"]]}
← {"type":"decide","game_id":"g17","n":144,"action":["roll"]}
→ {"type":"after","game_id":"g17","winner":2,"vp":[7,8,10,6],"truncated":false}
```

- **`hello` answer:**
  - `rulesets`: the presets the agent can play.
  - `options`: optionally, the values of single options it supports beyond them.
  - `observe`: whether it wants `step`.
  - `needs`: `view` (default; only what its seat sees), `world` (a full game drawn by the referee from the seat's view)
    or `truth` (every hand, for open-hand agents).
  - A referee that can't serve a game the agent supports, or won't hand out what it needs, doesn't start the game. It
    labels every game with what each seat received.
- **`n`:** counts the moves of the game, so answers and steps can't be mixed up.
- **`decide` answer:** a copy of one entry of `legal`. The referee matches it against its own list, as Catanatron's does,
  so an encoding that only looks right can't slip through.
- **Errors:** an answer that isn't legal, or a version the other side doesn't speak, ends the game for that agent with
  `{"type":"error","message":...}`.
- **Timeouts:** only if the ruleset sets `decide_timeout_ms`; evaluations run without one.

## Coordinates (SGN)

- **Hex:** `[q, r]` axial, pointy-top. `q` grows to the east, `r` to the south-east, and the center hex is `[0, 0]`. The
  classic board is the land within 2 steps of the center and the sea ring at 3.
- **Intersection:** the three hexes that meet there, sea included, e.g. `[[0,-1],[0,0],[1,-1]]`. Sent sorted by
  `(q, r)`, compared as a set. SGN logs list them clockwise; converting is a sort.
- **Path (edge):** its two intersections, sorted.
- **Engine conversions:**
  - Hexfarer: cube `(q, r, s)` without `s`.
  - Catanatron: cube `(x, y, z)` is Hexfarer's `(q, s, r)`, so `[x, z]`.
  - JSettlers: hex, node and edge numbers of its classic layout, each converted through its adjacency, never through
    hand-written tables.

## Board (in `before`)

```
{"tiles":[{"tile":[0,0],"tCode":"F","rNumber":11},...],          // SGN: H hills, FO forest, M mountains, F fields,
                                                                 //      P pastures, D desert, S sea
 "harbors":[{"path":[[[...]],[[...]]],"hCode":"HRW"},...],       // SGN: HR generic (3:1), HRB HRG HRL HRO HRW (2:1)
 "robber":[q,r]}
```

## Moves

A move is `[type]` or `[type, value]`. Cards are named `brick`, `lumber`, `ore`, `grain` and `wool`, and `knight`,
`road_building`, `year_of_plenty`, `monopoly` and `victory_point`. Card sets are objects like `{"ore":2,"grain":1}`.

| Move | Value | Result in `step` |
|---|---|---|
| `roll` | | `{"dice":[a,b]}` |
| `build_settlement` | intersection | |
| `build_road` | path | |
| `build_city` | intersection | |
| `buy_development` | | `{"card":name}` for the buyer, `{"card":null}` for the others |
| `play_knight` | | |
| `move_robber` | `{"hex":[q,r],"victim":seat or null}` | `{"stolen":card}` for thief and victim, `{"stolen":"unknown"}` for the others, `null` if nothing |
| `play_road_building` | | (the roads follow as `build_road`) |
| `play_year_of_plenty` | `["ore","grain"]` (one card if the bank has only one) | |
| `play_monopoly` | resource | `{"taken":{seat:count,...}}` |
| `discard` | card set (the whole discard as one decision) | |
| `bank_trade` | `{"give":set,"get":set}` | |
| `end_turn` | | |
| `offer` | `{"give":set,"get":set,"to":[seats]}` | |
| `respond` | `{"accept":true or false}` | |
| `counter` | `{"give":set,"get":set}`, in the offerer's terms | |
| `confirm` | `{"partner":seat}` | |
| `cancel` | | |

The setup placements are `build_settlement` and `build_road` in the setup prompts. The trading moves exist only where
the ruleset allows trading between players.

## State view (in `before` and `decide`)

```
{"prompt":"main",             // setup_settlement, setup_road, roll, main, discard, move_robber, answer_offer, pick_partner
 "current":1, "turn":37, "dice":[3,4],
 "robber":[q,r],
 "buildings":{"settlements":[[seat,intersection],...],"cities":[[seat,intersection],...],"roads":[[seat,path],...]},
 "bank":{"brick":12,...,"development":11},
 "players":[{"seat":0,"public_vp":5,"hand_size":7,"development_size":2,
             "played":{"knight":2,"road_building":0,"year_of_plenty":1,"monopoly":0},
             "longest_road":6,"has_longest_road":true,"has_largest_army":false,
             "pieces_left":{"roads":7,"settlements":2,"cities":3},
             "hand":{"ore":2,...} or null, "development":["knight","victory_point"] or null},
            ...],
 "offer":{"from":1,"give":set,"get":set,"to":[0,2],"answers":{"0":"accept"}} or null}
```

- `hand` and `development` are null for other seats under `view`.
- A `world` fills them with a game the referee drew from the seat's view and says so (`"drawn":true`).
- Under `truth` they are the real ones.

## Ruleset

The ruleset is a set of options; presets name common sets. `before` carries the preset and any options that differ from
it.

| Option | `standard` (the rulebook) | `catanatron` |
|---|---|---|
| `victory_points` | 10 | 10 |
| `discard_limit` (hand size above which half is discarded) | 7 | 7 |
| `player_trading` | `rulebook` (with the player on turn, after the roll) | `none` |
| `counteroffers` | true | false |
| `bank_shortage` | `rulebook` (nobody gets a short resource, unless a single player is owed it: they get the rest) | `nobody` |
| `win_timing` | `own_turn` | `any_time` |
| `robber_victims` | `adjacent` (any adjacent player, even without cards) | `with_cards` |
| `bank_rates` | `all` (4:1 also with a harbor) | `best_only` |
| `road_building` | `optional` (unplaced roads can be given up) | `forced` |
| `turn_limit` (player turns, then a draw) | none | 1000 |
| `decide_timeout_ms` | none | none |

New options get a default that matches the rulebook, so older agents keep playing the `standard` preset. An engine
that can't play an option value says so in `hello` by leaving the preset out.

## Versions

`version` is raised only when a change would break agents written for the old shape. New options and moves for rules
an agent didn't declare don't count: a referee never sends them to an agent that doesn't support them.
