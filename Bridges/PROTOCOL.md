# Open Seat Protocol (OSP)

Draft, version 2. One protocol through which a bot of any CATAN engine plays a seat under the referee of any other
engine, with the rules as an option of each game.

## Not another standard

OSP invents as little as possible:
- **Lifecycle** from Catanatron's protocol for bots in other languages (version 1, `catanatron/protocol.py`): one JSON
  object per line, a handshake, then `before`, `decide`, `step` and `after` per game, and an answer that must be one of
  the legal moves the referee listed.
- **Board coordinates** from the Settlers Game Notation (SGN 1.0, github.com/dnmfarrell/Settlers-Game-Notation): hexes in
  axial coordinates including the sea, an intersection as its three hexes, a path (edge) as its two intersections,
  harbors on paths, SGN's tile and harbor codes.
- **New in OSP:** the ruleset as explicit options, an engine-neutral move list and state view, a time budget per game,
  one way to report failures, decisions that send only a hash of the state, and one result record per game.
  Catanatron's version 1 carries its own node numbers and rules.

The goal is one adapter per engine, not one per pair of engines. If Catanatron takes OSP up as its next protocol
version, its adapter disappears.

## Roles and transport

- **Referee:** holds the true game and runs it (a Hexfarer match or Arena duel, a Catanatron game, a JSettlers server
  or a StacSettlers simulation through its adapter).
- **Agent:** a bot process that plays one or more seats. A program that plays seats over OSP is an **Open Seat Agent**:
  the adapters in this folder make Catanatron's and JSettlers' bots Open Seat Agents, and Hexfarer's agents play as one
  under other engines' referees.
- **Transport:** UTF-8 JSON objects, one per line, over the agent's standard input and output (or any line-based stream).
  The referee speaks first. Nothing but protocol lines on standard output; logs go to standard error. A referee may start
  an agent as a process per game and seat (Hexfarer's Arena does) or keep one for many games (Catanatron's driver does);
  then the handshake comes once and `before` to `after` repeats.
- **Seats:** numbered in turn order, seat 0 is the player who starts. An engine that draws who starts (JSettlers) numbers
  its players from that one.

## Messages

| Message | From | Answer | Content |
|---|---|---|---|
| `handshake` | referee | `handshake` | protocol name and version; the answer says what the agent plays and wants |
| `before` | referee | `ready` | a new game: seat, ruleset, budget, seed, board, the seat's state at the start |
| `decide` | referee | `decide` or `resync` | the seat's state (or only its hash), its legal moves and the trades it may propose |
| `step` | referee | none | a move someone made, with its result as the seat may see it (only to agents that observe) |
| `after` | referee | none | the end of a game, with its result record |
| `failure` | either | none | what went wrong; ends the game for the agent |

Every message that is answered may also be answered with `failure`.

```
→ {"type":"handshake","protocol":"osp","version":2}
← {"type":"handshake","protocol":"osp","version":2,"name":"hexfarer-ismcts","rulesets":["standard","catanatron"],
   "observe":true,"needs":"view","lean":true}
→ {"type":"before","game_id":"g17","seat":2,"players":4,"ruleset":{"preset":"standard","player_trading":"none"},
   "budget":{"decide_ms":5000,"game_ms":null,"threads":1,"gpu":false},"seed":1817,"board":{...},"state":{...}}
← {"type":"ready","game_id":"g17"}
→ {"type":"step","game_id":"g17","n":143,"seat":1,"action":["move_robber",{"hex":[0,-1],"victim":2}],
   "result":{"stolen":"ore"}}
→ {"type":"decide","game_id":"g17","n":144,"seat":2,"hash":"3fa0c9e1b27d4c58","legal":[["roll"],["play_knight"]]}
← {"type":"decide","game_id":"g17","n":144,"action":["roll"]}
→ {"type":"after","game_id":"g17","winner":2,"vp":[7,8,10,6],"truncated":false,"result":{...}}
```

### `handshake`

- The referee names the protocol and its version; the agent answers with the same, or with a `failure` (`version`).
  The versions have to match exactly.
- The answer:
  - `name`: the agent's name, for the results.
  - `rulesets`: the presets the agent can play (see `RULESET.md`).
  - `observe`: whether it wants `step`.
  - `needs`: `view` (default; only what its seat sees), `world` (a full game drawn by the referee from the seat's view)
    or `truth` (every hand, for open-hand agents).
  - `lean` (default false): whether `decide` may leave the state out and send only its hash. Only together with
    `observe`, and with `view` or `truth`.
- The referee serves `needs`, `observe` and `lean` as asked, or answers with a `failure` (`unsupported`) and starts no
  game. It never sends something else instead. It labels every game with what each seat received (the result record).
- Under `truth`, every `step` also carries `state`: the state before the move, with every hand. An agent that keeps its
  own copy of the game can then check it before following each move, and take over the referee's state where the two
  engines' rules differ (the referee decides the game). Under `world` a step carries the seat's view: worlds are drawn
  for decisions only.
- Under `truth` the results show everything (another seat's bought card, a steal between two others); otherwise as the
  seat sees them.
- JSettlers' robot sends steps; its server makes them from the players' requests, in the protocol's order (simultaneous
  discards from the lowest seat up, each with the state that order gives it). A change the server makes on its own (a
  slow player's turn ended) comes as no step; the next decision then carries the full state (see lean decisions).
- StacSettlers' robot sends steps the same way, from its server's requests. A trade its agents make in chat comes as
  the two steps it takes under `player_trading` `first_accept`: the offer of the player on turn to the partner, on
  exactly the terms that changed hands, and the partner's accept; the negotiation before it (offers declined, withdrawn
  or countered) makes no step. Where its robot trades in chat too (-Dosp.trade_in_chat), so do the seat's own trades,
  and a decision about an offer (`answer_offer`, `pick_partner`) shows the offer in its state though no step made it.

### `before` and `ready`

- `before` starts a game: `game_id`, `seat`, `players`, `ruleset`, `budget`, `seed`, `board` and `state`, the seat's
  state at the start as `needs` says.
- `seed`: a number from 0 to 2^31 - 1 for the agent's own randomness (drawn worlds, tie breaks). An agent that seeds
  from it plays the same game the same way, and the referee logs it, so every game replays exactly, as far as the
  agent's search depends on nothing else: a search with a time budget, or on several threads at once, also depends on
  timing (Hexfarer's agent replays exactly with an iteration budget and `--serial`).
- The agent takes the game with `{"type":"ready","game_id":"g17"}`, or turns it down with a `failure` (`unsupported`)
  whose `detail` names what it can't play: `{"preset":"catanatron"}` or `{"option":"road_building","value":"optional"}`.
  A referee sends only games of a preset the agent named, but an option can still be one it doesn't play.

### The budget

`"budget":{"decide_ms":5000,"game_ms":null,"threads":1,"gpu":false}` in every `before`:
- `decide_ms` (required): the time from sending `decide` until the referee reads the answer. Past it, the referee
  sends a `failure` (`timeout`). Agents keep their own margin for the transport.
- `game_ms` (or null): a clock for the whole game; each `decide` then carries `clock_ms`, the time the agent has left.
- `threads` (or null for no limit): the CPU threads the agent may use, for equal compute. A referee that pins the agent
  to cores enforces it; the others state it.
- `gpu` (true unless the game says otherwise): whether the agent may use a GPU.
- An agent that can't keep the budget (fewer threads than it uses, no GPU while its net runs on one) turns the game
  down instead of answering ready.
- The answers to `handshake` and `before` have 60 s, for starting a process and loading a net.
- Defaults, where whoever starts the game sets none: Hexfarer's referee 300 000 ms, Catanatron's driver 300 000 ms,
  JSettlers' robot 120 000 ms (its server also ends a robot's turn after its own limit), StacSettlers' robot 120 000 ms
  (its simulation should run with ForceEndTurns=false, else the server ends a turn after 30 s); any threads, a GPU
  allowed.

### `decide`, lean decisions and `resync`

- `decide` carries `game_id`, `n`, `seat`, `legal`, `hash`, and `state` unless the agent is lean; with a game clock
  also `clock_ms`; where the seat may propose a trade, `trades` (see trades between players).
- `hash`: the first 16 hex digits (lowercase) of the SHA-256 of the state's canonical JSON: keys sorted, no whitespace,
  UTF-8. The state holds only integers, booleans, null, lists, objects and strings of ASCII letters, digits and
  underscores, so its canonical form is unique. It is the hash of the state the agent would get: its view under `view`,
  the true state under `truth`.
- A lean agent keeps its own game from the steps and checks it against `hash`. If they differ, or it lost track, it
  answers `{"type":"resync","game_id":"g17","n":144}` and the referee sends the same decision again, with `state`.
- An agent that keeps its own game and can't make it the referee's again (from `state`) answers the decision with a
  `failure` (`internal`) rather than a move: a game played blind says nothing about the agent, and the failure goes
  into the result record. Hexfarer's agent first takes over what the referee's known rule differences change, else
  rebuilds its game from `state` (its fair agent's beliefs take the change in as a public edit); only a game that
  still differs then is lost. A board it doesn't write back as sent it turns down at `before` (`unsupported`).
- Lean saves something only where the agent's copy writes the referee's state exactly, in practice an agent of the
  referee's own engine (Hexfarer's agent under Hexfarer's referee). Engines fill the state's open fields differently
  (`turn`, `played`, an `offer` left out or null), so under another engine's referee every lean decision costs a
  resync.
- The referee may send `state` in any decision on its own (a forced resync), for example after a change no step
  explains. An agent that isn't lean gets `state` every time.
- The answer: `{"type":"decide","game_id":"g17","n":144,"action":...}`, a copy of one entry of `legal`. The referee
  matches it against its own list, as Catanatron's does, so an encoding that only looks right can't slip through. The
  one exception is a trade from `trades`, whose terms the agent writes; the referee checks those by the rules below.
- `n` counts the moves of the game, so answers and steps can't be mixed up.

### `after` and the result record

`after` ends a game for the agent: `game_id`, `winner` (a seat or null), `vp` per seat, `truncated` (the game was cut
short: an undo, a reset, a shutdown, a failure) and `result`, the game's result record. The referee writes the same
record into its log, the one result line of every referee:

```
"result":{"seed":1817,"moves":512,"turns":63,
          "seats":[{"seat":0,"agent":"hexfarer-ismcts","needs":"view","observe":true,"lean":true,
                    "decisions":118,"time_ms":5230,"max_ms":410,"resyncs":0,"fallbacks":0,"failures":[]},
                   {"seat":1,"agent":null,...},...]}
```

- `agent` is null for a seat the referee's own players play; their counts are the referee's own, where it has them.
- `turns`: the player turns after the setup, also those the server ended on its own.
- `time_ms` and `max_ms`: the total and the longest time of the seat's decisions, as the referee measured them.
- `fallbacks`: moves the referee chose for the seat after a failure, where it keeps a seat going.
- `failures`: each as `{"n":144,"code":"timeout","by":"referee","message":"..."}`.

### `failure`

`{"type":"failure","game_id":"g17","n":144,"code":"illegal","message":"...","detail":{...}}`. `game_id` is null before
a game, `n` outside a decision, `detail` optional. After a failure neither side sends more for that game; a failure
without a game ends the connection. The referee records every failure in the game's result record.

| Code | Sent by | When |
|---|---|---|
| `version` | either | the other side's protocol or version isn't one it speaks |
| `unsupported` | either | the referee can't serve the agent's `needs`, `observe` or `lean`; the agent can't play the game's preset or an option |
| `illegal` | referee | an answer that isn't one of the listed legal moves, or has another `n` |
| `malformed` | either | a line that isn't JSON, or a message without a field it needs |
| `timeout` | referee | no answer within the budget |
| `ended` | referee (recorded only) | the agent's process ended or closed its output, so it can't be told |
| `internal` | either | the side's own error, say an exception in the agent's search |

What happens to the seat is the referee's choice, recorded with the result: a Hexfarer match pauses (resuming starts
the agent again), the Arena drops the game, Catanatron's driver ends the game, and JSettlers' and StacSettlers' robots
play on with the first legal move (as `fallbacks`).

A game whose agent's seat has a failure or `fallbacks` in its result record says nothing about the agent's strength:
tallies leave it out of the win rate and count it beside it (Catanatron's `analyze.py` leaves out its whole map, which
`resume.py` plays again; JSettlers' `summary.py`; StacSettlers' `play.py`, which plays such games again until a run's
count holds). Many of them mean a broken setup,
so a run stops past a number of failed games (default 3, or one in a hundred of the run's games): Catanatron's driver
with `--max-failed`, JSettlers' server and StacSettlers' simulation with `-Dosp.max_failed_games`, where they write a
line starting with `osp: stopped:` to standard error and exit (code 3).

### Trades between players

Where the ruleset allows them (`player_trading` `rulebook`), trades are decisions like any other:
1. The player on turn, after the roll, may play `offer` (`{"give":set,"get":set,"to":[seats]}`), at most `offer_limit`
   times per turn.
2. Each seat the offer goes to then gets a decision with the prompt `answer_offer`, one after another starting from the
   offerer's left: `respond` (`{"accept":true or false}`) or, with `counteroffers`, `counter` in the offerer's terms.
   The state's `offer` shows the answers so far.

Offers and counteroffers can be any cards, so `legal` can't list them. A decision names them in `trades` instead,
only where the seat may propose one now: `{"offer":{"to":[1,2,3],"left":2}}` (the seats it may address, and the offers
it has left this turn, null without a limit), and `{"counter":{}}` while it answers an offer and counteroffers are
allowed. The answer is then a copy of an entry of `legal` or such a trade with its terms:
`["offer",{"give":{"wool":2},"get":{"ore":1},"to":[1,3]}]` or `["counter",{"give":{"ore":1},"get":{"grain":1}}]`.
The referee takes it if its terms are valid, and answers anything else with `illegal`:
- `give` and `get` are card sets of resources, each with at least one card, and no resource is on both sides.
- An offer's `give` is within the seat's hand, and `to` names at least one seat, each listed in `trades` and once.
- A counteroffer is in the offerer's terms, so its `get` (what the offerer would get) is within the seat's hand.

An agent that doesn't know `trades` ignores it, plays only what `legal` lists, and never proposes a trade.
3. When all have answered, the offerer gets the prompt `pick_partner`: `confirm` (`{"partner":seat}`) with a seat that
   accepted (on the offer's terms) or countered (on the counteroffer's terms), or `cancel`.
4. Every answer is a decision with the game's budget, and the steps tell observers each one.

With `player_trading` `first_accept` (JSettlers' flow) the moves and the state are the same, but the order is free:
- The addressed seats may answer in any order, as the referee sees the answers come; a step's seat can then be another
  than its state's `current` (which names the next addressee in seat order, then the offerer).
- An accept trades at once on the offer's terms and closes the offer: there is no partner to pick among plain accepts.
- A counteroffer stands until the offerer takes it (`confirm`, which trades at once) or the offer closes; the offerer may
  take one, or withdraw (`cancel`), while others are still asked.
- When every addressed seat declined, the offer closes, as under `rulebook`.

A referee whose players answer at the same time can't wait for a slow agent: a decision about an offer can expire before
its answer comes (another seat accepted, the offerer withdrew). The referee then drops the answer instead of playing it,
and the agent learns from the steps what happened instead.

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
{"prompt":"main",             // setup_settlement, setup_road, roll, main, road_building (placing the free roads of the
                              // card), discard, move_robber, answer_offer, pick_partner; a finished game's is left open
 "current":1, "on_turn":1,    // who acts now, and whose turn it is (a player discarding after a 7 acts on another's turn)
 "turn":37, "dice":[3,4], "free_roads":0,
 "robber":[q,r],
 "buildings":{"settlements":[[seat,intersection],...],"cities":[[seat,intersection],...],"roads":[[seat,path],...]},
 "bank":{"brick":12,...,"development":11},
 "players":[{"seat":0,"public_vp":5,"vp":6,"hand_size":7,"development_size":2,
             "played":{"knight":2,"road_building":0,"year_of_plenty":1,"monopoly":0},
             "longest_road":6,"has_longest_road":true,"has_largest_army":false,
             "pieces_left":{"roads":7,"settlements":2,"cities":3},
             "hand":{"brick":0,"ore":2,...} or null, "development":{"knight":1,"victory_point":1,...} or null},
            ...],
 "offer":{"from":1,"give":set,"get":set,"to":[0,2],"answers":{"0":"accept","2":"counter"},
          "counters":{"2":{"give":set,"get":set}}} or null}   // answers: accept, decline, counter; counteroffers in
                                                              // the offerer's terms
```

- `hand`, `development` and `vp` (the points with victory point cards) are null for other seats under `view`; every
  card type is listed, 0 included. `played` may count only knights (Hexfarer's engine keeps no other).
- A `world` fills them with a game the referee drew from the seat's view and says so (`"drawn":true`).
- Under `truth` they are the real ones.
- `bank` counts each resource as 19 less what the hands hold, never below 0. A game without a resource bank says so in
  its ruleset (`bank_shortage` `unlimited`): its counts stop at 0, but every production is paid.

## Ruleset

The ruleset is a set of options; presets name common sets (`standard`, the rulebook, and `catanatron`). `before`
carries the preset and every option that differs from it. New options default to the rulebook, so older agents keep
playing `standard`. Every option, its values and which engine plays what: `RULESET.md`.

## Versions

`version` is raised only when a change would break agents written for the old shape. New options and moves for rules
an agent didn't declare don't count: a referee never sends them to an agent that doesn't support them.
- Version 2: `hello` became `handshake` (without `options`), `before` is answered with `ready`, `failure` replaced
  `error`, and `budget`, `seed`, `lean`, `hash`, `resync` and the result record are new. `decide_timeout_ms` left the
  ruleset for the budget.
