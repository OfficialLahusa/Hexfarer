# Ruleset options (OSP)

The rules of an Open Seat Protocol game (see `PROTOCOL.md`) are a set of options. A preset names a common set, and
`before` carries the preset plus every option whose value differs from it:

```
"ruleset": {"preset": "standard", "player_trading": "none", "victory_points": 12}
```

- Every option has a default that matches the rulebook. A new option gets one too, so agents written before it keep
  playing the `standard` preset.
- An agent names the presets it can play in its handshake (`rulesets`). A referee sends an agent only games of a
  preset it named, and the agent turns a game down at `before` if it can't play one of its options.
- Values are JSON strings, numbers or booleans as listed; an option the agent doesn't know is one it doesn't play.

## Presets

- **`standard`**: the rulebook of the base game for 3 or 4 players. Every option at its default.
- **`catanatron`**: the rules of Catanatron's engine (commit `ecf9311`), as its referee plays them.
- JSettlers 2.6.10 and StacSettlers (a fork of JSettlers 1.1.16) have no preset: their games are `standard` with the
  options in the last two columns below.

| Option | `standard` | `catanatron` | JSettlers 2.6.10 | StacSettlers |
|---|---|---|---|---|
| `victory_points` | 10 | 10 | 10 | 10 |
| `discard_limit` | 7 | 7 | 7 | 7 |
| `player_trading` | `rulebook` | `none` | `rulebook`, `none` with game option NT | `none` for our seat (it declines their offers; its own agents trade among themselves in chat) |
| `counteroffers` | `true` | `false` | `true` | (no trades for our seat) |
| `bank_shortage` | `rulebook` | `nobody` | `unlimited` | `unlimited` |
| `win_timing` | `own_turn` | `any_time` | `own_turn`, by Largest Army only after the robber moved | `own_turn`, by Largest Army only after the robber moved |
| `robber_victims` | `adjacent` | `with_cards` | `with_cards` | `with_cards` |
| `bank_rates` | `all` | `best_only` | `best_only` | `all` |
| `road_building` | `optional` | `forced` | `forced` | `forced` |
| `turn_limit` | none | 1000 | none | none (its config's NoTurns unset) |
| `offer_limit` | none | none (no trading) | not measured (our robot plays without trades) | (no trades for our seat; its Stac agents keep their own limit) |

The JSettlers column comes from replaying 100 random JSettlers games through Hexfarer's rules (`Recorder` and
`OpenSeat replay`). Its `unlimited` bank and its Largest Army timing are what its engine does, not values an agent
can ask for; its robot names the bank in the ruleset, so an agent that can't play it turns the game down.

The StacSettlers column comes from reading its engine (JSettlers 1.1.16's `SOCGame` in StacSettlers @8185c5e) and from
its recorder's random games, whose every move passes the checks StacSettlers' server makes (`Recorder`,
`test_stacsettlers_game`): it pays every production and Year of Plenty without a bank; it robs only players holding a
resource card; it trades 4:1 with the bank always and 3:1 or 2:1 at the player's harbors (`canMakeBankTrade` takes
each, so not only the best rate, unlike JSettlers 2); a Road Building card gives two roads only with two roads left and
two places for them and its roads can't be given up; a player wins only on their own turn (`checkForWinner` looks at
the player on turn), and a knight that brings Largest Army and the points ends the game once the robber has moved.
Replaying Stac agents' games through Hexfarer's rules (`record.py`, games without trades between players, and
`OpenSeat replay`) shows the same two differences and no other: 20 games, 6311 moves, 4 games diverged, 3 by the
Largest Army's timing and 1 by the bank, whose missing limit also lets Year of Plenty and bank trades take a resource
of which the hands hold all 19. Later moves of a diverged game aren't compared, so a rarer difference may still hide
there. Its server also takes a second
setup road anywhere its player could build one, not only next to the settlement just placed; our robot keeps to the
rulebook there, the Stac agents did too in the replayed games.

## Options

- **`victory_points`** (number, default 10): the points a player needs to win.
- **`discard_limit`** (number, default 7): when a 7 is rolled, every player with more resource cards than this
  discards half of them, rounded down. The discards come one seat after another from the lowest, each as one decision
  (a card set).
- **`player_trading`**: trades between players.
  - `rulebook` (default): the player on turn trades with the others after the roll. An offer gives at least one card
    and asks for at least one, never the same resource on both sides; it goes to some or all other players, who
    accept, decline or counter in seat order, and the offerer then trades with one who accepted, or withdraws.
  - `none`: only trades with the bank and the harbors.
- **`counteroffers`** (boolean, default `true`): whether a player may answer an offer with one of their own terms
  (only with `player_trading` `rulebook`).
- **`bank_shortage`**: a production the bank can't pay in full.
  - `rulebook` (default): nobody gets the short resource, unless a single player is owed it; that player gets what is
    left.
  - `nobody`: nobody gets the short resource.
  - `unlimited`: there is no resource bank; every production is paid. The state's `bank` counts each resource as 19
    less what the hands hold and stops at 0, also when the hands hold more.
- **`win_timing`**: when the points win.
  - `own_turn` (default): a player wins during their own turn, as soon as they hold the points.
  - `any_time`: a player wins as soon as they hold the points, also on another player's turn.
- **`robber_victims`**: whom the robber may steal from.
  - `adjacent` (default): any player with a building at the robber's hex, even one without cards (nothing is stolen).
  - `with_cards`: only such players who hold a resource card; without one, the robber moves without a victim.
- **`bank_rates`**: which trade rates a player may use with the bank.
  - `all` (default): every rate the player has, 4:1 also next to a harbor.
  - `best_only`: only the best rate the player has for each resource.
- **`road_building`**: the Road Building card.
  - `optional` (default): the two roads can be placed or given up.
  - `forced`: both roads are placed if there is room; the card can be played only with a road left and a place for
    it.
- **`turn_limit`** (number or none, default none): after this many player turns the game ends as a draw.
- **`offer_limit`** (number or none, default none): the trade offers a player may make per turn (only with
  `player_trading` `rulebook`).

How long an agent may think is no rule of the game: it is the game's budget in `before` (see `PROTOCOL.md`).

## Who plays what

| | Presets | Options it sets or follows |
|---|---|---|
| Hexfarer as referee (client, server, Arena) | `standard` | sets `player_trading` (`rulebook`, `none`), `victory_points`, `discard_limit` and `offer_limit` (10 unless set otherwise); the rest at the default |
| Hexfarer's agent (`OpenSeat agent`) | `standard`, `catanatron` | follows `robber_victims`, `bank_rates` and `road_building` by narrowing its legal moves; takes the referee's bank, hands and Longest Road where its own rules would differ |
| Catanatron's referee (`drive.py`) | `catanatron` | |
| Catanatron's agent (`open_seat_agent.py`) | `standard`, `catanatron` | rebuilds its game from the referee's state at each decision |
| JSettlers' server with our robot | `standard` | sends `player_trading` `none`, `bank_shortage` `unlimited`, `bank_rates` `best_only`, `robber_victims` `with_cards`, `road_building` `forced` |
| StacSettlers' simulation with our robot | `standard` | sends `player_trading` `none`, `bank_shortage` `unlimited`, `robber_victims` `with_cards`, `road_building` `forced` |
