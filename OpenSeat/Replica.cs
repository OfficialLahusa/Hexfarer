using Common;
using Common.Actions;
using Common.OSP;
using System.Text.Json;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace OpenSeat
{
    /// <summary>
    /// Hexfarer's copy of a game another engine's referee runs over the Open Seat Protocol: built from the referee's board,
    /// it follows the referee's moves with their recorded results (dice, drawn cards, steals) and is compared with the
    /// referee's state. The referee's rules that narrow the legal moves (its ruleset: Catanatron's preset, or single options)
    /// are folded into the legal moves;
    /// the differences that change a state are named by FirstDifference and can be taken over from the referee (Patch),
    /// since the referee decides the game.
    /// </summary>
    internal sealed class Replica
    {
        public GameState State { get; private set; }
        public OSPCodec Codec { get; }

        /// <summary>
        /// Whether the referee plays Catanatron's rules (its ruleset's preset), whose known deviations the differences name
        /// </summary>
        public bool Catanatron { get; }

        /// <summary>
        /// The type of the last move followed, for explaining a difference it caused
        /// </summary>
        public string LastMove { get; private set; } = "";

        // The referee's rules that narrow Hexfarer's legal moves: robbing only victims with cards, only the best bank rate
        // per resource, Road Building's roads placed at once (and the card only with a road left and a place for it)
        private readonly bool _robbedNeedCards, _bestBankRateOnly, _forcedRoadBuilding;

        public Replica(JsonNode before)
        {
            Board board = OSPCodec.BuildBoard(before["board"]!);
            Codec = new OSPCodec(board);
            State = new GameState(board, (uint)before["players"]!.GetValue<int>());
            // The ruleset's options, by default its preset's: Catanatron's narrow the moves, the rulebook's ("standard") don't
            JsonNode? ruleset = before["ruleset"];
            bool catanatron = Catanatron = ruleset?["preset"]?.GetValue<string>() == "catanatron";
            string Option(string name, string rulebook, string catanatrons) => ruleset?[name]?.GetValue<string>() ?? (catanatron ? catanatrons : rulebook);
            _robbedNeedCards = Option("robber_victims", "adjacent", "with_cards") == "with_cards";
            _bestBankRateOnly = Option("bank_rates", "all", "best_only") == "best_only";
            _forcedRoadBuilding = Option("road_building", "optional", "forced") == "forced";
            // The options Hexfarer's engine plays itself: points to win, the discard limit, trades between players and offers
            State.Settings.VictoryPoints = ruleset?["victory_points"]?.GetValue<int>() ?? 10;
            State.Settings.RobberCardLimit = ruleset?["discard_limit"]?.GetValue<int>() ?? 7;
            string trading = Option("player_trading", "rulebook", "none");
            State.Settings.FreeTrade = trading is "rulebook" or "first_accept";
            // JSettlers' flow: the addressed players answer in any order and the first accept trades at once
            State.Settings.TradeFlow = trading == "first_accept" ? TradeFlow.FirstAccept : TradeFlow.Rulebook;
            State.Settings.MaxTradeOffersPerTurn = ruleset?["offer_limit"]?.GetValue<int>() ?? int.MaxValue;
        }

        // Every option of RULESET.md with the strings it takes (null: a number or none), and the boolean ones
        private static readonly Dictionary<string, string[]?> Options = new()
        {
            ["victory_points"] = null, ["discard_limit"] = null, ["turn_limit"] = null, ["offer_limit"] = null,
            ["player_trading"] = ["rulebook", "first_accept", "none"], ["bank_shortage"] = ["rulebook", "nobody", "unlimited"],
            ["win_timing"] = ["own_turn", "any_time"], ["robber_victims"] = ["adjacent", "with_cards"], ["bank_rates"] = ["all", "best_only"],
            ["road_building"] = ["optional", "forced"],
        };
        private static readonly HashSet<string> Flags = ["counteroffers"];

        /// <summary>
        /// What of a ruleset the copy can't play, as a failure's detail ({"preset": ...} or {"option": ..., "value": ...}),
        /// or null: the presets standard and catanatron and every option of RULESET.md with its values; a value Hexfarer's
        /// rules play differently is taken over from the referee's states (Patch)
        /// </summary>
        public static JsonObject? Unsupported(JsonNode? ruleset)
        {
            if (ruleset is not JsonObject options) return null;
            foreach ((string name, JsonNode? value) in options)
            {
                if (name == "preset")
                {
                    if (value?.ToJsonString() is not ("\"standard\"" or "\"catanatron\"")) return new JsonObject { ["preset"] = value?.DeepClone() };
                    continue;
                }
                // A value of the option's type: a boolean, a number (or none), or one of its strings
                JsonValueKind kind = value?.GetValueKind() ?? JsonValueKind.Null;
                bool known = Flags.Contains(name) ? kind is JsonValueKind.True or JsonValueKind.False
                    : Options.TryGetValue(name, out string[]? values) && (values == null
                        ? kind == JsonValueKind.Null || kind == JsonValueKind.Number && value!.AsValue().TryGetValue(out int count) && count >= 0
                        : kind == JsonValueKind.String && values.Contains(value!.GetValue<string>()));
                if (!known) return new JsonObject { ["option"] = name, ["value"] = value?.DeepClone() };
            }
            return null;
        }

        /// <summary>
        /// Takes the game from the referee's full state where following its moves can't (no steps came, or a rule
        /// difference changed the board): the copy becomes that state, with what the legal moves tell (OSPCodec.BuildState)
        /// </summary>
        public void Rebuild(JsonNode state, JsonArray legal)
        {
            // The state says where the game is, the ruleset how it is played: the settings stay the before's
            GameSettings settings = State.Settings;
            TurnState was = State.Turn;
            State = Codec.BuildState(state, legal);
            State.Settings = new GameSettings(settings);
            // The state doesn't say how many offers came this turn: within the same turn (the same player, after the roll in
            // both) the copy's count stays, so the offer limit and an agent's memory of its offers this turn hold
            TurnState now = State.Turn;
            if (was.PlayerIndex == now.PlayerIndex && was.TypeOfRound == TurnState.RoundType.Normal && now.TypeOfRound == TurnState.RoundType.Normal
                && !was.MustRoll && !now.MustRoll)
                now.TradeOffersThisTurn = Math.Max(now.TradeOffersThisTurn, was.TradeOffersThisTurn);
        }

        /// <summary>
        /// Whether the board as Hexfarer holds it writes back as the referee sent it
        /// </summary>
        public bool BoardMatches(JsonNode before) => OSPCodec.Canonical(Codec.BoardJson()) == OSPCodec.Canonical(before["board"]);

        /// <summary>
        /// The player who acts now: the lowest seat that can (as the agents and the Arena pick it), or -1
        /// </summary>
        public sbyte Acting()
        {
            for (sbyte seat = 0; seat < State.Players.Length; seat++)
                if (State.CanPlayerAct(seat)) return seat;
            return -1;
        }

        /// <summary>
        /// Hexfarer's legal moves with the referee's narrower rules folded in (Catanatron's, if its ruleset says so): a
        /// robber move against a victim without cards steals from nobody, only the best bank rate per resource is offered,
        /// the roads of a Road Building card have to be placed before anything else, and the card needs a road left and a
        /// place for it
        /// </summary>
        public IEnumerable<(Action action, JsonArray move)> Legal()
        {
            sbyte seat = Acting();
            if (seat < 0) yield break;
            List<Action> legal = LegalActionProvider.GetActionsForState(State, seat);
            PlayerState player = State.Players[seat];
            bool forcedRoads = _forcedRoadBuilding && player.BuildingStock.FreeRoads > 0 && legal.Any(a => a is RoadAction);
            // Road Building only with a road left and a place to build it
            bool roadsPossible = player.BuildingStock.RemainingRoads > 0 && State.Board.GetRoadSites(seat) != UInt128.Zero;
            foreach (Action action in legal)
            {
                if (forcedRoads && action is not RoadAction) continue;
                if (_forcedRoadBuilding && action is RoadBuildingAction && !roadsPossible) continue;
                if (_bestBankRateOnly && action is FourToOneTradeAction four && legal.Any(a => a is ThreeToOneTradeAction t && t.InputType == four.InputType || a is TwoToOneTradeAction s && s.InputType == four.InputType)) continue;
                if (_bestBankRateOnly && action is ThreeToOneTradeAction three && legal.Any(a => a is TwoToOneTradeAction s && s.InputType == three.InputType)) continue;
                JsonArray move = Codec.Move(action);
                if (_robbedNeedCards && action is RobberAction robber && robber.TargetPlayerIndex is sbyte victim && State.Players[victim].ResourceCards.Count() == 0)
                {
                    // A victim without cards isn't one: on a hex where someone has cards only they can be robbed,
                    // otherwise the robber moves there without stealing
                    bool othersWithCards = legal.Any(a => a is RobberAction r && r.TargetTileIndex == robber.TargetTileIndex
                        && r.TargetPlayerIndex is sbyte v && State.Players[v].ResourceCards.Count() > 0);
                    if (othersWithCards) continue;
                    move[1]!["victim"] = null;
                }
                yield return (action, move);
            }
        }

        /// <summary>
        /// The legal move of the protocol's move (after Legal's folding), also an offer or counteroffer the rules allow, or
        /// null if Hexfarer doesn't allow it
        /// </summary>
        public Action? Find(JsonNode move, sbyte? seat = null)
        {
            string wanted = OSPCodec.Canonical(move);
            // The legal list is the seat's whose turn it is in seat order: a move doesn't name its seat, so it is only looked up
            // there when the step comes from that seat (under JSettlers' flow an answer can come from another addressee)
            sbyte acting = Acting();
            Action? listed = seat is sbyte from && from != acting ? null
                : Legal().Where(m => OSPCodec.Canonical(m.move) == wanted).Select(m => m.action).FirstOrDefault();
            if (listed != null) return listed;
            // Offers and counteroffers can be any cards, so no legal list holds them; under JSettlers' flow an answer, a pick or
            // a withdrawal can come from another seat than the one in seat order. Read from the move for the seat that made it
            // (the step's, else the one to act), checked by the rules.
            sbyte by = seat ?? acting;
            return (OSPCodec.TradeAction(move, by) ?? OSPCodec.AnswerAction(move, by)) is Action trade && trade.IsValidFor(State) ? trade : null;
        }

        /// <summary>
        /// Plays the referee's move with its result; returns the kind and detail of the failure, or null
        /// </summary>
        public (string kind, string detail)? Follow(JsonNode step)
        {
            string at = $"move {step["n"]}";
            string move = OSPCodec.Canonical(step["action"]);
            Action? action = Find(step["action"]!, step["seat"] is JsonValue seat && seat.TryGetValue(out int by) ? (sbyte)by : null);
            if (action == null) return ($"step: Hexfarer doesn't allow {MoveType(move)}", $"{at}: {move}");

            // The referee's result decides the move's randomness
            Action? forced = Forced(action, Acting(), step["result"]);
            if (forced == null) return ($"result: Hexfarer can't reach the referee's result of {action.GetType().Name}", $"{at}: {step["result"]?.ToJsonString()}");
            forced.Apply(State);
            LastMove = MoveType(move);
            Applied = forced;
            return null;
        }

        /// <summary>
        /// The action Follow played last, with its outcome
        /// </summary>
        public Action? Applied { get; private set; }

        // The variant of a random move whose outcome is the referee's result
        private Action? Forced(Action action, sbyte seat, JsonNode? result)
        {
            if (action is not IOutputRandomnessAction random) return action;
            List<Action> variants = random.GetOutcomeVariants(State, seat).Select(v => v.Action).ToList();
            return action switch
            {
                // Only the total of the dice counts for the game
                RollAction => variants.Cast<RollAction>().FirstOrDefault(v => v.History!.RollResult.Total == result!["dice"]!.AsArray().Sum(d => d!.GetValue<int>())),
                BuyDevelopmentCardAction => variants.Cast<BuyDevelopmentCardAction>().FirstOrDefault(v => OSPCodec.CardNames[v.History!.DrawnType] == result!["card"]!.GetValue<string>()),
                // A robbery without a victim has no result or a null card
                RobberAction => variants.Cast<RobberAction>().FirstOrDefault(v => (v.History!.StolenCard is ResourceCardType card ? OSPCodec.ResourceNames[card] : null) == result?["stolen"]?.GetValue<string>()),
                _ => throw new NotSupportedException($"No forced outcome for {action.GetType().Name}."),
            };
        }

        /// <summary>
        /// The first difference between the referee's state and Hexfarer's, its kind (a known rule difference named as
        /// such) and detail, or null if they agree
        /// </summary>
        /// <param name="goesOn">Whether the referee's game goes on (it asks for a move), so Hexfarer's mustn't have ended</param>
        public (string kind, string detail)? FirstDifference(JsonNode theirs, string at, bool goesOn = true)
        {
            JsonObject ours = Codec.State(State);
            // Engines name a finished game's prompt differently: only while the game goes on does it count
            foreach (string field in (goesOn ? ["prompt"] : Array.Empty<string>()).Concat(["current", "free_roads", "robber"]))
                if (!Same(theirs[field], ours[field])) return ($"state: {field}", $"{at}: referee {theirs[field]?.ToJsonString()}, Hexfarer {ours[field]?.ToJsonString()}");
            foreach (string kind in new[] { "settlements", "cities", "roads" })
            {
                HashSet<string> a = theirs["buildings"]![kind]!.AsArray().Select(OSPCodec.Canonical).ToHashSet(), b = ours["buildings"]![kind]!.AsArray().Select(OSPCodec.Canonical).ToHashSet();
                if (!a.SetEquals(b)) return ($"state: {kind}", $"{at}: only the referee {string.Join(" ", a.Except(b))}; only Hexfarer {string.Join(" ", b.Except(a))}");
            }
            JsonNode bank = Bank(theirs);
            if (!Same(bank, ours["bank"]))
            {
                // Explained after a roll if Hexfarer emptied what the referee kept: the rulebook gives a single player owed a
                // short resource the rest, Catanatron gives nobody anything
                string[] differing = OSPCodec.ResourceNames.Values.Where(n => !Same(bank[n], ours["bank"]![n])).ToArray();
                bool shortage = Catanatron && LastMove == "roll" && differing.Length > 0 && differing.All(n => int.Parse(ours["bank"]![n]!.ToJsonString()) == 0);
                return (shortage ? "state: bank (known: Catanatron gives nobody a short resource, even a single player owed it)" : "state: bank",
                    $"{at}: referee {theirs["bank"]!.ToJsonString()}, Hexfarer {ours["bank"]!.ToJsonString()}");
            }
            for (int seat = 0; seat < ours["players"]!.AsArray().Count; seat++)
            {
                JsonNode a = theirs["players"]![seat]!, b = ours["players"]![seat]!;
                // Not the road length: Catanatron keeps it at 0 through the setup and updates it only on some moves; the
                // award and its points are what count. The award first, since it changes the points
                if (!Same(a["has_longest_road"], b["has_longest_road"]))
                {
                    // Explained if some road counts differently at an opponent's building, or if the award changed after a
                    // settlement cut a road (Catanatron then gives it to whoever is longest, ties by its own order, even under 5)
                    bool countsDiffer = Catanatron && Enumerable.Range(0, State.Players.Length).Any(s => LongestRoad(s, enterOpponents: true) != LongestRoad(s, enterOpponents: false));
                    string kind = countsDiffer ? "state: Longest Road (known: Catanatron doesn't count a road's segment ending at an opponent's building)"
                        : Catanatron && LastMove == "build_settlement" ? "state: Longest Road (known: Catanatron's recount after a settlement cuts a road)"
                        : "state: Longest Road";
                    return (kind, $"{at}, seat {seat}: referee {a["has_longest_road"]}, Hexfarer {b["has_longest_road"]}");
                }
                foreach (string field in new[] { "public_vp", "vp", "hand", "development", "has_largest_army", "pieces_left" })
                    if (!Same(a[field], b[field])) return ($"state: {field}", $"{at}, seat {seat}: referee {a[field]?.ToJsonString()}, Hexfarer {b[field]?.ToJsonString()}");
                // Hexfarer counts only the knights played
                if (!Same(a["played"]!["knight"], b["played"]!["knight"])) return ("state: knights played", $"{at}, seat {seat}: referee {a["played"]!["knight"]}, Hexfarer {b["played"]!["knight"]}");
            }
            // The referee asks for a move, so its game goes on (Hexfarer's may have ended by points the referee doesn't give)
            if (goesOn && State.HasEnded) return ("state: game over (Hexfarer's rules ended the game, the referee's go on)", $"{at}: seat {State.Turn.PlayerIndex} won by Hexfarer's rules");
            return null;
        }

        /// <summary>
        /// Takes over what the referee's known rule differences change, since the referee decides the game: every hand,
        /// the bank, who holds Longest Road (with its points), and that the game goes on. Only for a state the referee asks
        /// a move in. Returns whether the states agree afterwards.
        /// </summary>
        public bool Patch(JsonNode theirs)
        {
            JsonNode bank = Bank(theirs);
            foreach ((ResourceCardType type, string name) in OSPCodec.ResourceNames)
            {
                State.ResourceBank.Remove(type, State.ResourceBank.Get(type));
                State.ResourceBank.Add(type, uint.Parse(bank[name]!.ToJsonString()));
            }
            for (int seat = 0; seat < State.Players.Length; seat++)
            {
                JsonNode player = theirs["players"]![seat]!;
                PlayerState ours = State.Players[seat];
                foreach ((ResourceCardType type, string name) in OSPCodec.ResourceNames)
                {
                    ours.ResourceCards.Remove(type, ours.ResourceCards.Get(type));
                    ours.ResourceCards.Add(type, uint.Parse(player["hand"]![name]!.ToJsonString()));
                }
                ours.VictoryPoints.LongestRoadPoints = (byte)(player["has_longest_road"]!.GetValue<bool>() ? 2 : 0);
            }
            // A game Hexfarer's rules ended goes on as the referee's does (ending only marks the round as ended)
            if (State.HasEnded) State.Turn.TypeOfRound = TurnState.RoundType.Normal;
            return FirstDifference(theirs, "patch") == null;
        }

        /// <summary>
        /// The seat's longest road, each road segment used once: by the rulebook a road may end at an opponent's building
        /// (and stops there), with enterOpponents false it may not reach one at all (Catanatron's count)
        /// </summary>
        public int LongestRoad(int seat, bool enterOpponents)
        {
            // The seat's roads as intersection pairs, and the intersections with an opponent's building
            List<(int a, int b)> roads = [];
            for (int e = 0; e < State.Board.Edges.Count; e++)
                if (State.Board.HasRoadOf(e, seat)) roads.Add(State.Board.Adjacency.GetIntersectionIndices(e));
            bool Opponent(int i) => State.Board.GetBuilding(i) != Intersection.BuildingType.None && State.Board.GetBuildingOwner(i) != seat;

            int best = 0;
            bool[] used = new bool[roads.Count];
            void Walk(int node, int length, bool start)
            {
                best = Math.Max(best, length);
                // A road goes on past an intersection only if no opponent built there
                if (!start && Opponent(node)) return;
                for (int r = 0; r < roads.Count; r++)
                {
                    if (used[r] || roads[r].a != node && roads[r].b != node) continue;
                    int next = roads[r].a == node ? roads[r].b : roads[r].a;
                    if (!enterOpponents && Opponent(next)) continue;
                    used[r] = true;
                    Walk(next, length + 1, false);
                    used[r] = false;
                }
            }
            foreach (int start in roads.SelectMany(r => new[] { r.a, r.b }).Distinct())
                if (enterOpponents || !Opponent(start)) Walk(start, 0, true);
            return best;
        }

        public static bool Same(JsonNode? a, JsonNode? b) => OSPCodec.Canonical(a) == OSPCodec.Canonical(b);

        // The referee's bank as Hexfarer holds one: an unlimited bank's count below 0 (the hands hold more than 19, as in
        // JSettlers) is an empty bank
        private static JsonNode Bank(JsonNode theirs)
        {
            JsonObject bank = theirs["bank"]!.DeepClone().AsObject();
            foreach (string name in OSPCodec.ResourceNames.Values)
                if (bank[name] is JsonNode count && int.Parse(count.ToJsonString()) < 0) bank[name] = 0;
            return bank;
        }

        public static string MoveType(string move) => JsonNode.Parse(move)![0]!.GetValue<string>();
    }
}
