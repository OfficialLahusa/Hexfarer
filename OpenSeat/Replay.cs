using Common;
using Common.Actions;
using Common.Osp;
using System.IO.Compression;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace OpenSeat
{
    /// <summary>
    /// The replay harness: plays the games another engine's referee recorded as Open Seat Protocol messages (e.g.
    /// Catanatron's, by Bridges/catanatron/record.py) through Hexfarer's rules, with the recorded dice, cards and steals,
    /// and compares at every move what both engines hold: the full state and the legal moves. A game's replay stops at the
    /// first state that differs or the first recorded move Hexfarer doesn't allow (the games no longer agree after it);
    /// differences in the legal moves alone are counted and the replay goes on. Known differences of the referee's
    /// ruleset are folded before comparing (see Normalize), so every remaining difference is a rule difference not
    /// described yet, a mapping bug or a bug in either engine.
    /// </summary>
    internal sealed class Replay
    {
        /// <summary>
        /// Counts and first examples by kind of difference, over all games
        /// </summary>
        public sealed class Report
        {
            public int Games, Moves, Diverged;
            public Dictionary<string, int> Counts { get; } = [];
            public Dictionary<string, List<string>> Examples { get; } = [];

            public void Add(string kind, string example, int maxExamples)
            {
                Counts[kind] = Counts.GetValueOrDefault(kind) + 1;
                if (!Examples.TryGetValue(kind, out List<string>? examples)) Examples[kind] = examples = [];
                if (examples.Count < maxExamples) examples.Add(example);
            }

            public override string ToString()
            {
                List<string> lines = [$"{Games} games, {Moves} moves replayed, {Diverged} games diverged"];
                foreach ((string kind, int count) in Counts.OrderByDescending(c => c.Value))
                {
                    lines.Add($"{count,8}  {kind}");
                    foreach (string example in Examples[kind]) lines.Add($"            {example}");
                }
                return string.Join(Environment.NewLine, lines);
            }
        }

        private readonly int _maxExamples;

        public Report Result { get; } = new();

        public Replay(int maxExamples = 3) => _maxExamples = maxExamples;

        /// <summary>
        /// The messages of a log: one JSON object per line, gzipped if the name ends in .gz
        /// </summary>
        public static IEnumerable<JsonNode> Read(string path)
        {
            using Stream file = File.OpenRead(path);
            using Stream input = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? new GZipStream(file, CompressionMode.Decompress) : file;
            using StreamReader reader = new(input);
            while (reader.ReadLine() is string line)
                if (line.Length > 0) yield return JsonNode.Parse(line)!;
        }

        /// <summary>
        /// Replays every game of the messages
        /// </summary>
        public Report Run(IEnumerable<JsonNode> messages, int maxGames = int.MaxValue)
        {
            Game? game = null;
            foreach (JsonNode message in messages)
            {
                string type = message["type"]!.GetValue<string>();
                if (type == "before")
                {
                    if (Result.Games >= maxGames) break;
                    game = new Game(this, message);
                    Result.Games++;
                    continue;
                }
                if (game == null || game.Stopped) continue;
                switch (type)
                {
                    case "decide": game.Decide(message); break;
                    case "step": game.Step(message); Result.Moves++; break;
                    case "after": game.After(message); break;
                }
            }
            return Result;
        }

        /// <summary>
        /// One game's replay
        /// </summary>
        private sealed class Game
        {
            private readonly Replay _replay;
            private readonly string _id;
            private readonly GameState _state;
            private readonly OspCodec _codec;

            // The type of the last move replayed, for explaining a difference it caused
            private string _lastMove = "";

            public bool Stopped { get; private set; }

            public Game(Replay replay, JsonNode before)
            {
                _replay = replay;
                _id = before["game_id"]!.GetValue<string>();
                Board board = OspCodec.BuildBoard(before["board"]!);
                _codec = new OspCodec(board);
                _state = new GameState(board, before["players"]!.GetValue<uint>());
                // The board as Hexfarer holds it is the board the referee sent
                if (OspCodec.Canonical(_codec.BoardJson()) != OspCodec.Canonical(before["board"]))
                    Diverge("board", "the board built from the message writes back differently");
                Compare(before["state"]!, "before");
            }

            public void Decide(JsonNode decide)
            {
                string at = $"move {decide["n"]}";
                if (!Compare(decide["state"]!, at)) return;
                sbyte seat = Acting();
                if (seat != decide["seat"]!.GetValue<int>()) { Diverge("seat", $"{at}: seat {decide["seat"]} acts, Hexfarer expects {seat}"); return; }

                // The legal moves of both, as sets of protocol moves
                HashSet<string> ours = Normalize(LegalActionProvider.GetActionsForState(_state, seat)).Select(m => OspCodec.Canonical(m.move)).ToHashSet();
                HashSet<string> theirs = decide["legal"]!.AsArray().Select(OspCodec.Canonical).ToHashSet();
                string prompt = decide["state"]!["prompt"]!.GetValue<string>();
                foreach (string missing in theirs.Except(ours))
                    Note($"legal: only the referee allows {MoveType(missing)} ({prompt})", $"{at}: {missing}");
                foreach (string extra in ours.Except(theirs))
                    Note($"legal: only Hexfarer allows {MoveType(extra)} ({prompt})", $"{at}: {extra}");
            }

            public void Step(JsonNode step)
            {
                string at = $"move {step["n"]}";
                sbyte seat = Acting();
                string move = OspCodec.Canonical(step["action"]);
                Action? action = Normalize(LegalActionProvider.GetActionsForState(_state, seat))
                    .Where(m => OspCodec.Canonical(m.move) == move).Select(m => m.action).FirstOrDefault();
                if (action == null) { Diverge($"step: Hexfarer doesn't allow {MoveType(move)}", $"{at}: {move}"); return; }

                // The referee's result decides the move's randomness
                action = Forced(action, seat, step["result"], at);
                if (action == null) return;
                action.Apply(_state);
                _lastMove = MoveType(move);
            }

            public void After(JsonNode after)
            {
                if (!Compare(after["state"]!, "after")) return;
                int? winner = after["winner"] is JsonNode w ? w.GetValue<int>() : null;
                int? ours = _state.HasEnded ? _state.Turn.PlayerIndex : null;
                if (winner == ours) return;
                // Explained if the referee's winner isn't the player on turn: Catanatron lets anyone win at any time
                bool offTurn = winner != null && ours == null && winner != int.Parse(after["state"]!["current"]!.ToJsonString());
                Diverge(offTurn ? "winner (known: Catanatron counts a win outside the winner's own turn)" : "winner",
                    $"the referee's winner is seat {winner?.ToString() ?? "none"}, Hexfarer's {ours?.ToString() ?? "none"}");
            }

            // The player who acts now: the lowest seat that can (as the agents and the Arena pick it)
            private sbyte Acting()
            {
                for (sbyte seat = 0; seat < _state.Players.Length; seat++)
                    if (_state.CanPlayerAct(seat)) return seat;
                return -1;
            }

            /// <summary>
            /// Hexfarer's legal moves with the referee's known differences folded in (Catanatron's rules): a robber move
            /// against a victim without cards steals from nobody, only the best bank rate per resource is offered, and the
            /// roads of a Road Building card have to be placed before anything else
            /// </summary>
            private IEnumerable<(Action action, JsonArray move)> Normalize(List<Action> legal)
            {
                sbyte seat = Acting();
                PlayerState player = _state.Players[seat];
                bool forcedRoads = player.BuildingStock.FreeRoads > 0 && legal.Any(a => a is RoadAction);
                // Road Building only with a road left and a place to build it
                bool roadsPossible = player.BuildingStock.RemainingRoads > 0 && _state.Board.GetRoadSites(seat) != UInt128.Zero;
                foreach (Action action in legal)
                {
                    if (forcedRoads && action is not RoadAction) continue;
                    if (action is RoadBuildingAction && !roadsPossible) continue;
                    if (action is FourToOneTradeAction four && legal.Any(a => a is ThreeToOneTradeAction t && t.InputType == four.InputType || a is TwoToOneTradeAction s && s.InputType == four.InputType)) continue;
                    if (action is ThreeToOneTradeAction three && legal.Any(a => a is TwoToOneTradeAction s && s.InputType == three.InputType)) continue;
                    JsonArray move = _codec.Move(action);
                    if (action is RobberAction robber && robber.TargetPlayerIndex is sbyte victim && _state.Players[victim].ResourceCards.Count() == 0)
                    {
                        // A victim without cards isn't one: on a hex where someone has cards only they can be robbed,
                        // otherwise the robber moves there without stealing
                        bool othersWithCards = legal.Any(a => a is RobberAction r && r.TargetTileIndex == robber.TargetTileIndex
                            && r.TargetPlayerIndex is sbyte v && _state.Players[v].ResourceCards.Count() > 0);
                        if (othersWithCards) continue;
                        move[1]!["victim"] = null;
                    }
                    yield return (action, move);
                }
            }

            // The variant of a random move whose outcome is the referee's result
            private Action? Forced(Action action, sbyte seat, JsonNode? result, string at)
            {
                if (action is not IOutputRandomnessAction random) return action;
                List<Action> variants = random.GetOutcomeVariants(_state, seat).Select(v => v.Action).ToList();
                Action? chosen = action switch
                {
                    // Only the total of the dice counts for the game
                    RollAction => variants.Cast<RollAction>().FirstOrDefault(v => v.History!.RollResult.Total == result!["dice"]!.AsArray().Sum(d => d!.GetValue<int>())),
                    BuyDevelopmentCardAction => variants.Cast<BuyDevelopmentCardAction>().FirstOrDefault(v => OspCodec.CardNames[v.History!.DrawnType] == result!["card"]!.GetValue<string>()),
                    RobberAction => variants.Cast<RobberAction>().FirstOrDefault(v => (v.History!.StolenCard is ResourceCardType card ? OspCodec.ResourceNames[card] : null) == result!["stolen"]?.GetValue<string>()),
                    _ => throw new NotSupportedException($"No forced outcome for {action.GetType().Name}."),
                };
                if (chosen == null) Diverge($"result: Hexfarer can't reach the referee's result of {action.GetType().Name}", $"{at}: {result?.ToJsonString()}");
                return chosen;
            }

            // Compares the referee's state with Hexfarer's; at the first difference the game stops
            private bool Compare(JsonNode theirs, string at)
            {
                JsonObject ours = _codec.State(_state);
                foreach (string field in new[] { "prompt", "current", "free_roads", "robber" })
                    if (!Same(theirs[field], ours[field])) return Diverge($"state: {field}", $"{at}: referee {theirs[field]?.ToJsonString()}, Hexfarer {ours[field]?.ToJsonString()}");
                foreach (string kind in new[] { "settlements", "cities", "roads" })
                {
                    HashSet<string> a = theirs["buildings"]![kind]!.AsArray().Select(OspCodec.Canonical).ToHashSet(), b = ours["buildings"]![kind]!.AsArray().Select(OspCodec.Canonical).ToHashSet();
                    if (!a.SetEquals(b)) return Diverge($"state: {kind}", $"{at}: only the referee {string.Join(" ", a.Except(b))}; only Hexfarer {string.Join(" ", b.Except(a))}");
                }
                if (!Same(theirs["bank"], ours["bank"]))
                {
                    // Explained after a roll if Hexfarer emptied what the referee kept: the rulebook gives a single player
                    // owed a short resource the rest, Catanatron gives nobody anything
                    string[] differing = OspCodec.ResourceNames.Values.Where(n => !Same(theirs["bank"]![n], ours["bank"]![n])).ToArray();
                    bool shortage = _lastMove == "roll" && differing.Length > 0 && differing.All(n => int.Parse(ours["bank"]![n]!.ToJsonString()) == 0);
                    return Diverge(shortage ? "state: bank (known: Catanatron gives nobody a short resource, even a single player owed it)" : "state: bank",
                        $"{at}: referee {theirs["bank"]!.ToJsonString()}, Hexfarer {ours["bank"]!.ToJsonString()}");
                }
                for (int seat = 0; seat < ours["players"]!.AsArray().Count; seat++)
                {
                    JsonNode a = theirs["players"]![seat]!, b = ours["players"]![seat]!;
                    // Not the road length: Catanatron keeps it at 0 through the setup and updates it only on some moves, the
                    // award and its points are what count
                    // The award first, since it changes the points
                    if (!Same(a["has_longest_road"], b["has_longest_road"]))
                    {
                        // Explained if some road counts differently at an opponent's building, or if the award changed
                        // after a settlement cut a road (Catanatron then gives it to whoever is longest, ties by its own
                        // order, even under 5)
                        bool countsDiffer = Enumerable.Range(0, _state.Players.Length).Any(s => LongestRoad(s, enterOpponents: true) != LongestRoad(s, enterOpponents: false));
                        string kind = countsDiffer ? "state: Longest Road (known: Catanatron doesn't count a road's segment ending at an opponent's building)"
                            : _lastMove == "build_settlement" ? "state: Longest Road (known: Catanatron's recount after a settlement cuts a road)"
                            : "state: Longest Road";
                        return Diverge(kind, $"{at}, seat {seat}: referee {a["has_longest_road"]}, Hexfarer {b["has_longest_road"]}");
                    }
                    foreach (string field in new[] { "public_vp", "vp", "hand", "development", "has_largest_army", "pieces_left" })
                        if (!Same(a[field], b[field])) return Diverge($"state: {field}", $"{at}, seat {seat}: referee {a[field]?.ToJsonString()}, Hexfarer {b[field]?.ToJsonString()}");
                    // Hexfarer counts only the knights played
                    if (!Same(a["played"]!["knight"], b["played"]!["knight"])) return Diverge("state: knights played", $"{at}, seat {seat}: referee {a["played"]!["knight"]}, Hexfarer {b["played"]!["knight"]}");
                }
                return true;
            }

            /// <summary>
            /// The seat's longest road, each road segment used once: by the rulebook a road may end at an opponent's
            /// building (and stops there), with enterOpponents false it may not reach one at all (Catanatron's count)
            /// </summary>
            private int LongestRoad(int seat, bool enterOpponents)
            {
                // The seat's roads as intersection pairs, and the intersections with an opponent's building
                List<(int a, int b)> roads = [];
                for (int e = 0; e < _state.Board.Edges.Count; e++)
                    if (_state.Board.HasRoadOf(e, seat)) roads.Add(_state.Board.Adjacency.GetIntersectionIndices(e));
                bool Opponent(int i) => _state.Board.GetBuilding(i) != Intersection.BuildingType.None && _state.Board.GetBuildingOwner(i) != seat;

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

            private static bool Same(JsonNode? a, JsonNode? b) => OspCodec.Canonical(a) == OspCodec.Canonical(b);

            private static string MoveType(string move) => JsonNode.Parse(move)![0]!.GetValue<string>();

            private void Note(string kind, string example) => _replay.Result.Add(kind, $"{_id} {example}", _replay._maxExamples);

            private bool Diverge(string kind, string example)
            {
                Note(kind, example);
                _replay.Result.Diverged++;
                Stopped = true;
                return false;
            }
        }
    }
}
