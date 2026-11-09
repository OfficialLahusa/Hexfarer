using Common.Actions;
using System.Text.Json;
using System.Text.Json.Nodes;
using static Common.BoardGeometry;
using static Common.Tile;
using Action = Common.Actions.Action;

namespace Common.OSP
{
    /// <summary>
    /// Lanzenbrecher's boards, moves and states in the Open Seat Protocol (see Bridges/PROTOCOL.md): hexes as axial [q, r], an
    /// intersection as its three hexes, a path as its two intersections, cards by name. Seats are Lanzenbrecher's player
    /// indices, which are the turn order. One codec per board, since the keys depend only on its geometry.
    /// </summary>
    public sealed class OSPCodec
    {
        public static readonly IReadOnlyDictionary<ResourceCardType, string> ResourceNames = new Dictionary<ResourceCardType, string>
        {
            [ResourceCardType.Lumber] = "lumber", [ResourceCardType.Brick] = "brick", [ResourceCardType.Wool] = "wool",
            [ResourceCardType.Grain] = "grain", [ResourceCardType.Ore] = "ore",
        };

        public static readonly IReadOnlyDictionary<DevelopmentCardType, string> CardNames = new Dictionary<DevelopmentCardType, string>
        {
            [DevelopmentCardType.Knight] = "knight", [DevelopmentCardType.RoadBuilding] = "road_building",
            [DevelopmentCardType.YearOfPlenty] = "year_of_plenty", [DevelopmentCardType.Monopoly] = "monopoly",
            [DevelopmentCardType.VictoryPoint] = "victory_point",
        };

        // SGN's tile codes: forest, hills, pastures, fields, mountains, desert
        private static readonly IReadOnlyDictionary<TileType, string> TileCodes = new Dictionary<TileType, string>
        {
            [TileType.Lumber] = "FO", [TileType.Brick] = "H", [TileType.Wool] = "P", [TileType.Grain] = "F", [TileType.Ore] = "M",
            [TileType.Desert] = "D",
        };

        // SGN's harbor codes: HR is 3:1, the others 2:1 for their resource
        private static readonly IReadOnlyDictionary<Port.TradeType, string> HarborCodes = new Dictionary<Port.TradeType, string>
        {
            [Port.TradeType.Generic] = "HR", [Port.TradeType.Lumber] = "HRL", [Port.TradeType.Brick] = "HRB",
            [Port.TradeType.Wool] = "HRW", [Port.TradeType.Grain] = "HRG", [Port.TradeType.Ore] = "HRO",
        };

        public Board Board { get; }

        // Every intersection's and edge's key as canonical JSON, by index, and back
        private readonly string[] _intersections, _paths;
        private readonly Dictionary<string, int> _intersectionIndex = [], _pathIndex = [];

        public OSPCodec(Board board)
        {
            Board = board;
            Cube[][] keys = IntersectionKeys(board);
            _intersections = keys.Select(key => Canonical(new JsonArray(key.Select(cube => (JsonNode)Hex(cube)).ToArray()))).ToArray();
            for (int i = 0; i < _intersections.Length; i++) _intersectionIndex[_intersections[i]] = i;
            _paths = Enumerable.Range(0, board.Edges.Count).Select(e => PathJson(EdgeKey(board, e))).ToArray();
            for (int e = 0; e < _paths.Length; e++) _pathIndex[_paths[e]] = e;
        }

        // A path: its two intersections, sorted by their hexes
        private string PathJson((int a, int b) ends)
        {
            string[] sorted = new[] { _intersections[ends.a], _intersections[ends.b] }.Order(KeyOrder).ToArray();
            return $"[{sorted[0]},{sorted[1]}]";
        }

        // Orders keys as the protocol sorts them: element by element, numbers by value
        private static readonly IComparer<string> KeyOrder = Comparer<string>.Create((a, b) => CompareJson(JsonNode.Parse(a)!, JsonNode.Parse(b)!));

        private static int CompareJson(JsonNode a, JsonNode b)
        {
            if (a is JsonArray x && b is JsonArray y)
            {
                for (int i = 0; i < Math.Min(x.Count, y.Count); i++)
                {
                    int order = CompareJson(x[i]!, y[i]!);
                    if (order != 0) return order;
                }
                return x.Count.CompareTo(y.Count);
            }
            return a.GetValue<int>().CompareTo(b.GetValue<int>());
        }

        public static JsonArray Hex(Cube cube) => [cube.Q, cube.R];

        public JsonArray Hex(int tileIndex) => Hex(CubeOf(Board, tileIndex));

        public int TileIndex(JsonNode hex) => TileIndexOf(Board, new Cube(hex[0]!.GetValue<int>(), hex[1]!.GetValue<int>(), -hex[0]!.GetValue<int>() - hex[1]!.GetValue<int>()));

        public JsonNode Intersection(int index) => JsonNode.Parse(_intersections[index])!;

        public JsonNode Path(int edge) => JsonNode.Parse(_paths[edge])!;

        /// <summary>
        /// The intersection index of a key (its three hexes in any order), or -1
        /// </summary>
        public int IntersectionIndex(JsonNode key)
        {
            string sorted = Canonical(new JsonArray(key.AsArray().Select(h => (JsonNode)h!.DeepClone()).Order(Comparer<JsonNode>.Create(CompareJson)).ToArray()));
            return _intersectionIndex.TryGetValue(sorted, out int index) ? index : -1;
        }

        /// <summary>
        /// The edge index of a path (its two intersections in any order), or -1
        /// </summary>
        public int EdgeIndex(JsonNode path)
        {
            int a = IntersectionIndex(path[0]!), b = IntersectionIndex(path[1]!);
            if (a < 0 || b < 0) return -1;
            return _pathIndex.TryGetValue(PathJson((a, b)), out int edge) ? edge : -1;
        }

        /// <summary>
        /// A card set as the protocol writes it: {name: count}, only the cards held, in the order of ResourceNames
        /// </summary>
        public static JsonObject Cards(CardSet<ResourceCardType> cards)
        {
            JsonObject set = [];
            foreach ((ResourceCardType type, string name) in ResourceNames)
                if (cards.Get(type) > 0) set[name] = cards.Get(type);
            return set;
        }

        public static CardSet<ResourceCardType> Cards(JsonNode set)
        {
            CardSet<ResourceCardType> cards = new();
            foreach ((string name, JsonNode? count) in set.AsObject())
                cards.Add(ResourceNames.Single(n => n.Value == name).Key, count!.GetValue<uint>());
            return cards;
        }

        /// <summary>
        /// The protocol's move of a Lanzenbrecher action
        /// </summary>
        public JsonArray Move(Action action) => action switch
        {
            RollAction => ["roll"],
            EndTurnAction => ["end_turn"],
            FirstInitialSettlementAction a => ["build_settlement", Intersection(a.IntersectionIndex)],
            SecondInitialSettlementAction a => ["build_settlement", Intersection(a.IntersectionIndex)],
            SettlementAction a => ["build_settlement", Intersection(a.IntersectionIndex)],
            FirstInitialRoadAction a => ["build_road", Path(a.EdgeIndex)],
            SecondInitialRoadAction a => ["build_road", Path(a.EdgeIndex)],
            RoadAction a => ["build_road", Path(a.EdgeIndex)],
            CityAction a => ["build_city", Intersection(a.IntersectionIndex)],
            BuyDevelopmentCardAction => ["buy_development"],
            KnightAction => ["play_knight"],
            RoadBuildingAction => ["play_road_building"],
            YearOfPlentyAction a => ["play_year_of_plenty", new JsonArray(new[] { a.FirstChoice }.Concat(a.SecondChoice is ResourceCardType second ? [second] : [])
                .Select(type => ResourceNames[type]).Order(StringComparer.Ordinal).Select(name => (JsonNode)name).ToArray())],
            MonopolyAction a => ["play_monopoly", ResourceNames[a.ChosenType]],
            RobberAction a => ["move_robber", new JsonObject { ["hex"] = Hex(a.TargetTileIndex), ["victim"] = a.TargetPlayerIndex is sbyte victim ? victim : null }],
            DiscardAction a => ["discard", Cards(a.SelectedCards)],
            FourToOneTradeAction a => BankTrade(a.InputType, 4, a.OutputType),
            ThreeToOneTradeAction a => BankTrade(a.InputType, 3, a.OutputType),
            TwoToOneTradeAction a => BankTrade(a.InputType, 2, a.OutputType),
            // Trades between players: the offer to a set of seats (a bit per seat), the answers, the offerer's choice
            OfferTradeAction a => ["offer", new JsonObject { ["give"] = Cards(a.Give), ["get"] = Cards(a.Get), ["to"] = Seats(a.Recipients) }],
            RespondToTradeAction a => ["respond", new JsonObject { ["accept"] = a.Accept }],
            CounterTradeAction a => ["counter", new JsonObject { ["give"] = Cards(a.Give), ["get"] = Cards(a.Get) }],
            ConfirmTradeAction a => ["confirm", new JsonObject { ["partner"] = a.Partner }],
            CancelTradeAction => ["cancel"],
            _ => throw new NotSupportedException($"No protocol move for {action.GetType().Name} yet."),
        };

        /// <summary>
        /// The Lanzenbrecher action of an answer to an open offer, a pick of a partner or a withdrawal, made by the seat (under
        /// JSettlers' flow these can come from another seat than the one in seat order); null for any other move or one that
        /// doesn't read as one of these. Whether the rules allow it is the caller's to check.
        /// </summary>
        public static Action? AnswerAction(JsonNode? move, sbyte seat)
        {
            if (move is not JsonArray { Count: >= 1 } named || named[0] is not JsonValue first || !first.TryGetValue(out string? kindName)) return null;
            JsonObject? terms = named.Count > 1 ? named[1] as JsonObject : null;
            if (kindName == "respond" && terms is not null && terms["accept"] is JsonValue accept && accept.TryGetValue(out bool accepted))
                return new RespondToTradeAction(seat, accepted);
            if (kindName == "confirm" && terms is not null && terms["partner"] is JsonValue partner && partner.TryGetValue(out int with) && with is >= 0 and < 8)
                return new ConfirmTradeAction(seat, (sbyte)with);
            if (kindName == "cancel")
                return new CancelTradeAction(seat);
            return null;
        }

        /// <summary>
        /// The Lanzenbrecher action of a trade move that no legal list holds, since its cards can be any (an offer or a
        /// counteroffer), made by the seat; null for any other move or one that doesn't read as either. Whether the rules
        /// allow it is the caller's to check.
        /// </summary>
        public static Action? TradeAction(JsonNode? move, sbyte seat)
        {
            if (move is not JsonArray { Count: 2 } parts || parts[0] is not JsonValue kind || !kind.TryGetValue(out string? name)
                || parts[1] is not JsonObject terms || ReadCards(terms["give"]) is not CardSet<ResourceCardType> give || ReadCards(terms["get"]) is not CardSet<ResourceCardType> get)
                return null;
            switch (name)
            {
                case "offer":
                    if (terms["to"] is not JsonArray to) return null;
                    byte recipients = 0;
                    foreach (JsonNode? recipient in to)
                    {
                        // Each seat once
                        if (recipient is not JsonValue value || !value.TryGetValue(out int number) || number is < 0 or > 7 || (recipients & 1 << number) != 0) return null;
                        recipients |= (byte)(1 << number);
                    }
                    return new OfferTradeAction(seat, give, get, recipients);
                case "counter":
                    return new CounterTradeAction(seat, give, get);
                default:
                    return null;
            }
        }

        // A card set of the protocol's resource names and counts, or null if it isn't one
        private static CardSet<ResourceCardType>? ReadCards(JsonNode? set)
        {
            if (set is not JsonObject cards) return null;
            CardSet<ResourceCardType> read = new();
            foreach ((string name, JsonNode? count) in cards)
            {
                ResourceCardType type = ResourceNames.FirstOrDefault(n => n.Value == name).Key;
                if (type == ResourceCardType.Unknown || count is not JsonValue value || !value.TryGetValue(out uint amount)) return null;
                read.Add(type, amount);
            }
            return read;
        }

        // The seats of a bit set, in order
        private static JsonArray Seats(byte seats) => new(Enumerable.Range(0, 8).Where(seat => (seats & 1 << seat) != 0).Select(seat => (JsonNode)seat).ToArray());

        private static JsonArray BankTrade(ResourceCardType give, uint count, ResourceCardType get) =>
            ["bank_trade", new JsonObject { ["give"] = new JsonObject { [ResourceNames[give]] = count }, ["get"] = new JsonObject { [ResourceNames[get]] = 1 } }];

        /// <summary>
        /// The protocol's board: every hex within reach of an intersection with its SGN code and number, the harbors on
        /// their paths, the robber
        /// </summary>
        public JsonObject BoardJson()
        {
            JsonArray tiles = [];
            Export export = Of(Board);
            foreach (TileEntry tile in export.Tiles.OrderBy(t => (t.Cube[0], t.Cube[1])))
            {
                Tile t = Board.Map.Values[tile.Index];
                tiles.Add(new JsonObject { ["tile"] = new JsonArray(tile.Cube[0], tile.Cube[1]), ["tCode"] = tile.Land ? TileCodes[t.Type] : "S", ["rNumber"] = t.Number is byte n ? n : null });
            }
            JsonArray harbors = [];
            for (int p = 0; p < Board.Ports.Count; p++)
                harbors.Add(new JsonObject { ["path"] = JsonNode.Parse(PathJson((export.Ports[p].Intersections[0], export.Ports[p].Intersections[1]))), ["hCode"] = HarborCodes[Board.Ports[p].Type] });
            return new JsonObject { ["tiles"] = tiles, ["harbors"] = harbors, ["robber"] = Hex(Board.GetRobberTileIndex()!.Value) };
        }

        /// <summary>
        /// A Lanzenbrecher board from the protocol's board (a classic board: 19 land hexes, the sea ring, harbors on coastal paths)
        /// </summary>
        public static Board BuildBoard(JsonNode board)
        {
            // Positions on Lanzenbrecher's map by axial hex, through the cube coordinates of any classic board
            Board template = MapGenerator.GenerateRandomClassic();
            (int x, int y) Position(JsonNode hex)
            {
                int index = TileIndexOf(template, new Cube(hex[0]!.GetValue<int>(), hex[1]!.GetValue<int>(), -hex[0]!.GetValue<int>() - hex[1]!.GetValue<int>()));
                return (index % (int)template.Map.Width, index / (int)template.Map.Width);
            }

            Dictionary<(int x, int y), (TileType type, byte? number)> land = [];
            foreach (JsonNode? tile in board["tiles"]!.AsArray())
            {
                string code = tile!["tCode"]!.GetValue<string>();
                if (code == "S") continue;
                TileType type = TileCodes.Single(c => c.Value == code).Key;
                land[Position(tile["tile"]!)] = (type, tile["rNumber"] is JsonNode number ? number.GetValue<byte>() : null);
            }

            // A harbor's path lies between its water hex and the land it faces: the two hexes both its intersections touch
            List<Port> ports = [];
            foreach (JsonNode? harbor in board["harbors"]!.AsArray())
            {
                JsonArray path = harbor!["path"]!.AsArray();
                Cube[] shared = path[0]!.AsArray().Select(ToCube).Intersect(path[1]!.AsArray().Select(ToCube)).ToArray();
                if (shared.Length != 2) throw new ArgumentException($"The harbor path {path.ToJsonString()} is no path.");
                Cube water = shared.Single(c => c.Distance == 3), coast = shared.Single(c => c.Distance <= 2);
                Direction.Tile facing = Enum.GetValues<Direction.Tile>().Single(d => water.Shift(d) == coast);
                (int x, int y) = Position(Hex(water));
                ports.Add(new Port((byte)x, (byte)y, facing, HarborCodes.Single(c => c.Value == harbor["hCode"]!.GetValue<string>()).Key));
            }
            return MapGenerator.CreateClassic(land, ports, Position(board["robber"]!));
        }

        private static Cube ToCube(JsonNode? hex) => new Cube(hex![0]!.GetValue<int>(), hex[1]!.GetValue<int>(), -hex[0]!.GetValue<int>() - hex[1]!.GetValue<int>());

        /// <summary>
        /// The protocol's prompt: what the acting player has to do now
        /// </summary>
        public static string Prompt(GameState state)
        {
            if (state.Turn.PendingTrade is TradeOffer offer) return offer.NextResponder != null ? "answer_offer" : "pick_partner";
            if (state.Turn.TypeOfRound is TurnState.RoundType.FirstInitial or TurnState.RoundType.SecondInitial)
            {
                // A settlement first, then its road
                List<Action> legal = LegalActionProvider.GetActionsForState(state, (sbyte)state.Turn.PlayerIndex);
                return legal.Any(a => a is FirstInitialSettlementAction or SecondInitialSettlementAction) ? "setup_settlement" : "setup_road";
            }
            if (state.Turn.MustDiscard) return "discard";
            if (state.Turn.MustMoveRobber) return "move_robber";
            if (FreeRoads(state) > 0) return "road_building";
            return state.Turn.MustRoll ? "roll" : "main";
        }

        /// <summary>
        /// The free roads of a Road Building card the player on turn can still place (none without a place for a road)
        /// </summary>
        public static uint FreeRoads(GameState state)
        {
            int player = state.Turn.PlayerIndex;
            uint free = state.Players[player].BuildingStock.FreeRoads;
            return free > 0 && state.Board.GetRoadSites(player) != UInt128.Zero ? free : 0;
        }

        /// <summary>
        /// The full state in the protocol's shape (every hand, as the referee sees it). Of the played development cards
        /// Lanzenbrecher counts only the knights.
        /// </summary>
        public JsonObject State(GameState state)
        {
            // Who acts now: the lowest seat still discarding after a 7, with an open offer the next addressee in seat order (then
            // the offerer), else the player on turn
            int current = state.Turn.MustDiscard ? Array.IndexOf(state.Turn.AwaitedPlayerDiscards, true)
                : state.Turn.PendingTrade is TradeOffer offer ? offer.NextResponder ?? offer.Offerer : state.Turn.PlayerIndex;
            JsonArray settlements = [], cities = [], roads = [];
            for (int i = 0; i < Board.Intersections.Count; i++)
            {
                Intersection.BuildingType building = state.Board.GetBuilding(i);
                if (building == Common.Intersection.BuildingType.Settlement) settlements.Add(new JsonArray(state.Board.GetBuildingOwner(i), Intersection(i)));
                else if (building == Common.Intersection.BuildingType.City) cities.Add(new JsonArray(state.Board.GetBuildingOwner(i), Intersection(i)));
            }
            for (int e = 0; e < Board.Edges.Count; e++)
                if (state.Board.GetRoad(e) != Edge.BuildingType.None) roads.Add(new JsonArray(state.Board.GetRoadOwner(e), Path(e)));

            JsonObject bank = Cards(state.ResourceBank);
            foreach (string name in ResourceNames.Values) bank[name] ??= 0;
            bank["development"] = state.DevelopmentBank.Count();

            JsonArray players = [];
            for (int seat = 0; seat < state.Players.Length; seat++)
            {
                PlayerState player = state.Players[seat];
                JsonObject hand = [], development = [];
                foreach ((ResourceCardType type, string name) in ResourceNames) hand[name] = player.ResourceCards.Get(type);
                foreach ((DevelopmentCardType type, string name) in CardNames) development[name] = player.DevelopmentCards.Get(type);
                players.Add(new JsonObject
                {
                    ["seat"] = seat,
                    ["hand_size"] = player.ResourceCards.Count(),
                    ["development_size"] = player.DevelopmentCards.Count(),
                    ["public_vp"] = player.VictoryPoints.Total - player.VictoryPoints.DevelopmentCardPoints,
                    ["vp"] = player.VictoryPoints.Total,
                    ["hand"] = hand,
                    ["development"] = development,
                    ["played"] = new JsonObject { ["knight"] = player.PlayedKnights },
                    ["longest_road"] = player.LongestRoadLength,
                    ["has_longest_road"] = player.VictoryPoints.LongestRoadPoints > 0,
                    ["has_largest_army"] = player.VictoryPoints.LargestArmyPoints > 0,
                    // The roads of a Road Building card are taken from the stock when it is played; until placed they are
                    // still off the board
                    ["pieces_left"] = new JsonObject { ["roads"] = player.BuildingStock.RemainingRoads + player.BuildingStock.FreeRoads, ["settlements"] = player.BuildingStock.RemainingSettlements, ["cities"] = player.BuildingStock.RemainingCities },
                });
            }
            return new JsonObject
            {
                ["prompt"] = Prompt(state),
                ["current"] = current,
                // Whose turn it is (the current player may be another one, e.g. a player discarding after a 7)
                ["on_turn"] = state.Turn.PlayerIndex,
                ["free_roads"] = FreeRoads(state),
                ["robber"] = Hex(state.Board.GetRobberTileIndex()!.Value),
                ["buildings"] = new JsonObject { ["settlements"] = settlements, ["cities"] = cities, ["roads"] = roads },
                ["bank"] = bank,
                ["players"] = players,
                ["offer"] = Offer(state),
            };
        }

        /// <summary>
        /// The state as a seat sees it: the full state without the other seats' hands, development cards and points with
        /// victory point cards (their counts stay)
        /// </summary>
        public JsonObject View(GameState state, int seat)
        {
            JsonObject view = State(state);
            foreach (JsonNode? player in view["players"]!.AsArray())
            {
                if (player!["seat"]!.GetValue<int>() == seat) continue;
                player["hand"] = null;
                player["development"] = null;
                player["vp"] = null;
            }
            return view;
        }

        /// <summary>
        /// A move's result as a seat may see it, from the move as it was applied (its history), or null for a move
        /// without one: the dice for everyone, a bought card only for the buyer, a stolen card for the thief and the victim
        /// ("unknown" for the others, null if nothing was stolen), the cards a monopoly took from each player. A null seat
        /// sees everything, as the referee does (for open-hand agents).
        /// </summary>
        public static JsonNode? Result(Action applied, int? seat) => applied switch
        {
            RollAction roll => new JsonObject { ["dice"] = new JsonArray(roll.History!.RollResult.First, roll.History.RollResult.Second) },
            BuyDevelopmentCardAction buy => new JsonObject { ["card"] = seat == null || seat == buy.PlayerIndex ? CardNames[buy.History!.DrawnType] : null },
            RobberAction robber => new JsonObject
            {
                ["stolen"] = robber.History!.StolenCard is not ResourceCardType stolen ? null
                    : seat == null || seat == robber.PlayerIndex || seat == robber.TargetPlayerIndex ? ResourceNames[stolen] : "unknown",
            },
            MonopolyAction monopoly => new JsonObject
            {
                ["taken"] = new JsonObject(monopoly.History!.TransferredCards.Select(t => KeyValuePair.Create(t.Item1.ToString(), (JsonNode?)t.Item2))),
            },
            _ => null,
        };

        /// <summary>
        /// The game's rules as the protocol's ruleset: the rulebook ("standard") and the options that differ from it
        /// </summary>
        public static JsonObject Ruleset(GameSettings settings)
        {
            JsonObject ruleset = new() { ["preset"] = "standard" };
            if (!settings.FreeTrade) ruleset["player_trading"] = "none";
            // JSettlers' flow: answers in any order, the first accept trades at once
            else if (settings.TradeFlow == TradeFlow.FirstAccept) ruleset["player_trading"] = "first_accept";
            if (settings.VictoryPoints != 10) ruleset["victory_points"] = settings.VictoryPoints;
            if (settings.RobberCardLimit != 7) ruleset["discard_limit"] = settings.RobberCardLimit;
            // The rulebook has no limit on offers; Lanzenbrecher's referee has one (10 unless set otherwise)
            if (settings.FreeTrade) ruleset["offer_limit"] = settings.TradeOfferLimit;
            return ruleset;
        }

        // The development cards of a classic game
        private static readonly IReadOnlyDictionary<DevelopmentCardType, uint> DevelopmentDeck = new Dictionary<DevelopmentCardType, uint>
        {
            [DevelopmentCardType.Knight] = 14, [DevelopmentCardType.VictoryPoint] = 5, [DevelopmentCardType.RoadBuilding] = 2,
            [DevelopmentCardType.YearOfPlenty] = 2, [DevelopmentCardType.Monopoly] = 2,
        };

        // The move that plays each card that can be played
        private static readonly IReadOnlyDictionary<DevelopmentCardType, string> PlayMoves = new Dictionary<DevelopmentCardType, string>
        {
            [DevelopmentCardType.Knight] = "play_knight", [DevelopmentCardType.RoadBuilding] = "play_road_building",
            [DevelopmentCardType.YearOfPlenty] = "play_year_of_plenty", [DevelopmentCardType.Monopoly] = "play_monopoly",
        };

        /// <summary>
        /// A game in the protocol's full state (every hand), on this codec's board: the buildings, hands, development cards,
        /// banks, points, Longest Road and Largest Army, whose turn it is and what has to be done. What the state doesn't
        /// say comes from the legal moves the referee listed for the player on turn: which of its development cards it may
        /// play now (the others count as bought this turn, or as none playable after one was played). The development bank
        /// holds the cards nobody holds or played (Lanzenbrecher's state counts only the knights played), as many as the state
        /// says. A finished game, an open offer between players and the last roll aren't rebuilt.
        /// </summary>
        public GameState BuildState(JsonNode state, JsonArray legal)
        {
            JsonArray players = state["players"]!.AsArray();
            // A board of its own, emptied: the codec's board stays as the referee sent it
            Board board = new(Board);
            board.Clear();
            GameState game = new(board, (uint)players.Count);
            int Number(JsonNode? n) => int.Parse(n!.ToJsonString());

            // Buildings, and the points they give
            foreach (JsonNode? entry in state["buildings"]!["settlements"]!.AsArray())
            {
                game.Board.SetBuilding(IntersectionIndex(entry![1]!), Common.Intersection.BuildingType.Settlement, (sbyte)Number(entry[0]));
                game.Players[Number(entry[0])].VictoryPoints.SettlementPoints++;
            }
            foreach (JsonNode? entry in state["buildings"]!["cities"]!.AsArray())
            {
                game.Board.SetBuilding(IntersectionIndex(entry![1]!), Common.Intersection.BuildingType.City, (sbyte)Number(entry[0]));
                game.Players[Number(entry[0])].VictoryPoints.CityPoints += 2;
            }
            foreach (JsonNode? entry in state["buildings"]!["roads"]!.AsArray())
                game.Board.SetRoad(EdgeIndex(entry![1]!), Edge.BuildingType.Road, (sbyte)Number(entry[0]));
            game.Board.Robber = Board.Map.Values[TileIndex(state["robber"]!)];
            SettlementAction.FullyRecalculatePortPrivileges(game);

            // Whose turn it is and what has to be done
            string prompt = state["prompt"]!.GetValue<string>();
            int current = Number(state["current"]);
            int onTurn = state["on_turn"] is JsonNode turn ? Number(turn) : current;
            JsonNode me = players[onTurn]!;
            int settlementsOnTurn = state["buildings"]!["settlements"]!.AsArray().Count(e => Number(e![0]) == onTurn)
                + state["buildings"]!["cities"]!.AsArray().Count(e => Number(e![0]) == onTurn);
            HashSet<string> listed = legal.Select(m => m![0]!.GetValue<string>()).ToHashSet();
            TurnState t = game.Turn;
            t.PlayerIndex = onTurn;
            t.TypeOfRound = prompt switch
            {
                "setup_settlement" => settlementsOnTurn == 0 ? TurnState.RoundType.FirstInitial : TurnState.RoundType.SecondInitial,
                "setup_road" => settlementsOnTurn == 1 ? TurnState.RoundType.FirstInitial : TurnState.RoundType.SecondInitial,
                _ => TurnState.RoundType.Normal,
            };
            // The rounds the setup counts; in play only the first roll of the game matters (the value net's features). The
            // sizes of a hand: as the state gives them, or (from older referees) as its cards add up
            int Size(JsonNode p, string size, string cards) => p[size] is JsonNode n ? Number(n) : p[cards]!.AsObject().Sum(c => Number(c.Value));
            bool untouched = players.All(p => Number(p!["pieces_left"]!["settlements"]) == BuildingStock.MAX_SETTLEMENTS - 2
                && Number(p!["pieces_left"]!["roads"]) == BuildingStock.MAX_ROADS - 2 && Size(p!, "hand_size", "hand") + Size(p!, "development_size", "development") <= 3);
            t.RoundCounter = t.TypeOfRound switch
            {
                TurnState.RoundType.FirstInitial => 0,
                TurnState.RoundType.SecondInitial => 1,
                _ => prompt == "roll" && onTurn == 0 && untouched ? 2 : 3,
            };
            // Road Building may come before the roll, its roads then before it too (the legal moves tell); after a robber
            // moved by a knight the state can't tell, it counts as rolled
            t.MustRoll = prompt == "roll" || prompt == "road_building" && listed.Contains("roll");
            // After a 7 the seats discard from the lowest up, as the protocol orders them: the current one and every later seat
            // above the limit still have to, and the robber moves after them
            t.MustMoveRobber = prompt is "move_robber" or "discard";
            if (prompt == "discard")
                for (int seat = current; seat < players.Count; seat++)
                    t.AwaitedPlayerDiscards[seat] = seat == current || Size(players[seat]!, "hand_size", "hand") > game.Settings.RobberCardLimit;
            // An open offer between players, with its answers and counteroffers (Offer writes it); it counts as one offer of
            // the turn, since the state doesn't say how many came before it
            t.PendingTrade = TradeOfferOf(state["offer"], players.Count);
            if (t.PendingTrade != null) t.TradeOffersThisTurn = Math.Max(t.TradeOffersThisTurn, 1);

            // Hands, development cards, played knights, pieces, Longest Road and Largest Army; the cards held or played count
            // against the deck (of the played ones the referee may count only knights, as Lanzenbrecher does)
            uint[] heldCards = new uint[Enum.GetValues<DevelopmentCardType>().Length];
            for (int seat = 0; seat < players.Count; seat++)
            {
                JsonNode p = players[seat]!;
                PlayerState player = game.Players[seat];
                player.ResourceCards = Cards(p["hand"]!);
                player.DevelopmentCards = new CardSet<DevelopmentCardType>();
                foreach ((DevelopmentCardType type, string name) in CardNames)
                {
                    uint count = (uint)Number(p["development"]![name]);
                    player.DevelopmentCards.Add(type, count);
                    heldCards[(int)type] += count;
                }
                player.PlayedKnights = (uint)Number(p["played"]!["knight"]);
                foreach ((DevelopmentCardType type, string name) in CardNames)
                    if (p["played"]![name] is JsonNode played) heldCards[(int)type] += (uint)Number(played);
                player.VictoryPoints.LongestRoadPoints = (byte)(p["has_longest_road"]!.GetValue<bool>() ? 2 : 0);
                player.VictoryPoints.LargestArmyPoints = (byte)(p["has_largest_army"]!.GetValue<bool>() ? 2 : 0);
                player.VictoryPoints.DevelopmentCardPoints = (byte)player.DevelopmentCards.Get(DevelopmentCardType.VictoryPoint);
                uint free = seat == onTurn && prompt == "road_building" ? (uint)Number(state["free_roads"]) : 0;
                player.BuildingStock.FreeRoads = free;
                player.BuildingStock.RemainingRoads = (uint)Number(p["pieces_left"]!["roads"]) - free;
                player.BuildingStock.RemainingSettlements = (uint)Number(p["pieces_left"]!["settlements"]);
                player.BuildingStock.RemainingCities = (uint)Number(p["pieces_left"]!["cities"]);
            }

            // The development cards the player on turn may play: a listed card is playable; the others were bought this
            // turn, or none is playable because one was played already (Road Building's roads are placed after playing it)
            PlayerState mine = game.Players[onTurn];
            bool anyListed = PlayMoves.Values.Any(listed.Contains);
            bool heldPlayable = PlayMoves.Keys.Any(type => mine.DevelopmentCards.Get(type) > 0);
            t.HasPlayedDevelopmentCard = prompt == "road_building" || prompt is "roll" or "main" && heldPlayable && !anyListed;
            if (prompt is "roll" or "main" && anyListed)
                foreach ((DevelopmentCardType type, string move) in PlayMoves)
                    if (!listed.Contains(move) && mine.DevelopmentCards.Get(type) > 0)
                        mine.NewDevelopmentCards.Add(type, mine.DevelopmentCards.Get(type));

            // The banks: the resources the state names (an unlimited bank's count below 0, the hands holding more than 19, as
            // empty); the development cards nobody holds or played, cut to its count
            game.ResourceBank = new CardSet<ResourceCardType>();
            foreach ((ResourceCardType type, string name) in ResourceNames)
                game.ResourceBank.Add(type, (uint)Math.Max(0, Number(state["bank"]![name])));
            List<DevelopmentCardType> deck = [];
            foreach ((DevelopmentCardType type, uint total) in DevelopmentDeck)
                for (uint i = heldCards[(int)type]; i < total; i++) deck.Add(type);
            int size = Number(state["bank"]!["development"]);
            // Unknown plays (only knights are counted) leave the deck too large: the non-knight cards go first
            deck = deck.OrderBy(type => type == DevelopmentCardType.Knight ? 1 : 0).Skip(Math.Max(0, deck.Count - size)).ToList();
            while (deck.Count < size) deck.Add(DevelopmentCardType.Knight);
            game.DevelopmentBank = new CardSet<DevelopmentCardType>();
            foreach (DevelopmentCardType type in deck) game.DevelopmentBank.Add(type, 1);

            // Each road's length by Lanzenbrecher's count, as every move from here on counts it (a referee may count differently);
            // the award stays the referee's
            for (int seat = 0; seat < players.Count; seat++) game.CalculateLongestRoad(seat);
            for (int seat = 0; seat < players.Count; seat++)
                game.Players[seat].VictoryPoints.LongestRoadPoints = (byte)(players[seat]!["has_longest_road"]!.GetValue<bool>() ? 2 : 0);
            return game;
        }

        /// <summary>
        /// The open offer of a protocol state (as Offer writes it), or null without one
        /// </summary>
        public static TradeOffer? TradeOfferOf(JsonNode? offer, int players)
        {
            if (offer is not JsonObject open) return null;
            // Seats as numbers however they were written (parsed text, or values of any integer type)
            static int Seat(JsonNode n) => int.Parse(n.ToJsonString());
            byte recipients = 0;
            foreach (JsonNode? seat in open["to"]!.AsArray())
                recipients |= (byte)(1 << Seat(seat!));
            TradeOffer trade = new((sbyte)Seat(open["from"]!), Cards(open["give"]!), Cards(open["get"]!), recipients, players);
            foreach ((string seat, JsonNode? answer) in open["answers"]?.AsObject() ?? [])
                trade.Responses[int.Parse(seat)] = answer!.GetValue<string>() switch
                {
                    "accept" => TradeResponse.Accepted,
                    "decline" => TradeResponse.Rejected,
                    "counter" => TradeResponse.Countered,
                    _ => TradeResponse.Pending,
                };
            // Counteroffers in seat order, as Offer writes them (in the order they came: seat order under the rulebook)
            foreach ((string seat, JsonNode? terms) in (open["counters"]?.AsObject() ?? []).OrderBy(c => int.Parse(c.Key)))
                trade.Counters.Add(new TradeCounter(sbyte.Parse(seat), Cards(terms!["give"]!), Cards(terms["get"]!)));
            return trade;
        }

        /// <summary>
        /// The open offer between players: who offered what to whom, the answers so far (accept, decline or counter) and
        /// the counteroffers in the offerer's terms; null without one
        /// </summary>
        public static JsonObject? Offer(GameState state)
        {
            if (state.Turn.PendingTrade is not TradeOffer offer) return null;
            JsonObject answers = [], counters = [];
            for (int seat = 0; seat < offer.Responses.Length; seat++)
            {
                string? answer = offer.Responses[seat] switch
                {
                    TradeResponse.Accepted => "accept",
                    TradeResponse.Rejected => "decline",
                    TradeResponse.Countered => "counter",
                    _ => null,
                };
                if (answer != null) answers[seat.ToString()] = answer;
            }
            foreach (TradeCounter counter in offer.Counters)
                counters[counter.Player.ToString()] = new JsonObject { ["give"] = Cards(counter.Give), ["get"] = Cards(counter.Get) };
            JsonObject result = new()
            {
                ["from"] = offer.Offerer,
                ["give"] = Cards(offer.Give),
                ["get"] = Cards(offer.Get),
                ["to"] = new JsonArray(Enumerable.Range(0, offer.Responses.Length).Where(seat => offer.Responses[seat] != TradeResponse.NotAsked).Select(seat => (JsonNode)seat).ToArray()),
                ["answers"] = answers,
            };
            if (counters.Count > 0) result["counters"] = counters;
            return result;
        }

        /// <summary>
        /// JSON with every object's members sorted by name, so that equal values have equal text
        /// </summary>
        public static string Canonical(JsonNode? node) => JsonSerializer.Serialize(Sorted(node));

        /// <summary>
        /// A state's hash as the protocol's decisions carry it: the first 16 hex digits of the SHA-256 of its canonical
        /// JSON (keys sorted, no whitespace, UTF-8)
        /// </summary>
        public static string Hash(JsonNode? state) =>
            Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(Canonical(state))))[..16].ToLowerInvariant();

        private static JsonNode? Sorted(JsonNode? node) => node switch
        {
            JsonObject o => new JsonObject(o.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => KeyValuePair.Create(p.Key, Sorted(p.Value)))),
            JsonArray a => new JsonArray(a.Select(Sorted).ToArray()),
            null => null,
            _ => node.DeepClone(),
        };
    }
}
