using Common.Actions;
using System.Text.Json;
using System.Text.Json.Nodes;
using static Common.BoardGeometry;
using static Common.Tile;
using Action = Common.Actions.Action;

namespace Common.Osp
{
    /// <summary>
    /// Hexfarer's boards, moves and states in the Open Seat Protocol (see Bridges/PROTOCOL.md): hexes as axial [q, r], an
    /// intersection as its three hexes, a path as its two intersections, cards by name. Seats are Hexfarer's player
    /// indices, which are the turn order. One codec per board, since the keys depend only on its geometry.
    /// </summary>
    public sealed class OspCodec
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

        public OspCodec(Board board)
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
        /// The protocol's move of a Hexfarer action
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
            _ => throw new NotSupportedException($"No protocol move for {action.GetType().Name} yet."),
        };

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
        /// A Hexfarer board from the protocol's board (a classic board: 19 land hexes, the sea ring, harbors on coastal paths)
        /// </summary>
        public static Board BuildBoard(JsonNode board)
        {
            // Positions on Hexfarer's map by axial hex, through the cube coordinates of any classic board
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
        /// Hexfarer counts only the knights.
        /// </summary>
        public JsonObject State(GameState state)
        {
            int current = state.Turn.MustDiscard ? Array.IndexOf(state.Turn.AwaitedPlayerDiscards, true) : state.Turn.PlayerIndex;
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
                ["free_roads"] = FreeRoads(state),
                ["robber"] = Hex(state.Board.GetRobberTileIndex()!.Value),
                ["buildings"] = new JsonObject { ["settlements"] = settlements, ["cities"] = cities, ["roads"] = roads },
                ["bank"] = bank,
                ["players"] = players,
            };
        }

        /// <summary>
        /// The state as a seat sees it (stub)
        /// </summary>
        public JsonObject View(GameState state, int seat) => throw new NotImplementedException();

        /// <summary>
        /// A move's result as a seat may see it (stub)
        /// </summary>
        public static JsonNode? Result(Action applied, int seat) => throw new NotImplementedException();

        /// <summary>
        /// The game's rules as the protocol's ruleset (stub)
        /// </summary>
        public static JsonObject Ruleset(GameSettings settings) => throw new NotImplementedException();

        /// <summary>
        /// JSON with every object's members sorted by name, so that equal values have equal text
        /// </summary>
        public static string Canonical(JsonNode? node) => JsonSerializer.Serialize(Sorted(node));

        private static JsonNode? Sorted(JsonNode? node) => node switch
        {
            JsonObject o => new JsonObject(o.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => KeyValuePair.Create(p.Key, Sorted(p.Value)))),
            JsonArray a => new JsonArray(a.Select(Sorted).ToArray()),
            null => null,
            _ => node.DeepClone(),
        };
    }
}
