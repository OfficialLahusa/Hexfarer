using Agents;
using Common;
using Common.Actions;
using Common.Osp;
using OpenSeat;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Hexfarer's side of the Open Seat Protocol: the codec (boards, moves, states) and the replay harness, which plays
    /// Catanatron's recorded games (Fixtures/OpenSeat, from Bridges/catanatron/record.py --bots R,W,R,W --games 6 --seed 100)
    /// through Hexfarer's rules. Catanatron's log is the independent reference: a wrong board, move, result or state in
    /// the codec shows up as a difference there.
    /// </summary>
    public class OpenSeatTests
    {
        // A JSON number as int, whatever type it was written from
        private static int Int(JsonNode? number) => int.Parse(number!.ToJsonString());

        private static string Fixture => TestUtil.FixturePath("OpenSeat", "catanatron-rw.jsonl.gz");

        private static IEnumerable<GameState> RandomGames(int games = 3, int maxMoves = 3000)
        {
            for (int seed = 1; seed <= games; seed++)
            {
                TestUtil.Seed(seed);
                GameState state = TestUtil.NewGame();
                Agent[] agents = TestUtil.RandomAgents();
                for (int moves = 0; !state.HasEnded && moves < maxMoves; moves++)
                {
                    yield return state;
                    agents[TestUtil.ActingPlayer(state)].Act(state).Apply(state);
                }
            }
        }

        [Fact]
        public void CatanatronsGamesReplayThroughHexfarersRules()
        {
            Replay replay = new();
            Replay.Report report = replay.Run(Replay.Read(Fixture));
            int steps = Replay.Read(Fixture).Count(m => m["type"]!.GetValue<string>() == "step");

            Assert.Equal(6, report.Games);
            // One game reaches the known difference in Longest Road and stops there; the others agree to the end, every
            // state and every list of legal moves
            Assert.Equal(1, report.Diverged);
            Assert.Equal(["state: Longest Road (known: Catanatron doesn't count a road's segment ending at an opponent's building)"], report.Counts.Keys);
            Assert.InRange(report.Moves, steps * 3 / 4, steps);
        }

        /// <summary>
        /// The first game of the fixture (which agrees to the end), with one message changed
        /// </summary>
        private static Replay.Report ReplayChanged(Func<JsonNode, bool> where, Action<JsonNode> change)
        {
            string first = Replay.Read(Fixture).First()["game_id"]!.GetValue<string>();
            List<JsonNode> messages = Replay.Read(Fixture).TakeWhile(m => m["game_id"]!.GetValue<string>() == first).ToList();
            if (messages.FirstOrDefault(where) is JsonNode target) change(target);
            return new Replay().Run(messages);
        }

        private static bool At(JsonNode message, string type, int n) => message["type"]!.GetValue<string>() == type && message["n"] is JsonNode at && Int(at) == n;

        [Fact]
        public void TheHarnessFindsEveryKindOfDifference()
        {
            Assert.Equal(0, ReplayChanged(_ => false, _ => { }).Diverged);

            // A hand the referee holds differently
            Replay.Report hand = ReplayChanged(m => At(m, "decide", 60), m => m["state"]!["players"]![1]!["hand"]!["ore"] = Int(m["state"]!["players"]![1]!["hand"]!["ore"]) + 1);
            Assert.Equal((1, 1), (hand.Diverged, hand.Counts.GetValueOrDefault("state: hand")));

            // A legal move only one side has (the game goes on)
            Replay.Report missing = ReplayChanged(m => At(m, "decide", 40), m => m["legal"]!.AsArray().RemoveAt(0));
            Assert.Equal(0, missing.Diverged);
            Assert.Single(missing.Counts, c => c.Key.StartsWith("legal: only Hexfarer allows"));
            Replay.Report extra = ReplayChanged(m => At(m, "decide", 40), m => m["legal"]!.AsArray().Add(JsonNode.Parse("[\"play_monopoly\",\"ore\"]")));
            Assert.Equal(1, extra.Counts.GetValueOrDefault("legal: only the referee allows play_monopoly (main)") + extra.Counts.GetValueOrDefault("legal: only the referee allows play_monopoly (roll)"));

            // Other dice: the yields differ from the referee's from the next move on
            JsonNode firstRoll = Replay.Read(Fixture).First(m => m["type"]!.GetValue<string>() == "step" && m["action"]![0]!.GetValue<string>() == "roll" && m["result"]!["dice"]!.AsArray().Sum(d => Int(d)) is not 7 and not 2);
            int rollAt = Int(firstRoll["n"]);
            Replay.Report dice = ReplayChanged(m => At(m, "step", rollAt), m => m["result"]!["dice"] = new JsonArray(1, 1));
            Assert.Equal(1, dice.Diverged);
            Assert.True(dice.Counts.Keys.Single().StartsWith("state: "), dice.ToString());

            // A move Hexfarer doesn't allow
            Replay.Report illegal = ReplayChanged(m => At(m, "step", 0), m => m["action"] = JsonNode.Parse("[\"end_turn\"]"));
            Assert.Equal((1, 1), (illegal.Diverged, illegal.Counts.GetValueOrDefault("step: Hexfarer doesn't allow end_turn")));
        }

        [Fact]
        public void BoardsWriteAndReadBack()
        {
            for (int seed = 1; seed <= 5; seed++)
            {
                TestUtil.Seed(seed);
                Board board = MapGenerator.GenerateRandomClassic();
                JsonObject written = new OspCodec(board).BoardJson();
                Board read = OspCodec.BuildBoard(written);

                Assert.Equal(OspCodec.Canonical(written), OspCodec.Canonical(new OspCodec(read).BoardJson()));
                for (int i = 0; i < board.Map.Length; i++)
                    Assert.Equal((board.Map.Values[i].Type, board.Map.Values[i].Number), (read.Map.Values[i].Type, read.Map.Values[i].Number));
                Assert.Equal(board.Ports, read.Ports);
                Assert.Equal(board.GetRobberTileIndex(), read.GetRobberTileIndex());
                // 37 hexes, 9 harbors, SGN's codes
                Assert.Equal(37, written["tiles"]!.AsArray().Count);
                Assert.Equal(9, written["harbors"]!.AsArray().Count);
                Assert.Equal(18, written["tiles"]!.AsArray().Count(t => t!["tCode"]!.GetValue<string>() == "S"));
            }
        }

        [Fact]
        public void EveryLegalMoveHasItsOwnProtocolMoveAndKeysReadBack()
        {
            int decisions = 0;
            OspCodec? codec = null;
            foreach (GameState state in RandomGames())
            {
                if (codec?.Board.Map != state.Board.Map) codec = new OspCodec(state.Board);
                sbyte player = TestUtil.ActingPlayer(state);
                List<Action> legal = LegalActionProvider.GetActionsForState(state, player);
                // Year of Plenty lists both orders of two cards, the protocol one set
                List<string> moves = legal.Where(a => a is not YearOfPlentyAction).Select(a => OspCodec.Canonical(codec.Move(a))).ToList();
                Assert.Equal(moves.Count, moves.Distinct().Count());
                decisions++;
            }
            Assert.True(decisions > 1000, $"{decisions} decisions");

            // Every intersection and edge reads back from its key, in any order of its parts
            TestUtil.Seed(9);
            OspCodec keys = new OspCodec(MapGenerator.GenerateRandomClassic());
            for (int i = 0; i < keys.Board.Intersections.Count; i++)
            {
                JsonArray key = keys.Intersection(i).AsArray();
                Assert.Equal(i, keys.IntersectionIndex(key));
                Assert.Equal(i, keys.IntersectionIndex(new JsonArray(key.Reverse().Select(h => (JsonNode)h!.DeepClone()).ToArray())));
            }
            for (int e = 0; e < keys.Board.Edges.Count; e++)
            {
                JsonArray path = keys.Path(e).AsArray();
                Assert.Equal(e, keys.EdgeIndex(path));
                Assert.Equal(e, keys.EdgeIndex(new JsonArray(path[1]!.DeepClone(), path[0]!.DeepClone())));
            }
        }

        [Fact]
        public void TheStateMeansWhatHexfarerHolds()
        {
            // Names by meaning, written here rather than taken from the codec
            (string name, ResourceCardType type)[] resources = [("lumber", ResourceCardType.Lumber), ("brick", ResourceCardType.Brick), ("wool", ResourceCardType.Wool), ("grain", ResourceCardType.Grain), ("ore", ResourceCardType.Ore)];
            (string name, DevelopmentCardType type)[] cards = [("knight", DevelopmentCardType.Knight), ("road_building", DevelopmentCardType.RoadBuilding), ("year_of_plenty", DevelopmentCardType.YearOfPlenty), ("monopoly", DevelopmentCardType.Monopoly), ("victory_point", DevelopmentCardType.VictoryPoint)];
            OspCodec? codec = null;
            int checkedStates = 0;
            foreach (GameState state in RandomGames(maxMoves: 1500))
            {
                if (codec?.Board.Map != state.Board.Map) codec = new OspCodec(state.Board);
                JsonObject view = codec.State(state);
                foreach ((string name, ResourceCardType type) in resources)
                    Assert.Equal(state.ResourceBank.Get(type), (uint)Int(view["bank"]![name]));
                Assert.Equal(state.DevelopmentBank.Count(), (uint)Int(view["bank"]!["development"]));
                for (int seat = 0; seat < state.Players.Length; seat++)
                {
                    JsonNode player = view["players"]![seat]!;
                    foreach ((string name, ResourceCardType type) in resources)
                        Assert.Equal(state.Players[seat].ResourceCards.Get(type), (uint)Int(player["hand"]![name]));
                    foreach ((string name, DevelopmentCardType type) in cards)
                        Assert.Equal(state.Players[seat].DevelopmentCards.Get(type), (uint)Int(player["development"]![name]));
                    // The pieces on the board are the ones that left the stock
                    int Placed(string kind) => view["buildings"]![kind]!.AsArray().Count(b => Int(b![0]) == seat);
                    Assert.Equal(15, Placed("roads") + Int(player["pieces_left"]!["roads"]));
                    Assert.Equal(5, Placed("settlements") + Placed("cities") + Int(player["pieces_left"]!["settlements"]) - (4 - Int(player["pieces_left"]!["cities"])));
                    Assert.Equal(4, Placed("cities") + Int(player["pieces_left"]!["cities"]));
                }
                // The prompt to roll is exactly when rolling is legal, except that Hexfarer lets a player with a Road
                // Building card's roads unplaced roll and give them up (road_building: optional)
                sbyte acting = TestUtil.ActingPlayer(state);
                string prompt = view["prompt"]!.GetValue<string>();
                Assert.Equal(LegalActionProvider.GetActionsForState(state, acting).Any(a => a is RollAction), prompt == "roll" || prompt == "road_building" && state.Turn.MustRoll);
                checkedStates++;
            }
            Assert.True(checkedStates > 1000);
        }
    }
}
