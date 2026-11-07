using Common;
using Common.Osp;
using Common.Serialization;
using OpenSeat;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The relay (OpenSeat relay): an outside referee's games as Hexfarer save files, for the client to show. Each game
    /// is played through Hexfarer's rules with the recorded results, as the replay harness does; where the rules part
    /// (a state, a move or a result Hexfarer can't follow), the file ends there and a new part starts from the
    /// referee's state. A save file holds where its part ended and the moves since it began, so the client can step back
    /// through them to its start.
    /// </summary>
    public class OpenSeatRelayTests
    {
        private static int Int(JsonNode? number) => int.Parse(number!.ToJsonString());

        // Catanatron's games (see OpenSeatTests): five agree with Hexfarer's rules to the end, one reaches the known
        // difference in Longest Road
        private static string Fixture => TestUtil.FixturePath("OpenSeat", "catanatron-rw.jsonl.gz");

        private static List<JsonNode> FirstGame()
        {
            string first = Replay.Read(Fixture).First()["game_id"]!.GetValue<string>();
            return Replay.Read(Fixture).TakeWhile(m => m["game_id"]!.GetValue<string>() == first).ToList();
        }

        private static bool At(JsonNode message, string type, int n) => message["type"]!.GetValue<string>() == type && message["n"] is JsonNode at && Int(at) == n;

        // A game's state as the protocol writes it, without whose move is asked (a finished game asks nothing)
        private static string Written(OspCodec codec, GameState state)
        {
            JsonObject json = codec.State(state);
            json.Remove("prompt");
            return OspCodec.Canonical(json);
        }

        // The players as Hexfarer writes them, in the fields the referee's state has (older logs have fewer); of the cards
        // played Hexfarer keeps the knights, and it counts the roads' lengths itself
        private static void SamePlayers(JsonNode theirs, GameState state)
        {
            JsonArray ours = new OspCodec(state.Board).State(state)["players"]!.AsArray();
            for (int seat = 0; seat < ours.Count; seat++)
            {
                Assert.Equal(OspCodec.Canonical(theirs["players"]![seat]!["played"]!["knight"]), OspCodec.Canonical(ours[seat]!["played"]!["knight"]));
                foreach ((string field, JsonNode? value) in theirs["players"]![seat]!.AsObject().Where(f => f.Key is not ("played" or "longest_road")))
                    Assert.True(OspCodec.Canonical(value) == OspCodec.Canonical(ours[seat]![field]), $"seat {seat}, {field}: referee {value?.ToJsonString()}, Hexfarer {ours[seat]![field]?.ToJsonString()}");
            }
        }

        [Fact]
        public void EveryGameOfALogBecomesSaveFilesThatEndAsTheRefereesGame()
        {
            List<JsonNode> messages = Replay.Read(Fixture).ToList();
            List<Relay.Part> parts = new Relay().Run(messages).ToList();
            foreach (IGrouping<string, Relay.Part> game in parts.GroupBy(p => p.GameId))
            {
                List<JsonNode> own = messages.Where(m => m["game_id"]!.GetValue<string>() == game.Key).ToList();
                JsonNode after = own.Last();
                Assert.Equal("after", after["type"]!.GetValue<string>());
                // Every step is in a part, once, in order, and the parts are numbered from 1
                Assert.Equal(own.Count(m => m["type"]!.GetValue<string>() == "step"), game.Sum(p => p.Actions.Count));
                Assert.Equal(Enumerable.Range(1, game.Count()), game.Select(p => p.Number));
                Assert.Equal(0, game.First().FirstMove);
                // The last part ends as the referee's game did: the same points, the same winner
                Relay.Part last = game.Last();
                Assert.Null(last.Cut);
                Assert.Equal(after["vp"]!.AsArray().Select(Int), last.End.Players.Select(p => (int)p.VictoryPoints.Total));
                Assert.Equal(after["winner"] is JsonNode w ? Int(w) : -1, last.End.HasEnded ? last.End.Turn.PlayerIndex : -1);
            }
            Assert.Equal(6, parts.Select(p => p.GameId).Distinct().Count());
            // The game with the difference in Longest Road goes on in later parts, from the referee's state each time; the
            // others are one part each
            List<Relay.Part> cuts = parts.Where(p => p.Cut != null).ToList();
            Assert.NotEmpty(cuts);
            Assert.All(cuts, c => Assert.Contains("Longest Road", c.Cut));
            Assert.Single(cuts.Select(c => c.GameId).Distinct());
            Assert.Equal(5 + cuts.Count + 1, parts.Count);
            foreach (Relay.Part cut in cuts)
            {
                Relay.Part next = parts[parts.IndexOf(cut) + 1];
                Assert.Equal((cut.GameId, cut.Number + 1), (next.GameId, next.Number));
                Assert.Equal(cut.FirstMove + cut.Actions.Count, next.FirstMove);
            }
        }

        [Fact]
        public void ASaveFileStepsBackToWhereItsPartBegan()
        {
            string directory = Path.Combine(Path.GetTempPath(), $"relay-{Guid.NewGuid():N}");
            try
            {
                Relay relay = new();
                List<Relay.Part> parts = relay.Write(Replay.Read(Fixture), directory).ToList();
                Assert.Equal(parts.Count, Directory.GetFiles(directory, "*.yaml").Length);
                foreach (Relay.Part part in parts)
                {
                    // Read back as the client reads it: the part's end, and its moves undone one by one lead to its start
                    SaveFile save = SaveFileSerializer.Load(Path.Combine(directory, Relay.FileName(part)));
                    OspCodec codec = new(save.GameState.Board);
                    Assert.Equal(Written(codec, part.End), Written(codec, save.GameState));
                    Assert.Equal(part.Actions.Count, save.PlayedActions.Count);
                    Assert.Empty(save.UndoHistory);
                    GameState state = save.GameState;
                    for (int i = save.PlayedActions.Count - 1; i >= 0; i--) save.PlayedActions[i].Revert(state);
                    Assert.Equal(Written(codec, part.Start), Written(codec, state));
                }
                // A game's first part is named after it, the later ones with their number
                Assert.Contains(parts, p => p.Number == 2 && Relay.FileName(p).EndsWith("-part2.yaml"));
                Assert.All(parts.Where(p => p.Number == 1), p => Assert.Equal($"{p.GameId}.yaml", Relay.FileName(p)));
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
            }
        }

        [Fact]
        public void AResultThatLeadsElsewhereStartsANewPartFromTheRefereesState()
        {
            List<JsonNode> messages = FirstGame();
            // A roll of the middle game comes out as a seven: Hexfarer's game moves the robber, the referee's goes on as it was
            JsonNode roll = messages.First(m => m["type"]!.GetValue<string>() == "step" && Int(m["n"]) >= 40
                && m["action"]![0]!.GetValue<string>() == "roll" && m["result"]!["dice"]!.AsArray().Sum(d => Int(d)) != 7);
            int n = Int(roll["n"]);
            roll["result"]!["dice"] = new JsonArray(3, 4);
            Relay relay = new();
            List<Relay.Part> parts = relay.Run(messages).ToList();

            Assert.Equal(2, parts.Count);
            Assert.StartsWith("state", parts[0].Cut);
            // The first part ends with the seven, the second starts at the next decision from the referee's state
            Assert.Equal((0, n + 1), (parts[0].FirstMove, parts[0].Actions.Count));
            Assert.Equal(n + 1, parts[1].FirstMove);
            JsonNode decide = messages.First(m => At(m, "decide", n + 1));
            OspCodec codec = new(parts[1].Start.Board);
            SamePlayers(decide["state"]!, parts[1].Start);
            Assert.Equal(OspCodec.Canonical(decide["state"]!["robber"]), OspCodec.Canonical(codec.State(parts[1].Start)["robber"]));
            // The relay says where and why
            string line = Assert.Single(relay.Log);
            Assert.Contains($"move {n + 1}", line);
            Assert.Contains(parts[0].Cut!, line);
        }

        [Fact]
        public void AMoveHexfarerDoesntAllowEndsThePartBeforeIt()
        {
            List<JsonNode> messages = FirstGame();
            // The referee plays a move Hexfarer has no such thing as
            JsonNode step = messages.First(m => At(m, "step", 80));
            step["action"] = new JsonArray("build_settlement", new JsonArray(new JsonArray(99, 99, -198), new JsonArray(98, 99, -197), new JsonArray(99, 98, -197)));
            Relay relay = new();
            List<Relay.Part> parts = relay.Run(messages).ToList();

            Assert.Equal(2, parts.Count);
            Assert.StartsWith("step:", parts[0].Cut);
            // The part ends before the move; the next one starts at the decision after it, from the referee's state
            Assert.Equal(80, parts[0].Actions.Count);
            Assert.Equal(81, parts[1].FirstMove);
            JsonNode decide = messages.First(m => At(m, "decide", 81));
            SamePlayers(decide["state"]!, parts[1].Start);
            Assert.Single(relay.Log);
        }
    }
}
