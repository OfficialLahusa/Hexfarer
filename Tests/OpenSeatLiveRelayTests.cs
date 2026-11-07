using Common;
using Common.Actions;
using OpenSeat;
using Simulation;
using Common.OSP;
using System.Text.Json.Nodes;
using Xunit;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The live relay (OpenSeat relay --live): an outside referee's game, read from its log while the referee writes it,
    /// played into a hosted match that the client watches over the network (Join, Watch). Each move keeps the referee's
    /// outcome; where the rules part, the watcher's game starts again from the referee's state, as in the save files.
    /// </summary>
    public class OpenSeatLiveRelayTests
    {
        private static int Int(JsonNode? number) => int.Parse(number!.ToJsonString());

        private static string Fixture => TestUtil.FixturePath("OpenSeat", "catanatron-rw.jsonl.gz");

        // The fixture's games, each from its before to its after
        private static List<List<JsonNode>> Games() => Replay.Read(Fixture).GroupBy(m => m["game_id"]!.GetValue<string>()).Select(g => g.ToList()).ToList();

        private static bool At(JsonNode message, string type, int n) => message["type"]!.GetValue<string>() == type && message["n"] is JsonNode at && Int(at) == n;

        private static ServerConfig Config() => new() { Port = 0, AllowAnonymousSpectators = true };

        private static Task<RemoteSimulation> Watch(LiveRelay live) => RemoteSimulation.ConnectAsync(live.Uri, new Hello("Sam", null, Role.Spectator, null, null));

        // Applies the watcher's events until the condition holds
        private static bool Pump(RemoteSimulation client, Func<bool> condition, int milliseconds = 30000)
        {
            DateTime end = DateTime.UtcNow.AddMilliseconds(milliseconds);
            while (DateTime.UtcNow < end)
            {
                client.ProcessEvents();
                if (condition()) return true;
                Thread.Sleep(20);
            }
            client.ProcessEvents();
            return condition();
        }

        // Writes messages as the referee does, a line each
        private static void Append(string path, IEnumerable<JsonNode> messages) => File.AppendAllLines(path, messages.Select(m => m.ToJsonString()));

        // The state after a part's first moves
        private static GameState After(Relay.Part part, int moves)
        {
            GameState state = new(part.Start);
            foreach (Action action in part.Actions.Take(moves)) action.Clone().Apply(state);
            return state;
        }

        // What a watcher of the live relay ends with: the state, the entries since the last part started (the moves and the
        // differences taken over as edits) and the parts started
        private static (GameState End, int Entries, int Parts) Watched(IEnumerable<JsonNode> messages)
        {
            Relay relay = new(patch: true);
            (GameState? end, int entries, int parts) = (null, 0, 0);
            foreach (JsonNode message in messages)
                foreach (Relay.Event e in relay.Feed(message))
                    switch (e)
                    {
                        case Relay.Started:
                            (entries, parts) = (0, parts + 1);
                            break;
                        case Relay.Moved or Relay.Patched:
                            entries++;
                            break;
                        case Relay.Ended ended:
                            end = ended.Part.End;
                            break;
                    }
            return (end!, entries, parts);
        }

        private static string TempPath(string name)
        {
            string directory = Path.Combine(Path.GetTempPath(), "hexfarer-live-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            return Path.Combine(directory, name);
        }

        [Fact]
        public async Task AWatcherSeesTheRefereesGameWhileItIsWritten()
        {
            List<JsonNode> game = Games()[0];
            List<Relay.Part> parts = new Relay().Run(game).ToList();
            Assert.True(parts[0].Actions.Count > 60);
            int cut = game.FindIndex(m => At(m, "decide", 60));
            string path = TempPath("game.jsonl");
            Append(path, game.Take(cut));
            // The referee is in the middle of writing the next line
            string next = game[cut].ToJsonString();
            File.AppendAllText(path, next[..(next.Length / 2)]);

            await using LiveRelay live = await LiveRelay.StartAsync(path, Config(), pollMilliseconds: 20);
            using RemoteSimulation watcher = await Watch(live);
            // The game so far, with its moves to step back through
            Assert.True(Pump(watcher, () => watcher.Game.Record.State.Equals(After(parts[0], 60))), $"watcher at {watcher.Game.Record.History.Count} moves");
            Assert.Equal(60, watcher.Game.Record.History.Count);

            // The rest, as the referee writes it
            File.AppendAllText(path, next[(next.Length / 2)..] + "\n");
            Append(path, game.Skip(cut + 1));
            Assert.True(Pump(watcher, () => watcher.Game.Record.State.Equals(parts[^1].End)), $"watcher at {watcher.Game.Record.History.Count} moves");
            Assert.Equal(parts[^1].Actions.Count, watcher.Game.Record.History.Count);
            Assert.Equal(parts.Count, live.Parts);
        }

        [Fact]
        public async Task ARuleDifferenceStartsTheWatchersGameAgainFromTheRefereesState()
        {
            List<JsonNode> game = Games()[0];
            // A roll of the middle game comes out as a seven: Hexfarer's game moves the robber, the referee's goes on as it was
            JsonNode roll = game.First(m => m["type"]!.GetValue<string>() == "step" && Int(m["n"]) >= 40
                && m["action"]![0]!.GetValue<string>() == "roll" && m["result"]!["dice"]!.AsArray().Sum(d => Int(d)) != 7);
            roll["result"]!["dice"] = new JsonArray(3, 4);
            (GameState End, int Entries, int Parts) expected = Watched(game.Select(m => JsonNode.Parse(m.ToJsonString())!));
            Assert.True(expected.Parts >= 2);
            string path = TempPath("game.jsonl");
            Append(path, game);

            await using LiveRelay live = await LiveRelay.StartAsync(path, Config(), pollMilliseconds: 20);
            using RemoteSimulation watcher = await Watch(live);
            // The last part's moves only: it starts from the referee's state
            Assert.True(Pump(watcher, () => watcher.Game.Record.State.Equals(expected.End)), $"watcher at {watcher.Game.Record.History.Count} moves");
            Assert.Equal(expected.Entries, watcher.Game.Record.History.Count);
            Assert.Equal(expected.Parts, live.Parts);
        }

        [Fact]
        public async Task AHandTheRefereeHoldsDifferentlyIsTakenOverAsAnEditAndTheGameGoesOn()
        {
            List<JsonNode> game = Games()[0];
            List<Relay.Part> original = new Relay().Run(game.Select(m => JsonNode.Parse(m.ToJsonString())!)).ToList();
            Assert.Single(original);
            // At a roll (not a 7) the referee's state holds a card more in seat 0's small hand, one the bank has; its next
            // state doesn't
            int Count(JsonNode? cards, string name) => cards?[name] is JsonNode n ? Int(n) : 0;
            int at = Enumerable.Range(50, 400).First(n => game.FirstOrDefault(m => At(m, "decide", n)) is JsonNode decide && decide["state"]!["prompt"]!.GetValue<string>() == "roll"
                && game.FirstOrDefault(m => At(m, "step", n)) is JsonNode roll && roll["result"]!["dice"]!.AsArray().Sum(d => Int(d)) != 7
                && decide["state"]!["players"]![0]!["hand"]!.AsObject().Sum(c => Int(c.Value)) < 5);
            JsonNode state = game.First(m => At(m, "decide", at))["state"]!;
            string card = OSPCodec.ResourceNames.Values.First(name => Count(state["bank"], name) > 0);
            state["bank"]![card] = Count(state["bank"], card) - 1;
            JsonNode seat0 = state["players"]![0]!;
            seat0["hand"]![card] = Count(seat0["hand"], card) + 1;
            if (seat0["hand_size"] is JsonNode size) seat0["hand_size"] = Int(size) + 1;
            string path = TempPath("game.jsonl");
            Append(path, game);

            await using LiveRelay live = await LiveRelay.StartAsync(path, Config(), pollMilliseconds: 20);
            using RemoteSimulation watcher = await Watch(live);
            // One game from its first move: the card comes in as an edit at that decision and goes again at the next
            Assert.True(Pump(watcher, () => watcher.Game.Record.State.Equals(original[0].End)), $"watcher at {watcher.Game.Record.History.Count} entries");
            Assert.Equal(original[0].Actions.Count + 2, watcher.Game.Record.History.Count);
            Assert.Equal(2, watcher.Game.Record.History.Count(entry => entry.Edit != null));
            Assert.Equal(1, live.Parts);
        }

        [Fact]
        public async Task AWatcherOfADirectoryFollowsTheNewestGame()
        {
            List<List<JsonNode>> games = Games();
            string first = TempPath("a.jsonl"), directory = Path.GetDirectoryName(first)!;
            Append(first, games[0]);
            (GameState End, int Entries, int Parts) a = Watched(games[0]), b = Watched(games[1]);

            await using LiveRelay live = await LiveRelay.StartAsync(directory, Config(), pollMilliseconds: 20);
            using RemoteSimulation watcher = await Watch(live);
            Assert.True(Pump(watcher, () => watcher.Game.Record.State.Equals(a.End)));

            // A new game appears, as a server writes a file per game
            Append(Path.Combine(directory, "b.jsonl"), games[1]);
            Assert.True(Pump(watcher, () => watcher.Game.Record.State.Equals(b.End)), $"watcher at {watcher.Game.Record.History.Count} moves");
            Assert.Equal(b.Entries, watcher.Game.Record.History.Count);
        }

        [Fact]
        public async Task AWatcherOfADirectoryStartsWithItsNewestGame()
        {
            // Two games already written: the watcher sees the newer one, as it joins a server's games under way
            List<List<JsonNode>> games = Games();
            string older = TempPath("a.jsonl"), directory = Path.GetDirectoryName(older)!;
            Append(older, games[0]);
            Thread.Sleep(50);
            Append(Path.Combine(directory, "b.jsonl"), games[1]);
            (GameState End, int Entries, int Parts) b = Watched(games[1]);

            await using LiveRelay live = await LiveRelay.StartAsync(directory, Config(), pollMilliseconds: 20);
            using RemoteSimulation watcher = await Watch(live);
            Assert.True(Pump(watcher, () => watcher.Game.Record.State.Equals(b.End)));
            Assert.Equal(b.Parts, live.Parts);
        }

        [Fact]
        public void ARelayedMoveKeepsTheRefereesOutcome()
        {
            using Match match = new(new GameState(MapGenerator.GenerateRandomClassic(), 4), [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human], valueNetAvailable: false);
            SessionInfo admin = new("admin", "Admin", Role.Admin, null), spectator = new("sam", "Sam", Role.Spectator, null);
            Random random = new(5);
            // Through the setup to the first roll
            while (true)
            {
                GameState state = match.Record.State;
                int acting = match.ActingPlayer;
                List<Action> legal = LegalActionProvider.GetActionsForState(state, (sbyte)acting);
                if (legal.OfType<RollAction>().FirstOrDefault() is RollAction roll)
                {
                    // The referee rolled a twelve, which Hexfarer's dice give one time in 36
                    Action twelve = ((IOutputRandomnessAction)roll).GetOutcomeVariants(state, (sbyte)acting).Select(v => v.Action).Cast<RollAction>()
                        .First(v => v.History!.RollResult.Total == 12);
                    // Spectators can't relay; the match stays
                    match.Handle(spectator, new RelayMove(twelve));
                    Assert.Equal(16, match.Record.History.Count);
                    // Over the network too
                    Command sent = Protocol.DeserializeCommand(Protocol.Serialize(new RelayMove(twelve)));
                    match.Handle(admin, sent);
                    Assert.Equal(17, match.Record.History.Count);
                    Assert.Equal(12, ((RollAction)match.Record.History[^1].Action!).History!.RollResult.Total);
                    return;
                }
                match.Handle(admin, new PlayMove(legal[random.Next(legal.Count)]));
            }
        }
    }
}
