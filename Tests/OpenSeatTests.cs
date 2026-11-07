using Agents;
using Agents.Inference;
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

        [Fact]
        public void ALogCutShortReadsUpToTheCut()
        {
            // A stopped writer leaves a gzip stream cut off, its last line half written: read up to there, no failure
            byte[] whole = File.ReadAllBytes(Fixture);
            int messages = Replay.Read(Fixture).Count();
            string cut = Path.Combine(Path.GetTempPath(), $"cut-{Guid.NewGuid():N}.jsonl.gz");
            try
            {
                File.WriteAllBytes(cut, whole[..(whole.Length / 2)]);
                int read = Replay.Read(cut).Count();
                Assert.InRange(read, messages / 4, messages - 1);

                // The same for a plain log whose last line is half written
                string text = string.Join("\n", Replay.Read(Fixture).Take(100).Select(m => m.ToJsonString()));
                File.WriteAllText(cut.Replace(".gz", ""), text[..^20]);
                Assert.Equal(99, Replay.Read(cut.Replace(".gz", "")).Count());
            }
            finally
            {
                File.Delete(cut);
                File.Delete(cut.Replace(".gz", ""));
            }
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

        /// <summary>
        /// A recorded game's messages for one seat as the driver sends them to an agent with the full state: its before,
        /// every step with the state before the move (from that move's decide), its own decisions, the after
        /// </summary>
        private static IEnumerable<JsonNode> AsSentTo(string game, int seat)
        {
            Dictionary<int, JsonNode> states = [];
            foreach (JsonNode recorded in Replay.Read(Fixture).Where(m => m["game_id"]!.GetValue<string>() == game))
            {
                JsonNode message = recorded.DeepClone();
                switch (message["type"]!.GetValue<string>())
                {
                    case "before": message["seat"] = seat; yield return message; break;
                    case "decide":
                        states[Int(message["n"])] = message["state"]!.DeepClone();
                        if (Int(message["seat"]) == seat) yield return message;
                        break;
                    case "step": message["state"] = states[Int(message["n"])].DeepClone(); yield return message; break;
                    default: yield return message; break;
                }
            }
        }

        [Fact]
        public void TheAgentTakesOverTheRefereesStateBeforeFollowingAMove()
        {
            // Another seat's move whose state before it holds a hand Hexfarer's copy doesn't (as after Catanatron's bank
            // shortage): the copy takes it over at that step, before its own next decision, and follows on
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), provider: InferenceProvider.Cpu);
            StringWriter log = new();
            SeatAgent agent = new(net, 16, 0, log);
            bool changed = false;
            foreach (JsonNode message in AsSentTo("catanatron-100", 0))
            {
                if (!changed && message["type"]!.GetValue<string>() == "step" && Int(message["n"]) > 60 && Int(message["seat"]) != 0
                    && Int(message["state"]!["bank"]!["ore"]) > 0)
                {
                    JsonNode hand = message["state"]!["players"]![2]!["hand"]!, bank = message["state"]!["bank"]!;
                    hand["ore"] = Int(hand["ore"]) + 1;
                    bank["ore"] = Int(bank["ore"]) - 1;
                    agent.Handle(message);
                    changed = true;
                    break;
                }
                agent.Handle(message);
            }
            Assert.True(changed);
            agent.Handle(JsonNode.Parse("""{"type":"after"}""")!);
            JsonNode stats = JsonNode.Parse(log.ToString().Trim().Split('\n').Last())!;
            Assert.Equal((1, false), (Int(stats["patches"]), stats["lost"]!.GetValue<bool>()));

            // A whole game sent so (catanatron-101 reaches the Longest Road difference): never lost, taken over where it differs
            foreach ((string game, bool differs) in new[] { ("catanatron-100", false), ("catanatron-101", true) })
            {
                StringWriter whole = new();
                SeatAgent second = new(net, 16, 0, whole);
                foreach (JsonNode message in AsSentTo(game, 0))
                {
                    JsonNode? answer = second.Handle(message);
                    if (message["type"]!.GetValue<string>() == "decide")
                        Assert.Contains(OspCodec.Canonical(answer!["action"]), message["legal"]!.AsArray().Select(OspCodec.Canonical).ToHashSet());
                }
                JsonNode end = JsonNode.Parse(whole.ToString().Trim().Split('\n').Last())!;
                Assert.Equal((false, differs), (end["lost"]!.GetValue<bool>(), Int(end["patches"]) > 0));
            }
        }

        [Theory]
        [InlineData("catanatron-100", false)]
        [InlineData("catanatron-101", true)]
        public void TheAgentAnswersEveryDecisionLegallyAndTakesOverTheRefereesState(string game, bool differs)
        {
            // Seat 0 of a recorded game (catanatron-101 reaches the known Longest Road difference): the agent follows the
            // recorded moves and answers the seat's decisions (its answers don't change the recorded game)
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), provider: InferenceProvider.Cpu);
            StringWriter log = new();
            SeatAgent agent = new(net, 32, 0, log);
            Assert.Equal("osp", agent.Handle(JsonNode.Parse("""{"type":"hello","protocol":"osp","version":1}""")!)!["protocol"]!.GetValue<string>());
            int answered = 0, choices = 0;
            foreach (JsonNode recorded in Replay.Read(Fixture).Where(m => m["game_id"]!.GetValue<string>() == game))
            {
                JsonNode message = recorded.DeepClone();
                string type = message["type"]!.GetValue<string>();
                if (type == "before") message["seat"] = 0;
                if (type == "decide" && Int(message["seat"]) != 0) continue;
                JsonNode? answer = agent.Handle(message);
                if (type != "decide") continue;
                HashSet<string> legal = message["legal"]!.AsArray().Select(OspCodec.Canonical).ToHashSet();
                Assert.Contains(OspCodec.Canonical(answer!["action"]), legal);
                Assert.Equal(Int(message["n"]), Int(answer["n"]));
                answered++;
                if (legal.Count > 1) choices++;
            }
            Assert.True(answered > 30, $"{answered} decisions");

            // The copy never got lost; it took over the referee's state only where the rules differ
            JsonNode stats = JsonNode.Parse(log.ToString().Trim().Split('\n').Last())!;
            Assert.Equal((answered, false), (Int(stats["decisions"]), stats["lost"]!.GetValue<bool>()));
            // It searched every decision with a choice (single legal moves need none)
            Assert.Equal(choices, Int(stats["searched"]));
            Assert.Equal(differs, Int(stats["patches"]) > 0);
        }

        [Fact]
        public void TheAgentReportsItsSearchWorkPerGame()
        {
            // Budget 40 in batches of 16: each search evaluates whole batches, at least 48 leaves (three batches) and at
            // most a few batches more while the root waits for its moves; two games in a row, each with its own counts
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), provider: InferenceProvider.Cpu);
            StringWriter log = new();
            SeatAgent agent = new(net, 40, 0, log, leavesPerBatch: 16);
            Assert.EndsWith("-40it-b16", agent.Name);
            foreach (string game in new[] { "catanatron-100", "catanatron-102" })
            {
                foreach (JsonNode recorded in Replay.Read(Fixture).Where(m => m["game_id"]!.GetValue<string>() == game))
                {
                    JsonNode message = recorded.DeepClone();
                    string type = message["type"]!.GetValue<string>();
                    if (type == "before") message["seat"] = 0;
                    if (type == "decide" && Int(message["seat"]) != 0) continue;
                    agent.Handle(message);
                }
                JsonNode stats = JsonNode.Parse(log.ToString().Trim().Split('\n').Last())!;
                Assert.Equal(game, stats["game_id"]!.GetValue<string>());
                int searched = Int(stats["searched"]), searches = Int(stats["searches"]);
                long iterations = stats["iterations"]!.GetValue<long>(), distinct = stats["distinct_leaves"]!.GetValue<long>();
                // A search per decision with a choice, whole batches of 16 within the bounds, fewer distinct positions
                Assert.InRange(searches, searched - 2, searched);
                Assert.True(searches > 10, $"{searches} searches");
                Assert.Equal(0, iterations % 16);
                Assert.InRange(iterations, 48L * searches, (48L + 16 * 4) * searches);
                Assert.InRange(distinct, searches, iterations - 1);
            }
        }

        [Fact]
        public void AnAgentWithATimeBudgetReportsItsSearchTime()
        {
            // 20 ms per decision, batches of 16 selected on one thread: each search takes at least its budget and at most a
            // few batches longer, and the game's search time is the sum over its searches
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), intraOpThreads: 1, provider: InferenceProvider.Cpu);
            StringWriter log = new();
            SeatAgent agent = new(net, 0, 0.02, log, leavesPerBatch: 16, parallelSelection: false);
            Assert.EndsWith("-0.02s-b16-serial", agent.Name);
            List<(int searches, double seconds)> games = [];
            foreach (string game in new[] { "catanatron-100", "catanatron-102" })
            {
                foreach (JsonNode recorded in Replay.Read(Fixture).Where(m => m["game_id"]!.GetValue<string>() == game))
                {
                    JsonNode message = recorded.DeepClone();
                    string type = message["type"]!.GetValue<string>();
                    if (type == "before") message["seat"] = 0;
                    if (type == "decide" && Int(message["seat"]) != 0) continue;
                    agent.Handle(message);
                }
                JsonNode stats = JsonNode.Parse(log.ToString().Trim().Split('\n').Last())!;
                int searches = Int(stats["searches"]);
                double seconds = stats["search_seconds"]!.GetValue<double>();
                Assert.True(searches > 10, $"{searches} searches");
                Assert.InRange(seconds, 0.02 * searches, 0.25 * searches);
                Assert.Equal(0, stats["iterations"]!.GetValue<long>() % 16);
                games.Add((searches, seconds));
            }
            // The second game counts only its own searches: carried over, it would hold the first game's time on top
            Assert.True(games[1].seconds < games[0].seconds + 0.02 * games[1].searches, $"{games[0]} then {games[1]}");
        }

        [Theory]
        [InlineData("catanatron-100", false)]
        [InlineData("catanatron-101", false)]
        [InlineData("catanatron-100", true)]
        public void TheFairAgentPlaysFromWhatItsSeatSaw(string game, bool payout)
        {
            // Seat 0 of a recorded game with the fair agent (catanatron-101 takes over the referee's state): it answers
            // legally, its copy never drifts from the replica, its decisions come from its information-set tree, and its
            // beliefs at the end are what a tracker of seat 0 holds after the same moves, with every takeover as a public
            // edit (a reset would make the hands known). With payout, the referee's state at one of seat 0's decisions gives
            // an extra card to a player whose hand seat 0 doesn't fully know, as a payout by other rules would (the next state
            // takes it back)
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), provider: InferenceProvider.Cpu);
            StringWriter log = new();
            SeatAgent agent = new(net, 32, 0, log, leavesPerBatch: 16, fair: true);
            Assert.Contains("-ismcts-fair-", agent.Name);
            Replica? expected = null;
            BeliefTracker? seen = null;
            uint mostUnknown = 0;
            int patches = 0;
            bool paid = false;
            void TakeOver(JsonNode state)
            {
                if (expected!.FirstDifference(state, "test") == null) return;
                GameState untouched = new(expected.State);
                Assert.True(expected.Patch(state));
                seen!.Observe(untouched, null, expected.State);
                patches++;
            }
            foreach (JsonNode recorded in Replay.Read(Fixture).Where(m => m["game_id"]!.GetValue<string>() == game))
            {
                JsonNode message = recorded.DeepClone();
                string type = message["type"]!.GetValue<string>();
                if (type == "before") message["seat"] = 0;
                if (type == "decide" && Int(message["seat"]) != 0) continue;
                if (payout && !paid && type == "decide" && Enumerable.Range(1, 3).FirstOrDefault(p => seen!.Hand(p).Unknown > 0) is int player and > 0
                    && message["state"]!["bank"]!["ore"]!.GetValue<int>() > 0)
                {
                    JsonNode state = message["state"]!;
                    state["bank"]!["ore"] = state["bank"]!["ore"]!.GetValue<int>() - 1;
                    state["players"]![player]!["hand"]!["ore"] = state["players"]![player]!["hand"]!["ore"]!.GetValue<int>() + 1;
                    paid = true;
                }
                if (type == "after")
                {
                    // Before the end: the beliefs as the seat built them, against the independent ones
                    Assert.Equal(seen, agent.Fair!.Beliefs);
                    Assert.True(mostUnknown > 0, "the game has a steal seat 0 didn't see");
                    Assert.Equal(game == "catanatron-101" || payout, patches > 0);
                    Assert.Equal(payout, paid);
                }
                JsonNode? answer = agent.Handle(message);
                switch (type)
                {
                    case "before":
                        expected = new Replica(message);
                        seen = new BeliefTracker(0, expected.State);
                        break;
                    case "step":
                        if (message["state"] is JsonNode state) TakeOver(state);
                        GameState pre = new(expected!.State);
                        Assert.Null(expected.Follow(message));
                        seen!.Observe(pre, expected.Applied, expected.State);
                        mostUnknown = Math.Max(mostUnknown, Enumerable.Range(1, 3).Max(p => seen.Hand(p).Unknown));
                        break;
                    case "decide":
                        TakeOver(message["state"]!);
                        Assert.Contains(OspCodec.Canonical(answer!["action"]), message["legal"]!.AsArray().Select(OspCodec.Canonical).ToHashSet());
                        break;
                }
                // After every message the seat knows what the independent tracker knows, no more and no less
                if (type is "step" or "decide") Assert.Equal(seen, agent.Fair!.Beliefs);
            }
            JsonNode stats = JsonNode.Parse(log.ToString().Trim().Split('\n').Last())!;
            Assert.Equal((true, false, 0), (stats["fair"]!.GetValue<bool>(), stats["lost"]!.GetValue<bool>(), Int(stats["resyncs"])));
            Assert.True(Int(stats["searches"]) > 10);
            Assert.NotNull(agent.Fair!.LastTree);
            Assert.False(agent.Fair.FullKnowledge);
        }

        [Fact]
        public void ADecisionWithoutASearchAddsNoSearchWork()
        {
            // The same game twice; the second time the referee also lists a move Hexfarer's rules don't have next to every
            // single legal move, so the agent answers those decisions with a forced move and no search: its search work
            // stays the same, the earlier search isn't counted again
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), provider: InferenceProvider.Cpu);
            JsonNode Stats(bool padded)
            {
                StringWriter log = new();
                SeatAgent agent = new(net, 40, 0, log, leavesPerBatch: 16);
                foreach (JsonNode recorded in Replay.Read(Fixture).Where(m => m["game_id"]!.GetValue<string>() == "catanatron-100"))
                {
                    JsonNode message = recorded.DeepClone();
                    string type = message["type"]!.GetValue<string>();
                    if (type == "before") message["seat"] = 0;
                    if (type == "decide" && Int(message["seat"]) != 0) continue;
                    if (padded && type == "decide" && message["legal"]!.AsArray().Count == 1)
                        message["legal"]!.AsArray().Add(JsonNode.Parse("""["end_turn","made up"]"""));
                    agent.Handle(message);
                }
                return JsonNode.Parse(log.ToString().Trim().Split('\n').Last())!;
            }
            JsonNode plain = Stats(false), padded = Stats(true);
            Assert.True(Int(padded["searched"]) > Int(plain["searched"]) + 10, "the padded decisions reach the agent's search");
            Assert.Equal((Int(plain["searches"]), plain["iterations"]!.GetValue<long>()), (Int(padded["searches"]), padded["iterations"]!.GetValue<long>()));
        }

        [Fact]
        public void AGameHexfarersRulesEndedGoesOnWhenTheRefereeAsksForAMove()
        {
            // The copy of a recorded game up to a decision, then ended by Hexfarer's rules (as when its Longest Road gives a
            // player 10 points Catanatron doesn't): the referee's next decision revives it, and the agent searches on
            List<JsonNode> game = Replay.Read(Fixture).Where(m => m["game_id"]!.GetValue<string>() == "catanatron-100").ToList();
            JsonNode before = game[0].DeepClone();
            before["seat"] = 0;
            Replica replica = new(before);
            JsonNode decide = game.First(m => m["type"]!.GetValue<string>() == "decide" && Int(m["n"]) > 80 && Int(m["seat"]) == 0 && m["legal"]!.AsArray().Count > 1);
            foreach (JsonNode step in game.Where(m => m["type"]!.GetValue<string>() == "step" && Int(m["n"]) < Int(decide["n"])))
                Assert.Null(replica.Follow(step));
            Assert.Null(replica.FirstDifference(decide["state"]!, "test"));

            replica.State.Turn.TypeOfRound = TurnState.RoundType.MatchEnded;
            Assert.StartsWith("state: game over", replica.FirstDifference(decide["state"]!, "test")!.Value.kind);
            Assert.Empty(replica.Legal());
            Assert.True(replica.Patch(decide["state"]!));
            Assert.False(replica.State.HasEnded);
            Assert.NotEmpty(replica.Legal());
            Assert.Null(replica.FirstDifference(decide["state"]!, "test"));
        }

        [Fact]
        public void AFinishedGamesPromptIsntCompared()
        {
            // Engines name the prompt of a finished game differently (JSettlers "over", Hexfarer the last one): after the
            // game it doesn't count, while the game goes on it does
            List<JsonNode> game = Replay.Read(Fixture).Where(m => m["game_id"]!.GetValue<string>() == "catanatron-100").ToList();
            Replica replica = new(game[0]);
            JsonNode state = replica.Codec.State(replica.State).DeepClone();
            state["prompt"] = "over";
            Assert.Null(replica.FirstDifference(state, "after", goesOn: false));
            Assert.Equal("state: prompt", replica.FirstDifference(state, "test")!.Value.kind);
        }

        [Fact]
        public void ABankBelowZeroReadsAsEmpty()
        {
            // JSettlers keeps no resource bank: its count is 19 less what the hands hold, below 0 once they hold more
            GameState game = RandomGames(games: 1).Skip(150).First();
            OspCodec codec = new(OspCodec.BuildBoard(new OspCodec(game.Board).BoardJson()));
            JsonNode state = codec.State(game);
            int bank = Int(state["bank"]!["ore"]);
            JsonNode player = state["players"]![0]!;
            player["hand"]!["ore"] = Int(player["hand"]!["ore"]) + bank + 2;
            player["hand_size"] = Int(player["hand_size"]) + bank + 2;
            state["bank"]!["ore"] = -2;
            // As a message comes, through text
            state = JsonNode.Parse(state.ToJsonString())!;
            sbyte acting = TestUtil.ActingPlayer(game);
            JsonArray legal = new(LegalActionProvider.GetActionsForState(game, acting).Select(a => (JsonNode)codec.Move(a)).ToArray());

            // Rebuilt, the bank has no ore and the hand all of it
            GameState rebuilt = codec.BuildState(state, legal);
            Assert.Equal(0u, rebuilt.ResourceBank.Get(ResourceCardType.Ore));
            Assert.Equal((uint)Int(player["hand"]!["ore"]), rebuilt.Players[0].ResourceCards.Get(ResourceCardType.Ore));

            // A copy that follows the game takes such a bank over and then agrees with the referee
            JsonObject before = new() { ["type"] = "before", ["game_id"] = "bank", ["seat"] = 0, ["players"] = 4,
                ["ruleset"] = new JsonObject { ["preset"] = "standard" }, ["board"] = codec.BoardJson(), ["state"] = codec.State(game) };
            Replica replica = new(before);
            replica.Rebuild(codec.State(game), legal);
            Assert.True(replica.Patch(state));
            Assert.Equal(0u, replica.State.ResourceBank.Get(ResourceCardType.Ore));
            Assert.Null(replica.FirstDifference(state, "bank"));
        }

        [Fact]
        public void TheAgentTakesOverAHandAndBankTheRefereeHoldsDifferently()
        {
            // A referee that gave seat 2 an ore from the bank Hexfarer's rules wouldn't have (as Catanatron's bank shortage
            // does with the rest of a short resource): the agent's copy takes both over and plays on
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), provider: InferenceProvider.Cpu);
            StringWriter log = new();
            SeatAgent agent = new(net, 16, 0, log);
            int changedAt = -1;
            foreach (JsonNode recorded in Replay.Read(Fixture).Where(m => m["game_id"]!.GetValue<string>() == "catanatron-100"))
            {
                JsonNode message = recorded.DeepClone();
                string type = message["type"]!.GetValue<string>();
                if (type == "before") message["seat"] = 0;
                if (type == "decide" && Int(message["seat"]) != 0) continue;
                // From the first decision of seat 0 after move 60 on, the referee's states carry the extra ore
                if (type is "decide" or "after" && (changedAt >= 0 || type == "decide" && Int(message["n"]) > 60 && Int(message["state"]!["bank"]!["ore"]) > 0))
                {
                    if (changedAt < 0) changedAt = Int(message["n"]);
                    JsonNode hand = message["state"]!["players"]![2]!["hand"]!, bank = message["state"]!["bank"]!;
                    hand["ore"] = Int(hand["ore"]) + 1;
                    bank["ore"] = Int(bank["ore"]) - 1;
                }
                agent.Handle(message);
                if (changedAt >= 0) break;
            }
            JsonNode stats = JsonNode.Parse(agent.Handle(JsonNode.Parse("""{"type":"after"}""")!) is null ? log.ToString().Trim().Split('\n').Last() : "{}")!;
            Assert.True(changedAt > 0);
            Assert.Equal((1, false), (Int(stats["patches"]), stats["lost"]!.GetValue<bool>()));
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
