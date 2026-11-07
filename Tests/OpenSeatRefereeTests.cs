using Agents;
using Agents.Inference;
using Agents.OpenSeat;
using Common;
using Common.Actions;
using Common.Osp;
using OpenSeat;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Hexfarer as the referee of the Open Seat Protocol (see Bridges/PROTOCOL.md): what each seat may see of the states
    /// and the moves' results, the rules it announces, OpenSeatAgentProxy, which plays a seat by asking an Open Seat Agent,
    /// and OpenSeatAgentProcess, which runs one as a process
    /// </summary>
    public class OpenSeatRefereeTests
    {
        private static int Int(JsonNode? number) => int.Parse(number!.ToJsonString());

        private static string TinyNet => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        // Hexfarer's own Open Seat Agent, built next to the tests, run as a process
        private static string AgentCommand(string options = "") =>
            $"dotnet \"{Path.Combine(AppContext.BaseDirectory, "OpenSeat.dll")}\" agent --net \"{TinyNet}\" --iterations 16 --leaves-per-batch 16 --provider cpu {options}".Trim();

        // The moves of random games, each with the state before and after it
        private static IEnumerable<(GameState before, Action applied, GameState after)> RandomMoves(int games = 3, int maxMoves = 3000)
        {
            for (int seed = 1; seed <= games; seed++)
            {
                TestUtil.Seed(seed);
                GameState state = TestUtil.NewGame();
                Agent[] agents = TestUtil.RandomAgents();
                for (int moves = 0; !state.HasEnded && moves < maxMoves; moves++)
                {
                    GameState before = new(state);
                    Action action = agents[TestUtil.ActingPlayer(state)].Act(state);
                    action.Apply(state);
                    yield return (before, action, new GameState(state));
                }
            }
        }

        #region What a seat sees
        [Fact]
        public void AViewShowsOnlyTheSeatsOwnCards()
        {
            foreach ((GameState state, _, _) in RandomMoves(games: 2).Where((_, i) => i % 25 == 0))
            {
                OspCodec codec = new(state.Board);
                JsonObject truth = codec.State(state);
                for (int seat = 0; seat < 4; seat++)
                {
                    JsonObject view = codec.View(state, seat);
                    // Everything public is as in the full state, whose turn it is too
                    Assert.Equal(state.Turn.PlayerIndex, Int(truth["on_turn"]));
                    foreach (string field in new[] { "prompt", "current", "on_turn", "free_roads", "robber", "buildings", "bank", "offer" })
                        Assert.Equal(OspCodec.Canonical(truth[field]), OspCodec.Canonical(view[field]));
                    for (int other = 0; other < 4; other++)
                    {
                        JsonNode theirs = view["players"]![other]!, true_ = truth["players"]![other]!;
                        if (other == seat)
                        {
                            Assert.Equal(OspCodec.Canonical(true_), OspCodec.Canonical(theirs));
                            continue;
                        }
                        // Another seat's cards and its points with victory point cards stay hidden, its counts don't
                        Assert.Null(theirs["hand"]);
                        Assert.Null(theirs["development"]);
                        Assert.Null(theirs["vp"]);
                        foreach (string field in new[] { "seat", "public_vp", "played", "longest_road", "has_longest_road", "has_largest_army", "pieces_left" })
                            Assert.Equal(OspCodec.Canonical(true_[field]), OspCodec.Canonical(theirs[field]));
                        Assert.Equal(state.Players[other].ResourceCards.Count(), (uint)Int(theirs["hand_size"]));
                        Assert.Equal(state.Players[other].DevelopmentCards.Count(), (uint)Int(theirs["development_size"]));
                    }
                }
            }
        }

        [Fact]
        public void EachSeatSeesAMovesResultAsTheRulesShowIt()
        {
            bool sawSteal = false, sawEmptyRobber = false, sawCard = false, sawMonopoly = false;
            foreach ((GameState before, Action applied, GameState after) in RandomMoves(games: 4))
            {
                for (int seat = 0; seat < 4; seat++)
                {
                    JsonNode? result = OspCodec.Result(applied, seat);
                    // As the referee sees it (for open-hand agents): everything
                    Assert.Equal(OspCodec.Canonical(OspCodec.Result(applied, seat: null)), OspCodec.Canonical(OspCodec.Result(applied, applied switch
                    {
                        RobberAction robber => robber.PlayerIndex,
                        _ => applied.PlayerIndex,
                    })));
                    switch (applied)
                    {
                        case RollAction roll:
                            Assert.Equal($"[{roll.History!.RollResult.First},{roll.History.RollResult.Second}]", result!["dice"]!.ToJsonString());
                            break;
                        case BuyDevelopmentCardAction buy:
                            sawCard = true;
                            // Only the buyer sees the card
                            if (seat == buy.PlayerIndex) Assert.Equal(OspCodec.CardNames[buy.History!.DrawnType], result!["card"]!.GetValue<string>());
                            else Assert.Null(result!["card"]);
                            Assert.True(result.AsObject().ContainsKey("card"));
                            break;
                        case RobberAction robber when robber.History!.StolenCard is ResourceCardType stolen:
                            sawSteal = true;
                            // Thief and victim see the card, the others that one was stolen
                            bool party = seat == robber.PlayerIndex || seat == robber.TargetPlayerIndex;
                            Assert.Equal(party ? OspCodec.ResourceNames[stolen] : "unknown", result!["stolen"]!.GetValue<string>());
                            break;
                        case RobberAction:
                            sawEmptyRobber = true;
                            Assert.True(result!.AsObject().ContainsKey("stolen"));
                            Assert.Null(result["stolen"]);
                            break;
                        case MonopolyAction monopoly:
                            sawMonopoly = true;
                            // What each player gave up is public
                            JsonObject taken = result!["taken"]!.AsObject();
                            foreach ((int player, uint count) in monopoly.History!.TransferredCards)
                                Assert.Equal(count, (uint)Int(taken[player.ToString()]));
                            break;
                        default:
                            Assert.Null(result);
                            break;
                    }
                }
            }
            Assert.True(sawSteal && sawEmptyRobber && sawCard, "the random games should steal, rob nobody and buy a card");
            _ = sawMonopoly;
        }

        [Fact]
        public void TheRulesetNamesWhatDiffersFromTheRulebook()
        {
            GameSettings settings = new() { FreeTrade = true };
            Assert.Equal("""{"preset":"standard"}""", OspCodec.Canonical(OspCodec.Ruleset(settings)));
            settings.FreeTrade = false;
            Assert.Equal("""{"player_trading":"none","preset":"standard"}""", OspCodec.Canonical(OspCodec.Ruleset(settings)));
            settings.VictoryPoints = 12;
            settings.RobberCardLimit = 9;
            Assert.Equal("""{"discard_limit":9,"player_trading":"none","preset":"standard","victory_points":12}""", OspCodec.Canonical(OspCodec.Ruleset(settings)));
        }

        [Fact]
        public void TradesBetweenPlayersHaveProtocolMovesAndAnOpenOfferIsInTheState()
        {
            GameState state = TestUtil.EmptyNormalTurn(0);
            state.Settings.FreeTrade = true;
            TestUtil.GiveCards(state, 0, ResourceCardType.Ore, 2);
            TestUtil.GiveCards(state, 2, ResourceCardType.Wool, 1);
            OspCodec codec = new(state.Board);
            Assert.Null(codec.State(state)["offer"]);

            CardSet<ResourceCardType> ore = new(), wool = new();
            ore.Add(ResourceCardType.Ore, 1);
            wool.Add(ResourceCardType.Wool, 1);
            // Offered to seats 1 and 2 (a bit per seat)
            OfferTradeAction offer = new(0, ore, wool, 0b0110);
            Assert.Equal("""["offer",{"get":{"wool":1},"give":{"ore":1},"to":[1,2]}]""", OspCodec.Canonical(codec.Move(offer)));
            offer.Apply(state);
            JsonNode open = codec.State(state)["offer"]!;
            Assert.Equal("""{"answers":{},"from":0,"get":{"wool":1},"give":{"ore":1},"to":[1,2]}""", OspCodec.Canonical(open));

            RespondToTradeAction no = new(1, false);
            Assert.Equal("""["respond",{"accept":false}]""", OspCodec.Canonical(codec.Move(no)));
            no.Apply(state);
            RespondToTradeAction yes = new(2, true);
            Assert.Equal("""["respond",{"accept":true}]""", OspCodec.Canonical(codec.Move(yes)));
            yes.Apply(state);
            Assert.Equal("""{"1":"decline","2":"accept"}""", OspCodec.Canonical(codec.State(state)["offer"]!["answers"]));

            Assert.Equal("""["confirm",{"partner":2}]""", OspCodec.Canonical(codec.Move(new ConfirmTradeAction(0, 2))));
            Assert.Equal("""["cancel"]""", OspCodec.Canonical(codec.Move(new CancelTradeAction(0))));
            // A counteroffer in the offerer's terms
            CardSet<ResourceCardType> twoOre = new();
            twoOre.Add(ResourceCardType.Ore, 2);
            Assert.Equal("""["counter",{"get":{"wool":1},"give":{"ore":2}}]""", OspCodec.Canonical(codec.Move(new CounterTradeAction(1, twoOre, wool))));
        }
        #endregion

        #region The referee's messages (OpenSeatAgentProxy with a scripted agent)
        /// <summary>
        /// An Open Seat Agent in the test: records what the referee sends and answers hello and decide (by default the
        /// first legal move)
        /// </summary>
        private sealed class ScriptedAgent(string needs = "view", bool observe = true, Func<JsonNode, JsonNode>? choose = null, string[]? rulesets = null) : IOpenSeatChannel
        {
            public readonly List<JsonNode> Received = [];
            private readonly Queue<JsonNode> _answers = new();
            public bool Disposed { get; private set; }

            public void Send(JsonNode message)
            {
                Received.Add(message.DeepClone());
                switch (message["type"]!.GetValue<string>())
                {
                    case "hello":
                        _answers.Enqueue(new JsonObject { ["type"] = "hello", ["protocol"] = "osp", ["version"] = 1, ["name"] = "scripted",
                            ["rulesets"] = new JsonArray((rulesets ?? ["standard"]).Select(r => (JsonNode)r).ToArray()), ["observe"] = observe, ["needs"] = needs });
                        break;
                    case "decide":
                        JsonNode move = choose?.Invoke(message) ?? message["legal"]![0]!.DeepClone();
                        _answers.Enqueue(new JsonObject { ["type"] = "decide", ["game_id"] = message["game_id"]!.DeepClone(), ["n"] = message["n"]!.DeepClone(), ["action"] = move });
                        break;
                }
            }

            public JsonNode Receive(CancellationToken cancellation = default) =>
                _answers.Count > 0 ? _answers.Dequeue() : throw new InvalidOperationException("The scripted agent has nothing to answer.");

            public void Dispose() => Disposed = true;

            public IEnumerable<JsonNode> OfType(string type) => Received.Where(m => m["type"]!.GetValue<string>() == type);
        }

        /// <summary>
        /// Plays a game with the proxy in one seat and random agents in the others, as the Arena does (the move applied,
        /// then every agent told); the states before each move, in order
        /// </summary>
        private static List<GameState> PlayWithProxy(IOpenSeatChannel agent, int seed, sbyte seat = 0, Action<GameState>? setup = null, int maxMoves = 3000)
        {
            TestUtil.Seed(seed);
            GameState state = TestUtil.NewGame();
            setup?.Invoke(state);
            Agent[] agents = TestUtil.RandomAgents();
            using OpenSeatAgentProxy proxy = new(seat, state, agent, $"g{seed}", seed: seed);
            agents[seat] = proxy;
            List<GameState> befores = [];
            for (int moves = 0; !state.HasEnded && moves < maxMoves; moves++)
            {
                befores.Add(new GameState(state));
                Action action = agents[TestUtil.ActingPlayer(state)].Act(state);
                Assert.True(action.IsValidFor(state));
                action.Apply(state);
                foreach (Agent a in agents) a.HandleAction(action);
            }
            befores.Add(new GameState(state));
            return befores;
        }

        [Fact]
        public void TheRefereeGreetsAndStartsTheGameWithTheSeatsView()
        {
            ScriptedAgent agent = new(needs: "view");
            TestUtil.Seed(5);
            GameState start = TestUtil.NewGame();
            using (OpenSeatAgentProxy proxy = new(2, start, agent, "g5"))
            {
                Assert.Equal("scripted", proxy.Hello.Name);
                Assert.Equal(OpenSeatNeeds.View, proxy.Hello.Needs);
            }
            Assert.Equal(["hello", "before", "after"], agent.Received.Select(m => m["type"]!.GetValue<string>()));
            Assert.Equal("""{"protocol":"osp","type":"hello","version":1}""", OspCodec.Canonical(agent.Received[0]));
            JsonNode before = agent.Received[1];
            OspCodec codec = new(start.Board);
            Assert.Equal("g5", before["game_id"]!.GetValue<string>());
            Assert.Equal(2, Int(before["seat"]));
            Assert.Equal(4, Int(before["players"]));
            Assert.Equal(OspCodec.Canonical(OspCodec.Ruleset(start.Settings)), OspCodec.Canonical(before["ruleset"]));
            Assert.Equal(OspCodec.Canonical(codec.BoardJson()), OspCodec.Canonical(before["board"]));
            Assert.Equal(OspCodec.Canonical(codec.View(start, 2)), OspCodec.Canonical(before["state"]));
            // A game that never started is cut short
            Assert.True(agent.Received[2]["truncated"]!.GetValue<bool>());
            Assert.True(agent.Disposed);
        }

        [Fact]
        public void EveryDecisionListsTheLegalMovesAndTheAnswerIsPlayed()
        {
            // The agent picks a random entry, so its answers must come back as exactly those moves
            Random pick = new(7);
            List<string> chosen = [];
            ScriptedAgent agent = new(choose: decide =>
            {
                JsonArray legal = decide["legal"]!.AsArray();
                JsonNode move = legal[pick.Next(legal.Count)]!.DeepClone();
                chosen.Add(OspCodec.Canonical(move));
                return move;
            });
            List<GameState> befores = PlayWithProxy(agent, seed: 11);
            Assert.True(befores[^1].HasEnded);

            OspCodec codec = new(befores[0].Board);
            List<JsonNode> decides = agent.OfType("decide").ToList();
            Assert.Equal(chosen.Count, decides.Count);
            Assert.True(decides.Count > 20);
            // The steps of the proxy's own moves are the moves it chose, at the n it was asked at
            List<JsonNode> ownSteps = agent.OfType("step").Where(s => Int(s["seat"]) == 0).ToList();
            for (int d = 0; d < decides.Count; d++)
            {
                int n = Int(decides[d]["n"]);
                GameState state = befores[n];
                Assert.Equal(0, Int(decides[d]["seat"]));
                List<string> legal = LegalActionProvider.GetActionsForState(state, 0).Select(a => OspCodec.Canonical(codec.Move(a))).ToList();
                Assert.Equal(legal, decides[d]["legal"]!.AsArray().Select(OspCodec.Canonical));
                JsonNode step = ownSteps.Single(s => Int(s["n"]) == n);
                Assert.Equal(chosen[d], OspCodec.Canonical(step["action"]));
            }
            // Every move of the game reached the agent once, in order
            Assert.Equal(Enumerable.Range(0, befores.Count - 1), agent.OfType("step").Select(s => Int(s["n"])));
            JsonNode after = agent.OfType("after").Single();
            Assert.Equal(befores[^1].Turn.PlayerIndex, Int(after["winner"]));
            Assert.False(after["truncated"]!.GetValue<bool>());
            Assert.Equal(befores[^1].Players.Select(p => (int)p.VictoryPoints.Total), after["vp"]!.AsArray().Select(Int));
        }

        [Fact]
        public void AnAnswerOutsideTheListEndsTheSeatWithAnError()
        {
            ScriptedAgent agent = new(choose: _ => JsonNode.Parse("""["build_city",[[9,9],[9,9],[9,9]]]""")!);
            OpenSeatException error = Assert.Throws<OpenSeatException>(() => PlayWithProxy(agent, seed: 3));
            Assert.Contains("not one of the legal moves", error.Message);
            Assert.Contains(agent.Received, m => m["type"]!.GetValue<string>() == "error");
        }

        [Fact]
        public void AnAgentWithoutTheStandardRulesIsTurnedAway()
        {
            ScriptedAgent agent = new(rulesets: ["catanatron"]);
            TestUtil.Seed(1);
            OpenSeatException error = Assert.Throws<OpenSeatException>(() => new OpenSeatAgentProxy(0, TestUtil.NewGame(), agent, "g1"));
            Assert.Contains("standard", error.Message);
            Assert.Equal("error", agent.Received[^1]["type"]!.GetValue<string>());
            Assert.True(agent.Disposed);
        }

        [Fact]
        public void UnderViewNoMessageCarriesAnotherSeatsCards()
        {
            ScriptedAgent agent = new(needs: "view");
            PlayWithProxy(agent, seed: 4, seat: 1);
            foreach (JsonNode message in agent.Received)
            {
                if (message["state"] is JsonNode state)
                    for (int other = 0; other < 4; other++)
                        if (other != 1)
                        {
                            Assert.Null(state["players"]![other]!["hand"]);
                            Assert.Null(state["players"]![other]!["development"]);
                        }
                // Under view a step carries no state, and another seat's card or steal stays unknown
                if (message["type"]!.GetValue<string>() == "step")
                {
                    Assert.Null(message["state"]);
                    if (message["result"]?["card"] is JsonNode card && Int(message["seat"]) != 1) Assert.Fail($"saw another seat's card {card}");
                }
            }
            Assert.Contains(agent.OfType("step"), s => s["result"]?["stolen"]?.GetValue<string>() == "unknown");
        }

        [Fact]
        public void UnderTruthEveryStepCarriesTheStateBeforeTheMove()
        {
            ScriptedAgent agent = new(needs: "truth");
            List<GameState> befores = PlayWithProxy(agent, seed: 6);
            OspCodec codec = new(befores[0].Board);
            List<JsonNode> steps = agent.OfType("step").ToList();
            Assert.Equal(befores.Count - 1, steps.Count);
            foreach (JsonNode step in steps)
                Assert.Equal(OspCodec.Canonical(codec.State(befores[Int(step["n"])])), OspCodec.Canonical(step["state"]));
            foreach (JsonNode decide in agent.OfType("decide"))
                Assert.Equal(OspCodec.Canonical(codec.State(befores[Int(decide["n"])])), OspCodec.Canonical(decide["state"]));
            // An open-hand agent sees every result in full: another seat's bought card and steal too
            Assert.Contains(steps, s => Int(s["seat"]) != 0 && s["result"]?["card"] is JsonValue);
            Assert.DoesNotContain(steps, s => s["result"]?["card"] is null && s["action"]![0]!.GetValue<string>() == "buy_development");
            Assert.DoesNotContain(steps, s => s["result"]?["stolen"]?.GetValue<string>() == "unknown");
        }

        [Fact]
        public void AWorldAgreesWithTheSeatsViewAndIsMarkedDrawn()
        {
            ScriptedAgent agent = new(needs: "world");
            List<GameState> befores = PlayWithProxy(agent, seed: 8, seat: 3);
            OspCodec codec = new(befores[0].Board);
            bool hidden = false;
            foreach (JsonNode decide in agent.OfType("decide"))
            {
                GameState truth = befores[Int(decide["n"])];
                JsonNode world = decide["state"]!;
                JsonObject view = codec.View(truth, 3);
                Assert.True(world["drawn"]!.GetValue<bool>());
                foreach (string field in new[] { "prompt", "current", "free_roads", "robber", "buildings", "bank", "offer" })
                    Assert.Equal(OspCodec.Canonical(view[field]), OspCodec.Canonical(world[field]));
                for (int p = 0; p < 4; p++)
                {
                    JsonNode drawn = world["players"]![p]!;
                    // Every hand is there, with the true sizes; the seat's own is the true one
                    Assert.NotNull(drawn["hand"]);
                    Assert.Equal(truth.Players[p].ResourceCards.Count(), (uint)drawn["hand"]!.AsObject().Sum(c => Int(c.Value)));
                    Assert.Equal(truth.Players[p].DevelopmentCards.Count(), (uint)drawn["development"]!.AsObject().Sum(c => Int(c.Value)));
                    if (p == 3) Assert.Equal(OspCodec.Canonical(view["players"]![3]), OspCodec.Canonical(drawn));
                    else hidden |= OspCodec.Canonical(drawn["hand"]) != OspCodec.Canonical(codec.State(truth)["players"]![p]!["hand"]);
                    Assert.True(Int(drawn["vp"]) < truth.Settings.VictoryPoints);
                }
            }
            Assert.True(hidden, "a world should differ from the truth somewhere over a whole game");
            // Steps under world carry the seat's view, not a world
            foreach (JsonNode step in agent.OfType("step"))
                Assert.Equal(OspCodec.Canonical(codec.View(befores[Int(step["n"])], 3)), OspCodec.Canonical(step["state"]));
        }

        [Fact]
        public void AnAgentThatDoesntObserveGetsNoSteps()
        {
            ScriptedAgent agent = new(observe: false);
            PlayWithProxy(agent, seed: 9);
            Assert.Empty(agent.OfType("step"));
            Assert.NotEmpty(agent.OfType("decide"));
            Assert.Single(agent.OfType("after"));
        }
        #endregion

        #region Hexfarer's own Open Seat Agent under Hexfarer's referee
        /// <summary>
        /// Hexfarer's Open Seat Agent (SeatAgent) in the same process, answering through the channel
        /// </summary>
        private sealed class InProcessAgent(SeatAgent agent, bool dropSteps = false) : IOpenSeatChannel
        {
            private readonly Queue<JsonNode> _answers = new();
            // Through text both ways, as over a process's streams; without the steps if asked
            public void Send(JsonNode message)
            {
                if (dropSteps && message["type"]!.GetValue<string>() == "step") return;
                if (agent.Handle(JsonNode.Parse(message.ToJsonString())!) is JsonNode answer) _answers.Enqueue(JsonNode.Parse(answer.ToJsonString())!);
            }
            public JsonNode Receive(CancellationToken cancellation = default) => _answers.Dequeue();
            public void Dispose() { }
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void HexfarersOwnAgentPlaysASeatUnderHexfarersRefereeWithoutTakingAnythingOver(bool fair)
        {
            StringWriter log = new();
            SeatAgent seatAgent = new(new StateValueNet(TinyNet, provider: InferenceProvider.Cpu), iterations: 16, seconds: 0, log, leavesPerBatch: 16, fair: fair);
            List<GameState> befores = PlayWithProxy(new InProcessAgent(seatAgent), seed: 12, seat: 2);
            Assert.True(befores[^1].HasEnded);
            // The same rules on both sides: nothing to take over, no move replaced, never lost
            JsonNode stats = JsonNode.Parse(log.ToString().Trim().Split('\n')[^1])!;
            Assert.Equal(0, Int(stats["patches"]));
            Assert.Equal(0, Int(stats["fallbacks"]));
            Assert.False(stats["lost"]!.GetValue<bool>(), log.ToString());
            Assert.True(Int(stats["decisions"]) > 20);
        }
        #endregion

        [Fact]
        public void AGameRebuiltFromTheTrueStateWritesBackAsItAndListsTheSameMoves()
        {
            HashSet<string> prompts = [];
            foreach ((GameState state, _, _) in RandomMoves(games: 3).Where((_, i) => i % 3 == 0))
            {
                OspCodec codec = new(state.Board);
                sbyte acting = TestUtil.ActingPlayer(state);
                JsonObject truth = codec.State(state);
                List<string> legal = LegalActionProvider.GetActionsForState(state, acting).Select(a => OspCodec.Canonical(codec.Move(a))).Order().ToList();
                GameState rebuilt = codec.BuildState(truth, new JsonArray(legal.Select(m => JsonNode.Parse(m)).ToArray()));
                string at = $"{truth["prompt"]} with seat {acting}";
                Assert.Equal(OspCodec.Canonical(truth), OspCodec.Canonical(codec.State(rebuilt)));
                Assert.Equal(legal, LegalActionProvider.GetActionsForState(rebuilt, acting).Select(a => OspCodec.Canonical(codec.Move(a))).Order().ToList());
                Assert.True(TestUtil.ActingPlayer(rebuilt) == acting, at);
                prompts.Add(truth["prompt"]!.GetValue<string>());
            }
            Assert.Superset(new HashSet<string> { "setup_settlement", "setup_road", "roll", "main", "discard", "move_robber" }, prompts);
        }

        [Fact]
        public void HexfarersOpenHandAgentRebuildsItsGameWhenTheRefereeSendsNoSteps()
        {
            // As under JSettlers' robot: no steps, only each decision's true state
            StringWriter log = new();
            SeatAgent seatAgent = new(new StateValueNet(TinyNet, provider: InferenceProvider.Cpu), iterations: 16, seconds: 0, log, leavesPerBatch: 16);
            List<GameState> befores = PlayWithProxy(new InProcessAgent(seatAgent, dropSteps: true), seed: 14, seat: 1);
            Assert.True(befores[^1].HasEnded);
            JsonNode stats = JsonNode.Parse(log.ToString().Trim().Split('\n')[^1])!;
            Assert.False(stats["lost"]!.GetValue<bool>(), log.ToString());
            Assert.Equal(0, Int(stats["fallbacks"]));
            Assert.True(Int(stats["rebuilds"]) > 5, log.ToString());
        }

        #region Agents as processes
        [Fact]
        public void ACommandLineSplitsLikeAShellsWords()
        {
            Assert.Equal(["dotnet", "C:/Some Dir/OpenSeat.dll", "agent", "--iterations", "16"],
                OpenSeatAgentProcess.SplitCommandLine("dotnet \"C:/Some Dir/OpenSeat.dll\"  agent --iterations 16"));
            Assert.Equal(["python", "open_seat_agent.py", "--bot", "AB:2"], OpenSeatAgentProcess.SplitCommandLine("python open_seat_agent.py --bot AB:2"));
            Assert.Equal(["a", ""], OpenSeatAgentProcess.SplitCommandLine("a \"\""));
        }

        [Fact]
        public void AnAgentProcessPlaysAGameOverItsStandardStreams()
        {
            string errors = Path.Combine(Path.GetTempPath(), $"hexfarer-agent-{Guid.NewGuid():N}.log");
            try
            {
                OpenSeatAgentProcess process = OpenSeatAgentProcess.Start(AgentCommand(), TimeSpan.FromSeconds(60), errors);
                List<GameState> befores = PlayWithProxy(process, seed: 13, seat: 1);
                Assert.True(befores[^1].HasEnded);
                Assert.True(process.HasExited, "the agent should be closed after the game");
                // The agent's standard error went to the log, with its statistics line
                Assert.Contains("\"patches\":0", File.ReadAllText(errors));
            }
            finally { File.Delete(errors); }
        }

        [Fact]
        public void AProcessThatQuitsFailsWithWhatItSaid()
        {
            OpenSeatAgentProcess process = OpenSeatAgentProcess.Start($"dotnet \"{Path.Combine(AppContext.BaseDirectory, "OpenSeat.dll")}\" no-such-mode", TimeSpan.FromSeconds(60));
            OpenSeatException error = Assert.Throws<OpenSeatException>(() => OpenSeatHello.Exchange(process));
            Assert.Contains("ended", error.Message);
            Assert.Contains("Usage", error.Message);
            process.Dispose();
        }

        [Fact]
        public void LinesThatArentTheProtocolsAreSkippedAndKeptForTheErrorMessage()
        {
            // The help goes to standard output, as text, and the program ends
            OpenSeatAgentProcess process = OpenSeatAgentProcess.Start($"dotnet \"{Path.Combine(AppContext.BaseDirectory, "OpenSeat.dll")}\" --help", TimeSpan.FromSeconds(60));
            OpenSeatException error = Assert.Throws<OpenSeatException>(() => OpenSeatHello.Exchange(process));
            Assert.Contains("ended", error.Message);
            Assert.Contains("OpenSeat replay", error.Message);
            process.Dispose();
        }

        [Fact]
        public void AProcessThatDoesntAnswerTimesOut()
        {
            // The benchmark plays for a long time without reading its input
            OpenSeatAgentProcess process = OpenSeatAgentProcess.Start($"dotnet \"{Path.Combine(AppContext.BaseDirectory, "OpenSeat.dll")}\" bench --games 100000", TimeSpan.FromSeconds(2));
            OpenSeatException error = Assert.Throws<OpenSeatException>(() => OpenSeatHello.Exchange(process));
            Assert.Contains("no answer", error.Message);
            process.Dispose();
            Assert.True(process.HasExited);
        }
        #endregion
    }
}
