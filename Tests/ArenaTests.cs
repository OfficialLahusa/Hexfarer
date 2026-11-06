using Agents;
using Agents.Inference;
using Agents.MCTS;
using Arena;
using Common;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Tests
{
    /// <summary>
    /// The Arena duel tool: agent specs, the command line, the pair statistics with the SPRT, and a short mirror duel of two
    /// tiny untrained nets (only the plumbing, not playing strength)
    /// </summary>
    public class ArenaTests
    {
        private static string PolicyModelPath => TestUtil.FixturePath("ValueNet", "policy_tiny.onnx");
        private static string PlainModelPath => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        [Fact]
        public void AnAgentSpecWithoutOptionsHasTheDefaults()
        {
            AgentSpec spec = AgentSpec.Parse("I2=nets/i2.onnx");

            Assert.Equal("I2", spec.Name);
            Assert.Equal("nets/i2.onnx", spec.ModelPath);
            Assert.Equal(OpponentModel.MaxN, spec.OpponentModel);
            Assert.Equal(1.414, spec.Exploration);
            // No batch size given: small batches with an iteration budget, growing with it, large ones with a time budget
            Assert.Null(spec.LeavesPerBatch);
            Assert.Equal((16, 32, MCTSAgent.DefaultLeavesPerBatch), (spec.BatchFor(800), spec.BatchFor(5000), spec.BatchFor(0)));
            Assert.Equal(0.1, spec.FirstPlayReduction);
            Assert.Equal(0, spec.PlacementWeight);
            Assert.False(spec.UniformPriors);
            Assert.Equal("I2 i2.onnx MaxN", spec.ToString());
        }

        [Fact]
        public void AnAgentSpecReadsItsOptions()
        {
            AgentSpec spec = AgentSpec.Parse("c1=a.onnx=Paranoid=c1.0=b64=f0.25=p0.5=u");

            Assert.Equal(OpponentModel.Paranoid, spec.OpponentModel);
            Assert.Equal(1.0, spec.Exploration);
            Assert.Equal(64, spec.LeavesPerBatch);
            // A given batch size holds for either budget
            Assert.Equal((64, 64), (spec.BatchFor(800), spec.BatchFor(0)));
            Assert.Equal(0.25, spec.FirstPlayReduction);
            Assert.Equal(0.5, spec.PlacementWeight);
            Assert.True(spec.UniformPriors);
            Assert.Equal("c1 a.onnx Paranoid c1 b64 f0.25 p0.5 u", spec.ToString());
        }

        [Theory]
        [InlineData("A=x.onnx", 40, 0, 48)]
        [InlineData("A=x.onnx=b32", 40, 0, 64)]
        [InlineData("A=x.onnx", 0, 0.02, 256)]
        public void TheAgentSearchesInTheBatchesOfItsBudget(string spec, long iterations, double seconds, int multiple)
        {
            // The search runs whole batches: 40 iterations are three batches of 16 by default and two of 32 with b32; a time
            // budget searches in batches of 256
            AgentSpec agent = AgentSpec.Parse(spec.Replace("x.onnx", PlainModelPath));
            StateValueNet net = new(PlainModelPath, provider: InferenceProvider.Cpu);
            GameState state = new GameState(MapGenerator.GenerateRandomClassic(), 4);
            MCTSAgent search = Assert.IsType<MCTSAgent>(agent.Create(0, state, net, iterations, seconds));
            search.Act(state);
            long done = search.LastSearch!.Iterations;
            Assert.Equal(0, done % agent.BatchFor(iterations));
            if (iterations > 0) Assert.Equal(multiple, done);
            else Assert.True(done >= multiple, $"{done} iterations");
        }

        [Theory]
        [InlineData(40, 16)]
        [InlineData(800, 16)]
        [InlineData(3199, 16)]
        [InlineData(3200, 32)]
        [InlineData(5000, 32)]
        [InlineData(6399, 32)]
        [InlineData(6400, 64)]
        [InlineData(12800, 128)]
        [InlineData(25599, 128)]
        [InlineData(25600, 256)]
        [InlineData(1_000_000, 256)]
        public void TheDefaultBatchLeavesTheTreeAHundredUpdates(long iterations, int leaves)
        {
            // The largest power of two with at least 100 updates per decision, from 16 to 256
            Assert.Equal(leaves, AgentSpec.LeavesFor(iterations));
            Assert.Equal(leaves, AgentSpec.Parse("A=a.onnx").BatchFor(iterations));
        }

        [Fact]
        public void TheDefaultBatchOfPimcFollowsTheShareOfEachDeterminization()
        {
            // Four determinizations of 5000 iterations search trees of 1250, two of 12800 trees of 6400; bN holds as given
            Assert.Equal(16, AgentSpec.Parse("P=a.onnx=d4").BatchFor(5000));
            Assert.Equal(64, AgentSpec.Parse("P=a.onnx=d2").BatchFor(12800));
            Assert.Equal(128, AgentSpec.Parse("P=a.onnx=d1").BatchFor(12800));
            Assert.Equal(32, AgentSpec.Parse("P=a.onnx=d4=b32").BatchFor(5000));
            Assert.Equal(MCTSAgent.DefaultLeavesPerBatch, AgentSpec.Parse("P=a.onnx=d4").BatchFor(0));
        }

        [Theory]
        [InlineData(8, 1, 64)]
        [InlineData(16, 1, 64)]
        [InlineData(32, 1, 32)]
        [InlineData(48, 1, 21)]
        [InlineData(64, 1, 16)]
        [InlineData(256, 1, 16)]
        [InlineData(512, 1, 16)]
        [InlineData(8, 2, 128)]
        [InlineData(16, 2, 128)]
        [InlineData(32, 2, 64)]
        [InlineData(256, 2, 32)]
        public void TheDefaultGamesAtOnceKeepAThousandLeavesInFlightPerFile(int leavesPerBatch, int files, int games)
        {
            Assert.Equal(games, AgentSpec.GamesFor(leavesPerBatch, files));
        }

        [Fact]
        public void TheGamesAtOnceFollowTheSmallerBatchUnlessGiven()
        {
            DuelSettings Duel(string challenger, string baseline, long iterations, int? parallel = null, int maxGames = 4000) => new()
            {
                Challenger = AgentSpec.Parse(challenger),
                Baseline = AgentSpec.Parse(baseline),
                Iterations = iterations,
                Parallel = parallel,
                MaxGames = maxGames,
            };

            // One file, 5000 iterations search in batches of 32: 32 games; a side in batches of 16 needs 64
            Assert.Equal(32, Duel("C=a.onnx", "B=a.onnx", 5000).GamesAtOnce);
            Assert.Equal(64, Duel("C=a.onnx=b16", "B=a.onnx", 5000).GamesAtOnce);
            Assert.Equal(64, Duel("C=a.onnx", "B=a.onnx=b16", 5000).GamesAtOnce);
            Assert.Equal(64, Duel("C=a.onnx", "B=a.onnx", 800).GamesAtOnce);
            Assert.Equal(16, Duel("C=a.onnx=b256", "B=a.onnx=b64", 800).GamesAtOnce);
            // Two files: each side's evaluator gets about half the leaves, so twice the games
            Assert.Equal(64, Duel("C=c.onnx", "B=b.onnx", 5000).GamesAtOnce);
            Assert.Equal(128, Duel("C=c.onnx", "B=b.onnx=b16", 5000).GamesAtOnce);
            Assert.Equal(128, Duel("C=c.onnx", "B=b.onnx", 800).GamesAtOnce);
            // --parallel holds as given, never more games than the duel plays, one at a time with a time budget
            Assert.Equal(8, Duel("C=c.onnx", "B=b.onnx", 5000, parallel: 8).GamesAtOnce);
            Assert.Equal(10, Duel("C=c.onnx", "B=b.onnx", 800, maxGames: 10).GamesAtOnce);
            Assert.Equal(1, Duel("C=c.onnx", "B=b.onnx", 0, parallel: 8).GamesAtOnce);
            Assert.Equal(1, Duel("C=a.onnx", "B=a.onnx", 0).GamesAtOnce);
            Assert.Null(CommandLine.Parse(["mirror", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx"]).Settings!.Parallel);
        }

        [Fact]
        public void OnlyCudaRunsBatchesThatFillACallUnmerged()
        {
            AgentSpec small = AgentSpec.Parse("S=a.onnx"), b255 = AgentSpec.Parse("M=a.onnx=b255"), b256 = AgentSpec.Parse("L=a.onnx=b256"),
                b512 = AgentSpec.Parse("X=a.onnx=b512");
            Assert.True(AgentSpec.Merges(InferenceProvider.Cuda, 5000, small));
            Assert.True(AgentSpec.Merges(InferenceProvider.Cuda, 5000, b255));
            Assert.False(AgentSpec.Merges(InferenceProvider.Cuda, 5000, b256));
            Assert.False(AgentSpec.Merges(InferenceProvider.Cuda, 5000, b256, b512));
            // Without bN the budget sets the batch: 256 from 25600 iterations
            Assert.False(AgentSpec.Merges(InferenceProvider.Cuda, 25600, small));
            // Two sides sharing a file merge if either has small batches
            Assert.True(AgentSpec.Merges(InferenceProvider.Cuda, 5000, b512, small));
            Assert.True(AgentSpec.Merges(InferenceProvider.Cuda, 5000, small, b512));
            Assert.True(AgentSpec.Merges(InferenceProvider.DirectML, 5000, b256));
            Assert.True(AgentSpec.Merges(InferenceProvider.Cpu, 5000, b512));
        }

        [Fact]
        public void TheHeaderSaysWhoseLeafBatchesAreMerged()
        {
            StateValueNet net = new(PlainModelPath, provider: InferenceProvider.Cpu);
            using BatchingEvaluator small = new([net], 256), large = new([net], 512);
            SideNets alone = new(net, null), merged = new(net, small), mergedLarge = new(net, large);

            Assert.Equal("", MergedLeaves.Describe(alone, alone));
            Assert.Equal(", merged across games into net calls of up to 256 rows (shared by both sides)", MergedLeaves.Describe(merged, merged));
            Assert.Equal(", merged across games into net calls of up to 256 rows (the challenger's only)", MergedLeaves.Describe(merged, alone));
            Assert.Equal(", merged across games into net calls of up to 256 rows (the baseline's only)", MergedLeaves.Describe(alone, merged));
            // Two evaluators: the larger call size
            Assert.Equal(", merged across games into net calls of up to 512 rows", MergedLeaves.Describe(merged, mergedLarge));
            Assert.Equal(", merged across games into net calls of up to 512 rows", MergedLeaves.Describe(mergedLarge, merged));
        }

        [Fact]
        public void BothSidesShareANetOnlyWithTheSameFileAndBatchSize()
        {
            // One game at a time: no merging, every search runs its batches on its side's net
            AgentSpec a = AgentSpec.Parse($"a={PlainModelPath}"), b64 = AgentSpec.Parse($"b={PlainModelPath}=b64"), c = AgentSpec.Parse($"c={PolicyModelPath}");
            (SideNets first, SideNets second) = AgentSpec.Nets(a, AgentSpec.Parse($"a2={PlainModelPath}"), InferenceProvider.Cpu, 800, 1);
            Assert.Same(first.Net, second.Net);
            Assert.Null(first.Leaves);
            Assert.Null(second.Leaves);
            (first, second) = AgentSpec.Nets(a, b64, InferenceProvider.Cpu, 800, 1);
            Assert.NotSame(first.Net, second.Net);
            // With a time budget both default to 256: the same net again, unless the file differs; never merged, even with
            // several games allowed (a time budget plays one at a time)
            (first, second) = AgentSpec.Nets(AgentSpec.Parse($"t={PlainModelPath}=b256"), a, InferenceProvider.Cpu, 0, 16);
            Assert.Same(first.Net, second.Net);
            Assert.Null(first.Leaves);
            (first, second) = AgentSpec.Nets(a, c, InferenceProvider.Cpu, 0, 1);
            Assert.NotSame(first.Net, second.Net);
        }

        /// <summary>
        /// Both sides' searches sent their leaf batches to their own merging evaluator, and the duel's summary adds them up
        /// </summary>
        private static void AssertBothSidesMerged((SideNets Challenger, SideNets Baseline) nets, MergedLeaves merged)
        {
            BatchingEvaluator challenger = nets.Challenger.Leaves!, baseline = nets.Baseline.Leaves!;
            Assert.NotSame(challenger, baseline);
            Assert.True(challenger.Batches > 0, "the challenger's searches sent no batches");
            Assert.True(baseline.Batches > 0, "the baselines' searches sent no batches");
            Assert.Equal(challenger.Batches + baseline.Batches, merged.Calls);
            Assert.Equal(challenger.Rows + baseline.Rows, merged.Rows);
            Assert.Equal(challenger.RunRows + baseline.RunRows, merged.RunRows);
        }

        [Fact]
        public void GamesAtOnceMergeTheLeafBatchesOfEachFile()
        {
            AgentSpec a = AgentSpec.Parse($"a={PlainModelPath}"), b64 = AgentSpec.Parse($"b={PlainModelPath}=b64"), c = AgentSpec.Parse($"c={PolicyModelPath}");

            // The same file: one merging evaluator and one net for both sides, whatever their batch sizes
            (SideNets first, SideNets second) = AgentSpec.Nets(a, b64, InferenceProvider.Cpu, 800, 4);
            Assert.NotNull(first.Leaves);
            Assert.Same(first.Leaves, second.Leaves);
            Assert.Same(first.Net, second.Net);
            Assert.Equal(AgentSpec.MergedRows, first.Leaves!.MaxRows);
            // On the CPU a call runs only the rows in use (the full shape is for CUDA)
            first.Leaves.EvaluateAsync(new DenseTensor<float>([3, first.Net.InputDim]), false).GetAwaiter().GetResult();
            Assert.Equal(3, first.Leaves.Rows);
            Assert.Equal(3, first.Leaves.RunRows);
            MergedLeaves.Close(first, second);

            // A side's batches larger than a merged call set its size
            (first, second) = AgentSpec.Nets(AgentSpec.Parse($"big={PlainModelPath}=b512"), a, InferenceProvider.Cpu, 800, 4);
            Assert.Equal(512, first.Leaves!.MaxRows);
            MergedLeaves.Close(first, second);

            // Different files: one evaluator each, running its side's model
            (first, second) = AgentSpec.Nets(a, c, InferenceProvider.Cpu, 800, 4);
            Assert.NotNull(first.Leaves);
            Assert.NotNull(second.Leaves);
            Assert.NotSame(first.Leaves, second.Leaves);
            Assert.Null(first.Leaves!.Policy);
            Assert.NotNull(second.Leaves!.Policy);
            Assert.NotNull(second.Net.Policy);
            MergedLeaves.Close(first, second);
        }

        [Fact]
        public void AnAgentSpecReadsItsHiddenInformationSearch()
        {
            Assert.Equal(4, AgentSpec.Parse("P=a.onnx=d4").Determinizations);
            AgentSpec spec = AgentSpec.Parse("S=a.onnx=is");
            Assert.True(spec.InformationSets);
            Assert.Equal("S a.onnx MaxN is", spec.ToString());
            Assert.False(spec.FullKnowledge);
            AgentSpec full = AgentSpec.Parse("F=a.onnx=isfull");
            Assert.True(full.InformationSets && full.FullKnowledge);
            Assert.Equal("F a.onnx MaxN isfull", full.ToString());
            AgentSpec ablation = AgentSpec.Parse("A=a.onnx=isfull=ob=norm");
            Assert.True(ablation.BranchHiddenOutcomes);
            Assert.True(ablation.NormalizeValues);
            Assert.Equal("A a.onnx MaxN isfull norm ob", ablation.ToString());
            Assert.False(AgentSpec.Parse("M=a.onnx=nonorm").NormalizeValues);
            Assert.Null(AgentSpec.Parse("M=a.onnx").NormalizeValues);
            AgentSpec fresh = AgentSpec.Parse("N=a.onnx=noreuse");
            Assert.False(fresh.ReuseTree);
            Assert.Equal("N a.onnx MaxN noreuse", fresh.ToString());
            Assert.True(AgentSpec.Parse("L=a.onnx").EvaluateNewLeaves);
            AgentSpec old = AgentSpec.Parse("O=a.onnx=noleaf");
            Assert.False(old.EvaluateNewLeaves);
            Assert.Equal("O a.onnx MaxN noleaf", old.ToString());
            Assert.True(AgentSpec.Parse("F=a.onnx").PlayForcedMovesAtOnce);
            AgentSpec searching = AgentSpec.Parse("S=a.onnx=searchforced");
            Assert.False(searching.PlayForcedMovesAtOnce);
            Assert.Equal("S a.onnx MaxN searchforced", searching.ToString());
            Assert.Equal(1, AgentSpec.Parse("G=a.onnx=is").ReuseDecay);
            AgentSpec decaying = AgentSpec.Parse("G=a.onnx=is=g0.25");
            Assert.Equal(0.25, decaying.ReuseDecay);
            Assert.Equal("G a.onnx MaxN is g0.25", decaying.ToString());
            Assert.Throws<ArgumentException>(() => AgentSpec.Parse("G=a.onnx=g0"));
            Assert.Throws<ArgumentException>(() => AgentSpec.Parse("G=a.onnx=g1.5"));
        }

        [Theory]
        [InlineData("a.onnx")]
        [InlineData("=a.onnx")]
        [InlineData("A=")]
        [InlineData("A=a.onnx=x3")]
        public void AnInvalidAgentSpecIsRejected(string spec)
        {
            Assert.ThrowsAny<Exception>(() => AgentSpec.Parse(spec));
        }

        [Fact]
        public void TheCommandLineSetsTheDuel()
        {
            DuelSettings settings = CommandLine.Parse(["mirror", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx=c1.0", "--iterations", "800",
                "--parallel", "8", "--max-games", "1000", "--name", "loop_g009", "--sprt", "-10", "20", "--max-minutes", "90", "--provider", "cuda"]).Settings!;

            Assert.Equal("C", settings.Challenger.Name);
            Assert.Equal(1.0, settings.Baseline.Exploration);
            Assert.Equal(800, settings.Iterations);
            Assert.Equal(8, settings.Parallel);
            Assert.Equal(1000, settings.MaxGames);
            Assert.Equal(CommandLine.SeedOf("loop_g009"), settings.SeedBase);
            Assert.Equal(-10, settings.Elo0);
            Assert.Equal(20, settings.Elo1);
            Assert.Equal(TimeSpan.FromMinutes(90), settings.MaxDuration);
            Assert.Equal(InferenceProvider.Cuda, settings.Provider);
        }

        [Fact]
        public void ATimeBudgetReplacesTheIterations()
        {
            DuelSettings settings = CommandLine.Parse(["mirror", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx", "--seconds", "0.5"]).Settings!;

            Assert.Equal(0, settings.Iterations);
            Assert.Equal(0.5, settings.Seconds);
            Assert.Equal(30_000, settings.SeedBase);
        }

        [Fact]
        public void TheSeedOfANameIsStableAndSeparatesNames()
        {
            Assert.Equal(CommandLine.SeedOf("loop_g003"), CommandLine.SeedOf("loop_g003"));
            Assert.NotEqual(CommandLine.SeedOf("loop_g003"), CommandLine.SeedOf("loop_g006"));
            Assert.InRange(CommandLine.SeedOf("loop_g003"), 100_000, int.MaxValue - 10_000);
        }

        [Fact]
        public void AnAgentSpecReadsItsTradeOffers()
        {
            AgentSpec plain = AgentSpec.Parse("T=a.onnx=is");
            Assert.Equal((true, 0.005, 3, 8, false), (plain.MakeOffers, plain.OfferMargin, plain.MaxOffers, plain.OfferWorlds, plain.TwoForTwo));
            Assert.True(AgentSpec.Parse("T=a.onnx=o31").ThreeForOne);
            Assert.Equal((2, 0), (AgentSpec.Parse("T=a.onnx=oc2=op0").MaxCards, AgentSpec.Parse("T=a.onnx=oc2=op0").Prune));
            Assert.Equal((3, 16), (plain.MaxCards, plain.Prune));
            Assert.Throws<ArgumentException>(() => AgentSpec.Parse("T=a.onnx=oc4"));
            Assert.EndsWith("oc2 op0", AgentSpec.Parse("T=a.onnx=oc2=op0").ToString());
            Assert.EndsWith("is", plain.ToString());
            Assert.False(plain.ThreeForOne);
            AgentSpec spec = AgentSpec.Parse("T=a.onnx=is=om0.02=on1=ow4=o22=ob");
            Assert.Equal((true, 0.02, 1, 4, true, true), (spec.MakeOffers, spec.OfferMargin, spec.MaxOffers, spec.OfferWorlds, spec.TwoForTwo, spec.BranchHiddenOutcomes));
            Assert.EndsWith("is ob om0.02 on1 ow4 o22", spec.ToString());
            // Bots that only answer: neither offers nor counteroffers
            AgentSpec silent = AgentSpec.Parse("T=a.onnx=nooffers");
            Assert.Equal((false, false), (silent.MakeOffers, silent.Counters));
            Assert.EndsWith("nooffers noco", silent.ToString());
            Assert.Throws<ArgumentException>(() => AgentSpec.Parse("T=a.onnx=ow0"));
            Assert.Equal(0.5, AgentSpec.Parse("T=a.onnx=af0.5").AnswerFairness);
            Assert.Equal((1.0, 0.0), (plain.AnswerFairness, plain.PartnerFairness));
            Assert.Equal((0.0, 0.5), (AgentSpec.Parse("T=a.onnx=af0").AnswerFairness, AgentSpec.Parse("T=a.onnx=pf0.5").PartnerFairness));
            Assert.EndsWith("af0 pf0.5", AgentSpec.Parse("T=a.onnx=af0=pf0.5").ToString());
            AgentSpec answers = AgentSpec.Parse("T=a.onnx=am0.01=noco");
            Assert.Equal((0.01, false), (answers.AnswerMargin, answers.Counters));
            Assert.Equal((0.002, true), (plain.AnswerMargin, plain.Counters));
            Assert.EndsWith("am0.01 noco", answers.ToString());
            Assert.True(AgentSpec.Parse("T=a.onnx=noco=co").Counters);

            // The agent built from it offers as asked, with any search
            GameState start = new GameState(MapGenerator.GenerateRandomClassic(), 4);
            var net = new StateValueNet(PlainModelPath, provider: InferenceProvider.Cpu);
            foreach (string kind in new[] { "is", "d2", "c1.414" })
            {
                Agent agent = AgentSpec.Parse($"T={PlainModelPath}={kind}=om0.02=on1=ow4=o22=am0.02=noco=o31=af0.5=oc2=op8=pf0.25").Create(1, start, net, 16, 0);
                MCTSAgent search = agent switch { InformationSetMCTSAgent i => i.Search, DeterminizedMCTSAgent d => d.Search, _ => (MCTSAgent)agent };
                Assert.Equal((true, 0.02f, 1, 4, true), (search.Offers.Enabled, search.Offers.Margin, search.Offers.MaxPerTurn, search.Offers.Worlds, search.Offers.TwoForTwo));
                Assert.Equal((0.02f, false, true, 0.5f), (search.AnswerMargin, search.Offers.Counters, search.Offers.ThreeForOne, search.AnswerFairness));
                Assert.Equal((2, 8, 0.25f), (search.Offers.MaxCards, search.Offers.Prune, search.Offers.PartnerFairness));
                // An agent with the defaults gets the defaults of TradeOffers
                Agent plainAgent = AgentSpec.Parse($"T={PlainModelPath}={kind}").Create(1, start, net, 16, 0);
                MCTSAgent plainSearch = plainAgent switch { InformationSetMCTSAgent i => i.Search, DeterminizedMCTSAgent d => d.Search, _ => (MCTSAgent)plainAgent };
                TradeOffers defaults = new TradeOffers();
                Assert.Equal((defaults.Margin, defaults.MaxCards, defaults.Prune, defaults.Counters, defaults.Worlds, defaults.MaxPerTurn),
                    (plainSearch.Offers.Margin, plainSearch.Offers.MaxCards, plainSearch.Offers.Prune, plainSearch.Offers.Counters, plainSearch.Offers.Worlds, plainSearch.Offers.MaxPerTurn));
                Assert.Equal((defaults.PartnerFairness, new MCTSAgent(1, start, valueNet: net).AnswerFairness), (plainSearch.Offers.PartnerFairness, plainSearch.AnswerFairness));
            }
            Assert.False(((InformationSetMCTSAgent)AgentSpec.Parse($"T={PlainModelPath}=is=nooffers").Create(1, start, net, 16, 0)).Search.Offers.Enabled);

            // An agent that ignores what trades show keeps doing so after a reset
            InformationSetMCTSAgent deaf = (InformationSetMCTSAgent)AgentSpec.Parse($"T={PlainModelPath}=is=nolearn").Create(1, start, net, 16, 0);
            Assert.False(deaf.Beliefs.LearnsFromTrades);
            deaf.ResetToState(start);
            Assert.False(deaf.Beliefs.LearnsFromTrades);
            deaf.ResetTo(start, new BeliefTracker(1, start));
            Assert.False(deaf.Beliefs.LearnsFromTrades);
            Assert.True(((InformationSetMCTSAgent)AgentSpec.Parse($"T={PlainModelPath}=is").Create(1, start, net, 16, 0)).Beliefs.LearnsFromTrades);
        }

        [Fact]
        public void ATableDuelHasFreeTradeAndWinRateHypotheses()
        {
            CommandLine table = CommandLine.Parse(["table", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx=nooffers"]);
            Assert.True(table.Table);
            Assert.True(table.Settings!.FreeTrade);
            Assert.Equal((0.0, 3.0), (table.Settings.Elo0, table.Settings.Elo1));
            Assert.Equal((-1.0, 1.0), (CommandLine.Parse(["table", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx", "--sprt", "-1", "1"]).Settings!.Elo0, 1.0));

            CommandLine mirror = CommandLine.Parse(["mirror", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx"]);
            Assert.False(mirror.Table);
            Assert.False(mirror.Settings!.FreeTrade);
            Assert.True(CommandLine.Parse(["mirror", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx", "--free-trade"]).Settings!.FreeTrade);
        }

        [Fact]
        public void GroupResultsTestTheWinRate()
        {
            GroupStatistics strong = GroupStatistics.Of([.. Enumerable.Repeat(0.5, 40)], 0.25, 0.28);
            Assert.Equal(0.5, strong.Mean, 9);
            // No spread: the variance floor of 0.01
            Assert.Equal(40 * 0.03 * (1 - 0.53) / 0.02, strong.Llr, 6);
            Assert.Equal(1, strong.Verdict);

            double[] even = [.. Enumerable.Repeat(0.0, 15), .. Enumerable.Repeat(0.25, 10), .. Enumerable.Repeat(0.5, 10), .. Enumerable.Repeat(0.75, 5)];
            GroupStatistics s = GroupStatistics.Of(even, 0.25, 0.28);
            Assert.Equal(0.28125, s.Mean, 9);
            Assert.True(s.Low < 0.25 && s.High > 0.3);
            Assert.Equal(0, s.Verdict);
            Assert.Equal(0, GroupStatistics.Of([.. Enumerable.Repeat(1.0, 29)], 0.25, 0.28).Verdict);
        }

        [Fact]
        public void AShortTableDuelPutsTheChallengerInEverySeat()
        {
            var settings = new DuelSettings
            {
                Challenger = AgentSpec.Parse($"C={PolicyModelPath}=b8"),
                Baseline = AgentSpec.Parse($"B={PlainModelPath}=b8=nooffers"),
                Iterations = 16,
                MaxGames = 4,
                SeedBase = 321,
                Provider = InferenceProvider.Cpu,
                FreeTrade = true,
                Elo0 = 0,
                Elo1 = 3,
            };
            var log = new StringWriter();

            TableDuel duel = new(settings, log);
            GroupStatistics result = duel.Run();

            string text = log.ToString();
            // Four games at once (all the duel plays) with two files: the challenger's and the three baselines' leaf batches
            // each went through their side's merging evaluator, and the summary counts both
            Assert.Contains("16 iterations per decision, 4 games at once", text);
            Assert.Contains("merged across games into net calls of up to 256 rows;", text);
            AssertBothSidesMerged(duel.Nets!.Value, duel.Merged);
            Assert.Contains($"merged leaf batches: {duel.Merged}", text);
            Assert.Equal(1, result.Groups);
            string[] games = text.Split('\n').Where(line => line.Contains(" done): ")).ToArray();
            Assert.Equal(4, games.Length);
            for (int seat = 1; seat <= 4; seat++) Assert.Single(games, line => line.Contains($"challenger in seat {seat},"));
            Assert.All(games, line => Assert.Matches(@"offers C \d+ \(\d+ traded\), B 0 \(0 traded, 0 with C\)", line));
            Assert.Contains("all games played: 4 games", text);
        }

        [Theory]
        [InlineData("mirror", "--challenger", "A=a.onnx")]
        [InlineData("mirror", "--challenger", "A=a.onnx", "--baseline", "A=b.onnx")]
        [InlineData("mirror", "--challenger", "A=a.onnx", "--baseline", "B=b.onnx", "--sprt", "40", "0")]
        [InlineData("duel", "--challenger", "A=a.onnx", "--baseline", "B=b.onnx")]
        [InlineData("mirror", "--challenger", "A=a.onnx", "--baseline", "B=b.onnx", "--parallel")]
        public void AnInvalidCommandLineIsRejected(params string[] args)
        {
            Assert.Throws<ArgumentException>(() => CommandLine.Parse(args));
        }

        [Fact]
        public void ClearPairResultsAcceptH1()
        {
            double[] scores = [.. Enumerable.Repeat(1.0, 42), .. Enumerable.Repeat(0.0, 7), .. Enumerable.Repeat(0.5, 55)];

            PairStatistics s = PairStatistics.Of(scores, 0, 40);

            Assert.Equal((104, 42, 7, 55), (s.Pairs, s.ChallengerTwice, s.BaselineTwice, s.Split));
            Assert.Equal(9.211, s.Llr, 3);
            Assert.Equal(121.7, s.Elo, 1);
            Assert.True(s.EloLow > 0 && s.EloHigh > s.Elo);
            Assert.Equal(-2.944, s.LowerBound, 3);
            Assert.Equal(2.944, s.UpperBound, 3);
            Assert.Equal(1, s.Verdict);
        }

        [Fact]
        public void EvenPairResultsLeaveTheTestOpen()
        {
            double[] scores = [.. Enumerable.Repeat(1.0, 10), .. Enumerable.Repeat(0.0, 12), .. Enumerable.Repeat(0.5, 30)];

            PairStatistics s = PairStatistics.Of(scores, 0, 40);

            Assert.Equal(-1.328, s.Llr, 3);
            Assert.Equal(-13.4, s.Elo, 1);
            Assert.True(s.EloLow < 0 && s.EloHigh > 0);
            Assert.Equal(0, s.Verdict);
        }

        [Fact]
        public void IdenticalFirstPairsDecideNothing()
        {
            Assert.Equal(0, PairStatistics.Of([0.5, 0.5], 0, 40).Verdict);
            PairStatistics tenSplits = PairStatistics.Of([.. Enumerable.Repeat(0.5, 10)], 0, 40);
            Assert.InRange(tenSplits.Llr, tenSplits.LowerBound, 0);
            Assert.Equal(0, tenSplits.Verdict);
        }

        [Fact]
        public void FewerThanTwoPairsGiveNoEstimate()
        {
            PairStatistics s = PairStatistics.Of([1.0], 0, 40);

            Assert.Equal(1, s.Pairs);
            Assert.Equal(0, s.Llr);
            Assert.Equal(double.NegativeInfinity, s.EloLow);
            Assert.Equal(0, s.Verdict);
        }

        [Fact]
        public void AShortMirrorDuelPlaysAllPairs()
        {
            var settings = new DuelSettings
            {
                Challenger = AgentSpec.Parse($"P={PolicyModelPath}=b8"),
                Baseline = AgentSpec.Parse($"V={PlainModelPath}=b8"),
                Iterations = 16,
                MaxGames = 4,
                SeedBase = 123,
                Provider = InferenceProvider.Cpu,
            };
            var log = new StringWriter();

            MirrorDuel duel = new(settings, log);
            PairStatistics result = duel.Run();

            string text = log.ToString();
            // By default as many games at once as the duel plays (here fewer than GamesFor its batches of 8), with two files:
            // each side's leaf batches went through its own merging evaluator
            Assert.Contains("16 iterations per decision, 4 games at once", text);
            Assert.Contains("merged across games into net calls of up to 256 rows;", text);
            AssertBothSidesMerged(duel.Nets!.Value, duel.Merged);
            Assert.Contains($"merged leaf batches: {duel.Merged}", text);
            Assert.Equal(2, result.Pairs);
            Assert.Equal(4, text.Split('\n').Count(line => line.Contains(" game ")));
            Assert.Contains("seats C B C B", text);
            Assert.Contains("seats B C B C", text);
            Assert.Contains("all games played: 4 games", text);
            Assert.Matches(@"[+-]?\d+ Elo \[[+-]?\d+, [+-]?\d+\]", text.TrimEnd().Split('\n')[^1]);
        }

        // Hexfarer's own Open Seat Agent, built next to the tests, as an external command
        private static string OpenSeatAgentCommand => $"dotnet \"{Path.Combine(AppContext.BaseDirectory, "OpenSeat.dll")}\" agent --net \"{PlainModelPath}\" --iterations 16 --leaves-per-batch 16 --provider cpu";

        [Fact]
        public void AnAgentSpecCanNameAnOpenSeatAgentCommand()
        {
            // Everything after "command:" is the command line, equals signs included
            AgentSpec spec = AgentSpec.Parse("AB2=command:python open_seat_agent.py --bot AB:2 --needs=truth");
            Assert.Equal("AB2", spec.Name);
            Assert.True(spec.IsOpenSeatAgent);
            Assert.Equal("python open_seat_agent.py --bot AB:2 --needs=truth", spec.Command);
            Assert.Equal("AB2 Open Seat Agent: python open_seat_agent.py --bot AB:2 --needs=truth", spec.ToString());
            Assert.False(AgentSpec.Parse("I2=nets/i2.onnx").IsOpenSeatAgent);
            Assert.Throws<ArgumentException>(() => AgentSpec.Parse("AB2=command:"));
        }

        [Fact]
        public void OnlyTheSidesWithANetGetNetsAndSetTheGamesAtOnce()
        {
            AgentSpec hexfarer = AgentSpec.Parse($"H={PlainModelPath}"), external = AgentSpec.Parse("E=command:python x.py");
            (SideNets challenger, SideNets baseline) = AgentSpec.Nets(hexfarer, external, InferenceProvider.Cpu, 800, 4);
            Assert.NotNull(challenger.Net);
            Assert.Null(baseline.Net);
            Assert.Null(baseline.Leaves);
            MergedLeaves.Close(challenger, baseline);
            // One net file: the games at once follow its batches alone
            DuelSettings settings = new() { Challenger = hexfarer, Baseline = external, Iterations = 800 };
            Assert.Equal(AgentSpec.GamesFor(16, files: 1), settings.GamesAtOnce);
            settings = new() { Challenger = external, Baseline = AgentSpec.Parse($"H={PlainModelPath}=b64"), Iterations = 800 };
            Assert.Equal(AgentSpec.GamesFor(64, files: 1), settings.GamesAtOnce);
        }

        [Fact]
        public void ATableCanDoWithoutTradesBetweenPlayers()
        {
            Assert.True(CommandLine.Parse(["table", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx"]).Settings!.FreeTrade);
            Assert.False(CommandLine.Parse(["table", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx", "--no-free-trade"]).Settings!.FreeTrade);
        }

        [Fact]
        public void ATableDuelSeatsOpenSeatAgentProcesses()
        {
            var settings = new DuelSettings
            {
                Challenger = AgentSpec.Parse($"C={PlainModelPath}=b8"),
                Baseline = AgentSpec.Parse($"B=command:{OpenSeatAgentCommand}"),
                Iterations = 16,
                MaxGames = 4,
                SeedBase = 321,
                Provider = InferenceProvider.Cpu,
                FreeTrade = false,
                Elo0 = 0,
                Elo1 = 3,
            };
            var log = new StringWriter();
            GroupStatistics result = new TableDuel(settings, log).Run();

            string text = log.ToString();
            Assert.Contains("B Open Seat Agent: dotnet", text);
            // What the agent asked for is named once, so every game is labeled with what its seats received
            Assert.Contains("Open Seat Agent B: hexfarer-mcts-open-16it-b16, needs truth", text);
            Assert.DoesNotContain("failed", text);
            Assert.Equal(1, result.Groups);
            Assert.Contains("all games played: 4 games", text);
        }

        [Fact]
        public void AMirrorDuelSeatsOpenSeatAgentProcesses()
        {
            var settings = new DuelSettings
            {
                Challenger = AgentSpec.Parse($"C={PlainModelPath}=b8"),
                Baseline = AgentSpec.Parse($"B=command:{OpenSeatAgentCommand} --fair"),
                Iterations = 16,
                MaxGames = 2,
                SeedBase = 123,
                Provider = InferenceProvider.Cpu,
            };
            var log = new StringWriter();
            PairStatistics result = new MirrorDuel(settings, log).Run();
            string text = log.ToString();
            Assert.Contains("Open Seat Agent B: hexfarer-ismcts-fair-16it-b16, needs truth", text);
            Assert.DoesNotContain("failed", text);
            Assert.Equal(1, result.Pairs);
        }
    }
}
