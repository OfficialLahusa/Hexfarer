using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using SelfPlayCollector;
using System.Buffers.Binary;
using System.Globalization;
using System.Text.Json;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The self-play collector: command line, and a short run with the tiny untrained fixture net on the CPU whose output is
    /// checked against the games replayed from their seeds and save files
    /// </summary>
    public class SelfPlayCollectorTests
    {
        private static string TinyModelPath => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        [Fact]
        public void CommandLineSetsTheSearchBudgetsAndRejectsBadValues()
        {
            SelfPlaySettings settings = SelfPlayCollector.CommandLine.Parse(["--games", "20", "--full-iterations", "400", "--fast-iterations", "100",
                "--full-probability", "0.5", "--temperature-moves", "10", "--provider", "cpu", "--seed", "7"]).Settings;
            Assert.Equal(20, settings.Games);
            Assert.Equal(400, settings.FullIterations);
            Assert.Equal(100, settings.FastIterations);
            Assert.Equal(0.5, settings.FullSearchProbability);
            Assert.Equal(10, settings.TemperatureMoves);
            Assert.Equal(InferenceProvider.CPU, settings.Provider);
            Assert.Equal(7UL, settings.Seed);

            Assert.Throws<ArgumentException>(() => SelfPlayCollector.CommandLine.Parse(["--full-probability", "1.5"]));
            Assert.Throws<ArgumentException>(() => SelfPlayCollector.CommandLine.Parse(["--leaves", "64", "--batch", "32"]));
            Assert.Throws<ArgumentException>(() => SelfPlayCollector.CommandLine.Parse(["--unknown"]));
        }

        [Theory]
        [InlineData("placements_tiny.onnx")]
        [InlineData("policy_tiny.onnx")]
        public void RunRecordsPositionsThatReplayFromTheSavedGames(string model)
        {
            string modelPath = TestUtil.FixturePath("ValueNet", model);
            string root = Path.Combine(Path.GetTempPath(), "lanzenbrecher-selfplay-" + Guid.NewGuid().ToString("N"));
            try
            {
                SelfPlaySettings settings = new()
                {
                    Games = 3, ConcurrentGames = 3, LeavesPerBatch = 8, FullIterations = 32, FastIterations = 16, FullSearchProbability = 0.5,
                    TemperatureMoves = 10, MaxActions = 150, ModelPath = modelPath, Nets = 1, BatchRows = 64, Provider = InferenceProvider.CPU,
                    Seed = 11, OutputDirectory = root, Name = "run",
                };
                SelfPlayCollector.SelfPlayCollector collector = new(settings, TextWriter.Null);
                SelfPlayResult result = collector.Collect();
                Assert.Equal(0, result.Errors);
                Assert.Equal(3, result.Games);

                string directory = collector.RunDirectory;
                IStateVectorizer vectorizer = StateVectorizer.For(new StateValueNet(modelPath, provider: InferenceProvider.CPU));
                var games = ReadCSV(Path.Combine(directory, "games.csv"));
                var samples = ReadCSV(Path.Combine(directory, "samples.csv"));
                Assert.Equal(new long[] { 0, 1, 2 }, games.Select(game => long.Parse(game["game"])).Order());
                Assert.Equal(result.Positions, samples.Count);
                Assert.True(samples.Count > 0);

                byte[] states = File.ReadAllBytes(Path.Combine(directory, "states_c0000.f16"));
                Assert.Equal(samples.Count * vectorizer.FeatureDim * sizeof(ushort), states.Length);

                foreach (var game in games)
                {
                    long id = long.Parse(game["game"]);
                    ulong seed = ulong.Parse(game["seed"]);
                    Assert.Equal(Seeds.Derive(settings.Seed, (ulong)id), seed);

                    // The save file's actions replay from the seed's initial state to its final state
                    SaveFile saved = SelfPlayWriter.ReadRecord(directory, int.Parse(game["record_chunk"]), long.Parse(game["record_offset"]), int.Parse(game["record_length"]));
                    Assert.Equal(int.Parse(game["actions"]), saved.PlayedActions.Count);
                    Assert.Equal(game["finished"] == "1", saved.GameState.HasEnded);
                    Assert.Equal(saved.GameState.HasEnded ? saved.GameState.Turn.PlayerIndex : -1, int.Parse(game["winner"]));

                    GameState state = SelfPlayGame.InitialState(seed, settings.CenterDesert);
                    var positions = samples.Where(sample => sample["game"] == game["game"]).ToDictionary(sample => int.Parse(sample["action"]));
                    Assert.Equal(int.Parse(game["positions"]), positions.Count);

                    for (int i = 0; i <= saved.PlayedActions.Count; i++)
                    {
                        if (positions.TryGetValue(i, out var sample))
                            CheckPosition(sample, state, states, vectorizer, prunedVisits: model == "policy_tiny.onnx");
                        if (i == saved.PlayedActions.Count) break;

                        Action action = saved.PlayedActions[i];
                        Assert.True(action.IsValidFor(state), $"game {id}: action {i} doesn't replay");
                        action.Apply(state);
                    }
                    Assert.True(state.Equals(saved.GameState), $"game {id}: the replayed final state differs from the saved one");
                }

                // One entry of encoded moves per position, as many as the position's legal moves; writing them afterwards
                // by replaying the games gives the same files
                int[] counts = ReadInts(Path.Combine(directory, PolicyTargetWriter.CountsFile));
                Assert.Equal(samples.Count, counts.Length);
                Assert.Equal(counts.Sum() * ActionEncoding.MaxTokens * sizeof(short), new FileInfo(Path.Combine(directory, PolicyTargetWriter.TokensFile)).Length);

                // Per legal move the raw visits, the deciding player's value (NaN without visits) and the clean prior (NaN
                // without a policy head), per position the decider's value range: the inputs of Gumbel targets
                bool policy = model == "policy_tiny.onnx";
                float[] values = ReadFloats(Path.Combine(directory, PolicyValueWriter.ValuesFile));
                float[] ranges = ReadFloats(Path.Combine(directory, PolicyValueWriter.RangesFile));
                Assert.Equal(counts.Sum() * PolicyValueWriter.FloatsPerMove, values.Length);
                Assert.Equal(samples.Count * 2, ranges.Length);
                int offset = 0;
                for (int s = 0; s < samples.Count; s++)
                {
                    var pruned = samples[s]["visits"].Split(' ').Select(pair => pair.Split(':')).ToDictionary(pair => int.Parse(pair[0]), pair => uint.Parse(pair[1]));
                    double priorSum = 0;
                    long rawSum = 0;
                    for (int m = 0; m < counts[s]; m++)
                    {
                        float raw = values[(offset + m) * 3], value = values[(offset + m) * 3 + 1], prior = values[(offset + m) * 3 + 2];
                        Assert.Equal(raw == 0, float.IsNaN(value));
                        if (raw > 0) Assert.InRange(value, 0f, 1f);
                        // The recorded (pruned) visits never exceed the raw ones
                        uint recorded = pruned.TryGetValue(m, out uint kept) ? kept : 0;
                        Assert.True(raw >= recorded, $"position {s}, move {m}: fewer raw visits than recorded");
                        // Without a policy head nothing is pruned: the recorded visits are the raw ones
                        if (!policy) Assert.Equal(recorded, raw);
                        Assert.Equal(!policy, float.IsNaN(prior));
                        if (policy) priorSum += prior;
                        rawSum += (long)raw;
                    }
                    if (policy) Assert.Equal(1.0, priorSum, 3);
                    long rootVisits = long.Parse(samples[s]["root_visits"]);
                    Assert.InRange(rawSum, rootVisits - 1, rootVisits);
                    // The moves' values are the decider's: weighted by their visits they give the root's value of the decider
                    // (up to the root's own first visit)
                    double weighted = Enumerable.Range(0, counts[s]).Where(m => values[(offset + m) * 3] > 0)
                        .Sum(m => (double)values[(offset + m) * 3] * values[(offset + m) * 3 + 1]) / rawSum;
                    double rootValue = double.Parse(samples[s][$"q{samples[s]["actor"]}"], CultureInfo.InvariantCulture);
                    Assert.True(Math.Abs(weighted - rootValue) <= 1.5 / rootVisits + 1e-5, $"position {s}: the moves' values give {weighted}, the root {rootValue}");
                    (float low, float high) = (ranges[2 * s], ranges[2 * s + 1]);
                    Assert.Equal(float.IsNaN(low), float.IsNaN(high));
                    if (!float.IsNaN(low)) Assert.True(low < high);
                    offset += counts[s];
                }

                string[] policyFiles = [PolicyTargetWriter.CountsFile, PolicyTargetWriter.TokensFile, PolicyTargetWriter.WeightsFile, PolicyTargetWriter.LayoutFile];
                var written = policyFiles.ToDictionary(name => name, name => File.ReadAllBytes(Path.Combine(directory, name)));
                foreach (string name in policyFiles) File.Delete(Path.Combine(directory, name));
                Assert.Equal(samples.Count, PolicyTargetWriter.ConvertRun(directory, TextWriter.Null));
                foreach (string name in policyFiles)
                    Assert.Equal(written[name], File.ReadAllBytes(Path.Combine(directory, name)));
                Assert.Throws<InvalidOperationException>(() => PolicyTargetWriter.ConvertRun(directory, TextWriter.Null));
            }
            finally
            {
                if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
            }
        }

        [Theory]
        [InlineData("view_tiny.onnx")]
        [InlineData("policy_tiny.onnx")]
        public void UnderHiddenInformationEveryPositionIsRecordedAsItsActorsView(string model)
        {
            // Each seat searches a world drawn from its own view; the run records what the seat saw, whatever the net's layout
            string modelPath = TestUtil.FixturePath("ValueNet", model);
            string root = Path.Combine(Path.GetTempPath(), "lanzenbrecher-selfplay-" + Guid.NewGuid().ToString("N"));
            try
            {
                SelfPlaySettings settings = new()
                {
                    Games = 3, ConcurrentGames = 3, LeavesPerBatch = 8, FullIterations = 32, FastIterations = 16, FullSearchProbability = 0.5,
                    TemperatureMoves = 10, MaxActions = 400, ModelPath = modelPath, Nets = 1, BatchRows = 64, Provider = InferenceProvider.CPU,
                    Seed = 13, OutputDirectory = root, Name = "run", HiddenInformation = true,
                };
                SelfPlayCollector.SelfPlayCollector collector = new(settings, TextWriter.Null);
                Assert.Equal(0, collector.Collect().Errors);
                string directory = collector.RunDirectory;
                Assert.Contains($"\"feature_dim\": {ViewVectorizer.FeatureDim}", File.ReadAllText(Path.Combine(directory, "metadata.json")));
                Assert.Contains("\"hidden_information\": true", File.ReadAllText(Path.Combine(directory, "metadata.json")));

                var samples = ReadCSV(Path.Combine(directory, "samples.csv"));
                Assert.NotEmpty(samples);
                byte[] states = File.ReadAllBytes(Path.Combine(directory, "states_c0000.f16"));
                Assert.Equal(samples.Count * ViewVectorizer.FeatureDim * sizeof(ushort), states.Length);

                foreach (var game in ReadCSV(Path.Combine(directory, "games.csv")))
                {
                    SaveFile saved = SelfPlayWriter.ReadRecord(directory, int.Parse(game["record_chunk"]), long.Parse(game["record_offset"]), int.Parse(game["record_length"]));
                    GameState state = SelfPlayGame.InitialState(ulong.Parse(game["seed"]), settings.CenterDesert);
                    BeliefTracker[] trackers = Enumerable.Range(0, 4).Select(seat => new BeliefTracker(seat, state)).ToArray();
                    var positions = samples.Where(sample => sample["game"] == game["game"]).ToDictionary(sample => int.Parse(sample["action"]));
                    for (int i = 0; i <= saved.PlayedActions.Count; i++)
                    {
                        if (positions.TryGetValue(i, out var sample))
                        {
                            int actor = int.Parse(sample["actor"]);
                            Assert.True(state.CanPlayerAct(actor));
                            float[] expected = new float[ViewVectorizer.FeatureDim];
                            ViewVectorizer.VectorizeView(state, trackers[actor], expected);
                            int row = int.Parse(sample["row"]);
                            for (int f = 0; f < expected.Length; f++)
                                Assert.Equal((Half)expected[f], BinaryPrimitives.ReadHalfLittleEndian(states.AsSpan((row * expected.Length + f) * sizeof(ushort))));

                            // The searched moves are the true position's
                            int moves = MCTSTree.GetValidActions(state).Count;
                            Assert.All(sample["visits"].Split(' ').Select(pair => int.Parse(pair.Split(':')[0])), move => Assert.InRange(move, 0, moves - 1));
                        }
                        if (i == saved.PlayedActions.Count) break;
                        Action action = saved.PlayedActions[i];
                        Assert.True(action.IsValidFor(state), $"action {i} doesn't replay");
                        GameState before = new GameState(state);
                        action.Apply(state);
                        foreach (BeliefTracker tracker in trackers) tracker.Observe(before, action, state);
                    }
                    Assert.True(state.Equals(saved.GameState));
                }

                // The moves of every position can be encoded afterwards like those of any run
                Assert.Equal(samples.Count, ViewFeatureWriter.ConvertRun(directory, TextWriter.Null));
            }
            finally
            {
                if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
            }
        }

        [Fact]
        public void ViewFeaturesAreTheActorsViewOfEveryPosition()
        {
            string modelPath = TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");
            string root = Path.Combine(Path.GetTempPath(), "lanzenbrecher-selfplay-" + Guid.NewGuid().ToString("N"));
            try
            {
                SelfPlaySettings settings = new()
                {
                    Games = 3, ConcurrentGames = 3, LeavesPerBatch = 8, FullIterations = 32, FastIterations = 16, FullSearchProbability = 0.5,
                    TemperatureMoves = 10, MaxActions = 300, ModelPath = modelPath, Nets = 1, BatchRows = 64, Provider = InferenceProvider.CPU,
                    Seed = 12, OutputDirectory = root, Name = "run",
                };
                SelfPlayCollector.SelfPlayCollector collector = new(settings, TextWriter.Null);
                Assert.Equal(0, collector.Collect().Errors);
                string directory = collector.RunDirectory;
                var samples = ReadCSV(Path.Combine(directory, "samples.csv"));

                Assert.Equal(samples.Count, ViewFeatureWriter.ConvertRun(directory, TextWriter.Null));
                byte[] views = File.ReadAllBytes(Path.Combine(directory, ViewFeatureWriter.FileName));
                Assert.Equal(samples.Count * ViewVectorizer.FeatureDim * sizeof(ushort), views.Length);

                // Row i is the i-th line of samples.csv: the position as its actor sees it, with its tracker's beliefs
                var lineOf = samples.Select((sample, line) => (sample, line)).ToDictionary(x => (x.sample["game"], int.Parse(x.sample["action"])), x => x.line);
                foreach (var game in ReadCSV(Path.Combine(directory, "games.csv")))
                {
                    SaveFile saved = SelfPlayWriter.ReadRecord(directory, int.Parse(game["record_chunk"]), long.Parse(game["record_offset"]), int.Parse(game["record_length"]));
                    GameState state = SelfPlayGame.InitialState(ulong.Parse(game["seed"]), settings.CenterDesert);
                    BeliefTracker[] trackers = Enumerable.Range(0, 4).Select(seat => new BeliefTracker(seat, state)).ToArray();
                    for (int i = 0; i <= saved.PlayedActions.Count; i++)
                    {
                        if (lineOf.TryGetValue((game["game"], i), out int line))
                        {
                            int actor = int.Parse(samples[line]["actor"]);
                            float[] expected = new float[ViewVectorizer.FeatureDim];
                            ViewVectorizer.VectorizeView(state, trackers[actor], expected);
                            for (int f = 0; f < expected.Length; f++)
                                Assert.Equal((Half)expected[f], BinaryPrimitives.ReadHalfLittleEndian(views.AsSpan((line * expected.Length + f) * sizeof(ushort))));
                        }
                        if (i == saved.PlayedActions.Count) break;
                        GameState before = new GameState(state);
                        Action action = saved.PlayedActions[i];
                        action.Apply(state);
                        foreach (BeliefTracker tracker in trackers) tracker.Observe(before, action, state);
                    }
                }

                Assert.Throws<InvalidOperationException>(() => ViewFeatureWriter.ConvertRun(directory, TextWriter.Null));
            }
            finally
            {
                if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
            }
        }

        [Fact]
        public void TargetCheckSearchesRecordedPositionsShallowWithNoiseCleanAndDeep()
        {
            string modelPath = TestUtil.FixturePath("ValueNet", "policy_tiny.onnx");
            string root = Path.Combine(Path.GetTempPath(), "lanzenbrecher-targetcheck-" + Guid.NewGuid().ToString("N"));
            try
            {
                SelfPlaySettings settings = new()
                {
                    Games = 4, ConcurrentGames = 4, LeavesPerBatch = 8, FullIterations = 32, FastIterations = 16, FullSearchProbability = 1,
                    TemperatureMoves = 10, MaxActions = 500, ModelPath = modelPath, Nets = 1, BatchRows = 64, Provider = InferenceProvider.CPU,
                    Seed = 12, OutputDirectory = root, Name = "run",
                };
                SelfPlayCollector.SelfPlayCollector collector = new(settings, TextWriter.Null);
                Assert.Equal(0, collector.Collect().Errors);
                string directory = collector.RunDirectory;
                var samples = ReadCSV(Path.Combine(directory, "samples.csv"));
                var games = ReadCSV(Path.Combine(directory, "games.csv")).ToDictionary(game => game["game"]);

                // Every recorded position, so some are decided out of turn (discards, answers to trades)
                string output = Path.Combine(root, "check.jsonl");
                TargetCheckSettings check = TargetCheckSettings.Parse([directory, output, "--positions", "100000", "--shallow", "32", "--deep", "96", "--leaves", "8",
                    "--model", modelPath, "--provider", "cpu", "--nets", "1", "--batch", "64", "--concurrent", "8", "--seed", "5"]);
                Assert.Equal(100000, check.Positions);
                Assert.Equal(96, check.DeepIterations);
                Assert.Equal(samples.Count, TargetCheck.Run(check, TextWriter.Null));

                string[] lines = File.ReadAllLines(output);
                Assert.Equal(samples.Count, lines.Length);
                int outOfTurn = 0;
                var seen = new HashSet<(long, int)>();
                foreach (string line in lines)
                {
                    using JsonDocument document = JsonDocument.Parse(line);
                    JsonElement record = document.RootElement;
                    long game = record.GetProperty("game").GetInt64();
                    int action = record.GetProperty("action").GetInt32();
                    Assert.True(seen.Add((game, action)), "a position was checked twice");
                    var sample = samples.Single(s => long.Parse(s["game"]) == game && int.Parse(s["action"]) == action);
                    Assert.Equal(int.Parse(sample["actor"]), record.GetProperty("decider").GetInt32());
                    Assert.Equal(int.Parse(sample["round"]), record.GetProperty("round").GetInt32());

                    // The position replays from the saved game: its legal moves are the record's moves
                    var saved = games[sample["game"]];
                    SaveFile save = SelfPlayWriter.ReadRecord(directory, int.Parse(saved["record_chunk"]), long.Parse(saved["record_offset"]), int.Parse(saved["record_length"]));
                    GameState state = SelfPlayGame.InitialState(ulong.Parse(saved["seed"]), settings.CenterDesert);
                    for (int i = 0; i < action; i++) save.PlayedActions[i].Apply(state);
                    int moves = MCTSTree.GetValidActions(state).Count;
                    Assert.Equal(moves, record.GetProperty("moves").GetInt32());
                    if (record.GetProperty("decider").GetInt32() != state.Turn.PlayerIndex) outOfTurn++;

                    double[] prior = record.GetProperty("prior").EnumerateArray().Select(p => p.GetDouble()).ToArray();
                    Assert.Equal(moves, prior.Length);
                    Assert.Equal(1.0, prior.Sum(), 4);

                    JsonElement searches = record.GetProperty("searches");
                    long Total(string name) => searches.GetProperty(name).GetProperty("visits").EnumerateArray().Sum(v => v.GetInt64());
                    foreach (string name in new[] { "shallow", "clean", "deep" })
                    {
                        JsonElement search = searches.GetProperty(name);
                        long[] visits = search.GetProperty("visits").EnumerateArray().Select(v => v.GetInt64()).ToArray();
                        long[] pruned = search.GetProperty("pruned").EnumerateArray().Select(v => v.GetInt64()).ToArray();
                        JsonElement[] q = search.GetProperty("q").EnumerateArray().ToArray();
                        Assert.Equal(moves, visits.Length);
                        Assert.Equal(moves, pruned.Length);
                        Assert.Equal(moves, q.Length);
                        Assert.Equal(4, search.GetProperty("root").GetArrayLength());
                        for (int m = 0; m < moves; m++)
                        {
                            // A value exactly where the move has visits, a win probability
                            Assert.Equal(visits[m] == 0, q[m].ValueKind == JsonValueKind.Null);
                            if (visits[m] > 0) Assert.InRange(q[m].GetDouble(), 0.0, 1.0);
                            Assert.InRange(pruned[m], 0, visits[m]);
                        }
                        // Only the noisy search prunes
                        if (name != "shallow") Assert.Equal(visits, pruned);
                    }
                    Assert.True(Total("deep") > Total("clean"), "the deep search has more visits");
                    Assert.InRange(Total("clean"), 32 - 8, 32 + 8 * MCTSSearch.MaxBatchesForRoot);
                }
                Assert.True(outOfTurn > 0, "no position decided out of turn, so the deciding seat goes untested");

                // An existing output is never overwritten
                Assert.Throws<IOException>(() => TargetCheck.Run(check, TextWriter.Null));
                Assert.Throws<ArgumentException>(() => TargetCheckSettings.Parse([directory]));
                Assert.Throws<ArgumentException>(() => TargetCheckSettings.Parse([directory, output, "--deep", "0"]));
            }
            finally
            {
                if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
            }
        }

        /// <summary>
        /// A recorded position matches the replayed state: its features, round and moves
        /// </summary>
        private static void CheckPosition(Dictionary<string, string> sample, GameState state, byte[] states, IStateVectorizer vectorizer, bool prunedVisits)
        {
            Assert.Equal(state.Turn.RoundCounter, int.Parse(sample["round"]));
            Assert.InRange(int.Parse(sample["actor"]), 0, 3);
            Assert.Equal(1.0, Enumerable.Range(0, 4).Sum(p => double.Parse(sample[$"q{p}"], CultureInfo.InvariantCulture)), 1e-6);

            float[] expected = new float[vectorizer.FeatureDim];
            vectorizer.VectorizeInto(state, expected);
            int row = int.Parse(sample["row"]);
            for (int f = 0; f < expected.Length; f++)
            {
                Half stored = BinaryPrimitives.ReadHalfLittleEndian(states.AsSpan((row * expected.Length + f) * sizeof(ushort)));
                Assert.Equal((Half)expected[f], stored);
            }

            // Every visited move is a valid action of the position, and the visits add up to the root's (minus its own first visit at most)
            int moves = MCTSTree.GetValidActions(state).Count;
            var visits = sample["visits"].Split(' ').Select(pair => pair.Split(':')).Select(pair => (move: int.Parse(pair[0]), visits: uint.Parse(pair[1]))).ToArray();
            Assert.All(visits, v => Assert.InRange(v.move, 0, moves - 1));
            // (with a policy head the visits only the root noise caused are left out, so they can add up to less)
            Assert.InRange((long)visits.Sum(v => v.visits), prunedVisits ? 1 : long.Parse(sample["root_visits"]) - 1, long.Parse(sample["root_visits"]));
        }

        private static int[] ReadInts(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            return Enumerable.Range(0, bytes.Length / sizeof(int)).Select(i => BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * sizeof(int)))).ToArray();
        }

        private static float[] ReadFloats(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            return Enumerable.Range(0, bytes.Length / sizeof(float)).Select(i => BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * sizeof(float)))).ToArray();
        }

        private static List<Dictionary<string, string>> ReadCSV(string path)
        {
            string[] lines = File.ReadAllLines(path);
            string[] header = lines[0].Split(',');
            return lines.Skip(1).Select(line => header.Zip(line.Split(',')).ToDictionary(pair => pair.First, pair => pair.Second)).ToList();
        }
    }
}
