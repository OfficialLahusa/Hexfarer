using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace SelfPlayCollector
{
    /// <summary>
    /// Settings of a target check, from the command line: SelfPlayCollector target-check RUN_DIRECTORY OUTPUT [options]
    /// </summary>
    public sealed record TargetCheckSettings
    {
        public required string RunDirectory { get; init; }
        public required string OutputPath { get; init; }

        /// <summary>
        /// Recorded positions to check, drawn at random from the run's samples
        /// </summary>
        public int Positions { get; init; } = 2000;

        /// <summary>
        /// Iterations of the shallow searches (self-play's full search) and of the deep one
        /// </summary>
        public int ShallowIterations { get; init; } = 800;
        public int DeepIterations { get; init; } = 20000;

        /// <summary>
        /// Leaves per batch of the shallow searches (self-play's); the deep search takes the Arena's batch for its budget
        /// </summary>
        public int LeavesPerBatch { get; init; } = 16;

        /// <summary>
        /// Positions searched at once; their searches share the value net batches
        /// </summary>
        public int Concurrent { get; init; } = 256;

        public double Exploration { get; init; } = 1.414;
        public double RootNoise { get; init; } = 0.25;
        public double RootNoiseConcentration { get; init; } = 10;
        public string? ModelPath { get; init; }
        public InferenceProvider Provider { get; init; } = InferenceProvider.Auto;
        public int Nets { get; init; } = 2;
        public int BatchRows { get; init; } = 1024;

        /// <summary>
        /// Seed of the drawn positions and of the root noise
        /// </summary>
        public ulong Seed { get; init; } = 1;

        public const string Usage = """
            Usage: SelfPlayCollector target-check RUN_DIRECTORY OUTPUT [options]

            Searches recorded positions of a self-play run three times on one net: shallow with root noise as in
            self-play, shallow without noise, and deep. Writes per position (one JSON object per line) the net's priors
            and every search's visits, pruned visits, value of each legal move for the deciding player, root values and
            value range, for DatasetProcessing/target_check.py to score policy targets built from the shallow searches
            against the deep one.

              --positions N            Positions drawn from the run (default 2000)
              --shallow N              Iterations of the shallow searches (default 800)
              --deep N                 Iterations of the deep search (default 20000)
              --leaves L               Leaves per batch of the shallow searches (default 16)
              --concurrent N           Positions searched at once (default 256)
              --exploration C          PUCT exploration constant (default 1.414)
              --root-noise W           Weight of the shallow noisy search's root noise (default 0.25)
              --noise-concentration A  Total concentration of the root noise (default 10)
              --model PATH             Net with a policy head (default: the deployed model)
              --nets N                 Net sessions evaluating batches in turn (default 2)
              --batch N                Most rows per net batch (default 1024)
              --provider cpu|dml|cuda  Where the net runs (default: the GPU if available)
              --seed S                 Seed of the drawn positions and the noise (default 1)
            """;

        public static TargetCheckSettings Parse(string[] args)
        {
            if (args.Length < 2 || args[0].StartsWith("--") || args[1].StartsWith("--"))
                throw new ArgumentException("target-check needs a run directory and an output file.");
            TargetCheckSettings settings = new() { RunDirectory = args[0], OutputPath = args[1] };
            for (int i = 2; i < args.Length; i++)
            {
                string Value() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
                int IntValue() => int.Parse(Value(), CultureInfo.InvariantCulture);
                double DoubleValue() => double.Parse(Value(), CultureInfo.InvariantCulture);
                settings = args[i] switch
                {
                    "--positions" => settings with { Positions = IntValue() },
                    "--shallow" => settings with { ShallowIterations = IntValue() },
                    "--deep" => settings with { DeepIterations = IntValue() },
                    "--leaves" => settings with { LeavesPerBatch = IntValue() },
                    "--concurrent" => settings with { Concurrent = IntValue() },
                    "--exploration" => settings with { Exploration = DoubleValue() },
                    "--root-noise" => settings with { RootNoise = DoubleValue() },
                    "--noise-concentration" => settings with { RootNoiseConcentration = DoubleValue() },
                    "--model" => settings with { ModelPath = Value() },
                    "--nets" => settings with { Nets = IntValue() },
                    "--batch" => settings with { BatchRows = IntValue() },
                    "--provider" => settings with
                    {
                        Provider = Value() switch
                        {
                            "cpu" => InferenceProvider.CPU,
                            "dml" => InferenceProvider.DirectML,
                            "cuda" => InferenceProvider.CUDA,
                            string other => throw new ArgumentException($"Unknown provider \"{other}\" (cpu, dml or cuda)."),
                        }
                    },
                    "--seed" => settings with { Seed = ulong.Parse(Value(), CultureInfo.InvariantCulture) },
                    _ => throw new ArgumentException($"Unknown option \"{args[i]}\"."),
                };
            }
            settings.Validate();
            return settings;
        }

        public void Validate()
        {
            if (Positions < 1) throw new ArgumentException("--positions must be at least 1.");
            if (ShallowIterations < 1 || DeepIterations < 1) throw new ArgumentException("--shallow and --deep must be at least 1.");
            if (LeavesPerBatch < 1) throw new ArgumentException("--leaves must be at least 1.");
            if (Concurrent < 1) throw new ArgumentException("--concurrent must be at least 1.");
            if (Exploration < 0) throw new ArgumentException("--exploration must not be negative.");
            if (RootNoise is < 0 or > 1) throw new ArgumentException("--root-noise must be between 0 and 1.");
            if (RootNoiseConcentration <= 0) throw new ArgumentException("--noise-concentration must be positive.");
            if (Nets < 1) throw new ArgumentException("--nets must be at least 1.");
            if (BatchRows < Math.Max(LeavesPerBatch, MCTSAgent.LeavesFor(DeepIterations))) throw new ArgumentException("--batch must hold a search's batch.");
        }
    }

    /// <summary>
    /// The target check (see TargetCheckSettings.Usage): draws recorded positions of a self-play run, rebuilds each from its
    /// game's seed and save file, searches it shallow with noise, shallow without and deep, and writes the results sorted
    /// by game and action, next to a .meta.json with the settings, the net and the commit
    /// </summary>
    public static class TargetCheck
    {
        // A drawn position: its game and action index, the round and the deciding seat the run recorded, and the state
        private sealed record Position(long Game, int Action, int Round, int Actor, GameState State);

        /// <summary>
        /// Runs the check and returns the number of positions written; an existing output is never overwritten
        /// </summary>
        public static int Run(TargetCheckSettings settings, TextWriter log)
        {
            settings.Validate();
            if (File.Exists(settings.OutputPath))
                throw new IOException($"The output \"{Path.GetFullPath(settings.OutputPath)}\" already exists.");

            List<Position> positions = DrawPositions(settings);
            log.WriteLine($"{positions.Count} positions drawn from {settings.RunDirectory}");

            string modelPath = settings.ModelPath != null ? Path.GetFullPath(settings.ModelPath) : StateValueNet.DefaultModelPath;
            StateValueNet[] nets = Enumerable.Range(0, settings.Nets)
                .Select(_ => new StateValueNet(modelPath, provider: settings.Provider, fixedBatchSize: settings.BatchRows)).ToArray();
            if (nets[0].Policy == null) throw new InvalidOperationException("The target check needs a net with a policy head.");

            string[] lines = new string[positions.Count];
            int next = -1, done = 0;
            long iterations = 0;
            Exception? failure = null;
            var stopwatch = Stopwatch.StartNew();
            using (BatchingEvaluator evaluator = new(nets, settings.BatchRows))
            {
                async Task Worker()
                {
                    IStateVectorizer vectorizer = Vectorizers.For(nets[0], 0);
                    MCTSSearch shallow = new(evaluator, vectorizer, settings.LeavesPerBatch, parallelSelection: false);
                    MCTSSearch deep = new(evaluator, vectorizer, MCTSAgent.LeavesFor(settings.DeepIterations), parallelSelection: false);
                    int index;
                    while (Volatile.Read(ref failure) == null && (index = Interlocked.Increment(ref next)) < positions.Count)
                    {
                        try
                        {
                            (lines[index], long searched) = await CheckAsync(positions[index], index, settings, shallow, deep).ConfigureAwait(false);
                            Interlocked.Add(ref iterations, searched);
                            Interlocked.Increment(ref done);
                        }
                        catch (Exception e)
                        {
                            Interlocked.CompareExchange(ref failure, e, null);
                        }
                    }
                }

                Task[] workers = Enumerable.Range(0, Math.Min(settings.Concurrent, positions.Count)).Select(_ => Task.Run(Worker)).ToArray();
                while (!Task.WaitAll(workers, TimeSpan.FromSeconds(10)))
                    log.Write($"\r{Volatile.Read(ref done)}/{positions.Count} positions, {Interlocked.Read(ref iterations) / stopwatch.Elapsed.TotalSeconds:N0} iterations/s   ");
            }
            if (failure != null) throw new InvalidOperationException("A position's check failed.", failure);

            // Sorted by game and action, so the same settings give the same file
            var order = Enumerable.Range(0, positions.Count).OrderBy(i => positions[i].Game).ThenBy(i => positions[i].Action);
            using (StreamWriter writer = new(new FileStream(settings.OutputPath, FileMode.CreateNew, FileAccess.Write)))
                foreach (int i in order) writer.WriteLine(lines[i]);
            WriteMetadata(settings, modelPath, nets[0], positions.Count, stopwatch.Elapsed);
            log.WriteLine($"\r{positions.Count} positions checked in {stopwatch.Elapsed:hh\\:mm\\:ss}, {Interlocked.Read(ref iterations) / stopwatch.Elapsed.TotalSeconds:N0} iterations/s, written to {settings.OutputPath}");
            return positions.Count;
        }

        /// <summary>
        /// Draws the positions from samples.csv (at random, seeded) and rebuilds each by replaying its game's save file
        /// </summary>
        private static List<Position> DrawPositions(TargetCheckSettings settings)
        {
            string directory = settings.RunDirectory;
            using JsonDocument metadata = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "metadata.json")));
            bool centerDesert = metadata.RootElement.GetProperty("settings").GetProperty("center_desert").GetBoolean();
            if (metadata.RootElement.GetProperty("settings").GetProperty("hidden_information").GetBoolean())
                throw new InvalidOperationException("The target check searches the full state; this run was played under hidden information.");

            var records = new Dictionary<long, (ulong seed, int chunk, long offset, int length)>();
            foreach (string line in File.ReadLines(Path.Combine(directory, "games.csv")).Skip(1))
            {
                string[] f = line.Split(',');
                records[long.Parse(f[0], CultureInfo.InvariantCulture)] = (ulong.Parse(f[1], CultureInfo.InvariantCulture),
                    int.Parse(f[13], CultureInfo.InvariantCulture), long.Parse(f[14], CultureInfo.InvariantCulture), int.Parse(f[15], CultureInfo.InvariantCulture));
            }

            // game, action, round, actor of every recorded position (the first five columns but position)
            var samples = new List<(long game, int action, int round, int actor)>();
            foreach (string line in File.ReadLines(Path.Combine(directory, "samples.csv")).Skip(1))
            {
                string[] f = line.Split(',', 6);
                samples.Add((long.Parse(f[0], CultureInfo.InvariantCulture), int.Parse(f[2], CultureInfo.InvariantCulture),
                    int.Parse(f[3], CultureInfo.InvariantCulture), int.Parse(f[4], CultureInfo.InvariantCulture)));
            }
            Random random = new(Seeds.ToInt32(settings.Seed));
            var drawn = samples.OrderBy(_ => random.Next()).Take(settings.Positions).ToList();

            var byGame = drawn.GroupBy(sample => sample.game).ToArray();
            var rebuilt = new List<Position>[byGame.Length];
            try
            {
                Parallel.For(0, byGame.Length, g => rebuilt[g] = Rebuild(byGame[g].Key, byGame[g], records, directory, centerDesert));
            }
            catch (AggregateException e)
            {
                // The first failed game's own exception, so the caller sees its message
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
            }
            return rebuilt.SelectMany(found => found).ToList();
        }

        /// <summary>
        /// The drawn positions of one game, rebuilt by replaying its save file from the seed's initial state
        /// </summary>
        private static List<Position> Rebuild(long game, IEnumerable<(long game, int action, int round, int actor)> samples,
            Dictionary<long, (ulong seed, int chunk, long offset, int length)> records, string directory, bool centerDesert)
        {
            if (!records.TryGetValue(game, out var record))
                throw new InvalidOperationException($"Game {game} has positions but no line in games.csv.");
            SaveFile saved = SelfPlayWriter.ReadRecord(directory, record.chunk, record.offset, record.length);
            GameState state = SelfPlayGame.InitialState(record.seed, centerDesert);
            var found = new List<Position>();
            int applied = 0;
            foreach (var sample in samples.OrderBy(sample => sample.action))
            {
                if (sample.action > saved.PlayedActions.Count)
                    throw new InvalidOperationException($"Game {game}: the position at action {sample.action} lies beyond the game's {saved.PlayedActions.Count} actions.");
                while (applied < sample.action) saved.PlayedActions[applied++].Apply(state);
                // The replay must arrive where the run recorded the position
                if (state.Turn.RoundCounter != sample.round)
                    throw new InvalidOperationException($"Game {game}: action {sample.action} replays to round {state.Turn.RoundCounter}, the run recorded round {sample.round}.");
                found.Add(new Position(game, sample.action, sample.round, sample.actor, new GameState(state)));
            }
            return found;
        }

        /// <summary>
        /// The three searches of one position, as one line of JSON, and the iterations they took
        /// </summary>
        private static async Task<(string line, long iterations)> CheckAsync(Position position, int index, TargetCheckSettings settings, MCTSSearch shallow, MCTSSearch deep)
        {
            var buffer = new MemoryStream();
            using (Utf8JsonWriter json = new(buffer))
            {
                json.WriteStartObject();
                json.WriteNumber("game", position.Game);
                json.WriteNumber("action", position.Action);
                json.WriteNumber("round", position.Round);
                json.WriteNumber("decider", position.Actor);

                long iterations = 0;
                float[]? prior = null;
                json.WriteStartObject("searches");
                foreach ((string name, MCTSSearch search, double noise, int budget) in new[] {
                    ("shallow", shallow, settings.RootNoise, settings.ShallowIterations),
                    ("clean", shallow, 0.0, settings.ShallowIterations),
                    ("deep", deep, 0.0, settings.DeepIterations) })
                {
                    MCTSTree tree = new(new GameState(position.State), (sbyte)position.State.Turn.PlayerIndex, settings.Exploration, model: OpponentModel.MaxN);
                    search.RootNoiseWeight = noise;
                    search.RootNoiseConcentration = settings.RootNoiseConcentration;
                    // Noise of its own per position, whichever worker searches it
                    search.NoiseRandom = new Random(Seeds.ToInt32(Seeds.Derive(settings.Seed, (ulong)index + 1)));
                    await search.RunAsync(tree, double.PositiveInfinity, budget).ConfigureAwait(false);
                    iterations += search.LastSearch!.Iterations;

                    if (tree.RootNode.DeciderIndex != position.Actor)
                        throw new InvalidOperationException($"Game {position.Game}, action {position.Action}: seat {tree.RootNode.DeciderIndex} decides, the run recorded seat {position.Actor}.");
                    // The clean priors (before the noise) of the first search: every search evaluates the same root
                    float[] rootPolicy = search.LastRootPolicy ?? throw new InvalidOperationException("The search left no root priors.");
                    prior ??= rootPolicy;

                    (_, uint[] visits) = tree.LegalMoveVisits();
                    // Self-play's recorded visits: without those only the root noise caused (the same as the visits without noise)
                    uint[] pruned = noise > 0 ? tree.LegalMoveVisits(tree.PrunedRootVisits(rootPolicy)).visits : visits;
                    (_, double[] values) = tree.LegalMoveValues(position.Actor);
                    // The tree's range of the decider's values (PUCT normalizes with it), none before two distinct values
                    MinMaxStats.Range? range = tree.PlayerBounds?[position.Actor].GetRange();
                    if (range is MinMaxStats.Range seen && !(seen.Max > seen.Min)) range = null;

                    json.WriteStartObject(name);
                    json.WriteNumber("iterations", search.LastSearch.Iterations);
                    WriteArray(json, "visits", visits.Select(v => (double)v));
                    WriteArray(json, "pruned", pruned.Select(v => (double)v));
                    WriteArray(json, "q", values);
                    WriteArray(json, "root", tree.RootValues());
                    if (range is MinMaxStats.Range bounds) WriteArray(json, "range", [bounds.Min, bounds.Max]);
                    else json.WriteNull("range");
                    json.WriteEndObject();
                }
                json.WriteEndObject();

                json.WriteNumber("moves", prior!.Length);
                WriteArray(json, "prior", prior.Select(p => (double)p));
                json.WriteEndObject();
                json.Flush();
                return (Encoding.UTF8.GetString(buffer.ToArray()), iterations);
            }
        }

        // An array of numbers, NaN written as null
        private static void WriteArray(Utf8JsonWriter json, string name, IEnumerable<double> values)
        {
            json.WriteStartArray(name);
            foreach (double value in values)
            {
                if (double.IsNaN(value)) json.WriteNullValue();
                else json.WriteNumberValue(value);
            }
            json.WriteEndArray();
        }

        private static void WriteMetadata(TargetCheckSettings settings, string modelPath, StateValueNet net, int positions, TimeSpan elapsed)
        {
            string? version = typeof(TargetCheck).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
            int plus = version?.IndexOf('+') ?? -1;
            var metadata = new
            {
                commit = plus >= 0 ? version![(plus + 1)..] : null,
                run = Path.GetFullPath(settings.RunDirectory),
                model = new { file = Path.GetFileName(modelPath), sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(modelPath))), provider = net.Provider.ToString() },
                settings = new
                {
                    positions = settings.Positions, shallow_iterations = settings.ShallowIterations, deep_iterations = settings.DeepIterations,
                    leaves_per_batch = settings.LeavesPerBatch, deep_leaves_per_batch = MCTSAgent.LeavesFor(settings.DeepIterations),
                    exploration = settings.Exploration, root_noise = settings.RootNoise, noise_concentration = settings.RootNoiseConcentration,
                    seed = settings.Seed.ToString(CultureInfo.InvariantCulture),
                },
                result = new { positions, seconds = Math.Round(elapsed.TotalSeconds, 1) },
            };
            File.WriteAllText(settings.OutputPath + ".meta.json", JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
        }
    }
}
