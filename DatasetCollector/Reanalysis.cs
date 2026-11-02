using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Microsoft.Win32.SafeHandles;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Action = Common.Actions.Action;

namespace DatasetCollector
{
    /// <summary>
    /// Settings of a reanalysis: the net and search that give the policy targets, and which match runs of the index get them
    /// </summary>
    public sealed record ReanalysisSettings
    {
        public string ModelPath { get; init; } = "";
        public int Iterations { get; init; } = 1600;

        /// <summary>
        /// The match runs of the index (in row order) that are searched: from FirstRun, at most Runs of them; the other
        /// entries get no policy target
        /// </summary>
        public int FirstRun { get; init; }
        public int Runs { get; init; } = int.MaxValue;

        public int ConcurrentSearches { get; init; } = 1024;
        public int Nets { get; init; } = 3;
        public int BatchRows { get; init; } = 1024;
        public int LeavesPerBatch { get; init; } = 16;
        public double Exploration { get; init; } = 1.414;
        public InferenceProvider Provider { get; init; } = InferenceProvider.Auto;

        public void Validate()
        {
            if (string.IsNullOrEmpty(ModelPath)) throw new ArgumentException("--reanalyze needs --model.");
            if (Iterations < 1) throw new ArgumentException("--iterations must be at least 1.");
            if (FirstRun < 0) throw new ArgumentException("--first-run must not be negative.");
            if (Runs < 1) throw new ArgumentException("--runs must be at least 1.");
            if (ConcurrentSearches < 1) throw new ArgumentException("--concurrent must be at least 1.");
            if (Nets < 1) throw new ArgumentException("--nets must be at least 1.");
            if (LeavesPerBatch < 1) throw new ArgumentException("--leaves must be at least 1.");
            if (BatchRows < LeavesPerBatch) throw new ArgumentException("--batch must be at least --leaves.");
            if (Exploration < 0) throw new ArgumentException("--search-exploration must not be negative.");
        }
    }

    public sealed record ReanalysisResult(long Rows, long Searched, long Moves, TimeSpan Elapsed);

    /// <summary>
    /// Policy targets for a prepared dataset collected without searches (e.g. v2, labeled by playouts): every match is played
    /// again from its seed and each entry checked against its stored row (as in ViewRegeneration), then every entry with at
    /// least two legal moves is searched with MCTS (max^n, one tree per entry, no root noise) and the given net, and the share
    /// of the root visits per move becomes its policy target. Writes into the output directory, one entry per row of the index,
    /// the arrays of selfplay_to_dataset.py, so train.py reads them with --policy-dir:
    /// - policy_offsets.npy (n + 1,) int64: the moves of row i are rows offsets[i]:offsets[i + 1] of the arrays below (none
    ///   for an entry that wasn't searched: ended, a single legal move, or outside the chosen match runs)
    /// - policy_tokens.npy (moves, max_tokens) int16 and policy_weights.npy (moves, max_tokens) uint8: all legal moves in
    ///   the order of MCTSTree.GetValidActions (the root's children), encoded by ActionEncoding
    /// - policy_targets.npy (moves,) float32: the share of the root visits per move (where a move wins at once, the tree
    ///   searches only that one: it gets 1 and the other legal moves 0)
    /// - policy_layout.json: the token ranges; root_q.npy (n, 4) float32: the search's mean result per player at the root
    ///   (NaN where not searched); metadata.json: net, search, commit and result
    /// The directory is written as OUT.partial and renamed when complete.
    /// </summary>
    public static class Reanalysis
    {
        private const int RunsPerBatch = 2000;

        private sealed record Entry(long Row, GameState? State);
        private sealed record Searched(Action[] Moves, uint[] Visits, double[] RootValues);

        public static ReanalysisResult Run(string indexPath, string statesPath, string outputDirectory, ReanalysisSettings reanalysis,
            CollectorSettings settings, TextWriter log, CancellationToken cancellation = default)
        {
            reanalysis.Validate();
            if (Directory.Exists(outputDirectory)) throw new InvalidOperationException($"{outputDirectory} already exists.");
            List<(int match, List<(long row, string name)> entries)> groups = ViewRegeneration.ReadIndex(indexPath, out long rows);
            if (reanalysis.FirstRun >= groups.Count) throw new InvalidOperationException($"--first-run {reanalysis.FirstRun} lies beyond the index's {groups.Count} match runs.");
            var selected = groups.Skip(reanalysis.FirstRun).Take(reanalysis.Runs).ToList();

            using SafeFileHandle states = File.OpenHandle(statesPath);
            long statesData = ViewRegeneration.ReadNpyHeader(states, rows, StateVectorizer.FeatureDim);

            string modelPath = Path.GetFullPath(reanalysis.ModelPath);
            StateValueNet[] nets = Enumerable.Range(0, reanalysis.Nets)
                .Select(_ => new StateValueNet(modelPath, provider: reanalysis.Provider, fixedBatchSize: reanalysis.BatchRows)).ToArray();
            if (nets[0].InputDim != StateVectorizer.FeatureDim)
                throw new InvalidOperationException($"The reanalysis searches full states and needs a net with {StateVectorizer.FeatureDim} inputs, not {nets[0].InputDim}.");

            // An unfinished earlier attempt is started over
            string partial = outputDirectory + ".partial";
            if (Directory.Exists(partial)) Directory.Delete(partial, recursive: true);
            Directory.CreateDirectory(partial);
            DateTime started = DateTime.Now;
            WriteMetadata(partial, modelPath, nets[0], reanalysis, settings, indexPath, statesPath, selected, started, null, null);
            File.WriteAllText(Path.Combine(partial, "policy_layout.json"), JsonSerializer.Serialize(ActionEncoding.Layout, new JsonSerializerOptions { WriteIndented = true }));
            log.WriteLine($"Reanalyzing {selected.Sum(g => g.entries.Count)} of {rows} entries ({selected.Count} match runs from run {reanalysis.FirstRun}) with "
                + $"{Path.GetFileName(modelPath)} on {nets[0].Provider}, {reanalysis.Iterations} iterations per search, {reanalysis.ConcurrentSearches} at once");

            var stopwatch = Stopwatch.StartNew();
            long searched = 0, moves = 0, iterations = 0;
            using (NpyWriter offsets = new(Path.Combine(partial, "policy_offsets.npy"), "<i8", sizeof(long)))
            using (NpyWriter tokens = new(Path.Combine(partial, "policy_tokens.npy"), "<i2", sizeof(short), ActionEncoding.MaxTokens))
            using (NpyWriter weights = new(Path.Combine(partial, "policy_weights.npy"), "|u1", sizeof(byte), ActionEncoding.MaxTokens))
            using (NpyWriter targets = new(Path.Combine(partial, "policy_targets.npy"), "<f4", sizeof(float)))
            using (NpyWriter rootValues = new(Path.Combine(partial, "root_q.npy"), "<f4", sizeof(float), MCTSNode.PlayerCount))
            using (BatchingEvaluator evaluator = new(nets, reanalysis.BatchRows))
            {
                long row = 0;
                byte[] offsetBytes = new byte[sizeof(long)], valueBytes = new byte[MCTSNode.PlayerCount * sizeof(float)];
                byte[] tokenBytes = new byte[ActionEncoding.MaxTokens * sizeof(short)], weightBytes = new byte[ActionEncoding.MaxTokens];
                short[] moveTokens = new short[ActionEncoding.MaxTokens];
                float[] moveWeights = new float[ActionEncoding.MaxTokens];

                void WriteOffset()
                {
                    BinaryPrimitives.WriteInt64LittleEndian(offsetBytes, moves);
                    offsets.WriteRow(offsetBytes);
                }
                void WriteRow(Searched? result)
                {
                    if (result != null)
                    {
                        double total = result.Visits.Sum(v => (double)v);
                        for (int m = 0; m < result.Moves.Length; m++)
                        {
                            int used = ActionEncoding.Encode(result.Moves[m], moveTokens, moveWeights);
                            for (int t = 0; t < ActionEncoding.MaxTokens; t++)
                            {
                                BinaryPrimitives.WriteInt16LittleEndian(tokenBytes.AsSpan(t * sizeof(short)), t < used ? moveTokens[t] : (short)-1);
                                weightBytes[t] = t < used ? checked((byte)moveWeights[t]) : (byte)0;
                            }
                            tokens.WriteRow(tokenBytes);
                            weights.WriteRow(weightBytes);
                            Span<byte> share = stackalloc byte[sizeof(float)];
                            BinaryPrimitives.WriteSingleLittleEndian(share, (float)(result.Visits[m] / total));
                            targets.WriteRow(share);
                        }
                        moves += result.Moves.Length;
                        searched++;
                    }
                    WriteOffset();
                    for (int p = 0; p < MCTSNode.PlayerCount; p++)
                        BinaryPrimitives.WriteSingleLittleEndian(valueBytes.AsSpan(p * sizeof(float)), result != null ? (float)result.RootValues[p] : float.NaN);
                    rootValues.WriteRow(valueBytes);
                    row++;
                }

                WriteOffset();
                // The replay of the next batch of match runs overlaps the searches of the current one
                Task<Entry[]>? next = selected.Count > 0 ? Task.Run(() => Replay(selected.Take(RunsPerBatch).ToList(), settings, states, statesData)) : null;
                for (int start = 0; next != null; start += RunsPerBatch)
                {
                    Entry[] entries = next.GetAwaiter().GetResult();
                    var following = selected.Skip(start + RunsPerBatch).Take(RunsPerBatch).ToList();
                    next = following.Count > 0 ? Task.Run(() => Replay(following, settings, states, statesData)) : null;

                    while (row < entries[0].Row) WriteRow(null);
                    Searched?[] results = Search(entries, evaluator, nets[0], reanalysis, count => Interlocked.Add(ref iterations, count),
                        () => Report(log, stopwatch, start + entries.Length, selected.Sum(g => g.entries.Count), searched, Interlocked.Read(ref iterations)), cancellation);
                    if (cancellation.IsCancellationRequested) throw new OperationCanceledException(cancellation);
                    for (int i = 0; i < entries.Length; i++)
                    {
                        if (entries[i].Row != row) throw new InvalidOperationException($"Entry of row {entries[i].Row} came where row {row} was due.");
                        WriteRow(results[i]);
                    }
                    log.Write($"\r{Math.Min(start + RunsPerBatch, selected.Count)}/{selected.Count} match runs, {searched} entries searched, "
                        + $"{searched / Math.Max(stopwatch.Elapsed.TotalSeconds, 1e-9):F0}/s, {Interlocked.Read(ref iterations) / Math.Max(stopwatch.Elapsed.TotalSeconds, 1e-9):N0} iterations/s   ");
                }
                while (row < rows) WriteRow(null);

                offsets.Finish();
                tokens.Finish();
                weights.Finish();
                targets.Finish();
                rootValues.Finish();
                if (offsets.Rows != rows + 1 || rootValues.Rows != rows || tokens.Rows != moves || targets.Rows != moves)
                    throw new InvalidOperationException("The written arrays don't add up to one entry per row.");
            }
            stopwatch.Stop();
            log.WriteLine();

            ReanalysisResult result = new(rows, searched, moves, stopwatch.Elapsed);
            WriteMetadata(partial, modelPath, nets[0], reanalysis, settings, indexPath, statesPath, selected, started, DateTime.Now, result);
            Directory.Move(partial, outputDirectory);
            return result;
        }

        /// <summary>
        /// The entries of some match runs in row order, each replayed from its match seed and checked against its stored row;
        /// the state only where it is to be searched (at least two legal moves)
        /// </summary>
        private static Entry[] Replay(List<(int match, List<(long row, string name)> entries)> runs, CollectorSettings settings, SafeFileHandle states, long statesData)
        {
            Entry[][] byRun = new Entry[runs.Count][];
            try
            {
                Parallel.For(0, runs.Count, new ParallelOptions { MaxDegreeOfParallelism = settings.Threads }, i =>
                {
                    (int match, List<(long row, string name)> entries) = runs[i];
                    DatasetCollector.MatchSamples samples = DatasetCollector.SampleMatch(match, settings);
                    float[] full = new float[StateVectorizer.FeatureDim];
                    byte[] stored = new byte[StateVectorizer.FeatureDim * sizeof(ushort)];
                    byRun[i] = entries.Select(entry =>
                    {
                        GameState state = EntryState(entry.name, samples);
                        ViewRegeneration.CheckStoredRow(state, entry.name, entry.row, states, statesData, full, stored);
                        bool searchable = !state.HasEnded && MCTSTree.GetValidActions(state).Count >= 2;
                        return new Entry(entry.row, searchable ? state : null);
                    }).ToArray();
                });
            }
            catch (AggregateException e) when (e.InnerExceptions.Count > 0)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
            }
            return byRun.SelectMany(entries => entries).ToArray();
        }

        /// <summary>
        /// An entry's state: r{match}_s{sample} or r{match}_s{sample}_c{sibling}
        /// </summary>
        private static GameState EntryState(string name, DatasetCollector.MatchSamples samples)
        {
            string[] parts = name.Split('_');
            int sampleIdx = int.Parse(parts[1].AsSpan(1), CultureInfo.InvariantCulture);
            if (parts.Length == 2) return samples.States[sampleIdx].state;
            int siblingIdx = int.Parse(parts[2].AsSpan(1), CultureInfo.InvariantCulture);
            return samples.Groups[sampleIdx][siblingIdx].State;
        }

        /// <summary>
        /// Searches the entries with a state, ConcurrentSearches at once on the shared evaluator; null for the others
        /// </summary>
        private static Searched?[] Search(Entry[] entries, BatchingEvaluator evaluator, StateValueNet net, ReanalysisSettings reanalysis,
            System.Action<long> countIterations, System.Action report, CancellationToken cancellation)
        {
            Searched?[] results = new Searched?[entries.Length];
            int next = -1;
            int workers = Math.Max(1, Math.Min(reanalysis.ConcurrentSearches, entries.Count(entry => entry.State != null)));
            Task[] searches = Enumerable.Range(0, workers).Select(_ => Task.Run(async () =>
            {
                MCTSSearch search = new(evaluator, Vectorizers.For(net, 0), reanalysis.LeavesPerBatch, parallelSelection: false);
                int i;
                while (!cancellation.IsCancellationRequested && (i = Interlocked.Increment(ref next)) < entries.Length)
                {
                    if (entries[i].State is not GameState state) continue;
                    MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, reanalysis.Exploration, model: OpponentModel.MaxN);
                    await search.RunAsync(tree, double.PositiveInfinity, reanalysis.Iterations).ConfigureAwait(false);
                    countIterations(search.LastSearch!.Iterations);
                    Action[] legal = [.. MCTSTree.GetValidActions(state)];
                    results[i] = new Searched(legal, VisitsOfLegalMoves(legal, tree.RootNode.Children), tree.RootValues());
                }
            })).ToArray();
            while (!Task.WaitAll(searches, TimeSpan.FromSeconds(10)))
                report();
            return results;
        }

        /// <summary>
        /// The root visits per legal move: the root's children are the legal moves in the same order, except where a move
        /// wins at once, which the tree searches alone (MCTSTree.ExpandNode); its visits then go to that move and the others
        /// keep 0, so the target still teaches the policy to pick it among all legal moves
        /// </summary>
        internal static uint[] VisitsOfLegalMoves(Action[] legal, IReadOnlyList<MCTSNode> children)
        {
            uint[] visits = new uint[legal.Length];
            if (children.Count == legal.Length)
            {
                for (int m = 0; m < legal.Length; m++) visits[m] = children[m].VisitCount;
                return visits;
            }
            foreach (MCTSNode child in children)
            {
                int m = Array.FindIndex(legal, move => move.IsSameDecision(child.LastAction!));
                if (m < 0) throw new InvalidOperationException($"The root move {child.LastAction} is not among the legal moves.");
                visits[m] += child.VisitCount;
            }
            return visits;
        }

        private static void Report(TextWriter log, Stopwatch stopwatch, long done, long total, long searched, long iterations)
        {
            double seconds = Math.Max(stopwatch.Elapsed.TotalSeconds, 1e-9);
            log.Write($"\r{searched} entries searched (batch up to entry {done}/{total}), {searched / seconds:F0}/s, {iterations / seconds:N0} iterations/s   ");
        }

        private static void WriteMetadata(string directory, string modelPath, StateValueNet net, ReanalysisSettings reanalysis, CollectorSettings settings,
            string indexPath, string statesPath, List<(int match, List<(long row, string name)> entries)> selected, DateTime started, DateTime? finished, ReanalysisResult? result)
        {
            var metadata = new
            {
                format = 1,
                kind = "reanalysis",
                commit = CommitHash(),
                started = started.ToString("o", CultureInfo.InvariantCulture),
                finished = finished?.ToString("o", CultureInfo.InvariantCulture),
                status = result == null ? "running" : "finished",
                source = new { index = Path.GetFullPath(indexPath), states = Path.GetFullPath(statesPath), collection_seed = settings.Seed.ToString(CultureInfo.InvariantCulture) },
                model = new
                {
                    file = Path.GetFileName(modelPath),
                    sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(modelPath))),
                    feature_dim = net.InputDim,
                    policy = net.Policy != null,
                    provider = net.Provider.ToString(),
                },
                search = new
                {
                    iterations = reanalysis.Iterations,
                    exploration = reanalysis.Exploration,
                    opponent_model = "MaxN",
                    root_noise = 0,
                    leaves_per_batch = reanalysis.LeavesPerBatch,
                    concurrent_searches = reanalysis.ConcurrentSearches,
                    nets = reanalysis.Nets,
                    batch_rows = reanalysis.BatchRows,
                },
                match_runs = new { first = reanalysis.FirstRun, count = selected.Count, entries = selected.Sum(g => (long)g.entries.Count) },
                result = result == null ? null : new { rows = result.Rows, searched = result.Searched, moves = result.Moves, seconds = Math.Round(result.Elapsed.TotalSeconds, 1) },
            };
            File.WriteAllText(Path.Combine(directory, "metadata.json"), JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
        }

        private static string? CommitHash()
        {
            string? version = typeof(Reanalysis).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
            int plus = version?.IndexOf('+') ?? -1;
            return plus >= 0 ? version![(plus + 1)..] : null;
        }
    }

    /// <summary>
    /// Writes a C-ordered .npy array row by row whose row count is known only at the end: the header is reserved with a
    /// fixed size and filled in by Finish
    /// </summary>
    internal sealed class NpyWriter : IDisposable
    {
        private const int HeaderBytes = 128;
        private readonly FileStream _file;
        private readonly string _descr;
        private readonly int[] _rowShape;
        private readonly int _rowBytes;

        /// <param name="descr">The numpy type, e.g. "&lt;f4"</param>
        /// <param name="rowShape">The shape of a row; none for a 1-D array</param>
        public NpyWriter(string path, string descr, int itemBytes, params int[] rowShape)
        {
            _file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 1 << 20);
            _descr = descr;
            _rowShape = rowShape;
            _rowBytes = itemBytes * rowShape.Aggregate(1, (a, b) => a * b);
            _file.Write(new byte[HeaderBytes]);
        }

        public long Rows { get; private set; }

        public void WriteRow(ReadOnlySpan<byte> row)
        {
            if (row.Length != _rowBytes) throw new ArgumentException($"A row has {_rowBytes} bytes, not {row.Length}.");
            _file.Write(row);
            Rows++;
        }

        /// <summary>
        /// Writes the header with the final row count; the array is complete after this
        /// </summary>
        public void Finish()
        {
            _file.Flush();
            _file.Seek(0, SeekOrigin.Begin);
            _file.Write(Header(_descr, Rows, _rowShape));
            _file.Flush();
        }

        internal static byte[] Header(string descr, long rows, int[] rowShape)
        {
            string shape = rowShape.Length == 0 ? $"({rows},)" : $"({rows}, {string.Join(", ", rowShape)})";
            string dictionary = $"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}";
            int padding = HeaderBytes - 10 - dictionary.Length - 1;
            if (padding < 0) throw new InvalidOperationException("The .npy header doesn't fit its reserved size.");
            byte[] header = new byte[HeaderBytes];
            header[0] = 0x93;
            Encoding.ASCII.GetBytes("NUMPY").CopyTo(header, 1);
            header[6] = 1;
            BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(8), HeaderBytes - 10);
            Encoding.ASCII.GetBytes(dictionary + new string(' ', padding) + "\n").CopyTo(header, 10);
            return header;
        }

        public void Dispose() => _file.Dispose();
    }
}
