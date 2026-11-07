using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Agents.Inference
{
    /// <summary>
    /// Where the value net runs. Auto takes the GPU if it is available, through DirectML on Windows and CUDA on Linux, and
    /// the CPU otherwise; the environment variable HEXFARER_INFERENCE (cpu, dml or cuda; the older CATANSIM_INFERENCE works
    /// too) overrides it.
    /// </summary>
    public enum InferenceProvider { Auto, CPU, DirectML, CUDA }

    public class StateValueNet
    {
        // File name of the deployed model, set by StateValueNetModel in Agents.csproj, which also copies it next to the executable
        public static readonly string DefaultModelFile = typeof(StateValueNet).Assembly
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .Single(attribute => attribute.Key == "StateValueNetModel").Value!;

        public static string DefaultModelPath => ResolveModelPath(DefaultModelFile);

        // The model file is not tracked in git, so a fresh clone builds without it
        public static bool IsDefaultModelAvailable => File.Exists(DefaultModelPath);

        // Threads ONNX Runtime uses inside one Run call. The MCTS agent evaluates one batch while it selects the next,
        // so half of the logical cores run the value net and half the search (more threads starve the search, at all
        // cores both slow down by half; measured with batches of 256 on 32 logical cores).
        public static readonly int DefaultIntraOpThreads = Math.Max(1, Environment.ProcessorCount / 2);

        // Batch size the DirectML session is compiled for, that of the MCTS agent's leaf batches. DirectML runs a fixed
        // shape about 4x faster than a dynamic one (measured on an RTX 5080: 566k vs 136k states/s at 256, the CPU 89k).
        public const int DefaultFixedBatchSize = 256;

        private readonly InferenceSession _session;

        // DirectML doesn't allow parallel Run calls on one session
        private readonly object? _runLock;

        public const int PlayerCount = 4;

        /// <summary>
        /// The provider the net runs on (never Auto)
        /// </summary>
        public InferenceProvider Provider { get; }

        /// <summary>
        /// Batch size the session runs with (DirectML), null if any size runs as it is (CPU, CUDA). Other batch sizes still
        /// work: smaller batches are padded, larger ones split.
        /// </summary>
        public int? FixedBatchSize { get; }

        /// <summary>
        /// Whether the model also predicts each player's placement distribution (output "placements", models trained with
        /// auxiliary heads)
        /// </summary>
        public bool HasPlacements { get; }

        /// <summary>
        /// Number of features the model takes, the layout it was trained on (see StateVectorizer.For)
        /// </summary>
        public int InputDim { get; }

        /// <summary>
        /// The policy head of a model trained with one (outputs "policy_query" and "policy_embeddings", see ActionEncoding),
        /// or null
        /// </summary>
        public PolicyHead? Policy { get; }

        public StateValueNet(string modelPath, int intraOpThreads = 0, InferenceProvider provider = InferenceProvider.Auto, int fixedBatchSize = DefaultFixedBatchSize)
        {
            string resolvedPath = ResolveModelPath(modelPath);
            if (!File.Exists(resolvedPath))
                throw new FileNotFoundException($"Value net not found at \"{resolvedPath}\". Place it in Agents/Artifacts (see StateValueNetModel in Agents.csproj) and rebuild.", resolvedPath);

            provider = ResolveProvider(provider);
            if (provider == InferenceProvider.DirectML)
            {
                try
                {
                    _session = CreateDirectMLSession(resolvedPath, fixedBatchSize);
                    FixedBatchSize = fixedBatchSize;
                    _runLock = new object();
                }
                catch (Exception e) when (e is OnnxRuntimeException or EntryPointNotFoundException or DllNotFoundException)
                {
                    // No DirectX 12 GPU, or the CPU-only runtime (e.g. on Linux): fall back to the CPU
                    if (InferenceSetting == "dml") throw;
                    provider = InferenceProvider.CPU;
                }
            }
            else if (provider == InferenceProvider.CUDA)
            {
                try
                {
                    _session = CreateCUDASession(resolvedPath);
                }
                catch (Exception e) when (e is OnnxRuntimeException or EntryPointNotFoundException or DllNotFoundException)
                {
                    // No NVIDIA GPU, or the CUDA or cuDNN libraries are missing: fall back to the CPU
                    if (InferenceSetting == "cuda") throw;
                    provider = InferenceProvider.CPU;
                }
            }

            if (provider == InferenceProvider.CPU)
            {
                var options = new SessionOptions() {
                    IntraOpNumThreads = intraOpThreads > 0 ? intraOpThreads : DefaultIntraOpThreads
                };
                _session = new InferenceSession(resolvedPath, options);
            }

            Provider = provider;
            HasPlacements = _session!.OutputMetadata.ContainsKey("placements");
            InputDim = _session.InputMetadata["state"].Dimensions[1];

            if (_session.OutputMetadata.ContainsKey("policy_query") && _session.OutputMetadata.TryGetValue("policy_embeddings", out NodeMetadata? embeddings))
            {
                // The embedding table is the same for every batch: read it once from an empty batch
                int dim = embeddings.Dimensions[1];
                float[] table = RunSession(new DenseTensor<float>([FixedBatchSize ?? 1, InputDim]), ["policy_embeddings"])[0];
                if (table.Length != ActionEncoding.VocabularySize * dim)
                    throw new InvalidDataException($"The policy embeddings have {table.Length / dim} tokens, ActionEncoding {ActionEncoding.VocabularySize}.");
                Policy = new PolicyHead(table, dim);
            }
        }

        /// <summary>
        /// The provider forced by the environment, under the project's current or former name
        /// </summary>
        private static string? InferenceSetting =>
            Environment.GetEnvironmentVariable("HEXFARER_INFERENCE") ?? Environment.GetEnvironmentVariable("CATANSIM_INFERENCE");

        private static InferenceProvider ResolveProvider(InferenceProvider provider)
        {
            if (provider != InferenceProvider.Auto) return provider;
            return InferenceSetting switch
            {
                "cpu" => InferenceProvider.CPU,
                "dml" => InferenceProvider.DirectML,
                "cuda" => InferenceProvider.CUDA,
                _ => OperatingSystem.IsWindows() ? InferenceProvider.DirectML : OperatingSystem.IsLinux() ? InferenceProvider.CUDA : InferenceProvider.CPU,
            };
        }

        private static InferenceSession CreateDirectMLSession(string modelPath, int fixedBatchSize)
        {
            // DirectML needs sequential execution without memory patterns; a fixed batch dimension lets it compile the graph once
            var options = new SessionOptions()
            {
                EnableMemoryPattern = false,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            };
            options.AddFreeDimensionOverrideByName("batch", fixedBatchSize);
            options.AppendExecutionProvider_DML(0);
            return new InferenceSession(modelPath, options);
        }

        private static InferenceSession CreateCUDASession(string modelPath)
        {
            // CUDA runs any batch size as it is and allows parallel Run calls on one session; it needs the CUDA and cuDNN
            // libraries the ONNX Runtime GPU package was built against
            var options = new SessionOptions();
            options.AppendExecutionProvider_CUDA(0);
            return new InferenceSession(modelPath, options);
        }

        /// <summary>
        /// Relative model paths are resolved next to the executable instead of against the working directory
        /// </summary>
        private static string ResolveModelPath(string modelPath)
        {
            return Path.IsPathRooted(modelPath) ? modelPath : Path.Combine(AppContext.BaseDirectory, modelPath);
        }

        public float[] Run(DenseTensor<float> inputTensor)
        {
            float[] logits = RunOnlyLogits(inputTensor);
            return SoftmaxPerRow(logits, logits.Length / PlayerCount, PlayerCount);
        }

        public float[] RunOnlyLogits(DenseTensor<float> inputTensor)
        {
            // Only the value output is computed, models with a second output skip it here.
            // Row-major flat: [batch * NumPlayers], works for batch=1 or batched leaf-eval
            return RunOutputs(inputTensor, ["logits"])[0];
        }

        /// <summary>
        /// The named outputs of the batch (row-major flat), run in pieces of the fixed batch size if there is one
        /// </summary>
        private float[][] RunOutputs(DenseTensor<float> inputTensor, string[] outputNames)
        {
            int batchSize = inputTensor.Dimensions[0];
            if (FixedBatchSize is not int fixedSize || batchSize == fixedSize)
                return RunSession(inputTensor, outputNames);

            // Smaller batches are padded with empty rows, larger ones run in several pieces; only the real rows are kept
            int featureDim = inputTensor.Dimensions[1];
            var pieces = outputNames.Select(_ => new List<float>()).ToArray();
            for (int start = 0; start < batchSize; start += fixedSize)
            {
                int rows = Math.Min(fixedSize, batchSize - start);
                var piece = new DenseTensor<float>([fixedSize, featureDim]);
                inputTensor.Buffer.Span.Slice(start * featureDim, rows * featureDim).CopyTo(piece.Buffer.Span);

                float[][] outputs = RunSession(piece, outputNames);
                for (int o = 0; o < outputs.Length; o++)
                    pieces[o].AddRange(outputs[o].AsSpan(0, rows * (outputs[o].Length / fixedSize)));
            }
            return pieces.Select(piece => piece.ToArray()).ToArray();
        }

        private float[][] RunSession(DenseTensor<float> inputTensor, string[] outputNames)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("state", inputTensor)
            };

            if (_runLock == null) return Run(inputs, outputNames);
            lock (_runLock) return Run(inputs, outputNames);

            float[][] Run(List<NamedOnnxValue> inputs, string[] outputNames)
            {
                using var results = _session.Run(inputs, outputNames);
                return outputNames.Select(name => ToFloats(results.First(result => result.Name == name))).ToArray();
            }

            // The policy queries come as float16 (half the copy from the GPU), the other outputs as float
            static float[] ToFloats(DisposableNamedOnnxValue result)
            {
                if (result.ElementType != TensorElementType.Float16)
                    return result.AsTensor<float>().ToArray();
                // Converted in one vectorized pass over the tensor's memory (Float16 and Half share the IEEE layout)
                ReadOnlySpan<Float16> halves = ((DenseTensor<Float16>)result.AsTensor<Float16>()).Buffer.Span;
                float[] values = new float[halves.Length];
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(System.Runtime.InteropServices.MemoryMarshal.Cast<Float16, Half>(halves), values);
                return values;
            }
        }

        /// <summary>
        /// Win probabilities [batch * players] and placement distributions [batch * players * places], each row of 4 places
        /// summing to 1, of a model with HasPlacements
        /// </summary>
        public (float[] winProbabilities, float[] placements) RunWithPlacements(DenseTensor<float> inputTensor)
        {
            if (!HasPlacements)
                throw new InvalidOperationException("The value net has no placements output (it was trained without auxiliary heads).");

            float[][] outputs = RunOutputs(inputTensor, ["logits", "placements"]);
            float[] logits = outputs[0];
            float[] placementLogits = outputs[1];

            int batchSize = logits.Length / PlayerCount;
            return (SoftmaxPerRow(logits, batchSize, PlayerCount), SoftmaxPerRow(placementLogits, batchSize * PlayerCount, PlayerCount));
        }

        /// <summary>
        /// Win probabilities [batch * players] and, if requested, the placement distributions [batch * players * places] and
        /// the policy queries [batch * move types * dim] in one run
        /// </summary>
        public LeafValues Evaluate(DenseTensor<float> inputTensor, bool withPlacements, bool withPolicy)
        {
            int rows = inputTensor.Dimensions[0];
            return Evaluate(inputTensor, withPlacements, withPolicy, LeafValues.Allocate(rows, withPlacements, withPolicy ? Policy?.QueryLength ?? 0 : 0));
        }

        /// <summary>
        /// As Evaluate, written into the arrays of destination (at least as long as the batch needs) instead of new ones, so a
        /// search that keeps its buffers allocates nothing per batch; returns them with the outputs not asked for left out
        /// </summary>
        public LeafValues Evaluate(DenseTensor<float> inputTensor, bool withPlacements, bool withPolicy, LeafValues destination)
        {
            if (withPlacements && !HasPlacements)
                throw new InvalidOperationException("The value net has no placements output (it was trained without auxiliary heads).");
            if (withPolicy && Policy == null)
                throw new InvalidOperationException("The value net has no policy head.");

            int batchSize = inputTensor.Dimensions[0];
            if (FixedBatchSize is not int fixedSize || batchSize == fixedSize)
            {
                using BatchOutputs outputs = RunBatch(inputTensor, withPlacements, withPolicy);
                outputs.CopyRows(0, batchSize, destination, 0, withPlacements, withPolicy);
            }
            else
            {
                // Smaller batches are padded with empty rows, larger ones run in several pieces; only the real rows are kept
                int featureDim = inputTensor.Dimensions[1];
                var piece = new DenseTensor<float>([fixedSize, featureDim]);
                for (int start = 0; start < batchSize; start += fixedSize)
                {
                    int rows = Math.Min(fixedSize, batchSize - start);
                    piece.Buffer.Span.Clear();
                    inputTensor.Buffer.Span.Slice(start * featureDim, rows * featureDim).CopyTo(piece.Buffer.Span);
                    using BatchOutputs outputs = RunBatch(piece, withPlacements, withPolicy);
                    outputs.CopyRows(0, rows, destination, start, withPlacements, withPolicy);
                }
            }
            return destination.Only(withPlacements, withPolicy);
        }

        /// <summary>
        /// Runs one batch (of the fixed batch size, if there is one) and keeps the outputs in the runtime's memory until
        /// disposed, so callers copy out only the rows they need (see BatchingEvaluator)
        /// </summary>
        public BatchOutputs RunBatch(DenseTensor<float> inputTensor, bool withPlacements, bool withPolicy)
        {
            if (FixedBatchSize is int fixedSize && inputTensor.Dimensions[0] != fixedSize)
                throw new ArgumentException($"The session runs batches of {fixedSize} rows, not {inputTensor.Dimensions[0]}.", nameof(inputTensor));

            List<string> names = ["logits"];
            if (withPlacements) names.Add("placements");
            if (withPolicy) names.Add("policy_query");
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("state", inputTensor) };

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            if (_runLock == null)
                results = _session.Run(inputs, names);
            else
                lock (_runLock) results = _session.Run(inputs, names);
            return new BatchOutputs(results, Policy?.QueryLength ?? 0);
        }

        /// <summary>
        /// The outputs of one batch in the runtime's memory, read row by row: softmax for the win probabilities and placements,
        /// float16 policy queries widened to float
        /// </summary>
        public sealed class BatchOutputs : IDisposable
        {
            private readonly IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _results;
            private readonly DisposableNamedOnnxValue _logits;
            private readonly DisposableNamedOnnxValue? _placements, _queries;
            private readonly int _queryLength;

            internal BatchOutputs(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, int queryLength)
            {
                _results = results;
                _logits = results.First(result => result.Name == "logits");
                _placements = results.FirstOrDefault(result => result.Name == "placements");
                _queries = results.FirstOrDefault(result => result.Name == "policy_query");
                _queryLength = queryLength;
            }

            /// <summary>
            /// Copies `rows` rows from `sourceRow` on into destination from `destinationRow` on, the placements and queries only
            /// if asked for
            /// </summary>
            public void CopyRows(int sourceRow, int rows, LeafValues destination, int destinationRow, bool withPlacements, bool withPolicy)
            {
                SoftmaxRows(Floats(_logits).Slice(sourceRow * PlayerCount, rows * PlayerCount),
                    destination.WinProbabilities.AsSpan(destinationRow * PlayerCount, rows * PlayerCount), PlayerCount);

                if (withPlacements)
                {
                    const int Places = PlayerCount * PlayerCount;
                    SoftmaxRows(Floats(_placements!).Slice(sourceRow * Places, rows * Places),
                        destination.Placements!.AsSpan(destinationRow * Places, rows * Places), PlayerCount);
                }

                if (withPolicy)
                {
                    Span<float> queries = destination.PolicyQueries!.AsSpan(destinationRow * _queryLength, rows * _queryLength);
                    if (_queries!.ElementType == TensorElementType.Float16)
                    {
                        // Float16 and Half share the IEEE layout; one vectorized pass
                        ReadOnlySpan<Float16> halves = ((DenseTensor<Float16>)_queries.AsTensor<Float16>()).Buffer.Span.Slice(sourceRow * _queryLength, rows * _queryLength);
                        System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(System.Runtime.InteropServices.MemoryMarshal.Cast<Float16, Half>(halves), queries);
                    }
                    else
                        Floats(_queries).Slice(sourceRow * _queryLength, rows * _queryLength).CopyTo(queries);
                }
            }

            private static ReadOnlySpan<float> Floats(DisposableNamedOnnxValue result) => ((DenseTensor<float>)result.AsTensor<float>()).Buffer.Span;

            public void Dispose() => _results.Dispose();
        }

        /// <summary>
        /// Batchable numerically stable softmax implementation for row-major flat arrays.
        /// </summary>
        /// <param name="flat">Row-major flat array of shape [batchSize * numClasses]</param>
        /// <param name="batchSize">Batch size</param>
        /// <param name="numClasses">Number of classes </param>
        /// <returns>Row-major flat array of shape [batchSize * numClasses] with softmax distributed outputs</returns>
        private static float[] SoftmaxPerRow(float[] flat, int batchSize, int numClasses)
        {
            var result = new float[flat.Length];
            SoftmaxRows(flat.AsSpan(0, batchSize * numClasses), result, numClasses);
            return result;
        }

        /// <summary>
        /// Softmax of each row of numClasses values from source into destination (same length)
        /// </summary>
        private static void SoftmaxRows(ReadOnlySpan<float> source, Span<float> destination, int numClasses)
        {
            for (int offset = 0; offset < source.Length; offset += numClasses)
            {
                float max = float.NegativeInfinity;
                for (int c = 0; c < numClasses; c++)
                    max = Math.Max(max, source[offset + c]);

                float sumExp = 0f;
                for (int c = 0; c < numClasses; c++)
                {
                    float e = MathF.Exp(source[offset + c] - max);
                    destination[offset + c] = e;
                    sumExp += e;
                }

                for (int c = 0; c < numClasses; c++)
                    destination[offset + c] /= sumExp;
            }
        }
    }
}
