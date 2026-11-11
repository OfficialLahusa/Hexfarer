using Agents;
using Agents.Inference;
using Common;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Tests
{
    /// <summary>
    /// The value net on the CPU, with DirectML (GPU, Windows only) and with CUDA (GPU, Linux only; both skipped where they
    /// aren't available): the same outputs, also for batches the fixed DirectML batch size has to pad or split, and on CUDA
    /// the session run on a fixed set of threads
    /// </summary>
    public class InferenceProviderTests
    {
        private static string TinyModelPath => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        private static DenseTensor<float> RandomGameBatch(int size, int seed, StateVectorizer? vectorizer = null)
        {
            vectorizer ??= StateVectorizer.Current;
            int dim = vectorizer.Dimension;
            TestUtil.Seed(seed);
            DenseTensor<float> batch = new([size, dim]);
            int row = 0;
            while (row < size)
            {
                GameState state = TestUtil.NewGame();
                TestUtil.PlayRandomGame(state, afterApply: (s, _) =>
                {
                    if (row < size && Random.Shared.Next(4) == 0)
                        vectorizer.VectorizeInto(s, batch.Buffer.Span.Slice(row++ * dim, dim));
                }, maxActions: 400);
            }
            return batch;
        }

        private static StateValueNet DirectMLOrSkip(string modelPath)
        {
            Skip.IfNot(OperatingSystem.IsWindows(), "DirectML runs on Windows only.");
            StateValueNet net = new(modelPath, provider: InferenceProvider.DirectML);
            Skip.If(net.Provider != InferenceProvider.DirectML, "No DirectML device available.");
            return net;
        }

        [Fact]
        public void TheCPUProviderRunsAnyBatchAsItIs()
        {
            StateValueNet net = new(TinyModelPath, provider: InferenceProvider.CPU);
            Assert.Equal(InferenceProvider.CPU, net.Provider);
            Assert.Null(net.FixedBatchSize);
            Assert.Equal(3 * 4, net.Run(RandomGameBatch(3, 180)).Length);
        }

        [SkippableFact]
        public void DirectMLGivesTheSameValuesAsTheCPU()
        {
            StateValueNet gpu = DirectMLOrSkip(TinyModelPath);
            StateValueNet cpu = new(TinyModelPath, provider: InferenceProvider.CPU);
            Assert.Equal(StateValueNet.DefaultFixedBatchSize, gpu.FixedBatchSize);

            // Exactly the fixed size, padded (1 and 37) and split (300 = 256 + 44)
            foreach (int size in new[] { 256, 1, 37, 300 })
            {
                DenseTensor<float> batch = RandomGameBatch(size, 181 + size);
                float[] expected = cpu.Run(batch), actual = gpu.Run(batch);
                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-3, $"batch {size}, value {i}: CPU {expected[i]}, DirectML {actual[i]}");

                (float[] cpuWins, float[] cpuPlacements) = cpu.RunWithPlacements(batch);
                (float[] gpuWins, float[] gpuPlacements) = gpu.RunWithPlacements(batch);
                Assert.Equal(cpuPlacements.Length, gpuPlacements.Length);
                for (int i = 0; i < cpuPlacements.Length; i++)
                    Assert.True(Math.Abs(cpuPlacements[i] - gpuPlacements[i]) < 1e-3, $"batch {size}, placement {i}");
            }
        }

        [SkippableFact]
        public void TheDeployedNetRunsOnDirectML()
        {
            TestUtil.SkipWithoutModel();
            StateValueNet gpu = DirectMLOrSkip(TestUtil.ModelPath);
            StateValueNet cpu = new(TestUtil.ModelPath, provider: InferenceProvider.CPU);

            DenseTensor<float> batch = RandomGameBatch(50, 190, StateVectorizer.For(cpu));
            float[] expected = cpu.Run(batch), actual = gpu.Run(batch);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-3, $"value {i}: CPU {expected[i]}, DirectML {actual[i]}");
        }

        [SkippableFact]
        public void CUDARunsTheSessionOnAFixedSetOfThreadsWithTheCPUsValues()
        {
            // CUDA keeps state for every thread that ever ran the session and never frees it, so only a bounded set of threads
            // runs the session however many threads call it (here 64 fresh ones, 8 at a time, as a thread pool's come and go)
            TestUtil.SkipWithoutModel();
            Skip.IfNot(OperatingSystem.IsLinux(), "CUDA runs on Linux here.");
            StateValueNet gpu = new(TestUtil.ModelPath, provider: InferenceProvider.CUDA);
            Skip.If(gpu.Provider != InferenceProvider.CUDA, "No CUDA device available.");
            StateValueNet cpu = new(TestUtil.ModelPath, provider: InferenceProvider.CPU);
            Assert.Equal(0, cpu.SessionThreads);

            StateVectorizer vectorizer = StateVectorizer.For(cpu);
            DenseTensor<float>[] batches = Enumerable.Range(0, 8).Select(i => RandomGameBatch(1 + 37 * i, 200 + i, vectorizer)).ToArray();
            LeafValues[] expected = batches.Select(batch => cpu.Evaluate(batch, withPlacements: false, withPolicy: cpu.Policy != null)).ToArray();
            var failures = new System.Collections.Concurrent.ConcurrentBag<string>();
            for (int wave = 0; wave < 8; wave++)
            {
                var threads = Enumerable.Range(0, 8).Select(i => new Thread(() =>
                {
                    LeafValues actual = gpu.Evaluate(batches[i], withPlacements: false, withPolicy: gpu.Policy != null);
                    for (int j = 0; j < actual.WinProbabilities.Length; j++)
                        if (Math.Abs(actual.WinProbabilities[j] - expected[i].WinProbabilities[j]) >= 1e-3) { failures.Add($"batch {i}, value {j}"); break; }
                    for (int j = 0; j < (actual.PolicyQueries?.Length ?? 0); j++)
                        if (Math.Abs(actual.PolicyQueries![j] - expected[i].PolicyQueries![j]) >= 2e-2) { failures.Add($"batch {i}, query {j}"); break; }
                })).ToList();
                threads.ForEach(thread => thread.Start());
                threads.ForEach(thread => thread.Join());
            }

            Assert.Empty(failures);
            // The first threads ran their own calls, the later ones handed them over: at most twice CUDAThreads in all
            Assert.InRange(gpu.SessionThreads, StateValueNet.CUDAThreads + 1, 2 * StateValueNet.CUDAThreads);
        }
    }
}
