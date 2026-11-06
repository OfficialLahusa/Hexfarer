using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Agents.Inference
{
    /// <summary>
    /// Collects the leaf batches of many searches (e.g. games played at once) into large batches for the value net. Each net
    /// gets its own worker thread that takes waiting requests until the batch is full, runs them together and hands every
    /// search its rows back; with two nets one batch runs while the next one is collected. DirectML doesn't allow parallel
    /// Run calls on one session, so parallel workers need nets of their own (loaded from the same model).
    /// </summary>
    public sealed class BatchingEvaluator : ILeafEvaluator, IDisposable
    {
        private sealed record Request(DenseTensor<float> Features, int Rows, bool WithPlacements, bool WithPolicy, LeafValues? Destination, TaskCompletionSource<LeafValues> Completion);

        private readonly BlockingCollection<Request> _queue = new(new ConcurrentQueue<Request>());
        private readonly Thread[] _workers;
        private readonly int _maxRows;
        private readonly int _featureDim;
        private readonly bool _padToMaxRows;
        private long _batches, _rows, _runRows;

        /// <param name="nets">One net per worker, all of the same model</param>
        /// <param name="maxRows">Rows per net batch at most; nets with a fixed batch size (DirectML) need it at least this large</param>
        /// <param name="padToMaxRows">Nets without a fixed batch size run every batch with maxRows rows too (unused rows are
        /// ignored): one shape instead of one per batch size, since CUDA's memory pool grows with every new shape</param>
        public BatchingEvaluator(IReadOnlyList<StateValueNet> nets, int maxRows, bool padToMaxRows = false)
        {
            if (nets.Count == 0) throw new ArgumentException("At least one net is needed.", nameof(nets));
            if (nets.Any(net => net.FixedBatchSize < maxRows))
                throw new ArgumentException($"The nets' fixed batch size is below {maxRows} rows.", nameof(nets));
            if (nets.Any(net => net.InputDim != nets[0].InputDim || net.HasPlacements != nets[0].HasPlacements || (net.Policy == null) != (nets[0].Policy == null)))
                throw new ArgumentException("The nets must be of the same model.", nameof(nets));

            _maxRows = maxRows;
            _padToMaxRows = padToMaxRows;
            _featureDim = nets[0].InputDim;
            HasPlacements = nets[0].HasPlacements;
            Policy = nets[0].Policy;
            _workers = nets.Select((net, i) => new Thread(() => Work(net)) { IsBackground = true, Name = $"Value net worker {i}" }).ToArray();
            foreach (Thread worker in _workers)
                worker.Start();
        }

        public bool HasPlacements { get; }

        public PolicyHead? Policy { get; }

        /// <summary>
        /// Rows per net batch at most
        /// </summary>
        public int MaxRows => _maxRows;

        /// <summary>
        /// Net batches run so far
        /// </summary>
        public long Batches => Interlocked.Read(ref _batches);

        /// <summary>
        /// Rows evaluated so far (Rows / Batches is the mean batch size)
        /// </summary>
        public long Rows => Interlocked.Read(ref _rows);

        /// <summary>
        /// Rows the nets ran so far, padding included (RunRows / Batches is the shape they ran)
        /// </summary>
        public long RunRows => Interlocked.Read(ref _runRows);

        public Task<LeafValues> EvaluateAsync(DenseTensor<float> features, bool withPlacements, bool withPolicy = false, LeafValues? destination = null)
        {
            int rows = features.Dimensions[0];
            if (rows > _maxRows) throw new ArgumentException($"{rows} rows don't fit into a batch of {_maxRows}.", nameof(features));
            if (features.Dimensions[1] != _featureDim) throw new ArgumentException($"The net takes {_featureDim} features, not {features.Dimensions[1]}.", nameof(features));
            if (withPlacements && !HasPlacements)
                throw new InvalidOperationException("The value net has no placements output (it was trained without auxiliary heads).");
            if (withPolicy && Policy == null)
                throw new InvalidOperationException("The value net has no policy head.");

            // The search continues on the thread pool, not on the worker, which goes on with the next batch
            var completion = new TaskCompletionSource<LeafValues>(TaskCreationOptions.RunContinuationsAsynchronously);
            _queue.Add(new Request(features, rows, withPlacements, withPolicy, destination, completion));
            return completion.Task;
        }

        private void Work(StateValueNet net)
        {
            // A fixed batch size runs the whole buffer (unused rows are stale and ignored), otherwise only the rows in use
            int bufferRows = net.FixedBatchSize ?? _maxRows;
            float[] buffer = new float[bufferRows * _featureDim];
            var batch = new List<Request>();
            Request? carried = null;

            while (true)
            {
                batch.Clear();
                Request first;
                if (carried != null)
                    first = carried;
                else if (!_queue.TryTake(out first!, Timeout.Infinite))
                    return; // Disposed
                carried = null;

                // Take what is waiting until the batch is full; a request that no longer fits starts the next batch
                int rows = 0;
                Request? next = first;
                do
                {
                    if (rows + next.Rows > _maxRows)
                    {
                        carried = next;
                        break;
                    }
                    next.Features.Buffer.Span[..(next.Rows * _featureDim)].CopyTo(buffer.AsSpan(rows * _featureDim));
                    rows += next.Rows;
                    batch.Add(next);
                }
                while (_queue.TryTake(out next));

                try
                {
                    int runRows = net.FixedBatchSize ?? (_padToMaxRows ? _maxRows : rows);
                    var input = new DenseTensor<float>(buffer.AsMemory(0, runRows * _featureDim), [runRows, _featureDim]);
                    // The outputs stay in the runtime's memory, and each request gets its rows in its own arrays (those it brought,
                    // which a search reuses): no batch allocates large arrays, which would force full collections
                    using StateValueNet.BatchOutputs outputs = net.RunBatch(input, batch.Any(request => request.WithPlacements), batch.Any(request => request.WithPolicy));

                    int row = 0;
                    int queryLength = Policy?.QueryLength ?? 0;
                    foreach (Request request in batch)
                    {
                        LeafValues target = request.Destination ?? LeafValues.Allocate(request.Rows, request.WithPlacements, request.WithPolicy ? queryLength : 0);
                        outputs.CopyRows(row, request.Rows, target, 0, request.WithPlacements, request.WithPolicy);
                        request.Completion.SetResult(target.Only(request.WithPlacements, request.WithPolicy));
                        row += request.Rows;
                    }

                    Interlocked.Increment(ref _batches);
                    Interlocked.Add(ref _rows, rows);
                    Interlocked.Add(ref _runRows, runRows);
                }
                catch (Exception e)
                {
                    foreach (Request request in batch)
                        request.Completion.TrySetException(e);
                }
            }
        }

        /// <summary>
        /// Stops the workers after the requests already waiting
        /// </summary>
        public void Dispose()
        {
            _queue.CompleteAdding();
            foreach (Thread worker in _workers)
                worker.Join();
            _queue.Dispose();
        }
    }
}
