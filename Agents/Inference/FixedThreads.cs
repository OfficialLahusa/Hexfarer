using System.Collections.Concurrent;
using System.Runtime.ExceptionServices;

namespace Agents.Inference
{
    /// <summary>
    /// Runs work on a fixed set of long-lived threads, one piece of work per thread at a time; the caller waits for its work
    /// as if it had run it itself. A thread starts when first needed, and the thread freed last takes the next work, so a
    /// single caller keeps using one thread and only as many threads start as callers ever came at once (at most the count).
    /// For the CUDA session (see StateValueNet): ONNX Runtime's CUDA provider keeps state for every thread that ever ran the
    /// session and never frees it, so calls straight from a thread pool, which replaces its threads, make it grow without end.
    /// </summary>
    public sealed class FixedThreads : IDisposable
    {
        private sealed class Lane
        {
            public readonly BlockingCollection<Action> Work = new();
            public Thread? Thread;
        }

        private readonly Lane[] _lanes;
        // Free lanes, the one freed last on top
        private readonly Stack<int> _free = new();
        private readonly SemaphoreSlim _available;
        private readonly object _lock = new();
        private readonly string _name;
        private int _started;
        private volatile bool _disposed;

        public FixedThreads(int count, string name)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
            _lanes = Enumerable.Range(0, count).Select(_ => new Lane()).ToArray();
            // Lane 0 on top, so the first caller gets it
            for (int i = count - 1; i >= 0; i--)
                _free.Push(i);
            _available = new SemaphoreSlim(count, count);
            _name = name;
        }

        /// <summary>
        /// Threads that run work at most
        /// </summary>
        public int Count => _lanes.Length;

        /// <summary>
        /// Threads started so far (one per lane that ever got work)
        /// </summary>
        public int ThreadsStarted => Volatile.Read(ref _started);

        /// <summary>
        /// Runs work on one of the threads and returns its result, or throws its exception, to the caller; waits while all
        /// threads are busy
        /// </summary>
        public T Run<T>(Func<T> work)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _available.Wait();
            int index;
            lock (_lock)
                index = _free.Pop();
            try
            {
                // The lane is this caller's until it is pushed back, so starting its thread needs no lock
                Lane lane = _lanes[index];
                if (lane.Thread == null)
                {
                    lane.Thread = new Thread(() => { foreach (Action action in lane.Work.GetConsumingEnumerable()) action(); })
                    {
                        IsBackground = true,
                        Name = $"{_name} {index}",
                    };
                    lane.Thread.Start();
                    Interlocked.Increment(ref _started);
                }

                T result = default!;
                ExceptionDispatchInfo? error = null;
                using var done = new ManualResetEventSlim();
                lane.Work.Add(() =>
                {
                    try { result = work(); }
                    catch (Exception e) { error = ExceptionDispatchInfo.Capture(e); }
                    finally { done.Set(); }
                });
                done.Wait();
                // Rethrown with the work's own stack trace
                error?.Throw();
                return result;
            }
            finally
            {
                lock (_lock)
                    _free.Push(index);
                _available.Release();
            }
        }

        /// <summary>
        /// Ends the threads after the work they have; Run calls that come later throw
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (Lane lane in _lanes)
            {
                lane.Work.CompleteAdding();
                lane.Thread?.Join();
                lane.Work.Dispose();
            }
            _available.Dispose();
        }
    }
}
