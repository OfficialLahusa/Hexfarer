using System.Collections.Concurrent;
using System.Runtime.ExceptionServices;

namespace Agents.Inference
{
    /// <summary>
    /// Runs work on a bounded set of threads. The first threads to call (as many as the count) run their work themselves from
    /// then on, so long-lived callers (a BatchingEvaluator's workers, a game's thread) pay nothing extra; every later thread
    /// hands its work to one of as many fixed long-lived threads and waits for it, as if it had run it itself. However many
    /// threads call, at most twice the count ever run work, and no more than the count run work at once.
    /// For the CUDA session (see StateValueNet): ONNX Runtime's CUDA provider keeps state for every thread that ever ran the
    /// session and never frees it, so calls straight from a thread pool, which replaces its threads, make it grow without end;
    /// more than about 8 calls at once grow it too.
    /// </summary>
    public sealed class FixedThreads : IDisposable
    {
        private sealed class Lane
        {
            public readonly BlockingCollection<Action> Work = new();
            public Thread? Thread;
        }

        private readonly Lane[] _lanes;
        // Free lanes, the one freed last on top: a single caller keeps reusing one fixed thread
        private readonly Stack<int> _free = new();
        // Calls running at once, inline or handed over
        private readonly SemaphoreSlim _running;
        private readonly object _lock = new();
        // Whether the current thread is one of the threads that run their own work. Per thread, not by thread id: the runtime
        // reuses the ids of finished threads, and a new thread with an old id must not count as one of them
        private readonly ThreadLocal<bool> _isDirect = new();
        private readonly string _name;
        private int _direct, _started;
        private volatile bool _disposed;

        public FixedThreads(int count, string name)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
            _lanes = Enumerable.Range(0, count).Select(_ => new Lane()).ToArray();
            // Lane 0 on top, so the first work handed over gets it
            for (int i = count - 1; i >= 0; i--)
                _free.Push(i);
            _running = new SemaphoreSlim(count, count);
            _name = name;
        }

        /// <summary>
        /// Calls that run at once at most, and both the threads that run their own work and the fixed threads at most
        /// </summary>
        public int Count => _lanes.Length;

        /// <summary>
        /// Threads that ran their own work so far (the first callers, at most Count)
        /// </summary>
        public int DirectThreads => Volatile.Read(ref _direct);

        /// <summary>
        /// Fixed threads started so far (one per lane that ever got work, at most Count)
        /// </summary>
        public int ThreadsStarted => Volatile.Read(ref _started);

        /// <summary>
        /// Runs work, on this thread if it is one of the first callers, otherwise on one of the fixed threads, and returns its
        /// result or throws its exception; waits while Count calls run
        /// </summary>
        public T Run<T>(Func<T> work)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _running.Wait();
            try
            {
                if (IsDirect()) return work();
                return HandOver(work);
            }
            finally
            {
                _running.Release();
            }
        }

        /// <summary>
        /// Whether the current thread runs its own work: it already does, or it is among the first Count threads to call
        /// </summary>
        private bool IsDirect()
        {
            if (_isDirect.Value) return true;
            lock (_lock)
            {
                if (_direct >= _lanes.Length) return false;
                _direct++;
            }
            _isDirect.Value = true;
            return true;
        }

        private T HandOver<T>(Func<T> work)
        {
            // The running calls are at most Count, so a lane is always free here
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
            }
        }

        /// <summary>
        /// Ends the fixed threads after the work they have; Run calls that come later throw
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
            _running.Dispose();
            _isDirect.Dispose();
        }
    }
}
