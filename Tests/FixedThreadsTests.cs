using Agents.Inference;

namespace Tests
{
    /// <summary>
    /// FixedThreads, which runs the CUDA session's calls (see StateValueNet): the first threads to call (as many as its count)
    /// run their calls themselves from then on, every later thread hands its calls to the fixed threads, so however many
    /// threads call, at most twice the count ever run work; no more calls than the count run at once, inline or handed
    /// over; a single later caller keeps reusing one fixed thread; results and exceptions come back to the caller
    /// </summary>
    public class FixedThreadsTests
    {
        private static void OnFreshThreads(int count, int atOnce, Action<int> body)
        {
            // Callers come in waves of fresh threads, as a thread pool replaces its threads over hours
            for (int start = 0; start < count; start += atOnce)
            {
                var threads = Enumerable.Range(start, Math.Min(atOnce, count - start)).Select(i => new Thread(() => body(i))).ToList();
                threads.ForEach(thread => thread.Start());
                threads.ForEach(thread => thread.Join());
            }
        }

        [Fact]
        public void A_caller_that_keeps_calling_runs_its_work_itself()
        {
            // A long-lived worker (a BatchingEvaluator's, a self-play net's) pays no handover
            using var threads = new FixedThreads(4, "test");
            int self = Environment.CurrentManagedThreadId;
            for (int i = 0; i < 100; i++)
                Assert.Equal(self, threads.Run(() => Environment.CurrentManagedThreadId));
            Assert.Equal(1, threads.DirectThreads);
            Assert.Equal(0, threads.ThreadsStarted);
        }

        [Fact]
        public void However_many_threads_call_at_most_twice_its_count_ever_run_work()
        {
            using var threads = new FixedThreads(4, "test");
            var ran = new System.Collections.Concurrent.ConcurrentBag<(int caller, int runner)>();
            OnFreshThreads(200, 20, _ => ran.Add((Environment.CurrentManagedThreadId, threads.Run(() => Environment.CurrentManagedThreadId))));

            Assert.Equal(200, ran.Count);
            Assert.Equal(4, threads.DirectThreads);
            Assert.InRange(threads.ThreadsStarted, 1, 4);
            // Managed thread ids of finished threads may be reused, so count the work run elsewhere instead of distinct ids
            Assert.Equal(200 - 4, ran.Count(pair => pair.caller != pair.runner));
            Assert.InRange(ran.Where(pair => pair.caller != pair.runner).Select(pair => pair.runner).Distinct().Count(), 1, 4);
        }

        [Fact]
        public void Later_callers_one_at_a_time_share_one_fixed_thread()
        {
            using var threads = new FixedThreads(8, "test");
            var handedOver = new List<int>();
            OnFreshThreads(50, 1, i =>
            {
                int caller = Environment.CurrentManagedThreadId, runner = threads.Run(() => Environment.CurrentManagedThreadId);
                if (i >= 8) handedOver.Add(runner);
                else Assert.Equal(caller, runner);
            });

            Assert.Single(handedOver.Distinct());
            Assert.Equal(1, threads.ThreadsStarted);
            Assert.Equal(8, threads.DirectThreads);
        }

        [Fact]
        public void No_more_than_its_count_run_at_once_inline_or_handed_over()
        {
            using var threads = new FixedThreads(3, "test");
            int running = 0, most = 0;
            OnFreshThreads(12, 12, _ => threads.Run(() =>
            {
                int now = Interlocked.Increment(ref running);
                int seen;
                while (now > (seen = Volatile.Read(ref most)) && Interlocked.CompareExchange(ref most, now, seen) != seen) { }
                Thread.Sleep(40);
                Interlocked.Decrement(ref running);
                return 0;
            }));

            Assert.Equal(3, most);
        }

        [Fact]
        public void Results_and_exceptions_reach_the_caller_inline_and_handed_over()
        {
            using var threads = new FixedThreads(1, "test");
            // This thread is the one that runs its own calls
            Assert.Equal(42, threads.Run(() => 6 * 7));
            var inline = Assert.Throws<InvalidOperationException>(() => threads.Run<int>(() => throw new InvalidOperationException("inline")));
            Assert.Equal("inline", inline.Message);

            // Another thread's work goes to the fixed thread
            Exception? handedOver = null;
            string? next = null;
            var other = new Thread(() =>
            {
                handedOver = Record.Exception(() => threads.Run<int>(() => throw new InvalidOperationException("handed over")));
                next = threads.Run(() => "next");
            });
            other.Start();
            other.Join();

            Assert.IsType<InvalidOperationException>(handedOver);
            Assert.Equal("handed over", handedOver!.Message);
            Assert.Contains(nameof(Results_and_exceptions_reach_the_caller_inline_and_handed_over), handedOver.StackTrace);
            // The fixed thread that ran the failing work goes on with the next
            Assert.Equal("next", next);
            Assert.Equal(1, threads.ThreadsStarted);
        }

        [Fact]
        public void Dispose_ends_its_threads()
        {
            var threads = new FixedThreads(2, "test");
            var ran = new System.Collections.Concurrent.ConcurrentBag<Thread>();
            OnFreshThreads(8, 8, _ => ran.Add(threads.Run(() => Thread.CurrentThread)));
            Assert.InRange(threads.ThreadsStarted, 1, 2);
            threads.Dispose();

            Assert.All(ran.Distinct().Where(thread => thread.Name?.StartsWith("test") == true),
                thread => Assert.True(thread.Join(TimeSpan.FromSeconds(5)), $"{thread.Name} still runs"));
            Assert.Throws<ObjectDisposedException>(() => threads.Run(() => 0));
        }
    }
}
