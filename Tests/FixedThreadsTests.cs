using Agents.Inference;

namespace Tests
{
    /// <summary>
    /// FixedThreads, which runs the CUDA session's calls (see StateValueNet): however many threads call, the work runs on at
    /// most its fixed number of long-lived threads, a single caller keeps reusing one of them, no more run at once than that
    /// number, and results and exceptions come back to the caller
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
        public void Calls_from_many_fresh_threads_run_on_at_most_its_threads()
        {
            using var threads = new FixedThreads(4, "test");
            var ran = new System.Collections.Concurrent.ConcurrentBag<int>();
            var callers = new System.Collections.Concurrent.ConcurrentBag<int>();
            OnFreshThreads(200, 20, _ =>
            {
                callers.Add(Environment.CurrentManagedThreadId);
                ran.Add(threads.Run(() => Environment.CurrentManagedThreadId));
            });

            Assert.Equal(200, ran.Count);
            Assert.InRange(ran.Distinct().Count(), 1, 4);
            Assert.InRange(threads.ThreadsStarted, 1, 4);
            Assert.Empty(ran.Intersect(callers));
        }

        [Fact]
        public void One_caller_at_a_time_keeps_using_one_thread()
        {
            using var threads = new FixedThreads(8, "test");
            var ran = new List<int>();
            OnFreshThreads(50, 1, _ => ran.Add(threads.Run(() => Environment.CurrentManagedThreadId)));

            Assert.Single(ran.Distinct());
            Assert.Equal(1, threads.ThreadsStarted);
        }

        [Fact]
        public void No_more_than_its_threads_run_at_once()
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
        public void Results_and_exceptions_reach_the_caller()
        {
            using var threads = new FixedThreads(2, "test");
            Assert.Equal(42, threads.Run(() => 6 * 7));

            var error = Assert.Throws<InvalidOperationException>(() => threads.Run<int>(() => throw new InvalidOperationException("from the work")));
            Assert.Equal("from the work", error.Message);
            Assert.Contains(nameof(Results_and_exceptions_reach_the_caller), error.StackTrace);

            // The thread that ran the failing work goes on with the next
            Assert.Equal("next", threads.Run(() => "next"));
            Assert.Equal(1, threads.ThreadsStarted);
        }

        [Fact]
        public void Dispose_ends_its_threads()
        {
            var threads = new FixedThreads(4, "test");
            var ran = new System.Collections.Concurrent.ConcurrentBag<Thread>();
            OnFreshThreads(8, 8, _ => ran.Add(threads.Run(() => Thread.CurrentThread)));
            threads.Dispose();

            Assert.All(ran.Distinct(), thread => Assert.True(thread.Join(TimeSpan.FromSeconds(5)), $"{thread.Name} still runs"));
            Assert.Throws<ObjectDisposedException>(() => threads.Run(() => 0));
        }
    }
}
