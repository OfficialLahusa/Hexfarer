using Common;
using Simulation;
using System.Text;
using System.Text.Json.Nodes;

namespace OpenSeat
{
    /// <summary>
    /// An outside referee's games shown while they are played (OpenSeat relay --live): follows a referee log as the referee
    /// writes it (a file, or a directory with a file per game, as JSettlers' server writes them with -Dosp.record, where it
    /// follows the newest game), plays each game through Lanzenbrecher's rules (Relay) and into a match hosted for watchers
    /// (SimulationServer, which the client joins with the Watch role). A game or a part from the referee's state comes in
    /// as a loaded position, each move with the referee's outcome (RelayMove) and each known rule difference as an edit, so
    /// watchers can step back through the part.
    /// </summary>
    internal sealed class LiveRelay : IAsyncDisposable
    {
        // The relay plays every seat as the match's admin; watchers only watch
        private static readonly SessionInfo Feeder = new("relay", "Relay", Role.Admin, null);

        private readonly Match _match;
        private readonly SimulationServer _server;
        private readonly CancellationTokenSource _stop = new();
        private readonly Task _feeding;
        private readonly Relay _relay = new(patch: true);
        private readonly TextWriter? _log;

        public Uri Uri => _server.Uri;
        public int Port => _server.Port;

        /// <summary>
        /// Parts started so far (a game's first part counts)
        /// </summary>
        public int Parts { get; private set; }

        /// <summary>
        /// Moves played into the match so far
        /// </summary>
        public int Moves { get; private set; }

        private LiveRelay(Match match, SimulationServer server, string source, double delaySeconds, int pollMilliseconds, TextWriter? log)
        {
            _match = match;
            _server = server;
            _log = log;
            _feeding = Task.Run(() => Feed(source, delaySeconds, pollMilliseconds, _stop.Token));
        }

        /// <summary>
        /// Hosts the match and starts following the source: a log file or a directory of them
        /// </summary>
        public static async Task<LiveRelay> StartAsync(string source, ServerConfig config, double delaySeconds = 0, int pollMilliseconds = 200, TextWriter? log = null)
        {
            // Every seat is played from outside (by the relay), so no agent moves
            Match match = new(new GameState(MapGenerator.GenerateRandomClassic(), 4), [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human], valueNetAvailable: false);
            SimulationServer server = await SimulationServer.StartAsync(match, config);
            return new LiveRelay(match, server, source, delaySeconds, pollMilliseconds, log);
        }

        private async Task Feed(string source, double delaySeconds, int pollMilliseconds, CancellationToken stop)
        {
            try
            {
                foreach (JsonNode message in Directory.Exists(source) ? Newest(source, pollMilliseconds, stop) : Lines(source, pollMilliseconds, stop))
                {
                    foreach (Relay.Event e in _relay.Feed(message))
                    {
                        switch (e)
                        {
                            case Relay.Started started:
                                Parts++;
                                _match.Handle(Feeder, new LoadGame(new GameState(started.State), [], []));
                                _log?.WriteLine(started.Number == 1 ? $"{started.GameId}: started" : _relay.Log[^1]);
                                break;
                            case Relay.Moved moved:
                                Moves++;
                                _match.Handle(Feeder, new RelayMove(moved.Action));
                                if (delaySeconds > 0) await Task.Delay(TimeSpan.FromSeconds(delaySeconds), stop);
                                break;
                            case Relay.Patched patched:
                                // An edit in the game's history, so watchers keep the game from its first move
                                _match.Handle(Feeder, new ApplyEdit(new BoardEdit($"took over the referee's state ({patched.Difference})", patched.Seat, patched.Before, patched.After)));
                                break;
                            case Relay.Ended ended when ended.Part.Cut == null:
                                _log?.WriteLine($"{ended.Part.GameId}: ended after {ended.Part.FirstMove + ended.Part.Actions.Count} moves");
                                break;
                        }
                    }
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception e)
            {
                _log?.WriteLine($"The relay stopped: {e.Message}");
            }
        }

        /// <summary>
        /// The messages of a log file as the referee writes it: each whole line, waiting for more at its end, until stopped
        /// </summary>
        internal static IEnumerable<JsonNode> Lines(string path, int pollMilliseconds, CancellationToken stop, Func<bool>? giveUp = null)
        {
            while (!File.Exists(path))
            {
                if (stop.IsCancellationRequested || giveUp?.Invoke() == true) yield break;
                Thread.Sleep(pollMilliseconds);
            }
            using FileStream file = new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            byte[] buffer = new byte[64 * 1024];
            // A line the referee hasn't finished writing waits for its end
            List<byte> partial = [];
            while (!stop.IsCancellationRequested)
            {
                int read = file.Read(buffer, 0, buffer.Length);
                if (read == 0)
                {
                    if (giveUp?.Invoke() == true) yield break;
                    Thread.Sleep(pollMilliseconds);
                    continue;
                }
                int start = 0;
                for (int i = 0; i < read; i++)
                {
                    if (buffer[i] != (byte)'\n') continue;
                    partial.AddRange(new ArraySegment<byte>(buffer, start, i - start));
                    string line = Encoding.UTF8.GetString(partial.ToArray()).Trim();
                    partial.Clear();
                    start = i + 1;
                    if (line.Length > 0 && JsonNode.Parse(line) is JsonNode message) yield return message;
                }
                partial.AddRange(new ArraySegment<byte>(buffer, start, read - start));
            }
        }

        /// <summary>
        /// The messages of the newest game in a directory of logs (*.jsonl), then of the newest one started after it, and so on
        /// </summary>
        internal static IEnumerable<JsonNode> Newest(string directory, int pollMilliseconds, CancellationToken stop)
        {
            // The game shown last: only games started after it come next (older ones are over or under way for a while)
            FileInfo? current = null;
            static (DateTime, string) Order(FileInfo f) => (f.CreationTimeUtc, f.Name);
            IEnumerable<FileInfo> Later() => new DirectoryInfo(directory).GetFiles("*.jsonl")
                .Where(f => current == null || Order(f).CompareTo(Order(current)) > 0);
            while (!stop.IsCancellationRequested)
            {
                FileInfo? next = Later().OrderByDescending(Order).FirstOrDefault();
                if (next == null)
                {
                    Thread.Sleep(pollMilliseconds);
                    continue;
                }
                current = next;
                // Every game of the file, until its last game ended and a later file is there
                bool ended = false;
                bool LaterThere() => ended && Later().Any();
                foreach (JsonNode message in Lines(next.FullName, pollMilliseconds, stop, LaterThere))
                {
                    string type = message["type"]!.GetValue<string>();
                    if (type == "before") ended = false;
                    if (type == "after") ended = true;
                    yield return message;
                }
            }
        }

        public async ValueTask DisposeAsync()
        {
            _stop.Cancel();
            try { await _feeding.WaitAsync(TimeSpan.FromSeconds(5)); } catch (Exception) { }
            await _server.DisposeAsync();
            _match.Dispose();
        }
    }
}
