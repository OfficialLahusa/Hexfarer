using System.Collections.Concurrent;
using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Agents.OpenSeat
{
    /// <summary>
    /// An Open Seat Agent run as a process: messages go to its standard input, one JSON object per line, its answers come
    /// from its standard output. Lines on standard output that aren't JSON are kept for the error message (the protocol
    /// allows only its own lines there), standard error goes to a log file if given, and the last lines of both are kept,
    /// so a failure can say what the agent said last. An answer that doesn't come within the timeout fails, as does an
    /// agent that ended. Disposing closes its input (the agent ends when its input does) and stops it if it doesn't.
    /// </summary>
    public sealed class OpenSeatAgentProcess : IOpenSeatChannel
    {
        // Lines kept of what the agent said besides its answers
        private const int TailLines = 20;

        private readonly Process _process;
        private readonly TimeSpan _timeout;
        private readonly string _command;
        // The agent's standard output, line by line; null once it ended
        private readonly BlockingCollection<string?> _lines = new();
        private readonly Queue<string> _tail = new();
        private readonly StreamWriter? _errorLog;
        private bool _disposed, _exited;

        private OpenSeatAgentProcess(Process process, TimeSpan timeout, string command, StreamWriter? errorLog)
        {
            _process = process;
            _timeout = timeout;
            _command = command;
            _errorLog = errorLog;
        }

        /// <summary>
        /// Whether the agent's process has ended
        /// </summary>
        public bool HasExited => _disposed ? _exited : _process.HasExited;

        /// <summary>
        /// Starts the command (its words split as SplitCommandLine does); answers have to come within answerTimeout,
        /// standard error is appended to errorLog if given
        /// </summary>
        public static OpenSeatAgentProcess Start(string commandLine, TimeSpan answerTimeout, string? errorLog = null)
        {
            IReadOnlyList<string> words = SplitCommandLine(commandLine);
            if (words.Count == 0) throw new ArgumentException("An Open Seat Agent needs a command.", nameof(commandLine));
            ProcessStartInfo info = new(words[0])
            {
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                StandardInputEncoding = new UTF8Encoding(false),
            };
            foreach (string word in words.Skip(1)) info.ArgumentList.Add(word);

            StreamWriter? log = errorLog == null ? null : new StreamWriter(errorLog, append: true, new UTF8Encoding(false)) { AutoFlush = true };
            Process process = new() { StartInfo = info, EnableRaisingEvents = true };
            OpenSeatAgentProcess agent = new(process, answerTimeout, commandLine, log);
            process.OutputDataReceived += (_, e) => agent._lines.Add(e.Data);
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data == null) return;
                agent.Remember(e.Data);
                lock (agent._tail) agent._errorLog?.WriteLine(e.Data);
            };
            try
            {
                process.Start();
            }
            catch (Win32Exception e)
            {
                log?.Dispose();
                throw new OpenSeatException($"Couldn't start the Open Seat Agent \"{commandLine}\": {e.Message}", e);
            }
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            return agent;
        }

        /// <summary>
        /// The words of a command line: separated by spaces, double quotes group words with spaces (and give an empty word)
        /// </summary>
        public static IReadOnlyList<string> SplitCommandLine(string commandLine)
        {
            List<string> words = [];
            StringBuilder word = new();
            bool quoted = false, any = false;
            foreach (char c in commandLine)
            {
                if (c == '"') { quoted = !quoted; any = true; }
                else if (char.IsWhiteSpace(c) && !quoted)
                {
                    if (any) words.Add(word.ToString());
                    word.Clear();
                    any = false;
                }
                else { word.Append(c); any = true; }
            }
            if (any) words.Add(word.ToString());
            return words;
        }

        private void Remember(string line)
        {
            lock (_tail)
            {
                _tail.Enqueue(line);
                while (_tail.Count > TailLines) _tail.Dequeue();
            }
        }

        // What the agent said last, for an error message
        private string Said()
        {
            lock (_tail) return _tail.Count == 0 ? "" : $"; it said last:{Environment.NewLine}{string.Join(Environment.NewLine, _tail)}";
        }

        public void Send(JsonNode message)
        {
            try
            {
                _process.StandardInput.WriteLine(message.ToJsonString());
                _process.StandardInput.Flush();
            }
            catch (IOException e)
            {
                throw new OpenSeatException($"The Open Seat Agent \"{_command}\" ended before it read its messages{Ended()}{Said()}", e);
            }
        }

        public JsonNode Receive(CancellationToken cancellation = default)
        {
            Stopwatch waited = Stopwatch.StartNew();
            while (true)
            {
                TimeSpan left = _timeout - waited.Elapsed;
                string? line;
                try
                {
                    if (left <= TimeSpan.Zero || !_lines.TryTake(out line, (int)Math.Ceiling(left.TotalMilliseconds), cancellation))
                        throw new OpenSeatException($"The Open Seat Agent \"{_command}\" gave no answer within {_timeout.TotalSeconds:0.#} s{Said()}");
                }
                catch (OperationCanceledException) { throw; }
                if (line == null)
                {
                    // Its output ended: the agent did; wait for the rest of its standard error
                    _process.WaitForExit(2000);
                    _lines.Add(null);
                    throw new OpenSeatException($"The Open Seat Agent \"{_command}\" ended without an answer{Ended()}{Said()}");
                }
                if (line.Trim().Length == 0) continue;
                try
                {
                    if (JsonNode.Parse(line) is JsonObject answer) return answer;
                }
                catch (JsonException) { }
                // Not a protocol line: kept for the error message
                Remember(line);
            }
        }

        private string Ended() => _process.HasExited ? $" (exit code {_process.ExitCode})" : "";

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try
            {
                // The agent ends when its input does
                _process.StandardInput.Close();
            }
            catch (IOException) { }
            if (!_process.WaitForExit(5000))
            {
                try { _process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                _process.WaitForExit(5000);
            }
            _exited = _process.HasExited;
            lock (_tail) _errorLog?.Dispose();
            _process.Dispose();
        }
    }
}
