using System.Text.Json.Nodes;

namespace Agents.OpenSeat
{
    /// <summary>
    /// What an Open Seat Agent asks the referee for (see Bridges/PROTOCOL.md): only its seat's view, a full game the
    /// referee draws from that view, or every hand (for open-hand agents)
    /// </summary>
    public enum OpenSeatNeeds { View, World, Truth }

    /// <summary>
    /// A line-based connection to an Open Seat Agent: the referee sends a message and reads the agent's answers
    /// </summary>
    public interface IOpenSeatChannel : IDisposable
    {
        void Send(JsonNode message);

        /// <summary>
        /// The agent's next answer; throws an OpenSeatException if none comes
        /// </summary>
        JsonNode Receive(CancellationToken cancellation = default);
    }

    /// <summary>
    /// The protocol's failure codes and its failure message (see Bridges/PROTOCOL.md)
    /// </summary>
    public static class OpenSeatFailure
    {
        public const string Version = "version", Unsupported = "unsupported", Illegal = "illegal", Malformed = "malformed",
            Timeout = "timeout", Ended = "ended", Internal = "internal";

        /// <summary>
        /// The failure message: null game outside a game, null n outside a decision
        /// </summary>
        public static JsonObject Message(string? gameId, int? n, string code, string message, JsonNode? detail = null)
        {
            JsonObject failure = new() { ["type"] = "failure", ["game_id"] = gameId, ["n"] = n, ["code"] = code, ["message"] = message };
            if (detail != null) failure["detail"] = detail;
            return failure;
        }
    }

    /// <summary>
    /// A failure of an Open Seat Agent's seat: it broke the protocol, quit, didn't answer in time, turned a game down or
    /// reported a failure itself (FromAgent). Code is one of OpenSeatFailure's.
    /// </summary>
    public sealed class OpenSeatException(string message, string code = OpenSeatFailure.Internal, bool fromAgent = false, Exception? inner = null)
        : Exception(message, inner)
    {
        public string Code { get; } = code;
        public bool FromAgent { get; } = fromAgent;
    }

    /// <summary>
    /// A game's time budget (see Bridges/PROTOCOL.md): the time per decision, an optional clock for the whole game, the
    /// CPU threads the agent may use (null: no limit) and whether it may use a GPU
    /// </summary>
    public sealed record OpenSeatBudget(int DecideMs, int? GameMs = null, int? Threads = null, bool Gpu = false)
    {
        /// <summary>
        /// Hexfarer's referee when whoever starts the game sets none: 5 minutes per decision
        /// </summary>
        public static readonly OpenSeatBudget Default = new(300_000);

        public JsonObject Json() => new() { ["decide_ms"] = DecideMs, ["game_ms"] = GameMs, ["threads"] = Threads, ["gpu"] = Gpu };
    }

    /// <summary>
    /// The agent's handshake: its name, the presets it plays, whether it wants every move (step), what it needs to see and
    /// whether its decisions may come with only the state's hash (lean)
    /// </summary>
    public sealed record OpenSeatHandshake(string Name, IReadOnlyList<string> Rulesets, bool Observe, OpenSeatNeeds Needs, bool Lean = false)
    {
        /// <summary>
        /// The protocol's name and version this referee speaks
        /// </summary>
        public const string Protocol = "osp";
        public const int Version = 2;

        /// <summary>
        /// How long an agent has for its handshake and its ready: starting a process and loading a net
        /// </summary>
        public static readonly TimeSpan AnswerTime = TimeSpan.FromSeconds(60);

        /// <summary>
        /// Greets the agent and reads its handshake: the same protocol and version, fields of the protocol's types, and
        /// what this referee can serve (lean only with observe, and with view or truth). Otherwise the agent is told with a
        /// failure and the exception says why. The agent needs its seat's view and doesn't observe unless it says otherwise.
        /// </summary>
        public static OpenSeatHandshake Exchange(IOpenSeatChannel channel, CancellationToken cancellation = default)
        {
            channel.Send(new JsonObject { ["type"] = "handshake", ["protocol"] = Protocol, ["version"] = Version });
            JsonNode answer = ReceiveWithin(channel, AnswerTime, cancellation);
            if (Text(answer["type"]) == "failure") throw AgentFailure(answer);
            if (Text(answer["type"]) != "handshake" || Text(answer["protocol"]) != Protocol || answer["version"]?.ToJsonString() != Version.ToString())
                throw Refuse(channel, OpenSeatFailure.Version, $"The agent didn't answer the handshake in protocol {Protocol} version {Version}: {answer.ToJsonString()}");
            OpenSeatException Malformed(string field) => Refuse(channel, OpenSeatFailure.Malformed, $"The agent's handshake has a {field} of the wrong type: {answer.ToJsonString()}");
            OpenSeatNeeds needs = answer["needs"] switch
            {
                null => OpenSeatNeeds.View,
                JsonNode node => Text(node) switch
                {
                    "view" => OpenSeatNeeds.View,
                    "world" => OpenSeatNeeds.World,
                    "truth" => OpenSeatNeeds.Truth,
                    _ => throw Refuse(channel, OpenSeatFailure.Unsupported, $"The agent needs {node.ToJsonString()}, which isn't view, world or truth."),
                },
            };
            string[] rulesets = answer["rulesets"] switch
            {
                null => [],
                JsonArray list => list.Select(r => Text(r) ?? throw Malformed("rulesets")).ToArray(),
                _ => throw Malformed("rulesets"),
            };
            bool Flag(string field) => answer[field] switch
            {
                null => false,
                JsonValue value when value.TryGetValue(out bool flag) => flag,
                _ => throw Malformed(field),
            };
            bool observe = Flag("observe"), lean = Flag("lean");
            if (lean && (!observe || needs == OpenSeatNeeds.World))
                throw Refuse(channel, OpenSeatFailure.Unsupported, "A lean agent has to observe and need its view or the truth: it keeps its own game from the steps.");
            return new OpenSeatHandshake(Text(answer["name"]) ?? "unnamed", rulesets, observe, needs, lean);
        }

        /// <summary>
        /// The agent's next answer within the time, or a timeout failure; the caller's own cancellation passes through
        /// </summary>
        public static JsonNode ReceiveWithin(IOpenSeatChannel channel, TimeSpan time, CancellationToken cancellation = default)
        {
            using CancellationTokenSource limit = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            limit.CancelAfter(time);
            try
            {
                return channel.Receive(limit.Token);
            }
            catch (OperationCanceledException) when (!cancellation.IsCancellationRequested)
            {
                throw new OpenSeatException($"The agent gave no answer within {time.TotalSeconds:0.#} s.", OpenSeatFailure.Timeout);
            }
        }

        /// <summary>
        /// The failure an agent reported, as the exception for it
        /// </summary>
        public static OpenSeatException AgentFailure(JsonNode failure) =>
            new($"The agent reported a failure ({Text(failure["code"]) ?? "no code"}): {Text(failure["message"]) ?? failure.ToJsonString()}",
                Text(failure["code"]) ?? OpenSeatFailure.Internal, fromAgent: true);

        // Tells the agent why the handshake failed, if it still listens, and returns the exception for it
        private static OpenSeatException Refuse(IOpenSeatChannel channel, string code, string message)
        {
            try { channel.Send(OpenSeatFailure.Message(null, null, code, message)); }
            catch (OpenSeatException) { }
            return new OpenSeatException(message, code);
        }

        public static string? Text(JsonNode? node) => node is JsonValue value && value.TryGetValue(out string? text) ? text : null;
    }

    /// <summary>
    /// A seat's line in a game's result record (see Bridges/PROTOCOL.md): who played it (null for the referee's own
    /// players), what it received, its decisions and their time as the referee measured them, resyncs, moves the referee
    /// chose for it, and its failures
    /// </summary>
    public sealed class OpenSeatSeatRecord(int seat, OpenSeatHandshake? agent = null)
    {
        public int Seat { get; } = seat;
        public OpenSeatHandshake? Agent { get; } = agent;
        public int Decisions { get; private set; }
        public long TimeMs { get; private set; }
        public long MaxMs { get; private set; }
        public int Resyncs { get; set; }
        public int Fallbacks { get; set; }
        public List<JsonObject> Failures { get; } = [];

        public void Decided(long ms)
        {
            Decisions++;
            TimeMs += ms;
            MaxMs = Math.Max(MaxMs, ms);
        }

        public void Failed(int? n, string code, bool byAgent, string message) =>
            Failures.Add(new JsonObject { ["n"] = n, ["code"] = code, ["by"] = byAgent ? "agent" : "referee", ["message"] = message });

        public JsonObject Json() => new()
        {
            ["seat"] = Seat,
            ["agent"] = Agent?.Name,
            ["needs"] = Agent?.Needs.ToString().ToLowerInvariant(),
            ["observe"] = Agent?.Observe,
            ["lean"] = Agent?.Lean,
            ["decisions"] = Decisions,
            ["time_ms"] = TimeMs,
            ["max_ms"] = MaxMs,
            ["resyncs"] = Resyncs,
            ["fallbacks"] = Fallbacks,
            ["failures"] = new JsonArray(Failures.Select(f => (JsonNode)f.DeepClone()).ToArray()),
        };

        /// <summary>
        /// A game's result record: the seed its agents got, its moves and player turns, and a line per seat in seat order
        /// </summary>
        public static JsonObject Result(int seed, int moves, int turns, IEnumerable<OpenSeatSeatRecord> seats) => new()
        {
            ["seed"] = seed,
            ["moves"] = moves,
            ["turns"] = turns,
            ["seats"] = new JsonArray(seats.OrderBy(s => s.Seat).Select(s => (JsonNode)s.Json()).ToArray()),
        };
    }
}
