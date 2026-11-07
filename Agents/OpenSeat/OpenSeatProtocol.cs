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
    /// An Open Seat Agent that broke the protocol, quit or didn't answer in time
    /// </summary>
    public sealed class OpenSeatException(string message, Exception? inner = null) : Exception(message, inner);

    /// <summary>
    /// The agent's answer to the referee's hello: its name, the rulesets it plays, whether it wants every move (step) and
    /// what it needs to see
    /// </summary>
    public sealed record OpenSeatHello(string Name, IReadOnlyList<string> Rulesets, bool Observe, OpenSeatNeeds Needs)
    {
        /// <summary>
        /// The protocol's name and version this referee speaks
        /// </summary>
        public const string Protocol = "osp";
        public const int Version = 1;

        /// <summary>
        /// Greets the agent and reads its answer: it has to answer with the same protocol and version; it needs its seat's
        /// view and doesn't observe unless it says otherwise. A field of another type than the protocol's breaks it.
        /// </summary>
        public static OpenSeatHello Exchange(IOpenSeatChannel channel, CancellationToken cancellation = default)
        {
            channel.Send(new JsonObject { ["type"] = "hello", ["protocol"] = Protocol, ["version"] = Version });
            JsonNode answer = channel.Receive(cancellation);
            if (Text(answer["type"]) != "hello" || Text(answer["protocol"]) != Protocol || answer["version"]?.ToJsonString() != Version.ToString())
                throw new OpenSeatException($"The agent didn't answer hello in protocol {Protocol} version {Version}: {answer.ToJsonString()}");
            OpenSeatNeeds needs = Text(answer["needs"]) switch
            {
                null or "view" => OpenSeatNeeds.View,
                "world" => OpenSeatNeeds.World,
                "truth" => OpenSeatNeeds.Truth,
                string other => throw new OpenSeatException($"The agent needs \"{other}\", which isn't view, world or truth."),
            };
            OpenSeatException Malformed(string field) => new($"The agent's hello has a {field} of the wrong type: {answer.ToJsonString()}");
            string[] rulesets = answer["rulesets"] switch
            {
                null => [],
                JsonArray list => list.Select(r => Text(r) ?? throw Malformed("rulesets")).ToArray(),
                _ => throw Malformed("rulesets"),
            };
            bool observe = answer["observe"] switch
            {
                null => false,
                JsonValue value when value.TryGetValue(out bool flag) => flag,
                _ => throw Malformed("observe"),
            };
            return new OpenSeatHello(Text(answer["name"]) ?? "unnamed", rulesets, observe, needs);
        }

        private static string? Text(JsonNode? node) => node is JsonValue value && value.TryGetValue(out string? text) ? text : null;
    }
}
