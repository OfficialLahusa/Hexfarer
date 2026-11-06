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
        /// Greets the agent and reads its answer
        /// </summary>
        public static OpenSeatHello Exchange(IOpenSeatChannel channel) => throw new NotImplementedException();
    }
}
