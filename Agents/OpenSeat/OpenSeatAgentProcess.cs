using System.Text.Json.Nodes;

namespace Agents.OpenSeat
{
    /// <summary>
    /// An Open Seat Agent run as a process, speaking over its standard input and output
    /// </summary>
    public sealed class OpenSeatAgentProcess : IOpenSeatChannel
    {
        public bool HasExited => throw new NotImplementedException();

        public static OpenSeatAgentProcess Start(string commandLine, TimeSpan answerTimeout, string? errorLog = null) => throw new NotImplementedException();

        public static IReadOnlyList<string> SplitCommandLine(string commandLine) => throw new NotImplementedException();

        public void Send(JsonNode message) => throw new NotImplementedException();

        public JsonNode Receive(CancellationToken cancellation = default) => throw new NotImplementedException();

        public void Dispose() { }
    }
}
