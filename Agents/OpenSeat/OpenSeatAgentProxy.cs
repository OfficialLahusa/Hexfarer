using Common;
using Action = Common.Actions.Action;

namespace Agents.OpenSeat
{
    /// <summary>
    /// Plays a seat of a Hexfarer game by asking an Open Seat Agent
    /// </summary>
    public sealed class OpenSeatAgentProxy : Agent, IDisposable
    {
        public OpenSeatHello Hello { get; } = null!;

        public OpenSeatAgentProxy(sbyte seat, GameState start, IOpenSeatChannel channel, string gameId, int? seed = null) : base(seat)
        {
            throw new NotImplementedException();
        }

        public override Action Act(GameState state) => throw new NotImplementedException();

        public void Dispose() { }
    }
}
