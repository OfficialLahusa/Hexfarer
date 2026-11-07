using Agents;
using Agents.Inference;
using Agents.MCTS;
using Agents.OpenSeat;
using Common;

namespace Simulation
{
    /// <summary>
    /// Who plays a seat. The Pimc kinds are the MCTS presets: they see only what their seat can know and search its
    /// information sets (see InformationSetMCTSAgent); the Mcts kinds are the open-hand presets, which see every hand. The
    /// names stay as they were for saved seats.
    /// </summary>
    public enum SeatKind { Human, MctsEasy, MctsMedium, MctsHard, Greedy, Builder, Simple, Random, PimcEasy, PimcMedium, PimcHard, OpenSeatAgent }

    /// <summary>
    /// What a connection may do: an admin controls the match (seats, undo, edits, playback), a player and an agent (bridge)
    /// move for their seat, a spectator only watches
    /// </summary>
    public enum Role { Admin, Player, Agent, Spectator }

    /// <summary>
    /// A connection to a match: its role and, for players and agents, its seat
    /// </summary>
    public sealed record SessionInfo(string Id, string Name, Role Role, int? Seat, string Address = "");

    /// <summary>
    /// How agents move: paused or not, whether human seats are played by stand-in agents (watching all seats play), and the
    /// pause between moves so the game can be followed
    /// </summary>
    public sealed record PlaybackState(bool Paused, bool HumanStandIns, float PaceSeconds)
    {
        public static readonly PlaybackState Default = new(false, false, 0.6f);
    }

    /// <summary>
    /// Search settings of the MCTS seats: the presets' thinking times, or one time or iteration budget for all, and the
    /// opponent model; and the Open Seat Agent of each seat of that kind, by name (null for the others)
    /// </summary>
    public sealed record AgentSettings(double? SecondsForAll, long IterationsForAll, OpponentModel OpponentModel, IReadOnlyList<string?>? OpenSeatAgents = null)
    {
        public static readonly AgentSettings Default = new(null, 0, OpponentModel.MaxN);
    }

    /// <summary>
    /// An Open Seat Agent a simulation can seat (see Bridges/PROTOCOL.md): its name in the seat pickers and the command
    /// that starts it. Each simulation has its own list, from its host's settings; a seat picks one by name (the
    /// AgentSettings' OpenSeatAgents), never by a command from a client.
    /// </summary>
    public sealed record ExternalAgent(string Name, string Command);

    public static class Seats
    {
        /// <summary>
        /// The seat kinds in the order the pickers list them: the human, the MCTS presets (the default bots), the open-hand
        /// MCTS presets that see every hand, then the simpler agents from strong to weak
        /// </summary>
        public static readonly SeatKind[] Ordered =
        [
            SeatKind.Human, SeatKind.PimcEasy, SeatKind.PimcMedium, SeatKind.PimcHard, SeatKind.MctsEasy, SeatKind.MctsMedium, SeatKind.MctsHard,
            SeatKind.Greedy, SeatKind.Builder, SeatKind.Simple, SeatKind.Random
        ];

        /// <summary>
        /// The bot a new game gets by default: the hard MCTS preset, or the builder without the value net
        /// </summary>
        public static SeatKind DefaultBot(bool valueNetAvailable) => valueNetAvailable ? SeatKind.PimcHard : SeatKind.Builder;

        public static string Name(SeatKind kind) => kind switch
        {
            SeatKind.Human => "Human",
            SeatKind.MctsEasy => "Open-hand MCTS Easy",
            SeatKind.MctsMedium => "Open-hand MCTS Medium",
            SeatKind.MctsHard => "Open-hand MCTS Hard",
            SeatKind.Greedy => "Greedy",
            SeatKind.Builder => "Builder",
            SeatKind.Simple => "Simple",
            SeatKind.Random => "Random",
            SeatKind.PimcEasy => "MCTS Easy",
            SeatKind.PimcMedium => "MCTS Medium",
            SeatKind.PimcHard => "MCTS Hard",
            SeatKind.OpenSeatAgent => "Open Seat Agent",
            _ => kind.ToString()
        };

        /// <summary>
        /// Whether the seat's agent evaluates states with the value net (a human seat's stand-in does)
        /// </summary>
        public static bool NeedsValueNet(SeatKind kind) => kind is SeatKind.MctsEasy or SeatKind.MctsMedium or SeatKind.MctsHard or SeatKind.Greedy or SeatKind.Human
            or SeatKind.PimcEasy or SeatKind.PimcMedium or SeatKind.PimcHard;

        public static bool IsFair(SeatKind kind) => kind is SeatKind.PimcEasy or SeatKind.PimcMedium or SeatKind.PimcHard;

        public static double PresetSeconds(SeatKind kind) => kind switch
        {
            SeatKind.MctsEasy or SeatKind.PimcEasy => 0.1,
            SeatKind.MctsHard or SeatKind.PimcHard => 2,
            _ => 0.5
        };

        /// <summary>
        /// The agent of a seat; a human seat gets a stand-in that plays like the medium preset (the builder without the net)
        /// </summary>
        public static Agent CreateAgent(SeatKind kind, sbyte seat, GameState state, AgentSettings settings, bool valueNetAvailable)
        {
            if (kind == SeatKind.Human) kind = valueNetAvailable ? SeatKind.MctsMedium : SeatKind.Builder;
            if (NeedsValueNet(kind) && !valueNetAvailable) kind = SeatKind.Builder;

            MCTSAgent Search() => new MCTSAgent(seat, state, timeLimitSeconds: settings.SecondsForAll ?? PresetSeconds(kind), opponentModel: settings.OpponentModel,
                iterationLimit: settings.IterationsForAll);
            return kind switch
            {
                SeatKind.Random => new RandomAgent(seat),
                SeatKind.Simple => new SimpleAgent(seat),
                SeatKind.Builder => new BuilderAgent(seat),
                SeatKind.Greedy => new GreedyAgent(seat),
                // An Open Seat Agent needs its simulation's list (see the overload below); without it the Builder plays
                SeatKind.OpenSeatAgent => new BuilderAgent(seat),
                _ when IsFair(kind) => new InformationSetMCTSAgent(Search(), state),
                _ => Search()
            };
        }

        /// <summary>
        /// How long an Open Seat Agent may take for an answer before its seat fails
        /// </summary>
        public static readonly TimeSpan OpenSeatAgentTimeout = TimeSpan.FromMinutes(5);

        /// <summary>
        /// The agent of a seat, an Open Seat Agent from the simulation's list among them (a process, its game named gameId,
        /// knowing what beliefs says the seat knows): an unknown one, or one that doesn't start, leaves the seat to the
        /// Builder and says why in error
        /// </summary>
        public static Agent CreateAgent(SeatKind kind, sbyte seat, GameState state, AgentSettings settings, bool valueNetAvailable,
            IReadOnlyList<ExternalAgent> externalAgents, string gameId, BeliefTracker? beliefs, out string? error)
        {
            error = null;
            if (kind != SeatKind.OpenSeatAgent) return CreateAgent(kind, seat, state, settings, valueNetAvailable);
            string? name = settings.OpenSeatAgents is { } names && seat < names.Count ? names[seat] : null;
            ExternalAgent? agent = name == null ? null : externalAgents.FirstOrDefault(a => a.Name == name);
            if (agent == null)
                error = name == null ? $"Seat {seat + 1} has no Open Seat Agent chosen; the Builder plays it." : $"No Open Seat Agent \"{name}\" is set up here; the Builder plays seat {seat + 1}.";
            else
            {
                try
                {
                    return new OpenSeatAgentProxy(seat, state, OpenSeatAgentProcess.Start(agent.Command, OpenSeatAgentTimeout), gameId, beliefs: beliefs);
                }
                catch (OpenSeatException e)
                {
                    error = $"The Open Seat Agent \"{name}\" didn't start ({e.Message}); the Builder plays seat {seat + 1}.";
                }
            }
            return new BuilderAgent(seat);
        }

        public static bool ValueNetAvailable => StateValueNet.IsDefaultModelAvailable;
    }
}
