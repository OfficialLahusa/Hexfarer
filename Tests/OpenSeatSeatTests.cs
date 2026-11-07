using Agents;
using Agents.OpenSeat;
using Common;
using Common.Actions;
using Simulation;

namespace Tests
{
    /// <summary>
    /// Open Seat Agents as seats of a simulated match (SeatKind.OpenSeatAgent): the simulation knows its agents from its
    /// own settings (ExternalAgent: a name and the command that starts it), a seat picks one by name in the agent
    /// settings, and the match plays it like any agent; an unknown or failing agent leaves the seat to the Builder with an
    /// error, a replaced one is closed
    /// </summary>
    public class OpenSeatSeatTests
    {
        private static string TinyNet => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        // Hexfarer's own Open Seat Agent, built next to the tests
        private static ExternalAgent Hexfarer => new("Hexfarer test agent",
            $"dotnet \"{Path.Combine(AppContext.BaseDirectory, "OpenSeat.dll")}\" agent --net \"{TinyNet}\" --iterations 16 --leaves-per-batch 16 --provider cpu");

        private static bool WaitFor(Func<bool> condition, int milliseconds)
        {
            DateTime end = DateTime.UtcNow.AddMilliseconds(milliseconds);
            while (DateTime.UtcNow < end)
            {
                if (condition()) return true;
                Thread.Sleep(20);
            }
            return condition();
        }

        private static Match NewMatch(SeatKind[] seats, params ExternalAgent[] agents)
        {
            TestUtil.Seed(21);
            return new Match(TestUtil.NewGame(), seats, valueNetAvailable: false, externalAgents: agents);
        }

        [Fact]
        public void AnOpenSeatAgentPlaysItsSeatInAMatch()
        {
            using Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random, SeatKind.Random], Hexfarer);
            using LocalSimulation admin = new(match);
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = [Hexfarer.Name, null, null, null] }));
            admin.Send(new SetPlayback(PlaybackState.Default with { PaceSeconds = 0 }));
            Assert.True(WaitFor(() => match.AgentOf(0) is OpenSeatAgentProxy, 30000), "the seat should get the agent");
            Assert.Empty(admin.ProcessEvents().OfType<ErrorEvent>());

            match.Start();
            Assert.True(WaitFor(() => match.Record.State.HasEnded, 120000), "the game should end");
            // The agent's moves are in the game's history like any agent's
            Assert.Contains(match.Record.ActionsSinceLastEdit(), action => action.PlayerIndex == 0 && action is not RollAction);
        }

        [Fact]
        public void AnUnknownOpenSeatAgentLeavesTheSeatToTheBuilderWithAnError()
        {
            using Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random, SeatKind.Random], Hexfarer);
            using LocalSimulation admin = new(match);
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = ["nobody", null, null, null] }));
            List<MatchEvent> events = [];
            Assert.True(WaitFor(() => { events.AddRange(admin.ProcessEvents()); return events.OfType<ErrorEvent>().Any(); }, 30000));
            Assert.Contains("nobody", events.OfType<ErrorEvent>().First().Message);
            Assert.IsType<Agents.BuilderAgent>(match.AgentOf(0));
        }

        [Fact]
        public void AnAgentThatCantStartLeavesTheSeatToTheBuilderWithAnError()
        {
            ExternalAgent broken = new("broken", "no-such-program-for-hexfarer");
            using Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random, SeatKind.Random], broken);
            using LocalSimulation admin = new(match);
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = ["broken", null, null, null] }));
            List<MatchEvent> events = [];
            Assert.True(WaitFor(() => { events.AddRange(admin.ProcessEvents()); return events.OfType<ErrorEvent>().Any(); }, 30000));
            Assert.Contains("broken", events.OfType<ErrorEvent>().First().Message);
            Assert.IsType<Agents.BuilderAgent>(match.AgentOf(0));
        }

        [Fact]
        public void AReplacedOrDisposedOpenSeatAgentIsClosed()
        {
            using (Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random], Hexfarer))
            {
                using LocalSimulation admin = new(match);
                admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = [Hexfarer.Name, Hexfarer.Name, null, null] }));
                Assert.True(WaitFor(() => match.AgentOf(0) is OpenSeatAgentProxy && match.AgentOf(1) is OpenSeatAgentProxy, 30000));
                Assert.Equal(2, OpenSeatAgentProcess.Running);
                // Another seat kind: its agent goes
                admin.Send(new SetSeat(0, SeatKind.Random));
                Assert.True(WaitFor(() => OpenSeatAgentProcess.Running == 1, 30000), $"{OpenSeatAgentProcess.Running} agents still run");
            }
            // The match's end closes the rest
            Assert.Equal(0, OpenSeatAgentProcess.Running);
        }

        [Fact]
        public void AnUndoStartsTheAgentsGameAgainAndItPlaysOn()
        {
            using Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random, SeatKind.Random], Hexfarer);
            using LocalSimulation admin = new(match);
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = [Hexfarer.Name, null, null, null] }));
            admin.Send(new SetPlayback(PlaybackState.Default with { PaceSeconds = 0 }));
            Assert.True(WaitFor(() => match.AgentOf(0) is OpenSeatAgentProxy, 30000));
            match.Start();
            Assert.True(WaitFor(() => match.Record.ActionsSinceLastEdit().Count > 40, 60000));
            admin.Send(new SetPlayback(PlaybackState.Default with { Paused = true }));
            Assert.True(WaitFor(() => !match.IsAgentThinking, 30000));
            for (int i = 0; i < 10; i++) admin.Send(new UndoCommand());
            admin.Send(new SetPlayback(PlaybackState.Default with { PaceSeconds = 0 }));
            Assert.True(WaitFor(() => match.Record.State.HasEnded, 120000), "the game should end after the undos");
            Assert.Empty(admin.ProcessEvents().OfType<ErrorEvent>());
        }
    }
}
