using Agents;
using Agents.OpenSeat;
using Common;
using Common.Actions;
using Simulation;
using System.Diagnostics;

namespace Tests
{
    /// <summary>
    /// Open Seat Agents as seats of a simulated match (SeatKind.OpenSeatAgent): the simulation knows its agents from its
    /// own settings (ExternalAgent: a name and the command that starts it), a seat picks one by name in the agent
    /// settings, and the match plays it like any agent, started beside the match; an unknown agent or one that can't start
    /// leaves the seat to the Builder with an error, one that fails during the game pauses the match until it is resumed
    /// (which starts the agent again), a replaced one is closed
    /// </summary>
    public class OpenSeatSeatTests
    {
        private static string TinyNet => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        // Lanzenbrecher's own Open Seat Agent, built next to the tests
        private static ExternalAgent Lanzenbrecher => new("Lanzenbrecher test agent",
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
            using Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random, SeatKind.Random], Lanzenbrecher);
            using LocalSimulation admin = new(match);
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = [Lanzenbrecher.Name, null, null, null] }));
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
            using Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random, SeatKind.Random], Lanzenbrecher);
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
            ExternalAgent broken = new("broken", "no-such-program-for-lanzenbrecher");
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
            using (Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random], Lanzenbrecher))
            {
                using LocalSimulation admin = new(match);
                admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = [Lanzenbrecher.Name, Lanzenbrecher.Name, null, null] }));
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
        public void AFailedAgentPausesTheMatchOnceAndResumingStartsItAgain()
        {
            using Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random, SeatKind.Random], Lanzenbrecher);
            using LocalSimulation admin = new(match);
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = [Lanzenbrecher.Name, null, null, null] }));
            admin.Send(new SetPlayback(PlaybackState.Default with { PaceSeconds = 0 }));
            Assert.True(WaitFor(() => match.AgentOf(0) is OpenSeatAgentProxy, 30000), "the seat should get the agent");
            // The agent's process ends before its first move, as a crash would
            OpenSeatAgentProxy broken = (OpenSeatAgentProxy)match.AgentOf(0);
            broken.Channel.Dispose();
            List<MatchEvent> events = [.. admin.ProcessEvents()];

            match.Start();
            Assert.True(WaitFor(() => { events.AddRange(admin.ProcessEvents()); return match.Playback.Paused; }, 30000), "the match should pause");
            // Paused, the seat isn't asked again: one error, and no move played for it
            Thread.Sleep(1500);
            events.AddRange(admin.ProcessEvents());
            Assert.Contains("seat 1", Assert.Single(events.OfType<ErrorEvent>()).Message);
            Assert.Empty(match.Record.History);

            // Resuming starts the agent again at the position, and the game goes on
            admin.Send(new SetPlayback(PlaybackState.Default with { PaceSeconds = 0 }));
            Assert.True(WaitFor(() => match.Record.History.Count > 8, 60000), "the game should go on");
            Assert.IsType<OpenSeatAgentProxy>(match.AgentOf(0));
            Assert.NotSame(broken, match.AgentOf(0));
            Assert.Empty(admin.ProcessEvents().OfType<ErrorEvent>());
        }

        [Fact]
        public void AnAgentReplacedWhileItThinksFailsWithoutAWord()
        {
            // Lanzenbrecher's agent with 30 s per decision: still thinking when its seat gets another kind
            ExternalAgent slow = new("slow", Lanzenbrecher.Command + " --seconds 30");
            using Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random, SeatKind.Random], slow);
            using LocalSimulation admin = new(match);
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = ["slow", null, null, null] }));
            admin.Send(new SetPlayback(PlaybackState.Default with { PaceSeconds = 0 }));
            Assert.True(WaitFor(() => match.AgentOf(0) is OpenSeatAgentProxy, 30000), "the seat should get the agent");
            match.Start();
            Assert.True(WaitFor(() => match.IsAgentThinking, 30000), "the agent should think about its first move");
            // The agent doesn't read its input while it searches: closing it takes seconds, which the seat change doesn't wait for
            Stopwatch replacing = Stopwatch.StartNew();
            admin.Send(new SetSeat(0, SeatKind.Random));
            Assert.True(replacing.Elapsed < TimeSpan.FromSeconds(2), $"the seat change took {replacing.Elapsed.TotalSeconds:0.0} s");
            // Its search ends with the closed agent: no error, no pause, and the game goes on
            Assert.True(WaitFor(() => match.Record.History.Count > 8, 30000), "the game should go on");
            Assert.False(match.Playback.Paused);
            Assert.Empty(admin.ProcessEvents().OfType<ErrorEvent>());
        }

        [Fact]
        public void AnOpenSeatAgentStartsWithoutHoldingUpTheMatch()
        {
            // The benchmark never answers hello: its seat waits, the match goes on taking commands
            ExternalAgent silent = new("silent", $"dotnet \"{Path.Combine(AppContext.BaseDirectory, "OpenSeat.dll")}\" bench --games 100000");
            using Match match = NewMatch([SeatKind.Random, SeatKind.Random, SeatKind.Random, SeatKind.Random], silent);
            using LocalSimulation admin = new(match);
            Stopwatch watch = Stopwatch.StartNew();
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = ["silent", null, null, null] }));
            admin.Send(new SetSeat(0, SeatKind.OpenSeatAgent));
            admin.Send(new SetPlayback(PlaybackState.Default with { PaceSeconds = 0 }));
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(10), $"the commands took {watch.Elapsed.TotalSeconds:0.0} s");
            // Seat 0 places first: nothing is played while its agent starts
            match.Start();
            Thread.Sleep(1000);
            Assert.Empty(match.Record.History);
            Assert.IsNotType<OpenSeatAgentProxy>(match.AgentOf(0));
            Assert.IsNotType<BuilderAgent>(match.AgentOf(0));
            // The seat isn't asked meanwhile, so nothing fails
            Assert.Empty(admin.ProcessEvents().OfType<ErrorEvent>());
        }

        [Fact]
        public void ANewGameWithItsAgentsStartsTheSeatsAgents()
        {
            using Match match = NewMatch([SeatKind.Random, SeatKind.Random, SeatKind.Random, SeatKind.Random], Lanzenbrecher);
            using LocalSimulation admin = new(match);
            admin.Send(new NewGame([SeatKind.Random, SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random], false,
                Agents: AgentSettings.Default with { OpenSeatAgents = [null, Lanzenbrecher.Name, null, null] }));
            Assert.True(WaitFor(() => match.AgentOf(1) is OpenSeatAgentProxy, 30000), "the seat should get the agent");
            Assert.Equal([null, Lanzenbrecher.Name, null, null], match.AgentSettings.OpenSeatAgents!);
            Assert.Empty(admin.ProcessEvents().OfType<ErrorEvent>());
        }

        [Fact]
        public void TheSeatPickersListTheConfiguredAgentsAfterTheBuiltInKinds()
        {
            ExternalAgent other = new("Catanatron AB:2", "python open_seat_agent.py");
            IReadOnlyList<SeatChoice> choices = Simulation.Seats.Choices([Lanzenbrecher, other]);
            // The built-in kinds first, in their order, then each agent by its name
            Assert.Equal(Simulation.Seats.Ordered, choices.Take(Simulation.Seats.Ordered.Length).Select(c => c.Kind));
            Assert.All(choices.Take(Simulation.Seats.Ordered.Length), c => Assert.Null(c.Agent));
            Assert.Equal([Lanzenbrecher.Name, other.Name], choices.Skip(Simulation.Seats.Ordered.Length).Select(c => c.Agent));
            Assert.All(choices.Skip(Simulation.Seats.Ordered.Length), c => Assert.Equal(SeatKind.OpenSeatAgent, c.Kind));
            Assert.Contains(other.Name, choices[^1].Label);
            // A seat finds its choice by kind and, for an agent, its name; an agent no longer set up finds none
            Assert.Equal(Array.IndexOf(Simulation.Seats.Ordered, SeatKind.Random), Simulation.Seats.ChoiceIndex(choices, SeatKind.Random, null));
            Assert.Equal(choices.Count - 1, Simulation.Seats.ChoiceIndex(choices, SeatKind.OpenSeatAgent, other.Name));
            Assert.Equal(choices.Count - 2, Simulation.Seats.ChoiceIndex(choices, SeatKind.OpenSeatAgent, Lanzenbrecher.Name));
            Assert.Equal(-1, Simulation.Seats.ChoiceIndex(choices, SeatKind.OpenSeatAgent, "gone"));
            // Without agents the pickers are the built-in kinds
            Assert.Equal(Simulation.Seats.Ordered.Length, Simulation.Seats.Choices([]).Count);
        }

        [Fact]
        public void ASeatThatGetsAnAgentStartsOneGameOfIt()
        {
            using Match match = NewMatch([SeatKind.Random, SeatKind.Random, SeatKind.Random, SeatKind.Random], Lanzenbrecher);
            using LocalSimulation admin = new(match);
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = [null, Lanzenbrecher.Name, null, null] }));
            admin.Send(new SetSeat(1, SeatKind.OpenSeatAgent));
            Assert.True(WaitFor(() => match.AgentOf(1) is OpenSeatAgentProxy, 30000));
            // A new agent isn't reset right away, which would end its first game and start another
            OpenSeatAgentProxy proxy = (OpenSeatAgentProxy)match.AgentOf(1);
            Assert.EndsWith("-seat2", proxy.GameId);
            Assert.Empty(admin.ProcessEvents().OfType<ErrorEvent>());
        }

        [Fact]
        public void AnUndoStartsTheAgentsGameAgainAndItPlaysOn()
        {
            using Match match = NewMatch([SeatKind.OpenSeatAgent, SeatKind.Random, SeatKind.Random, SeatKind.Random], Lanzenbrecher);
            using LocalSimulation admin = new(match);
            admin.Send(new ConfigureAgents(AgentSettings.Default with { OpenSeatAgents = [Lanzenbrecher.Name, null, null, null] }));
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
