using Agents;
using Agents.MCTS;
using Common;
using Common.Actions;
using Simulation;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The fair seats (Fair MCTS Easy/Medium/Hard): agents that play under hidden information (InformationSetMCTSAgent) with the
    /// presets' thinking times, chosen separately from the default seats. The match keeps their beliefs as it keeps its own,
    /// through undo, redo, loading and seat changes.
    /// </summary>
    public class FairSeatTests
    {
        private static readonly SessionInfo Admin = new("admin", "Admin", Role.Admin, null);

        [Fact]
        public void FairSeatsHaveTheirNamesAndTimes()
        {
            Assert.Equal("MCTS Easy", Seats.Name(SeatKind.PIMCEasy));
            Assert.Equal("MCTS Medium", Seats.Name(SeatKind.PIMCMedium));
            Assert.Equal("MCTS Hard", Seats.Name(SeatKind.PIMCHard));
            Assert.Equal("Open-hand MCTS Hard", Seats.Name(SeatKind.MCTSHard));
            foreach ((SeatKind fair, SeatKind plain) in new[] { (SeatKind.PIMCEasy, SeatKind.MCTSEasy), (SeatKind.PIMCMedium, SeatKind.MCTSMedium), (SeatKind.PIMCHard, SeatKind.MCTSHard) })
            {
                Assert.True(Seats.NeedsValueNet(fair));
                Assert.Equal(Seats.PresetSeconds(plain), Seats.PresetSeconds(fair));
            }

            // The presets are the fair ones; the open-hand ones are asked for by name
            Assert.Equal(SeatKind.PIMCHard, ServerConsole.ParseKind("hard"));
            Assert.Equal(SeatKind.PIMCHard, ServerConsole.ParseKind("fair-hard"));
            Assert.Equal(SeatKind.MCTSHard, ServerConsole.ParseKind("open-hand-hard"));
            Assert.Equal(SeatKind.PIMCEasy, ServerConsole.ParseKind("PIMCEasy"));

            // New games get the hard fair preset, the builder without the value net
            Assert.Equal(SeatKind.PIMCHard, Seats.DefaultBot(valueNetAvailable: true));
            Assert.Equal(SeatKind.Builder, Seats.DefaultBot(valueNetAvailable: false));
        }

        [SkippableFact]
        public void AFairSeatsAgentPlaysUnderHiddenInformation()
        {
            TestUtil.SkipWithoutModel();
            GameState state = TestUtil.NewGame();
            InformationSetMCTSAgent hard = Assert.IsType<InformationSetMCTSAgent>(Seats.CreateAgent(SeatKind.PIMCHard, 2, state, AgentSettings.Default, true));
            Assert.Equal(2, hard.PlayerIndex);
            Assert.False(hard.FullKnowledge);
            Assert.Equal(2, hard.Search.TimeLimitSeconds);

            InformationSetMCTSAgent easy = Assert.IsType<InformationSetMCTSAgent>(Seats.CreateAgent(SeatKind.PIMCEasy, 1, state, new AgentSettings(null, 500, OpponentModel.MaxN), true));
            Assert.Equal(500, easy.Search.IterationLimit);

            // Without the net they fall back to the builder, like the other seats that need it
            Assert.IsType<BuilderAgent>(Seats.CreateAgent(SeatKind.PIMCMedium, 0, state, AgentSettings.Default, false));
        }

        [SkippableFact]
        public void FairSeatsPlayAWholeMatchAgainstEachOther()
        {
            // The client's path: the match runs the four fair agents with its runner until the game ends
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(43);
            SeatKind[] fair = [SeatKind.PIMCEasy, SeatKind.PIMCMedium, SeatKind.PIMCHard, SeatKind.PIMCEasy];
            using Match match = new Match(TestUtil.NewGame(), fair, valueNetAvailable: true);
            match.Handle(Admin, new ConfigureAgents(new AgentSettings(null, 256, OpponentModel.MaxN)));
            match.Handle(Admin, new SetPlayback(new PlaybackState(false, false, 0)));
            match.Start();

            System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
            while (!match.Record.State.HasEnded && watch.Elapsed < TimeSpan.FromMinutes(5)) Thread.Sleep(100);
            Assert.True(match.Record.State.HasEnded, "the fair seats didn't finish the game");
            for (int seat = 0; seat < 4; seat++)
                Assert.IsType<InformationSetMCTSAgent>(match.AgentOf(seat));
        }

        [SkippableFact]
        public void TheMatchGivesFairSeatsItsBeliefsThroughUndoLoadAndSeatChanges()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(41);
            using Match match = new Match(TestUtil.NewGame(), [SeatKind.PIMCMedium, SeatKind.Human, SeatKind.PIMCEasy, SeatKind.Human], valueNetAvailable: true);
            Random random = new Random(41);

            void AssertBeliefs()
            {
                for (int seat = 0; seat < 4; seat++)
                {
                    if (match.AgentOf(seat) is InformationSetMCTSAgent fair)
                        Assert.Equal(match.Beliefs.For(seat), fair.Beliefs);
                }
            }

            for (int i = 0; i < 600 && !match.Record.State.HasEnded; i++)
            {
                double r = random.NextDouble();
                if (r < 0.08) match.Handle(Admin, new UndoCommand());
                else if (r < 0.12) match.Handle(Admin, new RedoCommand());
                else
                {
                    GameState state = match.Record.State;
                    List<Action> legal = LegalActionProvider.GetActionsForState(state, TestUtil.ActingPlayer(state));
                    match.Handle(Admin, new PlayMove(legal[random.Next(legal.Count)]));
                }
                AssertBeliefs();
            }

            // A seat that turns fair in the middle of the game knows only what the seat could know
            match.Handle(Admin, new SetSeat(1, SeatKind.PIMCHard));
            Assert.IsType<InformationSetMCTSAgent>(match.AgentOf(1));
            AssertBeliefs();

            // Loading a game: the beliefs follow its moves from its start
            match.Handle(Admin, new LoadGame(new GameState(match.Record.State), match.Record.ActionsSinceLastEdit(), []));
            AssertBeliefs();
        }
    }
}
