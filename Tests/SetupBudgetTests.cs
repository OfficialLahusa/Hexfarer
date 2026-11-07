using Agents;
using Agents.Inference;
using Agents.MCTS;
using Arena;
using Common;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The setup budget (MCTSAgent.SetupBudgetFactor, Arena option suX): the initial placements search with a multiple of the
    /// budget, every other decision with the plain one, in the open-hand, information set and PIMC searches alike
    /// </summary>
    public class SetupBudgetTests
    {
        private static readonly Lazy<StateValueNet> Net = new(() =>
            new StateValueNet(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), provider: InferenceProvider.CPU));

        private static string ModelPath => TestUtil.FixturePath("ValueNet", "plain_tiny.onnx");

        /// <summary>
        /// Positions of a random game where the acting player has a choice: in the first and the second round of the
        /// initial placements, and the first one after them
        /// </summary>
        private static (GameState first, GameState second, GameState normal) Positions()
        {
            TestUtil.Seed(5);
            GameState state = TestUtil.NewGame();
            Agent[] agents = TestUtil.RandomAgents();
            GameState? first = null, second = null, normal = null;
            for (int moves = 0; normal == null && !state.HasEnded && moves < 3000; moves++)
            {
                if (MCTSTree.GetValidActions(state).Count > 1)
                {
                    switch (state.Turn.TypeOfRound)
                    {
                        case TurnState.RoundType.FirstInitial: first ??= new GameState(state); break;
                        case TurnState.RoundType.SecondInitial: second ??= new GameState(state); break;
                        case TurnState.RoundType.Normal: normal = new GameState(state); break;
                    }
                }
                agents[TestUtil.ActingPlayer(state)].Act(state).Apply(state);
            }
            Assert.True(first != null && second != null && normal != null, "missing a position");
            return (first!, second!, normal!);
        }

        [Fact]
        public void OnlyTheInitialPlacementsGetTheLargerBudget()
        {
            (GameState first, GameState second, GameState normal) = Positions();
            Assert.True(MCTSAgent.IsSetup(first) && MCTSAgent.IsSetup(second) && !MCTSAgent.IsSetup(normal));

            MCTSAgent agent = new MCTSAgent(0, first, valueNet: Net.Value, timeLimitSeconds: 0.5, iterationLimit: 100) { SetupBudgetFactor = 2.5 };
            Assert.Equal((1.25, 250L), agent.BudgetFor(first));
            Assert.Equal((1.25, 250L), agent.BudgetFor(second));
            Assert.Equal((0.5, 100L), agent.BudgetFor(normal));

            // Iterations round up; a search for time keeps 0 iterations
            agent.IterationLimit = 3;
            Assert.Equal(8L, agent.BudgetFor(first).iterations);
            agent.IterationLimit = 0;
            Assert.Equal((1.25, 0L), agent.BudgetFor(second));

            // By default every decision has the same budget
            MCTSAgent plain = new MCTSAgent(0, first, valueNet: Net.Value, timeLimitSeconds: 0.5, iterationLimit: 100);
            Assert.Equal(1, plain.SetupBudgetFactor);
            Assert.Equal((0.5, 100L), plain.BudgetFor(first));
            Assert.Throws<ArgumentOutOfRangeException>(() => plain.SetupBudgetFactor = 0);
        }

        [Theory]
        [InlineData("mcts")]
        [InlineData("ismcts")]
        [InlineData("pimc")]
        public void EverySearchSpendsTheLargerBudgetOnlyOnTheInitialPlacements(string kind)
        {
            (GameState first, GameState second, GameState normal) = Positions();
            foreach ((GameState position, bool setup) in new[] { (first, true), (second, true), (normal, false) })
            {
                sbyte player = TestUtil.ActingPlayer(position);
                // Enough iterations that no search has to run on until its root has moves (see MCTSSearch.MaxBatchesForRoot)
                MCTSAgent search = new MCTSAgent(player, position, valueNet: Net.Value, iterationLimit: 256, leavesPerBatch: 8) { SetupBudgetFactor = 4 };
                Agent agent = kind switch
                {
                    "ismcts" => new InformationSetMCTSAgent(search, position, seed: 1),
                    "pimc" => new DeterminizedMCTSAgent(search, position, 2, seed: 1),
                    _ => search,
                };
                // PIMC splits its iterations between its worlds (one while nothing is hidden); the statistics are those of
                // the last world's search
                long perSearch = 256 / (agent is DeterminizedMCTSAgent pimc ? pimc.Determinize(position).Count : 1);
                Action move = agent.Act(new GameState(position));
                Assert.True(move.IsValidFor(position), $"{kind} played the illegal {move}");

                long iterations = search.LastSearch!.Iterations;
                if (setup) Assert.True(iterations >= 4 * perSearch, $"{kind}: {iterations} iterations in the setup, expected {4 * perSearch}");
                else Assert.True(iterations < 2 * perSearch, $"{kind}: {iterations} iterations after the setup, expected {perSearch}");
            }
        }

        [Fact]
        public void TheArenaOptionReachesTheSearchOfEveryAgent()
        {
            AgentSpec plain = AgentSpec.Parse($"T={ModelPath}");
            AgentSpec larger = AgentSpec.Parse($"T={ModelPath}=su4");
            Assert.Equal((1.0, 4.0), (plain.SetupBudget, larger.SetupBudget));
            Assert.DoesNotContain("su", plain.ToString());
            Assert.EndsWith("su4", larger.ToString());
            Assert.Equal(2.5, AgentSpec.Parse($"T={ModelPath}=su2.5").SetupBudget);
            Assert.Throws<ArgumentException>(() => AgentSpec.Parse($"T={ModelPath}=su0"));

            GameState start = TestUtil.NewGame();
            foreach (string kind in new[] { "is", "d2", "c1.414" })
            {
                foreach ((string option, double factor) in new[] { ("su4", 4.0), ("c1.414", 1.0) })
                {
                    Agent agent = AgentSpec.Parse($"T={ModelPath}={kind}={option}").Create(1, start, Net.Value, 16, 0);
                    MCTSAgent search = agent switch { InformationSetMCTSAgent i => i.Search, DeterminizedMCTSAgent d => d.Search, _ => (MCTSAgent)agent };
                    Assert.Equal(factor, search.SetupBudgetFactor);
                }
            }
        }
    }
}
