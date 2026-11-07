using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// A search with a budget of a batch or two still ends with a move: it goes on until the root has its moves (a new leaf
    /// is evaluated before it is expanded, so the first batch alone never does), and a search cancelled before that falls back
    /// on the first legal move
    /// </summary>
    public class SmallBudgetTests
    {
        private static readonly Lazy<StateValueNet> Net = new(() =>
            new StateValueNet(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), provider: InferenceProvider.CPU));

        /// <summary>
        /// Positions of a random game where the acting player has a choice, the first placement among them
        /// </summary>
        private static List<GameState> Choices()
        {
            TestUtil.Seed(21);
            GameState state = TestUtil.NewGame();
            Agent[] agents = TestUtil.RandomAgents();
            List<GameState> positions = [];
            int last = -100;
            for (int moves = 0; positions.Count < 8 && !state.HasEnded && moves < 3000; moves++)
            {
                if (moves - last >= 20 && MCTSTree.GetValidActions(state).Count > 1)
                {
                    positions.Add(new GameState(state));
                    last = moves;
                }
                agents[TestUtil.ActingPlayer(state)].Act(state).Apply(state);
            }
            Assert.True(positions.Count == 8, $"only {positions.Count} positions");
            return positions;
        }

        // One leaf per batch: the first batch can only evaluate the root, which is expanded when it is selected again (with
        // larger batches and parallel selection that happens within the first batch or not, depending on the threads)
        private static MCTSAgent Search(GameState state, long iterations, double seconds = 5) =>
            new MCTSAgent(TestUtil.ActingPlayer(state), state, valueNet: Net.Value, timeLimitSeconds: seconds, iterationLimit: iterations, leavesPerBatch: 1);

        [Theory]
        [InlineData("mcts")]
        [InlineData("ismcts")]
        [InlineData("pimc")]
        public void TheSmallestIterationBudgetEndsWithALegalMoveFromAnExpandedRoot(string kind)
        {
            foreach (GameState position in Choices())
            {
                // One iteration, one batch: before the fix the root was only evaluated, not expanded
                MCTSAgent search = Search(position, iterations: 1);
                Agent agent = kind switch
                {
                    "ismcts" => new InformationSetMCTSAgent(search, position, seed: 1),
                    "pimc" => new DeterminizedMCTSAgent(search, position, 2, seed: 1),
                    _ => search,
                };
                Action move = agent.Act(new GameState(position));
                Assert.True(move.IsValidFor(position), $"{kind} played the illegal {move}");

                // The move came from a search that expanded the root (PIMC drops its trees after deciding)
                if (agent is DeterminizedMCTSAgent) continue;
                int rootMoves = agent is InformationSetMCTSAgent ismcts ? ismcts.LastTree!.RootNode.Children.Count : search.GetRootMoves().moves.Count;
                Assert.True(rootMoves > 0, $"{kind}: the root has no moves");
            }
        }

        [Fact]
        public void AnAlmostZeroTimeBudgetEndsWithALegalMoveFromAnExpandedRoot()
        {
            foreach (GameState position in Choices())
            {
                MCTSAgent search = Search(position, iterations: 0, seconds: 1e-9);
                Action move = search.Act(new GameState(position));
                Assert.True(move.IsValidFor(position), $"illegal {move}");
                Assert.NotEmpty(search.GetRootMoves().moves);
            }
        }

        [Fact]
        public void ATreeWithoutMovesPicksTheFirstLegalMove()
        {
            foreach (GameState position in Choices())
            {
                sbyte player = TestUtil.ActingPlayer(position);
                string first = MCTSTree.GetValidActions(position)[0].ToString();

                MCTSTree tree = new MCTSTree(new GameState(position), player, 1.414);
                Assert.Equal(first, tree.GetMostPromisingMove().ToString());

                ISMCTSTree informed = new ISMCTSTree(new GameState(position), new BeliefTracker(player, position), 1.414, 1, false, null);
                Assert.Equal(first, informed.GetMostPromisingMove().ToString());
                List<Action> legal = MCTSTree.GetValidActions(position);
                Assert.Equal(legal[^1].ToString(), informed.GetMostPromisingMove([legal[^1]]).ToString());
            }
        }

        [Fact]
        public void ASearchCancelledBeforeItStartsStillPlaysALegalMove()
        {
            using CancellationTokenSource cancelled = new();
            cancelled.Cancel();
            foreach (GameState position in Choices())
            {
                MCTSAgent search = Search(position, iterations: 64);
                Action move = search.Act(new GameState(position), cancelled.Token);
                Assert.True(move.IsValidFor(position), $"illegal {move}");

                InformationSetMCTSAgent ismcts = new InformationSetMCTSAgent(Search(position, iterations: 64), position, seed: 1);
                Action informed = ismcts.Act(new GameState(position), cancelled.Token);
                Assert.True(informed.IsValidFor(position), $"illegal {informed}");
            }
        }
    }
}
