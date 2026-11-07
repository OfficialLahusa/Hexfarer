using Agents.Inference;
using Agents.MCTS;
using Common;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// A seeded search: with selection on one thread the same seed searches the same tree, so a game can be played again
    /// from its seed; with selection on all cores the tree keeps the thread-safe shared random source (the threads' order
    /// decides there anyway)
    /// </summary>
    public class SearchSeedTests
    {
        private static StateValueNet Net() => new(TestUtil.FixturePath("ValueNet", "placements_tiny.onnx"), provider: InferenceProvider.CPU);

        // Positions of a seeded random game where the player acting has a choice
        private static List<GameState> Positions(int seed, int count)
        {
            TestUtil.Seed(seed);
            GameState state = TestUtil.NewGame();
            Agents.Agent[] agents = TestUtil.RandomAgents();
            List<GameState> positions = [];
            for (int move = 0; positions.Count < count && !state.HasEnded && move < 2000; move++)
            {
                if (move % 25 == 10 && MCTSTree.GetValidActions(state).Count > 1) positions.Add(new GameState(state));
                Action action = agents[TestUtil.ActingPlayer(state)].Act(state);
                action.Apply(state);
            }
            return positions;
        }

        private static (MCTSAgent agent, uint[] visits) Search(StateValueNet net, GameState position, bool parallel, int? seed)
        {
            MCTSAgent agent = new(TestUtil.ActingPlayer(position), new GameState(position), valueNet: net, iterationLimit: 200, leavesPerBatch: 16,
                parallelSelection: parallel, seed: seed);
            agent.Act(new GameState(position));
            return (agent, agent.GetRootMoves().moves.Select(move => move.Visits).ToArray());
        }

        [Fact]
        public void TheSameSeedSearchesTheSameTreeOnOneThreadAndAnotherSeedAnotherTree()
        {
            StateValueNet net = Net();
            List<GameState> positions = Positions(31, 8);
            Assert.Equal(8, positions.Count);
            int differ = 0;
            foreach (GameState position in positions)
            {
                uint[] first = Search(net, position, parallel: false, seed: 5).visits;
                Assert.Equal(first, Search(net, position, parallel: false, seed: 5).visits);
                if (!first.SequenceEqual(Search(net, position, parallel: false, seed: 6).visits)) differ++;
            }
            // The seed reaches the search: another one changes the tree somewhere
            Assert.True(differ > 0, "every position searched the same tree with another seed");
        }

        [Fact]
        public void ASeededAgentGivesEachNewTreeItsOwnSourceAndKeepsTheSharedOneOnAllCores()
        {
            StateValueNet net = Net();
            GameState position = Positions(32, 1)[0];
            // On one thread: a source of the tree's own, drawn from the agent's seed (a new one for the next tree)
            MCTSAgent serial = Search(net, position, parallel: false, seed: 5).agent;
            Random first = serial.Tree!.Random;
            Assert.NotSame(Random.Shared, first);
            serial.ResetToState(new GameState(position));
            serial.Act(new GameState(position));
            Assert.NotSame(first, serial.Tree!.Random);
            // On all cores, and without a seed: the shared source (many threads select at once)
            Assert.Same(Random.Shared, Search(net, position, parallel: true, seed: 5).agent.Tree!.Random);
            Assert.Same(Random.Shared, Search(net, position, parallel: false, seed: null).agent.Tree!.Random);
        }

        [Fact]
        public void ASeededSearchOnAllCoresStaysValid()
        {
            // The shared source is safe from many threads: the searches choose legal moves and spread their visits
            StateValueNet net = Net();
            foreach (GameState position in Positions(33, 4))
            {
                (MCTSAgent agent, uint[] visits) = Search(net, position, parallel: true, seed: 5);
                Assert.True(visits.Sum(v => v) >= 200);
                Assert.True(visits.Count(v => v > 0) > 1);
                Assert.True(agent.Act(new GameState(position)).IsValidFor(position));
            }
        }
    }
}
