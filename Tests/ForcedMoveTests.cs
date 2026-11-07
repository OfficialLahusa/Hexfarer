using Agents.Inference;
using Agents.MCTS;
using Common;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// A decision with a single legal move (a roll without a playable development card, the end of a turn with nothing
    /// affordable) is played at once by the open-hand MCTS and by PIMC, unless the old search is asked for; a decision with
    /// two moves is still searched
    /// </summary>
    public class ForcedMoveTests
    {
        private static readonly Lazy<StateValueNet> Net = new(() =>
            new StateValueNet(TestUtil.FixturePath("ValueNet", "plain_tiny.onnx"), provider: InferenceProvider.CPU));

        private static MCTSAgent NewSearch(GameState state) =>
            new MCTSAgent(0, state, valueNet: Net.Value, iterationLimit: 64, leavesPerBatch: 16, parallelSelection: false);

        public static IEnumerable<object[]> ForcedPositions()
        {
            yield return ["roll"];
            yield return ["end of turn"];
        }

        private static GameState Position(string name)
        {
            TestUtil.Seed(5);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: name == "roll");
            if (name == "two moves")
            {
                // Exactly the cards for a development card: buying one or ending the turn
                TestUtil.GiveCards(state, 0, ResourceCardType.Ore, 1);
                TestUtil.GiveCards(state, 0, ResourceCardType.Wool, 1);
                TestUtil.GiveCards(state, 0, ResourceCardType.Grain, 1);
            }
            return state;
        }

        private static long RootVisits(MCTSAgent search) => search.GetRootMoves().moves.Sum(move => (long)move.Visits);

        // Iterations of the last search, which PIMC's reset of the inner search after each decision leaves alone; 0 before the first
        private static long Searched(MCTSAgent search) => search.LastSearch?.Iterations ?? 0;

        [Theory]
        [MemberData(nameof(ForcedPositions))]
        public void TheOnlyLegalMoveIsPlayedWithoutASearch(string position)
        {
            GameState state = Position(position);
            Action only = Assert.Single(MCTSTree.GetValidActions(state));
            MCTSAgent search = NewSearch(state);

            Action played = search.Act(state);

            Assert.True(played.IsSameDecision(only));
            Assert.Equal(0, Searched(search));
            Assert.Null(search.GetRootMoves().state);
        }

        [Theory]
        [MemberData(nameof(ForcedPositions))]
        public void TheOldSearchStillSearchesTheOnlyLegalMove(string position)
        {
            GameState state = Position(position);
            Action only = Assert.Single(MCTSTree.GetValidActions(state));
            MCTSAgent search = NewSearch(state);
            search.PlayForcedMovesAtOnce = false;

            Action played = search.Act(state);

            Assert.True(played.IsSameDecision(only));
            Assert.True(Searched(search) > 0);
            Assert.Equal(state, search.GetRootMoves().state);
            Assert.True(RootVisits(search) > 0);
        }

        [Fact]
        public void ADecisionWithTwoMovesIsSearched()
        {
            GameState state = Position("two moves");
            Assert.Equal(2, MCTSTree.GetValidActions(state).Count);
            MCTSAgent search = NewSearch(state);

            search.Act(state);

            Assert.True(Searched(search) > 0);
            Assert.Equal(state, search.GetRootMoves().state);
            Assert.Equal(2, search.GetRootMoves().moves.Count);
            Assert.True(RootVisits(search) > 0);
        }

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void PIMCPlaysTheOnlyLegalMoveAtOnceUnlessTheOldSearchIsAskedFor(bool atOnce)
        {
            GameState state = Position("roll");
            Action only = Assert.Single(MCTSTree.GetValidActions(state));
            MCTSAgent search = NewSearch(state);
            search.PlayForcedMovesAtOnce = atOnce;
            DeterminizedMCTSAgent pimc = new DeterminizedMCTSAgent(search, state, determinizations: 4, seed: 1);

            Action played = pimc.Act(state);

            Assert.True(played.IsSameDecision(only));
            if (atOnce) Assert.Equal(0, Searched(search));
            else Assert.True(Searched(search) > 0);
        }
    }
}
