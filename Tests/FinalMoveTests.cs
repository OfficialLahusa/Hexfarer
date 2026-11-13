using Agents;
using Agents.Inference;
using Agents.MCTS;
using Arena;
using Common;

namespace Tests
{
    /// <summary>
    /// The move played after a search: the most visited move, or with a final move scale the top of the improved policy of
    /// Gumbel MuZero (MCTSTree.ImprovedPolicyChoice, the argmax of the targets DatasetProcessing/policy_targets.py's
    /// gumbel_targets trains on).
    /// </summary>
    public class FinalMoveTests
    {
        private static string PolicyModelPath => TestUtil.FixturePath("ValueNet", "policy_tiny.onnx");

        private static int Choice(float[] priors, uint[] visits, double[] values, double rootValue, double scale, MinMaxStats.Range? range = null) =>
            MCTSTree.ImprovedPolicyChoice(priors, visits, values, rootValue, scale, range);

        [Fact]
        public void TheImprovedPolicyPrefersTheBetterValueOverTheMoreVisitedMove()
        {
            // log prior + (50 + most visits) * 0.1 * value: -0.69 + 4.4 against -1.20 + 5.5 and -1.61 + 3.3
            Assert.Equal(1, Choice([0.5f, 0.3f, 0.2f], [60, 30, 10], [0.40, 0.50, 0.30], 0.42, 0.1));
        }

        [Fact]
        public void TheImprovedPolicyKeepsThePriorsLogarithm()
        {
            // Equal values: the highest prior wins, not the most visited move
            Assert.Equal(1, Choice([0.2f, 0.5f, 0.3f], [70, 20, 10], [0.5, 0.5, 0.5], 0.5, 0.1));
        }

        [Fact]
        public void TheValueWeightGrowsWithTheMostVisitsNotTheTotal()
        {
            // The better value is worth 0.1 * (50 + 60) * 0.1 = 1.1, the better prior log 4 = 1.39: the prior wins. With the
            // total (50 + 120) the value would (1.7)
            Assert.Equal(1, Choice([0.2f, 0.8f], [60, 60], [0.6, 0.5], 0.55, 0.1));
            // From 89 visits on the value is worth more
            Assert.Equal(0, Choice([0.2f, 0.8f], [90, 90], [0.6, 0.5], 0.55, 0.1));
        }

        [Fact]
        public void AnUnvisitedMoveCountsWithTheMixedValue()
        {
            // v_mix = (root + N * prior-weighted mean of the visited values) / (1 + N): (0.5 + 100 * 0.4125) / 101 = 0.413,
            // so the unvisited move's -0.22 + 4.13 stays below -3.00 + 7.5. With the plain mean (0.525) it would win (5.02)
            Assert.Equal(0, Choice([0.05f, 0.15f, 0.8f], [50, 50, 0], [0.75, 0.3, double.NaN], 0.5, 0.1));
            // With an unvisited move worth 0 instead it could never win: here it does with v_mix 0.5
            Assert.Equal(2, Choice([0.1f, 0.1f, 0.8f], [50, 50, 0], [0.6, 0.4, double.NaN], 0.5, 0.1));
        }

        [Fact]
        public void TheRootValueEntersTheMixedValue()
        {
            // Three visits: the root weighs a quarter, v_mix = (1 + 3 * 0.5) / 4 = 0.625, and -1.20 + 5.2 * 0.625 = 2.05 beats
            // -0.69 + 5.2 * 0.5 = 1.91. Without the root (0.5) the unvisited move would get 1.40
            Assert.Equal(2, Choice([0.5f, 0.2f, 0.3f], [2, 1, 0], [0.5, 0.5, double.NaN], 1.0, 0.1));
        }

        [Fact]
        public void TheValuesAreNormalizedToTheTreesRange()
        {
            // Raw, 0.05 more value is worth 6 * 0.05 = 0.3, less than the prior's log(7/3) = 0.85; normalized to [0.2, 0.4]
            // it is 0.25 more, worth 1.5
            Assert.Equal(0, Choice([0.7f, 0.3f], [10, 10], [0.30, 0.35], 0.32, 0.1));
            Assert.Equal(1, Choice([0.7f, 0.3f], [10, 10], [0.30, 0.35], 0.32, 0.1, new MinMaxStats.Range(0.2, 0.4)));
            // Clipped to [0, 1] after the mixing: a value above the range counts as 1, so the first move's 1.2 (clipped to 1)
            // ties the second's 1 in value and the prior decides
            Assert.Equal(0, Choice([0.6f, 0.4f], [10, 10], [0.44, 0.40], 0.40, 0.1, new MinMaxStats.Range(0.2, 0.4)));
            Assert.Equal(1, Choice([0.4f, 0.6f], [10, 10], [0.44, 0.40], 0.40, 0.1, new MinMaxStats.Range(0.2, 0.4)));
        }

        [Fact]
        public void ASingleVisitedMoveIsPlayed()
        {
            // As in gumbel_targets: the search's shortcut for a winning move visits only that move
            Assert.Equal(1, Choice([0.9f, 0.1f], [0, 5], [double.NaN, 0.1], 0.1, 0.1));
        }

        [Fact]
        public void AnInvalidChoiceThrows()
        {
            Assert.Throws<ArgumentException>(() => Choice([0.5f, 0.5f], [0, 0], [double.NaN, double.NaN], 0.5, 0.1));
            Assert.Throws<ArgumentException>(() => Choice([0.5f, 0.5f], [1, 1], [0.5], 0.5, 0.1));
            Assert.Throws<ArgumentOutOfRangeException>(() => Choice([0.5f, 0.5f], [1, 1], [0.5, 0.5], 0.5, -0.1));
        }

        [Fact]
        public void TheTreePlaysTheImprovedPolicysTopForTheDecidersValues()
        {
            // Real searches: the tree's choice with a final move scale is ImprovedPolicyChoice over its root's priors, visits,
            // the decider's mean values and root value, normalized to the decider's range in the tree. A long search tells
            // the values apart, a short one (many unvisited moves, the root a large part of v_mix) the root value
            StateValueNet net = new(PolicyModelPath, provider: InferenceProvider.CPU);
            bool anotherPlayerDiffers = false, rawDiffers = false, flippedRootDiffers = false, differsFromVisits = false;
            foreach ((int iterations, int batch) in new[] { (600, 8), (6, 2) })
            {
                // The first initial placement: a move per free intersection
                TestUtil.Seed(4129);
                GameState state = TestUtil.NewGame();
                int decider = state.Turn.PlayerIndex;
                var search = new MCTSSearch(new DirectEvaluator(net), new StateVectorizer(), batch, parallelSelection: false);
                MCTSTree tree = new(new GameState(state), (sbyte)decider, 1.414, model: OpponentModel.MaxN);
                search.Run(tree, 0, iterations);

                MCTSNode[] children = [.. tree.RootNode.Children];
                Assert.True(children.Length > 20);
                float[] priors = children.Select(child => child.Prior).ToArray();
                uint[] visits = children.Select(child => child.VisitCount).ToArray();
                int Expected(int player, double scale, bool normalize, bool flipRoot = false)
                {
                    double[] values = children.Select(child => child.ResolvedVisitCount == 0 ? double.NaN : child.PlayerScore(player) / child.ResolvedVisitCount).ToArray();
                    double root = tree.RootNode.PlayerScore(player) / tree.RootNode.ResolvedVisitCount;
                    if (flipRoot) root = 1 - root;
                    return MCTSTree.ImprovedPolicyChoice(priors, visits, values, root, scale, normalize ? tree.PlayerBounds![player].GetRange() : null);
                }
                int mostVisited = Array.IndexOf(visits, visits.Max());

                // Without a scale the most visited move
                Assert.True(tree.GetMostPromisingMove().IsSameDecision(children[mostVisited].LastAction!));
                foreach (double scale in new[] { 0.05, 0.1, 0.3, 1, 3, 10 })
                {
                    tree.FinalMoveScale = scale;
                    int expected = Expected(decider, scale, normalize: true);
                    Assert.True(tree.GetMostPromisingMove().IsSameDecision(children[expected].LastAction!), $"{iterations} iterations, scale {scale}");
                    anotherPlayerDiffers |= Expected((decider + 1) % 4, scale, normalize: true) != expected;
                    rawDiffers |= Expected(decider, scale, normalize: false) != expected;
                    flippedRootDiffers |= Expected(decider, scale, normalize: true, flipRoot: true) != expected;
                    differsFromVisits |= expected != mostVisited;
                }

                tree.FinalMoveScale = 0;
                Assert.True(tree.GetMostPromisingMove().IsSameDecision(children[mostVisited].LastAction!));
                Assert.Throws<ArgumentOutOfRangeException>(() => tree.FinalMoveScale = -1);
            }
            // The positions tell these apart, so a choice from another seat's or unnormalized values, or with the root value
            // turned around, would have failed above
            Assert.True(anotherPlayerDiffers);
            Assert.True(rawDiffers);
            Assert.True(flippedRootDiffers);
            Assert.True(differsFromVisits);
        }

        [Fact]
        public void AnAgentSpecReadsAFinalMoveScaleBesideAFirstPlayReduction()
        {
            // fq is read before the one-letter f (first play reduction); 0, the most visited move, is the default and not shown
            AgentSpec spec = AgentSpec.Parse("T=a.onnx=fq0.1=f0.3");
            Assert.Equal(0.1, spec.FinalMoveScale);
            Assert.Equal(0.3, spec.FirstPlayReduction);
            Assert.Equal("T a.onnx MaxN f0.3 fq0.1", spec.ToString());
            Assert.Equal(0, AgentSpec.Parse("T=a.onnx").FinalMoveScale);
            Assert.Throws<ArgumentException>(() => AgentSpec.Parse("T=a.onnx=fq0"));

            // The agent it creates plays with it, and its tree gets it at the next decision
            StateValueNet net = new(PolicyModelPath, provider: InferenceProvider.CPU);
            TestUtil.Seed(4130);
            GameState state = TestUtil.NewGame();
            var agent = (MCTSAgent)AgentSpec.Parse($"T={PolicyModelPath}=fq0.1").Create((sbyte)state.Turn.PlayerIndex, state, net, 64, 0);
            Assert.Equal(0.1, agent.FinalMoveScale);
            agent.Act(state);
            (_, IReadOnlyList<MCTSAgent.RootMove> moves) = agent.GetRootMoves();
            Assert.NotEmpty(moves);
            Assert.Equal(0.1, agent.CurrentTree!.FinalMoveScale);

            // Not under hidden information, whose agents pick their moves otherwise
            Assert.Throws<ArgumentException>(() => AgentSpec.Parse($"T={PolicyModelPath}=fq0.1=d4").Create(0, state, net, 64, 0));
            Assert.Throws<ArgumentException>(() => AgentSpec.Parse($"T={PolicyModelPath}=fq0.1=is").Create(0, state, net, 64, 0));
        }
    }
}
