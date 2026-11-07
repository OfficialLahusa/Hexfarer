using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Common.Actions;
using Microsoft.ML.OnnxRuntime.Tensors;
using static Tests.HiddenInformationTests;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Information set MCTS (ISMCTSTree, single observer): one tree for the searching seat whose nodes are what it can tell
    /// apart, a world drawn from its beliefs for every iteration. Moves are keyed by what the seat sees, a child counts as
    /// available only in the worlds where it is legal, and the search depends on nothing the seat can't see.
    /// </summary>
    public class ISMCTSTests
    {
        /// <summary>
        /// A value net stand-in that gives every position the same deterministic utilities from its features, so searches
        /// repeat exactly
        /// </summary>
        private sealed class FixedEvaluator : ILeafEvaluator
        {
            public bool HasPlacements => false;
            public PolicyHead? Policy => null;

            public Task<LeafValues> EvaluateAsync(DenseTensor<float> features, bool withPlacements, bool withPolicy = false, LeafValues? destination = null)
            {
                int rows = features.Dimensions[0], width = features.Dimensions[1];
                float[] probabilities = new float[rows * 4];
                ReadOnlySpan<float> buffer = features.Buffer.Span;
                for (int row = 0; row < rows; row++)
                {
                    double sum = 0;
                    for (int f = 0; f < width; f += 7) sum += buffer[row * width + f] * (f % 13);
                    for (int p = 0; p < 4; p++) probabilities[row * 4 + p] = (float)(1 + Math.Abs(Math.Sin(sum * (p + 1))));
                }
                return Task.FromResult(new LeafValues(probabilities, null));
            }
        }

        private static ISMCTSTree Search(GameState truth, BeliefTracker beliefs, int iterations = 3000, int seed = 5)
        {
            ISMCTSTree tree = new ISMCTSTree(truth, beliefs, seed: seed);
            MCTSSearch search = new MCTSSearch(new FixedEvaluator(), new ViewVectorizer { Observer = beliefs.Observer }, 16, parallelSelection: false);
            search.Run(tree, double.PositiveInfinity, iterations);
            return tree;
        }

        /// <summary>
        /// Positions of a few random games where the given seat is about to decide, with its beliefs
        /// </summary>
        private static IEnumerable<(GameState truth, BeliefTracker beliefs)> Decisions(int seat, int count)
        {
            int found = 0;
            for (int seed = 1; found < count && seed < 50; seed++)
            {
                Followed game = new Followed(NewGame(seed));
                Agent[] agents = TestUtil.RandomAgents();
                while (!game.Truth.HasEnded && found < count && game.Played.Count < 3000)
                {
                    sbyte acting = TestUtil.ActingPlayer(game.Truth);
                    if (acting == seat && game.Played.Count > 60 && game.Played.Count % 23 == 0 && MCTSTree.GetValidActions(game.Truth).Count > 1)
                    {
                        found++;
                        yield return (new GameState(game.Truth), new BeliefTracker(game.Trackers[seat]));
                    }
                    game.Play(agents[acting].Act(game.Truth));
                }
            }
        }

        private static IEnumerable<ISNode> Nodes(ISNode node)
        {
            yield return node;
            foreach (ISNode child in node.Children)
                foreach (ISNode below in Nodes(child)) yield return below;
        }

        [Fact]
        public void TheRootsMovesAreTheSeatsLegalMoves()
        {
            foreach ((GameState truth, BeliefTracker beliefs) in Decisions(1, 6))
            {
                ISMCTSTree tree = Search(truth, beliefs, 800);
                List<Action> legal = MCTSTree.GetValidActions(truth);
                Assert.Equal(legal.Count, tree.RootNode.Children.Count);
                Assert.All(tree.RootNode.Children, child => Assert.Contains(legal, move => move.IsSameDecision(child.Move!)));
                Assert.True(legal.Any(move => move.IsSameDecision(tree.GetMostPromisingMove())));
                Assert.Null(tree.GetMostPromisingMove() is IReplayAction replay && replay.HasHistory() ? replay : null);
            }
        }

        [Fact]
        public void TheTreeKeysMovesByWhatTheSeatSees()
        {
            int hiddenDraws = 0, ownDraws = 0, availableInSomeWorlds = 0;
            foreach ((GameState truth, BeliefTracker beliefs) in Decisions(2, 10))
            {
                ISMCTSTree tree = Search(truth, beliefs);
                foreach (ISNode node in Nodes(tree.RootNode))
                {
                    // Visits never exceed the times a node was available, which never exceed its parent's visits
                    if (node.Parent != null && !node.Parent.IsChance)
                    {
                        Assert.True(node.Visits <= node.Availability, "a child was visited more often than it was available");
                        Assert.True(node.Availability <= node.Parent.Visits, "a child was available more often than its parent was visited");
                        // (a node's first visit is its own evaluation, before it has children: compare with the most available sibling)
                        if (node.Availability < node.Parent.Children.Max(sibling => sibling.Availability)) availableInSomeWorlds++;
                    }
                    Assert.True(node.Children.Sum(child => (long)child.Visits) <= node.Visits);

                    if (node.IsChance)
                    {
                        // An outcome the seat can't see is one child, whatever the world drew
                        bool hidden = node.Move!.PlayerIndex != 2 && node.Move is BuyDevelopmentCardAction
                            || node.Move is RobberAction { TargetPlayerIndex: sbyte victim } robber && robber.PlayerIndex != 2 && victim != 2;
                        if (hidden)
                        {
                            Assert.True(node.Children.Count <= 1, $"{node.Move} branches on what seat 2 can't see");
                            if (node.Move is BuyDevelopmentCardAction) hiddenDraws++;
                        }
                        else if (node.Move is BuyDevelopmentCardAction && node.Children.Count > 1) ownDraws++;
                        Assert.All(node.Children, outcome => Assert.Equal(node.Move.GetType(), outcome.Move!.GetType()));
                    }
                    else
                    {
                        // Children of a decision are distinct decisions
                        for (int a = 0; a < node.Children.Count; a++)
                            for (int b = a + 1; b < node.Children.Count; b++)
                                Assert.False(node.Children[a].Move!.IsSameDecision(node.Children[b].Move!));
                    }
                }
            }
            Assert.True(hiddenDraws > 0, "no other seat bought a development card in the searched trees");
            Assert.True(availableInSomeWorlds > 0, "no move was legal in only some of the worlds");
        }

        [Fact]
        public void TheSearchDependsOnlyOnWhatTheSeatSees()
        {
            // The same position played on with two different stolen cards: a seat outside the steal searches exactly alike
            (Followed game, Action next) = FindNext((state, action) => IsSteal(state, action)
                && DistinctTypes(state.Players[((RobberAction)action).TargetPlayerIndex!.Value].ResourceCards) >= 2);
            (Followed a, Followed b) = TwoOutcomes(game, next, move => ((RobberAction)move).History!.StolenCard);
            int outsider = Enumerable.Range(0, 4).First(p => p != next.PlayerIndex && p != ((RobberAction)next).TargetPlayerIndex);

            ISMCTSTree treeA = Search(a.Truth, a.Trackers[outsider], 1500, seed: 9), treeB = Search(b.Truth, b.Trackers[outsider], 1500, seed: 9);
            Assert.Equal(treeA.RootVisits().Select(v => (v.move.ToString(), v.visits)), treeB.RootVisits().Select(v => (v.move.ToString(), v.visits)));
            Assert.Equal(treeA.RootValues(), treeB.RootValues());
        }

        [Fact]
        public void AFullKnowledgeTreeSearchesTheTruePosition()
        {
            int hidden = 0;
            foreach ((GameState truth, BeliefTracker beliefs) in Decisions(3, 4))
            {
                if (beliefs.HasHiddenInformation) hidden++;
                ISMCTSTree tree = new ISMCTSTree(truth, beliefs, seed: 1, fullKnowledge: true);
                for (int p = 0; p < 4; p++)
                {
                    Assert.Equal(truth.Players[p].ResourceCards, tree.RootState.Players[p].ResourceCards);
                    Assert.Equal(truth.Players[p].DevelopmentCards, tree.RootState.Players[p].DevelopmentCards);
                }
                Assert.Equal(truth.DevelopmentBank, tree.RootState.DevelopmentBank);
            }
            Assert.True(hidden > 0, "seat 3 never missed anything in the positions");
        }

        [Fact]
        public void IterationSeedsAreTheSameInEveryProcess()
        {
            // HashCode.Combine changes with every process, so a search with the same seed went differently in every run;
            // fixed values here can't
            Assert.Equal(-788417095, ISMCTSTree.IterationSeed(5, 1));
            Assert.Equal(1058536233, ISMCTSTree.IterationSeed(5, 2));
            Assert.Equal(-1258074185, ISMCTSTree.IterationSeed(1000, 1));
            Assert.Equal(323208503, ISMCTSTree.IterationSeed(-3, 7));
        }

        [Fact]
        public void OnlyVisitedNodesCacheTheirLegalMoves()
        {
            // Most children are never visited: a cache on every node (~1.5 KB on 32 threads) made the client stall in the GC;
            // only the nodes a world passed through get one, and keep it
            System.Reflection.FieldInfo cache = typeof(ISNode).GetField("_legalByHand", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
            (GameState truth, BeliefTracker beliefs) = Decisions(0, 1).First();
            ISMCTSTree tree = new ISMCTSTree(truth, beliefs, seed: 5);
            new MCTSSearch(new FixedEvaluator(), new StateVectorizer(), 16, parallelSelection: false).Run(tree, double.PositiveInfinity, 3000);

            List<ISNode> nodes = Nodes(tree.RootNode).ToList();
            Assert.Contains(nodes, node => node.Visits == 0);
            Assert.All(nodes.Where(node => node.Visits == 0), node => Assert.Null(cache.GetValue(node)));
            Assert.NotNull(cache.GetValue(tree.RootNode));
            Assert.Same(tree.RootNode.LegalByHand, tree.RootNode.LegalByHand);
        }

        [Fact]
        public void TheAblationsBranchOnHiddenOutcomesAndNormalize()
        {
            // Outcome branching splits another seat's development card draw by the drawn card; both still play legal moves
            int branched = 0;
            foreach ((GameState truth, BeliefTracker beliefs) in Decisions(2, 8))
            {
                ISMCTSTree tree = new ISMCTSTree(truth, beliefs, seed: 5, fullKnowledge: true) { BranchHiddenOutcomes = true, NormalizeValues = true };
                new MCTSSearch(new FixedEvaluator(), new StateVectorizer(), 16, parallelSelection: false).Run(tree, double.PositiveInfinity, 3000);
                Assert.Contains(MCTSTree.GetValidActions(truth), move => move.IsSameDecision(tree.GetMostPromisingMove()));
                branched += Nodes(tree.RootNode).Count(node => node.IsChance && node.Move is BuyDevelopmentCardAction && node.Move.PlayerIndex != 2 && node.Children.Count > 1);
            }
            Assert.True(branched > 0, "no hidden draw was branched on");
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void TheFullViewTreeEvaluatesNewLeavesLikeISMCTSOnlyWhenAsked(bool evaluateNewLeaves)
        {
            // With the switch a node is expanded only on a visit after the one that evaluated it, so every expanded decision
            // node has two visits or more; without it a new leaf is expanded at once and one of its moves evaluated instead
            (GameState truth, BeliefTracker beliefs) = Decisions(0, 1).First();
            MCTSTree tree = new MCTSTree(new GameState(truth), 0, 1.414, model: OpponentModel.MaxN) { EvaluateNewLeaves = evaluateNewLeaves };
            new MCTSSearch(new FixedEvaluator(), new StateVectorizer(), 16, parallelSelection: false).Run(tree, double.PositiveInfinity, 2000);

            List<MCTSNode> expanded = [];
            void Walk(MCTSNode node)
            {
                if (node.IsExpanded && !node.IsOutputRandomnessGroup) expanded.Add(node);
                foreach (MCTSNode child in node.Children) Walk(child);
            }
            Walk(tree.RootNode);
            Assert.NotEmpty(expanded);
            if (evaluateNewLeaves) Assert.All(expanded, node => Assert.True(node.VisitCount >= 2, $"a node was expanded on its first visit ({node.LastAction})"));
            else Assert.Contains(expanded, node => node.VisitCount == 1);
        }

        [Fact]
        public void TheSameSeedSearchesTheSame()
        {
            (GameState truth, BeliefTracker beliefs) = Decisions(0, 1).First();
            ISMCTSTree first = Search(truth, beliefs, 1000, seed: 3), second = Search(truth, beliefs, 1000, seed: 3);
            Assert.Equal(first.RootVisits().Select(v => v.visits), second.RootVisits().Select(v => v.visits));
        }

        [Fact]
        public void TheAgentPlaysLegalMovesUnderHiddenInformation()
        {
            GameState state = NewGame(36);
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "view_tiny.onnx"), provider: InferenceProvider.CPU);
            InformationSetMCTSAgent agent = new(new MCTSAgent(1, state, valueNet: net, iterationLimit: 256, leavesPerBatch: 16), state, seed: 2);
            Agent[] agents = TestUtil.RandomAgents();
            int decisions = 0;
            while (!state.HasEnded && decisions < 40)
            {
                sbyte acting = TestUtil.ActingPlayer(state);
                Action action = acting == 1 ? agent.Act(new GameState(state)) : agents[acting].Act(state);
                if (acting == 1)
                {
                    decisions++;
                    Assert.True(action.IsValidFor(state), $"{action} is not legal");
                }
                action.Apply(state);
                agent.HandleAction(action);
            }
            Assert.Equal(40, decisions);
            Assert.Equal(new BeliefTracker(agent.Beliefs), agent.Beliefs);
        }

        [Fact]
        public void TheNextDecisionGoesOnWithTheSubtreeOfTheMovesPlayedSince()
        {
            // The agent follows the game down its last tree: at its next decision the search starts at the node the game
            // reached, cut off from the old tree, with the visits it had, and still plays legal moves
            GameState state = NewGame(36);
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "view_tiny.onnx"), provider: InferenceProvider.CPU);
            InformationSetMCTSAgent agent = new(new MCTSAgent(1, state, valueNet: net, iterationLimit: 256, leavesPerBatch: 16, parallelSelection: false), state, seed: 2);
            Assert.True(agent.ReuseTree);
            Agent[] agents = TestUtil.RandomAgents();
            int decisions = 0, reused = 0;
            while (!state.HasEnded && decisions < 40)
            {
                sbyte acting = TestUtil.ActingPlayer(state);
                Action action;
                if (acting == 1)
                {
                    bool searches = MCTSTree.GetValidActions(state).Count > 1;
                    ISNode? carried = agent.ReusableRoot;
                    uint carriedVisits = carried?.Visits ?? 0;
                    action = agent.Act(new GameState(state));
                    Assert.True(action.IsValidFor(state), $"{action} is not legal");
                    if (searches)
                    {
                        decisions++;
                        ISNode root = agent.LastTree!.RootNode;
                        Assert.Null(root.Parent);
                        Assert.Same(agent.ReusableRoot, root);
                        if (carried != null)
                        {
                            Assert.Same(carried, root);
                            Assert.True(root.Visits >= carriedVisits + 256, $"{root.Visits} visits after {carriedVisits} carried");
                            if (carriedVisits > 0) reused++;
                        }
                        else Assert.Null(root.Move);
                    }
                }
                else action = agents[acting].Act(state);
                action.Apply(state);
                agent.HandleAction(action);
            }
            Assert.Equal(40, decisions);
            Assert.True(reused >= 5, $"only {reused} of {decisions} decisions went on with an earlier subtree");
        }

        [Fact]
        public void DecayScalesTheSubtreesCountsAndKeepsItsMeans()
        {
            (GameState truth, BeliefTracker beliefs) = Decisions(2, 1).First();
            ISMCTSTree tree = Search(truth, beliefs, 2000);
            var before = Nodes(tree.RootNode).Select(node => (node, node.Visits, node.ResolvedVisits, node.Availability,
                means: Enumerable.Range(0, MCTSNode.PlayerCount).Select(node.MeanValue).ToArray(), children: node.Children.ToArray())).ToList();
            Assert.True(before.Count > 100);

            tree.RootNode.Decay(0.3);

            foreach ((ISNode node, uint visits, uint resolved, uint availability, double[] means, ISNode[] children) in before)
            {
                uint scaled = (uint)Math.Round(resolved * 0.3, MidpointRounding.AwayFromZero);
                Assert.Equal(scaled, node.ResolvedVisits);
                Assert.Equal(Math.Max((uint)Math.Round(visits * 0.3, MidpointRounding.AwayFromZero), scaled), node.Visits);
                Assert.Equal((uint)Math.Round(availability * 0.3, MidpointRounding.AwayFromZero), node.Availability);
                for (int player = 0; player < MCTSNode.PlayerCount; player++)
                    Assert.Equal(scaled == 0 ? 0 : means[player], node.MeanValue(player), 1e-9);
                Assert.Equal(children, node.Children);
            }
        }

        [Fact]
        public void TheAgentDecaysTheSubtreeItGoesOnWith()
        {
            GameState state = NewGame(36);
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "view_tiny.onnx"), provider: InferenceProvider.CPU);
            InformationSetMCTSAgent agent = new(new MCTSAgent(1, state, valueNet: net, iterationLimit: 256, leavesPerBatch: 16, parallelSelection: false), state, seed: 2)
            {
                ReuseDecay = 0.5
            };
            Agent[] agents = TestUtil.RandomAgents();
            int decisions = 0, decayed = 0;
            while (!state.HasEnded && decisions < 40)
            {
                sbyte acting = TestUtil.ActingPlayer(state);
                Action action;
                if (acting == 1)
                {
                    bool searches = MCTSTree.GetValidActions(state).Count > 1;
                    uint carried = agent.ReusableRoot?.Visits ?? 0;
                    action = agent.Act(new GameState(state));
                    if (searches)
                    {
                        decisions++;
                        // The carried visits at half weight, then one visit per new iteration
                        Assert.Equal((uint)Math.Round(carried * 0.5, MidpointRounding.AwayFromZero) + agent.Search.LastSearch!.Iterations, agent.LastTree!.RootNode.Visits);
                        if (carried > 1) decayed++;
                    }
                }
                else action = agents[acting].Act(state);
                action.Apply(state);
                agent.HandleAction(action);
            }
            Assert.True(decayed >= 5, $"only {decayed} decisions went on with a decayed subtree");
        }

        [Theory]
        [InlineData(0.0)]
        [InlineData(-0.5)]
        [InlineData(1.5)]
        [InlineData(double.NaN)]
        public void TheDecayIsAboveZeroAndAtMostOne(double decay)
        {
            GameState state = NewGame(36);
            MCTSAgent search = new(1, state, valueNet: new StateValueNet(TestUtil.FixturePath("ValueNet", "view_tiny.onnx"), provider: InferenceProvider.CPU));
            Assert.Throws<ArgumentOutOfRangeException>(() => new InformationSetMCTSAgent(search, state) { ReuseDecay = decay });
            Assert.Equal(1, new InformationSetMCTSAgent(search, state).ReuseDecay);
        }

        [Fact]
        public void WithoutReuseEveryDecisionStartsANewTree()
        {
            GameState state = NewGame(36);
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "view_tiny.onnx"), provider: InferenceProvider.CPU);
            MCTSAgent search = new(1, state, valueNet: net, iterationLimit: 64, leavesPerBatch: 16) { ReuseTree = false };
            InformationSetMCTSAgent agent = new(search, state, seed: 2);
            Agent[] agents = TestUtil.RandomAgents();
            int decisions = 0;
            while (!state.HasEnded && decisions < 10)
            {
                sbyte acting = TestUtil.ActingPlayer(state);
                Action action = acting == 1 ? agent.Act(new GameState(state)) : agents[acting].Act(state);
                if (acting == 1 && agent.LastTree != null && MCTSTree.GetValidActions(state).Count > 1)
                {
                    decisions++;
                    Assert.Null(agent.LastTree.RootNode.Move);
                }
                action.Apply(state);
                agent.HandleAction(action);
                Assert.Null(agent.ReusableRoot);
            }
            Assert.Equal(10, decisions);
        }

        [Fact]
        public void AResetOrAMoveOutsideTheTreeEndsTheReuse()
        {
            (GameState truth, BeliefTracker beliefs) = Decisions(0, 1).First();
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "view_tiny.onnx"), provider: InferenceProvider.CPU);
            InformationSetMCTSAgent agent = new(new MCTSAgent(0, truth, valueNet: net, iterationLimit: 64, leavesPerBatch: 16), truth, seed: 2);
            agent.ResetTo(truth, beliefs);
            agent.Act(new GameState(truth));
            Assert.NotNull(agent.ReusableRoot);
            agent.ResetToState(truth);
            Assert.Null(agent.ReusableRoot);

            agent.ResetTo(truth, beliefs);
            agent.Act(new GameState(truth));
            Assert.NotNull(agent.ReusableRoot);
            agent.ResetTo(truth, beliefs);
            Assert.Null(agent.ReusableRoot);

            agent.Act(new GameState(truth));
            Assert.NotNull(agent.ReusableRoot);
            // A move that is no child of the tree (another player's end of turn) leaves it
            Assert.Null(agent.ReusableRoot!.Follow(new EndTurnAction((sbyte)((truth.Turn.PlayerIndex + 1) % 4)), 0, false));
        }
    }
}
