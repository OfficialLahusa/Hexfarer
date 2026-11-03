using Agents;
using Agents.MCTS;
using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The MCTS tree (without the value net) and the MCTSAgent (with the value net, skipped when it is missing)
    /// </summary>
    public class MCTSTests
    {
        // ================================================================ Value normalization

        [Fact]
        public void MinMaxStatsNormalizesToTheSeenRange()
        {
            MinMaxStats stats = new();
            Assert.Equal(0.3, stats.Normalize(0.3));

            stats.Update(0.4);
            Assert.Equal(0.3, stats.Normalize(0.3));

            stats.Update(0.2);
            stats.Update(0.6);
            Assert.Equal(0.5, stats.Normalize(0.4), 12);
            Assert.Equal(0.0, stats.Normalize(0.2), 12);
            Assert.Equal(1.0, stats.Normalize(0.6), 12);
        }

        [Fact]
        public void UctPrefersUnvisitedChildrenAndUsesNormalizedValues()
        {
            MCTSNode root = new(0);
            MCTSNode visited = new(0, root);
            MCTSNode unvisited = new(0, root);
            for (int i = 0; i < 10; i++) root.IncrementVisitCount();
            for (int i = 0; i < 5; i++) visited.IncrementVisitCount();
            visited.AddResult(0.05);

            Assert.Equal(1e9, unvisited.GetUCT(1.414));

            MinMaxStats bounds = new();
            bounds.Update(0.0);
            bounds.Update(0.02);
            double exploration = 1.414 * Math.Sqrt(Math.Log(10) / 5);
            Assert.Equal(0.01 + exploration, visited.GetUCT(1.414), 12);
            Assert.Equal(0.5 + exploration, visited.GetUCT(1.414, bounds), 12);
        }

        // ================================================================ Tree structure and chance nodes

        [Fact]
        public void RollGroupsHaveElevenOutcomesSampledWithTheDiceDistribution()
        {
            TestUtil.Seed(140);
            GameState state = PreRollState();
            MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, 1.414);
            tree.ExpandNode(tree.RootNode, new GameState(state));

            MCTSNode group = tree.RootNode.Children.Single(child => child.IsOutputRandomnessGroup && child.LastAction is RollAction);
            Assert.Equal(11, group.Children.Count);

            const int Samples = 60_000;
            Dictionary<int, int> counts = [];
            for (int i = 0; i < Samples; i++)
            {
                int total = ((RollAction)tree.SelectChild(group).LastAction!).History!.RollResult.Total;
                counts[total] = counts.GetValueOrDefault(total) + 1;
            }

            for (int total = 2; total <= 12; total++)
                Assert.InRange(counts.GetValueOrDefault(total) / (double)Samples, (6 - Math.Abs(total - 7)) / 36.0 - 0.01, (6 - Math.Abs(total - 7)) / 36.0 + 0.01);
        }

        [Fact]
        public void RootValuesVisitsAndSampledMovesFollowTheSearch()
        {
            TestUtil.Seed(148);
            GameState state = TestUtil.EmptyNormalTurn(1);
            TestUtil.GiveCards(state, 1, ResourceCardType.Grain, 4);
            TestUtil.GiveCards(state, 1, ResourceCardType.Ore, 4);
            MCTSTree tree = new(state, ownerIdx: 0, explorationParameter: 1.414, model: OpponentModel.MaxN);
            tree.ExpandNode(tree.RootNode, new GameState(state));
            MCTSNode[] children = tree.RootNode.Children.ToArray();
            Assert.True(children.Length >= 3);

            // Three visits of the first child, one of the second, none of the others
            for (int i = 0; i < 3; i++) tree.Backpropagate(children[0], [1, 0, 0, 0]);
            tree.Backpropagate(children[1], [0, 1, 0, 0]);

            Assert.Equal([0.75, 0.25, 0, 0], tree.RootValues());

            var visits = tree.RootVisits();
            Assert.Equal(children.Length, visits.Count);
            Assert.True(children[0].LastAction!.IsSameDecision(visits[0].move));
            Assert.Equal([3u, 1u], visits.Take(2).Select(v => v.visits));
            Assert.All(visits.Skip(2), v => Assert.Equal(0u, v.visits));

            // Temperature 0 is the most visited move, 1 samples by visits (3:1), 0.5 by squared visits (9:1)
            Random random = new(5);
            Assert.True(children[0].LastAction!.IsSameDecision(tree.SampleMove(0, random)));
            foreach ((double temperature, double expected) in new[] { (1.0, 0.75), (0.5, 0.9) })
            {
                const int Samples = 4000;
                int first = 0;
                for (int i = 0; i < Samples; i++)
                {
                    Action move = tree.SampleMove(temperature, random);
                    Assert.True(children[0].LastAction!.IsSameDecision(move) || children[1].LastAction!.IsSameDecision(move));
                    if (children[0].LastAction!.IsSameDecision(move)) first++;
                }
                Assert.InRange(first / (double)Samples, expected - 0.03, expected + 0.03);
            }
        }

        [Fact]
        public void PlayedActionsAreMatchedToTheirChildIncludingEquivalentRolls()
        {
            TestUtil.Seed(141);
            GameState state = PreRollState();
            sbyte roller = (sbyte)state.Turn.PlayerIndex;
            MCTSTree tree = new(new GameState(state), roller, 1.414);
            tree.ExpandNode(tree.RootNode, new GameState(state));

            for (byte first = 1; first <= 6; first++)
            {
                for (byte second = 1; second <= 6; second++)
                {
                    RollAction played = new(roller);
                    ActionTests.SetHistory(played, new RollAction.RollActionHistory(new RollResult { First = first, Second = second }, state.Turn.LastRoll));

                    MCTSNode? child = tree.FindChildForAction(played);
                    Assert.NotNull(child);
                    Assert.Equal(first + second, ((RollAction)child!.LastAction!).History!.RollResult.Total);
                }
            }

            // An action that is not among the children is not matched
            Assert.Null(tree.FindChildForAction(new EndTurnAction(roller)));
        }

        [Fact]
        public void ReRootingKeepsTheSubtreeAndDetachesIt()
        {
            TestUtil.Seed(142);
            GameState state = PreRollState();
            MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, 1.414);
            RunFakeIterations(tree, 400);

            MCTSNode group = tree.RootNode.Children.Single(child => child.IsOutputRandomnessGroup);
            MCTSNode outcome = group.Children.OrderByDescending(child => child.VisitCount).First();
            uint visits = outcome.VisitCount;
            GameState next = new GameState(state);
            outcome.LastAction!.Clone().Apply(next);

            tree.ReRoot(outcome, next);

            Assert.Same(outcome, tree.RootNode);
            Assert.Null(tree.RootNode.Parent);
            Assert.Equal(visits, tree.RootNode.VisitCount);
            Assert.Same(next, tree.RootState);
        }

        [Fact]
        public void PendingVisitsDoNotWidenTheValueRange()
        {
            TestUtil.Seed(143);
            GameState state = TestUtil.EmptyNormalTurn(0);
            TestUtil.GiveCards(state, 0, ResourceCardType.Grain, 4);
            MCTSTree tree = new(new GameState(state), 0, 1.414);
            tree.ExpandNode(tree.RootNode, new GameState(state));
            MCTSNode[] children = tree.RootNode.Children.ToArray();

            // Ten leaves in flight below the first child, then one result for the second: only that result counts for the means
            for (int i = 0; i < 10; i++)
                tree.MarkPendingPath(children[0]);
            tree.MarkPendingPath(children[1]);
            tree.Backpropagate(children[1], 0.5, pathAlreadyMarked: true);

            Assert.Equal(11u, tree.RootNode.VisitCount);
            Assert.Equal(1u, tree.RootNode.ResolvedVisitCount);
            Assert.Equal(new MinMaxStats.Range(0.5, 0.5), tree.ValueBounds!.GetRange());
        }

        [Fact]
        public void SearchNeverChangesTheTreesOwnActions()
        {
            // Regression test for shared tree actions being re-applied (and their history rewritten) by every iteration
            TestUtil.Seed(143);
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => TestUtil.IsFreeTurn(s) && s.Players[s.Turn.PlayerIndex].ResourceCards.Count() >= 4));
            MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, 1.414);

            RunFakeIterations(tree, 500);
            List<(Action action, object? history)> snapshot = AllNodes(tree.RootNode)
                .Where(node => node.LastAction != null)
                .Select(node => (node.LastAction!, node.LastAction!.GetType().GetProperty("History")?.GetValue(node.LastAction)))
                .ToList();

            RunFakeIterations(tree, 2000, parallel: true);

            Assert.True(snapshot.Count > 100);
            Assert.All(snapshot, entry => Assert.Same(entry.history, entry.action.GetType().GetProperty("History")?.GetValue(entry.action)));
        }

        [Fact]
        public void MostPromisingMoveIsACopyWithoutHistory()
        {
            TestUtil.Seed(144);
            GameState state = PreRollState();
            MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, 1.414);
            RunFakeIterations(tree, 300);

            Action move = tree.GetMostPromisingMove();

            Assert.DoesNotContain(tree.RootNode.Children, child => ReferenceEquals(child.LastAction, move));
            if (move is IReplayAction replay)
                Assert.False(replay.HasHistory());
        }

        [Fact]
        public void AWinningMoveIsTheOnlyChildOfItsNode()
        {
            TestUtil.Seed(146);
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => TestUtil.IsFreeTurn(s) && TestUtil.CountBuildings(s, s.Turn.PlayerIndex, Intersection.BuildingType.Settlement) > 0));
            int playerIdx = state.Turn.PlayerIndex;
            PlayerState player = state.Players[playerIdx];
            foreach ((ResourceCardType type, uint amount) in new[] { (ResourceCardType.Ore, 3u), (ResourceCardType.Grain, 2u) })
                if (player.ResourceCards.Get(type) < amount)
                    TestUtil.GiveCards(state, playerIdx, type, amount - player.ResourceCards.Get(type));

            // Short of the city's point: nothing wins, all moves stay
            player.VictoryPoints.SettlementPoints += (byte)(8 - player.VictoryPoints.Total);
            List<Action> moves = MCTSTree.GetValidActions(state);
            Assert.Null(MCTSTree.WinningMove(state, moves));
            MCTSTree open = new(new GameState(state), (sbyte)playerIdx, 1.414, model: OpponentModel.MaxN);
            open.ExpandNode(open.RootNode, new GameState(state));
            Assert.Equal(moves.Count, open.RootNode.Children.Count);

            // One city away from ten points: the city is the only child
            player.VictoryPoints.SettlementPoints += 1;
            Action? winning = MCTSTree.WinningMove(state, MCTSTree.GetValidActions(state));
            Assert.IsType<CityAction>(winning);
            MCTSTree tree = new(new GameState(state), (sbyte)playerIdx, 1.414, model: OpponentModel.MaxN);
            tree.ExpandNode(tree.RootNode, new GameState(state));
            Assert.Single(tree.RootNode.Children);
            Assert.True(tree.RootNode.Children[0].LastAction!.IsSameDecision(winning!));
        }

        [Fact]
        public void PolicyTargetsCoverAllLegalMovesAlsoWhereOnlyTheWinningMoveIsSearched()
        {
            TestUtil.Seed(146);
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => TestUtil.IsFreeTurn(s) && TestUtil.CountBuildings(s, s.Turn.PlayerIndex, Intersection.BuildingType.Settlement) > 0));
            int playerIdx = state.Turn.PlayerIndex;
            PlayerState player = state.Players[playerIdx];
            foreach ((ResourceCardType type, uint amount) in new[] { (ResourceCardType.Ore, 3u), (ResourceCardType.Grain, 2u) })
                if (player.ResourceCards.Get(type) < amount)
                    TestUtil.GiveCards(state, playerIdx, type, amount - player.ResourceCards.Get(type));

            // Without a winning move: the children are the legal moves, their visits stay in place
            player.VictoryPoints.SettlementPoints += (byte)(8 - player.VictoryPoints.Total);
            List<Action> legal = MCTSTree.GetValidActions(state);
            MCTSTree open = new(new GameState(state), (sbyte)playerIdx, 1.414, model: OpponentModel.MaxN);
            open.ExpandNode(open.RootNode, new GameState(state));
            uint[] counts = Enumerable.Range(1, legal.Count).Select(i => (uint)i).ToArray();
            (Action[] moves, uint[] visits) = open.LegalMoveVisits(counts);
            Assert.Equal(legal.Count, moves.Length);
            Assert.All(Enumerable.Range(0, legal.Count), m => Assert.True(moves[m].IsSameDecision(legal[m])));
            Assert.Equal(counts, visits);
            Assert.Throws<ArgumentException>(() => open.LegalMoveVisits([1, 2]));

            // Children in another order: the visits stay with their moves
            Array.Reverse(open.RootNode.ChildArray);
            (moves, visits) = open.LegalMoveVisits(counts);
            Assert.All(Enumerable.Range(0, legal.Count), m => Assert.True(moves[m].IsSameDecision(legal[m])));
            Assert.Equal(counts.Reverse(), visits);

            // Two children of the same decision would put the visits of both on one move: that stops instead
            MCTSNode[] openChildren = open.RootNode.ChildArray;
            openChildren[1] = new MCTSNode(openChildren[0].ActivePlayerIndex, open.RootNode, openChildren[0].LastAction);
            Assert.Throws<InvalidOperationException>(() => open.LegalMoveVisits(counts));

            // Self-play records the pruned visits of a PUCT search (without those only the root noise caused) on their moves
            IStateVectorizer vectorizer = StateVectorizer.Current;
            float[] features = new float[vectorizer.FeatureDim];
            double[] win = new double[4], loss = new double[4];
            win[playerIdx] = 1;
            loss[(playerIdx + 1) % 4] = 1;
            MCTSTree puct = new(new GameState(state), (sbyte)playerIdx, 1.414, model: OpponentModel.MaxN) { UsesPolicy = true };
            puct.ExpandNode(puct.RootNode, new GameState(state));
            MCTSNode[] children = puct.RootNode.Children.ToArray();
            for (int i = 0; i < 50; i++) puct.Backpropagate(children[0], win);
            for (int i = 0; i < 10; i++) puct.Backpropagate(children[1], loss);
            for (int i = 0; i < 10; i++) puct.Backpropagate(children[2], win);
            float[] clean = Enumerable.Repeat(1f / children.Length, children.Length).ToArray();
            uint[] pruned = puct.PrunedRootVisits(clean);
            Assert.InRange(pruned[1], 0u, 2u);
            (int, uint)[] expected = Enumerable.Range(0, legal.Count).Where(m => pruned[m] > 0).Select(m => (m, pruned[m])).ToArray();
            Assert.Equal(expected, SelfPlayCollector.SelfPlayGame.Record(state, 0, puct, 70, vectorizer, features, clean).Visits);
            Array.Reverse(puct.RootNode.ChildArray);
            Assert.Equal(expected, SelfPlayCollector.SelfPlayGame.Record(state, 0, puct, 70, vectorizer, features, clean).Visits);

            // One city away from ten points: the tree searches only the city, the targets still list every legal move
            player.VictoryPoints.SettlementPoints += 1;
            legal = MCTSTree.GetValidActions(state);
            Action winning = MCTSTree.WinningMove(state, legal)!;
            MCTSTree tree = new(new GameState(state), (sbyte)playerIdx, 1.414, model: OpponentModel.MaxN);
            tree.ExpandNode(tree.RootNode, new GameState(state));
            Assert.Single(tree.RootNode.Children);
            (moves, visits) = tree.LegalMoveVisits([7]);
            Assert.Equal(legal.Count, moves.Length);
            int winningIdx = legal.FindIndex(move => move.IsSameDecision(winning));
            Assert.Equal(Enumerable.Range(0, legal.Count).Select(m => m == winningIdx ? 7u : 0u), visits);

            // Self-play records the same from the tree's visits: all legal moves, the visits on the winning one, also pruned
            for (int i = 0; i < 7; i++) tree.Backpropagate(tree.RootNode.Children[0], win);
            SelfPlayCollector.RecordedPosition position = SelfPlayCollector.SelfPlayGame.Record(state, 0, tree, 7, vectorizer, features, null);
            Assert.Equal(legal.Count, position.Moves.Length);
            Assert.Equal([(winningIdx, 7u)], position.Visits);
            Assert.Equal([(winningIdx, 7u)], SelfPlayCollector.SelfPlayGame.Record(state, 0, tree, 7, vectorizer, features, [1f]).Visits);
        }

        // ================================================================ MCTSAgent with the value net

        [SkippableFact]
        public void AgentFindsAnImmediateWin()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(145);

            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => TestUtil.IsFreeTurn(s) && TestUtil.CountBuildings(s, s.Turn.PlayerIndex, Intersection.BuildingType.Settlement) > 0));
            int playerIdx = state.Turn.PlayerIndex;
            PlayerState player = state.Players[playerIdx];

            // One city away from ten points
            player.VictoryPoints.SettlementPoints += (byte)(9 - player.VictoryPoints.Total);
            foreach ((ResourceCardType type, uint amount) in new[] { (ResourceCardType.Ore, 3u), (ResourceCardType.Grain, 2u) })
                if (player.ResourceCards.Get(type) < amount)
                    TestUtil.GiveCards(state, playerIdx, type, amount - player.ResourceCards.Get(type));

            MCTSAgent agent = new((sbyte)playerIdx, state, timeLimitSeconds: 1.0);
            Action move = agent.Act(state);

            GameState after = new GameState(state);
            move.Apply(after);
            Assert.True(after.HasEnded && after.Turn.PlayerIndex == playerIdx, $"agent chose {move} instead of a winning move");
        }

        [SkippableFact]
        public void CancellingStopsTheSearchEarlyWithAValidMove()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(146);
            GameState state = TestUtil.NewGame();
            TestUtil.PlayRandomGame(state, maxActions: 60);
            sbyte acting = TestUtil.ActingPlayer(state);

            MCTSAgent agent = new(acting, state, timeLimitSeconds: 60);
            using CancellationTokenSource cancellation = new(TimeSpan.FromMilliseconds(200));
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            Action move = agent.Act(state, cancellation.Token);

            Assert.True(stopwatch.Elapsed.TotalSeconds < 10, $"search ran {stopwatch.Elapsed.TotalSeconds:F1}s after the cancellation");
            Assert.True(move.IsValidFor(state));
        }

        [SkippableFact]
        public void AnIterationLimitReplacesTheTimeLimit()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(147);
            GameState state = TestUtil.NewGame();
            TestUtil.PlayRandomGame(state, maxActions: 60);
            // A decision with a choice (a single legal move is played without a search)
            Assert.True(TestUtil.PlayUntil(state, s => MCTSTree.GetValidActions(s).Count >= 2));
            sbyte acting = TestUtil.ActingPlayer(state);

            // A time limit far too long for the test, so only the iteration limit can end the search
            MCTSAgent agent = new(acting, state, timeLimitSeconds: 600, iterationLimit: 3000);
            Action move = agent.Act(state);

            Assert.True(move.IsValidFor(state));
            Assert.InRange(agent.LastSearch!.Iterations, 3000, 3000 + 256);
        }

        [SkippableFact]
        public void AgentKeepsItsTreeInSyncWithTheGame()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(146);

            GameState state = TestUtil.NewGame();
            ProbeAgent mcts = new(0, state);
            Agent[] agents = [mcts, new RandomAgent(1), new RandomAgent(2), new RandomAgent(3)];
            int reusedAfterOwnMove = 0, ownFollowUps = 0, actions = 0;
            sbyte lastActor = -1;
            bool lastSearched = false;

            while (!state.HasEnded && actions < 400)
            {
                sbyte acting = TestUtil.ActingPlayer(state);
                // After a searched own move the tree goes on with its subtree; a single legal move is played without one
                if (acting == 0 && lastActor == 0 && lastSearched)
                {
                    ownFollowUps++;
                    if (mcts.Tree != null && mcts.Tree.RootState.Equals(state)) reusedAfterOwnMove++;
                }

                bool searched = acting == 0 && MCTSTree.GetValidActions(state).Count >= 2;
                Action action = agents[acting].Act(state);
                if (searched)
                    Assert.DoesNotContain(mcts.Tree!.RootNode.Children, child => ReferenceEquals(child.LastAction, action));

                action.Apply(state);
                foreach (Agent agent in agents)
                    agent.HandleAction(action);

                if (mcts.Tree != null)
                {
                    Assert.True(mcts.Tree.RootState.Equals(state), $"tree root state differs from the game after {action}");
                    Assert.Null(mcts.Tree.RootNode.Parent);
                }

                lastActor = acting;
                lastSearched = searched;
                actions++;
            }

            Assert.True(ownFollowUps > 0);
            Assert.Equal(ownFollowUps, reusedAfterOwnMove);
        }

        [SkippableFact]
        public void WithoutTreeReuseEveryDecisionStartsANewTree()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(146);

            GameState state = TestUtil.NewGame();
            ProbeAgent mcts = new(0, state) { ReuseTree = false };
            Agent[] agents = [mcts, new RandomAgent(1), new RandomAgent(2), new RandomAgent(3)];
            int decisions = 0;
            while (!state.HasEnded && decisions < 8)
            {
                sbyte acting = TestUtil.ActingPlayer(state);
                Action action = agents[acting].Act(state);
                if (acting == 0) decisions++;
                action.Apply(state);
                foreach (Agent agent in agents)
                    agent.HandleAction(action);
                Assert.Null(mcts.Tree);
            }
            Assert.Equal(8, decisions);
        }

        [SkippableFact]
        public void ResettingOrDivergingStatesStartANewTree()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(147);

            GameState state = TestUtil.NewGame();
            // A free turn with a choice (a single legal move is played without a tree)
            Assert.True(TestUtil.PlayUntil(state, s => TestUtil.IsFreeTurn(s) && MCTSTree.GetValidActions(s).Count >= 2));
            ProbeAgent agent = new((sbyte)state.Turn.PlayerIndex, state);

            agent.Act(state);
            Assert.NotNull(agent.Tree);

            agent.ResetToState(state);
            Assert.Null(agent.Tree);

            // A state the tree does not know (e.g. after a manual edit) starts over on a copy of that state
            agent.Act(state);
            GameState edited = new GameState(state);
            TestUtil.GiveCards(edited, edited.Turn.PlayerIndex, ResourceCardType.Ore, 1);
            agent.Act(edited);
            Assert.True(agent.Tree!.RootState.Equals(edited));
            Assert.NotSame(edited, agent.Tree.RootState);
        }

        // ================================================================ Helpers

        private class ProbeAgent : MCTSAgent
        {
            public ProbeAgent(sbyte playerIdx, GameState state) : base(playerIdx, state, timeLimitSeconds: 0.05) { }
            public MCTSTree? Tree => _tree;
        }

        private static GameState PreRollState()
        {
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => s.Turn.TypeOfRound == TurnState.RoundType.Normal && s.Turn.MustRoll && RollAction.GetActionsForState(s, (sbyte)s.Turn.PlayerIndex).Count > 0));
            return state;
        }

        /// <summary>
        /// Selection, expansion and backpropagation with a constant leaf value instead of the value net
        /// </summary>
        private static void RunFakeIterations(MCTSTree tree, int iterations, bool parallel = false)
        {
            void Iterate(int _)
            {
                (MCTSNode leaf, GameState leafState) = tree.SelectLeafNode();
                if (!leaf.IsTerminal)
                {
                    tree.ExpandNode(leaf, leafState);
                    (leaf, leafState) = tree.SelectLeafNode(leaf, leafState, suppressStateCopy: true);
                }
                tree.Backpropagate(leaf, leafState.GetHashCode() % 2 == 0 ? 0.4 : 0.6);
            }

            if (parallel)
                Parallel.For(0, iterations, Iterate);
            else
                for (int i = 0; i < iterations; i++) Iterate(i);
        }

        private static IEnumerable<MCTSNode> AllNodes(MCTSNode root)
        {
            Stack<MCTSNode> pending = new([root]);
            while (pending.TryPop(out MCTSNode? node))
            {
                yield return node;
                foreach (MCTSNode child in node.Children)
                    pending.Push(child);
            }
        }
    }
}
