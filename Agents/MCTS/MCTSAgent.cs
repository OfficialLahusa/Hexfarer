using Agents.Inference;
using Common;
using Common.Actions;
using Common.Serialization;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    /// <summary>
    /// Plays by MCTS over the true position (open-hand), the leaves valued by the net. Trades between players are not part of
    /// its search tree: before any search, Act decides them by the value net alone (an answer or partner pick in
    /// TradeDecisions, a counteroffer or offer in TradeOffers), and only a move that is no trade is searched (see MCTSTree).
    /// The information set and PIMC agents use this agent's search and its trade decisions the same way.
    /// </summary>
    public class MCTSAgent : Agent, IShowsSearch
    {
        public const double TIME_LIMIT_SECONDS = 5;
        // Leaves selected before their evaluation, unless the constructor is given another number
        public const int DefaultLeavesPerBatch = 256;

        /// <summary>
        /// Tree updates per decision that LeavesFor keeps with an iteration budget: the tree takes in the evaluations after
        /// every batch, so small batches search much better at the same budget (800 iterations: 16 leaves, 50 updates,
        /// against 256 +111 Elo; 5000: 32 leaves, 156 updates, against 256 +54), while larger ones fill the net calls faster
        /// </summary>
        public const int UpdatesPerDecision = 100;

        /// <summary>
        /// Fewest leaves per batch LeavesFor gives: at the same games at once batches of 8 played about half the games per
        /// hour of 16, and what they would gain in strength is unmeasured
        /// </summary>
        public const int MinLeavesPerBatch = 16;

        /// <summary>
        /// Leaves per batch by default for a search of this many iterations (the Arena's and the Open Seat Agent's): the
        /// largest power of two that still gives the tree UpdatesPerDecision updates, at least MinLeavesPerBatch and at most
        /// DefaultLeavesPerBatch (800 and 1600 iterations: 16, 3200 and 5000: 32, 6400: 64, from 25600: 256)
        /// </summary>
        public static int LeavesFor(long iterations)
        {
            long most = Math.Max(1, iterations / UpdatesPerDecision);
            long leaves = Math.Min(DefaultLeavesPerBatch, 1L << System.Numerics.BitOperations.Log2((ulong)most));
            return (int)Math.Max(MinLeavesPerBatch, leaves);
        }

        // Search tree, kept across moves by HandleAction (null until the next Act builds a new one)
        protected MCTSTree? _tree;
        protected double _explorationParameter;
        protected double _timeLimitSeconds;
        protected long _iterationLimit;
        // Random playouts (SimulatePlayout): the shared source, or one of its own with a seed and selection on one thread
        protected Random _random = Random.Shared;
        // With a seed and selection on one thread: draws the seed of each new tree's own source. A source per tree, not one
        // for the agent, so the k-th tree's numbers depend only on the seed and k, not on how many the trees before it drew
        private readonly Random? _seeds;

        // The default net is shared by all agents that aren't given another one, and loaded on first use
        private static readonly Lazy<StateValueNet> DefaultValueNet = new(() => new StateValueNet(StateValueNet.DefaultModelFile));

        protected readonly StateValueNet _stateValueNet;
        protected readonly IStateVectorizer _stateVectorizer;

        // Value of places 1 to 4 on [0, 1], the same as PLACEMENT_SCORES in state_value_net.py
        private static readonly float[] PlacementScores = [1f, 2f / 3f, 1f / 3f, 0f];
        private double _placementWeight;
        private OpponentModel _opponentModel;

        protected readonly object _treeLock = new object();

        /// <summary>
        /// The encoding of the leaves for the agent's net (see Vectorizers)
        /// </summary>
        public IStateVectorizer Vectorizer => _stateVectorizer;

        public double ExplorationParameter => _explorationParameter;

        /// <summary>
        /// Searches another tree (an information set tree, see InformationSetMCTSAgent) with this agent's net, budget and
        /// settings
        /// </summary>
        public void RunSearch(ISearchTree tree, CancellationToken cancellation = default)
        {
            // The budget of the root's decision (more for the initial placements, see SetupBudgetFactor)
            (double seconds, long iterations) = BudgetFor(tree.RootState);
            if (iterations > 0) _search.Run(tree, double.PositiveInfinity, iterations, cancellation);
            else _search.Run(tree, seconds, 0, cancellation);
        }

        /// <summary>
        /// Factor on the search budget (iterations or time) of the initial placements, where the search barely reaches past
        /// the player's own move (1: the same budget as every other decision). Applies to every search of this agent, so also
        /// to the information set and PIMC agents built on it (PIMC splits the larger budget between its worlds).
        /// </summary>
        public double SetupBudgetFactor
        {
            get => _setupBudgetFactor;
            set => _setupBudgetFactor = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "The setup budget factor must be above 0.");
        }

        private double _setupBudgetFactor = 1;

        /// <summary>
        /// Whether the position is in the initial placements (both rounds of settlements and their roads)
        /// </summary>
        public static bool IsSetup(GameState state) =>
            state.Turn.TypeOfRound is TurnState.RoundType.FirstInitial or TurnState.RoundType.SecondInitial;

        /// <summary>
        /// The search budget of a decision in the position: the time and the iteration limit, both times SetupBudgetFactor in
        /// the initial placements (iterations rounded up; 0 iterations stays 0, a search for time)
        /// </summary>
        public (double seconds, long iterations) BudgetFor(GameState state)
        {
            double factor = IsSetup(state) ? _setupBudgetFactor : 1;
            return (_timeLimitSeconds * factor, (long)Math.Ceiling(_iterationLimit * factor));
        }

        /// <summary>
        /// The agent's answer or pick in an open trade between players, decided outside the tree (see TradeDecisions), or null
        /// </summary>
        public Action? DecideTrade(GameState state) => TradeDecisions.Decide(state, PlayerIndex, TradeUtility, AnswerMargin, AnswerFairness);

        /// <summary>
        /// The share of the offerer's gain an offer has to give the agent for it to accept (0: any gain above AnswerMargin; by
        /// default as much as the offerer gains)
        /// </summary>
        public float AnswerFairness { get; set; } = TradeDecisions.DefaultFairness;

        /// <summary>
        /// How much a trade has to raise the agent's value for it to accept an offer
        /// </summary>
        public float AnswerMargin { get; set; } = TradeDecisions.ValueMargin;

        /// <summary>
        /// The agent's counteroffer to an offer it turns down, judged over worlds as its offers (see TradeOffers.DecideCounter)
        /// </summary>
        public CounterTradeAction? DecideCounter(GameState state, Func<int, IReadOnlyList<GameState>> worlds) =>
            Offers.DecideCounter(state, PlayerIndex, worlds, Utilities);

        /// <summary>
        /// The agent's trade offers on turn (see TradeOffers), made before the search
        /// </summary>
        public TradeOffers Offers { get; } = new();

        /// <summary>
        /// The agent's offer now, judged by the value net over worlds it can't tell from the true position (the true position
        /// for the open-hand search), or null
        /// </summary>
        public OfferTradeAction? DecideOffer(GameState state, Func<int, IReadOnlyList<GameState>> worlds) =>
            Offers.Decide(state, PlayerIndex, worlds, Utilities);

        /// <summary>
        /// Statistics of the last search (RunForTime or RunBatchedIterations)
        /// </summary>
        public SearchStatistics? LastSearch => _search.LastSearch;

        // The batched search, with large batches selected on all cores and evaluated by the agent's net
        private readonly MCTSSearch _search;


        /// <summary>
        /// How much lower than its parent an unvisited move counts in PUCT (see MCTSTree.FirstPlayReduction)
        /// </summary>
        public double FirstPlayReduction { get; set; } = 0.1;

        /// <summary>
        /// Whether selection scales each player's values to the range seen in the tree (see MCTSTree, MinMaxStats)
        /// </summary>
        public bool NormalizeValues { get; set; } = true;

        /// <summary>
        /// Whether the next decision goes on with the subtree of the moves played since (see HandleAction) instead of a new tree
        /// </summary>
        public bool ReuseTree { get; set; } = true;

        /// <summary>
        /// Evaluate a leaf on its first visit instead of one of its moves (see MCTSTree.EvaluateNewLeaves)
        /// </summary>
        public bool EvaluateNewLeaves { get; set; } = true;

        /// <summary>
        /// Play the only legal move at once instead of searching (a roll without a playable development card, the end of a
        /// turn without anything affordable); the search would only grow the subtree behind it, of which the next decision
        /// keeps one branch
        /// </summary>
        public bool PlayForcedMovesAtOnce { get; set; } = true;

        /// <summary>
        /// Search by PUCT with uniform priors instead of the policy head's (see MCTSSearch.UniformPriors)
        /// </summary>
        public bool UniformPriors
        {
            get => _search.UniformPriors;
            set => _search.UniformPriors = value;
        }

        /// <param name="valueNet">Value net for the leaves, the default model if null</param>
        /// <param name="vectorizer">Encoding the value net was trained on, the StateVectorizer for the net's input size if null</param>
        /// <summary>
        /// Weight of the placement score in the search value: the value of a state is the win probability plus this weight
        /// times the expected placement score (1 for first place to 0 for last), so the search can tell options apart when it
        /// is far behind and all of them have a win probability of about 0. 0 searches for the win probability alone; above 0
        /// needs a value net with placements (HasPlacements). Can be changed between decisions.
        /// </summary>
        public double PlacementWeight
        {
            get => _placementWeight;
            set
            {
                if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "The placement weight must not be negative.");
                if (value > 0 && !_stateValueNet.HasPlacements)
                    throw new ArgumentException("A placement weight above 0 needs a value net with a placements output.", nameof(value));
                _placementWeight = value;
                _search.PlacementWeight = value;
            }
        }

        /// <summary>
        /// How the search picks the other players' moves (see OpponentModel). Changing it starts a new tree at the next decision,
        /// so it can be set while the agent is thinking.
        /// </summary>
        public OpponentModel OpponentModel
        {
            get => _opponentModel;
            set => _opponentModel = value;
        }

        /// <summary>
        /// Search time per decision, can be changed between decisions; not used while IterationLimit is set
        /// </summary>
        public double TimeLimitSeconds
        {
            get => _timeLimitSeconds;
            set => _timeLimitSeconds = value;
        }

        /// <summary>
        /// Iterations per decision instead of the time limit (0 searches for TimeLimitSeconds). Unlike time, the search quality
        /// then doesn't depend on the load, e.g. with several games running at once. Rounded up to whole batches; can be
        /// changed between decisions.
        /// </summary>
        public long IterationLimit
        {
            get => _iterationLimit;
            set
            {
                if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "The iteration limit must not be negative.");
                _iterationLimit = value;
            }
        }

        public MCTSAgent(sbyte playerIndex, GameState initialState, double explorationParameter = 1.414, double timeLimitSeconds = TIME_LIMIT_SECONDS,
            StateValueNet? valueNet = null, IStateVectorizer? vectorizer = null, double placementWeight = 0, OpponentModel opponentModel = OpponentModel.MaxN,
            long iterationLimit = 0, int leavesPerBatch = DefaultLeavesPerBatch, bool parallelSelection = true, ILeafEvaluator? evaluator = null,
            int? seed = null)
            : base(playerIndex)
        {
            // A seed makes a search with selection on one thread repeat exactly: each new tree (and the agent's playouts) gets a
            // random source of its own, drawn from it. With selection on all cores the seed is ignored and all stay on the shared
            // thread-safe source: an unlocked source would break under many threads, a lock would slow the hot path, and the
            // order in which the threads draw would decide anyway
            // new Random(seed) runs .NET's older generator, kept for compatibility: 0.2 ns slower per draw than the shared one,
            // without a measurable difference in the search's speed or visits
            if (seed is int s && !parallelSelection)
            {
                _seeds = new Random(s);
                _random = new Random(_seeds.Next());
            }
            _explorationParameter = explorationParameter;
            _timeLimitSeconds = timeLimitSeconds;
            IterationLimit = iterationLimit;
            _stateValueNet = valueNet ?? DefaultValueNet.Value;
            _stateVectorizer = vectorizer ?? Vectorizers.For(_stateValueNet, playerIndex);
            // The leaf batches go to the net directly, or to an evaluator shared with other searches (e.g. a BatchingEvaluator
            // that merges the batches of games played at once), which has to run the same model
            if (evaluator != null && (evaluator.HasPlacements != _stateValueNet.HasPlacements || (evaluator.Policy == null) != (_stateValueNet.Policy == null)))
                throw new ArgumentException("The evaluator has to run the value net's model.", nameof(evaluator));
            // Selection on one thread repeats a search exactly from a seed, on all cores it is faster
            _search = new MCTSSearch(evaluator ?? new DirectEvaluator(_stateValueNet), _stateVectorizer, leavesPerBatch, parallelSelection);
            PlacementWeight = placementWeight;
            _opponentModel = opponentModel;

            ResetToState(initialState);
        }

        public override Action Act(GameState state)
        {
            return Act(state, CancellationToken.None);
        }

        /// <summary>
        /// Searches until the time or iteration limit or until the token is cancelled (after the batch in progress)
        /// </summary>
        public override Action Act(GameState state, CancellationToken cancellation)
        {
            // Trades between players are not part of the search tree: they are decided here, before and outside the search, by
            // the value net alone (the position with and without the trade). The tree below never contains a trade.
            // An open offer it has to act in: its answer or pick, and a counteroffer instead of a rejection if one is worth it.
            // The open-hand agent sees every hand, so its only world is the true position
            if (DecideTrade(state) is Action trade)
                return trade is RespondToTradeAction { Accept: false } && DecideCounter(state, _ => [state]) is CounterTradeAction counter ? counter : trade;
            // On turn: an offer of its own, if one is worth it
            if (DecideOffer(state, _ => [state]) is OfferTradeAction offer) return offer;

            // A single legal move needs no search
            if (PlayForcedMovesAtOnce)
            {
                List<Action> legal = MCTSTree.GetValidActions(state);
                if (legal.Count == 1) return legal[0].Clone();
            }

            // Reuse the tree kept up to date by HandleAction as long as it describes exactly this state.
            // Otherwise (first move, missed or manual actions, undo, another opponent model) start over on a private copy of the state.
            if (_tree == null || _tree.Model != _opponentModel || !_tree.RootState.Equals(state))
            {
                // With a seed the new tree gets a source of its own, the next one from the seed's stream; else the shared one
                _tree = new MCTSTree(new GameState(state), PlayerIndex, _explorationParameter, NormalizeValues, _opponentModel,
                    random: _seeds != null ? new Random(_seeds.Next()) : null);
            }
            _tree.FirstPlayReduction = FirstPlayReduction;
            _tree.EvaluateNewLeaves = EvaluateNewLeaves;

            // The budget of this decision (more for the initial placements, see SetupBudgetFactor)
            (double seconds, long iterations) = BudgetFor(state);
            if (iterations > 0) RunBatchedIterations(iterations, cancellation);
            else RunForTime(seconds, cancellation);

            return _tree.GetMostPromisingMove();
        }

        public override void HandleAction(Action action)
        {
            // Called after the action was applied to the real game, so its output randomness is resolved in its history
            if (!ReuseTree) _tree = null;
            if (_tree == null) return;

            MCTSNode? nextRoot = _tree.FindChildForAction(action);
            GameState nextState = new GameState(_tree.RootState);
            Action replay = action.Clone();

            // Without a matching explored child, or if the tree lost track of the real game, the next Act starts over
            if (nextRoot == null || !replay.IsValidFor(nextState))
            {
                _tree = null;
                return;
            }

            // Advance the root state with the real action instead of the tree's: equivalent outcomes keep their actual
            // details (e.g. the dice pair behind a roll total), so the root state stays identical to the real game state
            replay.Apply(nextState);
            _tree.ReRoot(nextRoot, nextState);
        }

        /// <summary>
        /// The current tree (for tests)
        /// </summary>
        internal MCTSTree? Tree => _tree;

        public override void ResetToState(GameState state)
        {
            // The tree is rebuilt for the given state by the next Act
            _tree = null;
        }

        /// <summary>
        /// A root move of the search: its visits, its prior and its mean value for each player
        /// </summary>
        public sealed record RootMove(Action Move, uint Visits, float Prior, double[] MeanValues);

        /// <summary>
        /// The root moves of the current tree and the state they belong to, for showing the search (also while it runs: the
        /// counts are read without a lock, so they can lag a batch behind); empty without a tree or for a chance node
        /// </summary>
        public (GameState? state, IReadOnlyList<RootMove> moves) GetRootMoves()
        {
            MCTSTree? tree = _tree;
            if (tree == null) return (null, []);

            List<RootMove> moves = [];
            foreach (MCTSNode child in tree.RootNode.Children)
            {
                if (child.LastAction == null) return (tree.RootState, []);
                uint resolved = child.ResolvedVisitCount;
                double[] means = new double[MCTSNode.PlayerCount];
                for (int player = 0; player < means.Length; player++)
                    means[player] = resolved == 0 ? 0 : child.PlayerScore(player) / resolved;
                moves.Add(new RootMove(child.LastAction, child.VisitCount, child.Prior, means));
            }
            return (tree.RootState, moves);
        }

        public void RunIterations(int iterations)
        {
            Console.WriteLine($"Running {iterations} MCTS iterations:");
            Stopwatch stopwatch = Stopwatch.StartNew();

            for (int i = 0; i < iterations; i++)
            {
                RunOneIteration();
            }

            TimeSpan runtime = stopwatch.Elapsed;

            Console.WriteLine($"Simulation took {runtime.TotalSeconds}s.");
        }

        public void RunForTime(double timeLimitSeconds, CancellationToken cancellation = default)
        {
            Console.WriteLine($"Running MCTS for {timeLimitSeconds}s:");
            RunSearch(timeLimitSeconds, 0, cancellation);
        }

        /// <summary>
        /// The batched search of RunForTime, for a number of iterations (rounded up to whole batches) instead of a time
        /// </summary>
        public void RunBatchedIterations(long iterations, CancellationToken cancellation = default)
        {
            Console.WriteLine($"Running MCTS for {iterations} iterations:");
            RunSearch(double.PositiveInfinity, iterations, cancellation);
        }

        private void RunSearch(double timeLimitSeconds, long iterationLimit, CancellationToken cancellation)
        {
            // Only called while Act holds a tree
            _search.Run(_tree!, timeLimitSeconds, iterationLimit, cancellation);
            Console.WriteLine($"Simulation ran for {LastSearch!.Iterations} iterations.");
        }

        protected void RunOneIteration()
        {
            // Only called while Act holds a tree
            MCTSTree tree = _tree!;

            // Selection
            // Find a leaf node to expand, starting from the root node and traversing down the tree using the UCT formula
            (MCTSNode leafNode, GameState stateAtLeaf) = tree.SelectLeafNode();

            // Expansion
            // If the selected leaf isn't terminal, expand its children and select a leaf node among them
            if (!leafNode.IsTerminal)
            {
                tree.ExpandNode(leafNode, stateAtLeaf);
                (leafNode, stateAtLeaf) = tree.SelectLeafNode(leafNode, stateAtLeaf, suppressStateCopy: true);
            }

            // Simulation
            // Ended matches have a known result, everything else is estimated by the value net
            Span<double> utilities = stackalloc double[MCTSNode.PlayerCount];
            if (stateAtLeaf.HasEnded) TerminalUtilities(stateAtLeaf, utilities);
            else SimulateUtilities(stateAtLeaf, utilities);

            // Backpropagation
            // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root
            if (tree.Model == OpponentModel.Random) tree.Backpropagate(leafNode, utilities[PlayerIndex]);
            else tree.Backpropagate(leafNode, utilities);
        }

        private void LeafUtilities(ReadOnlySpan<float> winProbabilities, ReadOnlySpan<float> placements, Span<double> destination)
        {
            MCTSSearch.LeafUtilities(winProbabilities, placements, _placementWeight, destination);
        }

        private void TerminalUtilities(GameState state, Span<double> destination)
        {
            MCTSSearch.TerminalUtilities(state, _placementWeight, destination);
        }

        protected double TerminalResult(GameState state)
        {
            Span<double> utilities = stackalloc double[MCTSNode.PlayerCount];
            TerminalUtilities(state, utilities);
            return utilities[PlayerIndex];
        }

        /// <summary>
        /// Expected placement score of a placement distribution over places 1 to 4
        /// </summary>
        public static float PlacementScore(ReadOnlySpan<float> placementDistribution)
        {
            float score = 0;
            for (int place = 0; place < PlacementScores.Length; place++)
                score += placementDistribution[place] * PlacementScores[place];
            return score;
        }

        /// <summary>
        /// Placement score of a player in an ended match: the winner is first, the others are ranked by victory points, and
        /// players with equal victory points share their places (the same rule as the placement labels in playout_results.py)
        /// </summary>
        public static float TerminalPlacementScore(GameState state, int player)
        {
            int winner = state.Turn.PlayerIndex;
            if (player == winner) return PlacementScores[0];

            int points = state.Players[player].VictoryPoints.Total;
            int better = 1, tied = 0;
            for (int p = 0; p < state.Players.Length; p++)
            {
                if (p == winner) continue;
                int other = state.Players[p].VictoryPoints.Total;
                if (other > points) better++;
                else if (other == points) tied++;
            }

            float score = 0;
            for (int place = better; place < better + tied; place++)
                score += PlacementScores[place];
            return score / tied;
        }

        protected double SimulatePlayout(GameState state)
        {
            //Console.WriteLine("Sim state hash: " + state.GetHashCode());

            while (!state.HasEnded)
            {
                List<Action> actions = MCTSTree.GetValidActions(state);
                if (actions.Count == 0)
                {
                    Console.WriteLine("No actions available but not over: Saving state");

                    // Write GameState to file
                    SaveFile saveFile = new SaveFile(state, [], []);
                    string saveData = SaveFileSerializer.Serialize(saveFile);
                    File.WriteAllText("messup.yaml", saveData);

                    throw new InvalidOperationException("No valid actions available.");
                }
                // A random move from the agent's source (seeded with a seed and selection on one thread)
                Action action = actions[_random.Next(actions.Count)];
                action.Apply(state);
                //Console.WriteLine("In sim state hash: " + state.GetHashCode());
            }
            return state.Turn.PlayerIndex == PlayerIndex ? 1 : 0;
        }

        protected double SimulateML(GameState state)
        {
            Span<double> utilities = stackalloc double[MCTSNode.PlayerCount];
            SimulateUtilities(state, utilities);
            return utilities[PlayerIndex];
        }

        private void SimulateUtilities(GameState state, Span<double> destination)
        {
            float[] features = new float[_stateVectorizer.FeatureDim];
            _stateVectorizer.VectorizeInto(state, features);
            var input = new DenseTensor<float>(features, [1, features.Length]);

            if (_placementWeight == 0)
            {
                LeafUtilities(_stateValueNet.Run(input), default, destination);
                return;
            }

            (float[] winProbabilities, float[] placements) = _stateValueNet.RunWithPlacements(input);
            LeafUtilities(winProbabilities, placements, destination);
        }

        /// <summary>
        /// A player's utility of a position by the value net, as the search scores its leaves
        /// </summary>
        private float TradeUtility(GameState state, int player)
        {
            Span<double> utilities = stackalloc double[MCTSNode.PlayerCount];
            SimulateUtilities(state, utilities);
            return (float)utilities[player];
        }

        // Features of the positions an offer is judged on, kept between decisions
        private float[] _offerFeatures = [];

        /// <summary>
        /// Positions per run of the net when judging trades: every run has this shape (the last one padded with empty rows),
        /// since CUDA's memory pool grows with every new batch shape until the GPU runs out of memory
        /// </summary>
        public const int UtilityBatch = 256;

        /// <summary>
        /// Every player's utility of each position by the value net, as the search scores its leaves, in batches of
        /// UtilityBatch: PlayerCount values per position
        /// </summary>
        public double[] Utilities(IReadOnlyList<GameState> states)
        {
            int dim = _stateVectorizer.FeatureDim, players = MCTSNode.PlayerCount;
            // One feature buffer for a full batch, reused across decisions (no per-call allocation on the large object heap)
            if (_offerFeatures.Length < UtilityBatch * dim) _offerFeatures = new float[UtilityBatch * dim];
            double[] utilities = new double[states.Count * players];
            for (int start = 0; start < states.Count; start += UtilityBatch)
            {
                // Positions in this chunk; the rest of the batch stays zero and its outputs are ignored
                int rows = Math.Min(UtilityBatch, states.Count - start);
                Span<float> features = _offerFeatures.AsSpan(0, UtilityBatch * dim);
                features.Clear();
                for (int i = 0; i < rows; i++)
                    _stateVectorizer.VectorizeInto(states[start + i], features.Slice(i * dim, dim));
                // Always the full shape [UtilityBatch, dim], whatever the number of positions
                var input = new DenseTensor<float>(_offerFeatures.AsMemory(0, UtilityBatch * dim), [UtilityBatch, dim]);

                // Utilities as at the search's leaves: win probabilities alone, or mixed with the placements
                if (_placementWeight == 0)
                {
                    float[] winProbabilities = _stateValueNet.Run(input);
                    for (int i = 0; i < rows; i++)
                        LeafUtilities(winProbabilities.AsSpan(i * players, players), default, utilities.AsSpan((start + i) * players, players));
                }
                else
                {
                    // Placements come as four probabilities (1st to 4th) per player
                    (float[] winProbabilities, float[] placements) = _stateValueNet.RunWithPlacements(input);
                    for (int i = 0; i < rows; i++)
                        LeafUtilities(winProbabilities.AsSpan(i * players, players), placements.AsSpan(i * players * 4, players * 4), utilities.AsSpan((start + i) * players, players));
                }
            }
            return utilities;
        }

        protected float ValuationToScore(ReadOnlySpan<float> valuation)
        {
            float sum = 0;
            foreach (float value in valuation)
                sum += value;

            return valuation[PlayerIndex] / sum;
        }

        protected float[] StateValueFunc(GameState state)
        {
            float[] features = new float[_stateVectorizer.FeatureDim];
            _stateVectorizer.VectorizeInto(state, features);
            return _stateValueNet.Run(new DenseTensor<float>(features, [1, features.Length]));
        }
    }
}
