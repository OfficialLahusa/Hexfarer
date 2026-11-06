using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    /// <summary>
    /// Plays under hidden information with information set MCTS (ISMCTSTree): it follows the game as its seat sees it
    /// (BeliefTracker) and searches one tree over its information sets, a world drawn for every iteration, with the net, the
    /// time or iteration budget and the settings of the MCTS agent of its seat. Trades between players are decided outside
    /// the tree: answers on a world drawn from the view, offers over several (see TradeOffers). The true state it is given is
    /// only used for its public parts.
    /// </summary>
    public sealed class InformationSetMCTSAgent : Agent, IShowsSearch
    {
        private readonly MCTSAgent _search;
        private readonly Random _random;

        // The game as the agent followed it, to give the tracker the positions before and after each move
        private GameState _mirror;
        private BeliefTracker _beliefs;

        // The node of the last tree for the position the game has reached (moves followed as the seat saw them), whose subtree
        // the next search goes on with; null once the game left the tree
        private ISNode? _reusable;

        /// <param name="search">The MCTS agent of the same seat, whose net, budget and settings the search uses</param>
        /// <param name="seed">Seed of the worlds, random if null</param>
        public InformationSetMCTSAgent(MCTSAgent search, GameState initialState, int? seed = null)
            : base(search.PlayerIndex)
        {
            _search = search;
            _random = seed is int s ? new Random(s) : new Random();
            _mirror = new GameState(initialState);
            _beliefs = new BeliefTracker(PlayerIndex, initialState);
        }

        /// <summary>
        /// What the agent knows about the game
        /// </summary>
        public BeliefTracker Beliefs => _beliefs;

        /// <summary>
        /// The game as the agent followed it (for checks that it still follows the real game; not to be changed)
        /// </summary>
        public GameState Mirror => _mirror;

        public MCTSAgent Search => _search;

        /// <summary>
        /// Searches the true position instead of worlds drawn from the beliefs (an ablation: the same search without hidden
        /// information)
        /// </summary>
        public bool FullKnowledge { get; init; }

        /// <summary>
        /// Ablations of the tree (see ISMCTSTree): branching on outcomes the seat can't see, normalized values
        /// </summary>
        public bool BranchHiddenOutcomes { get; init; }

        public bool NormalizeValues { get; init; }

        /// <summary>
        /// Whether its beliefs take in the cards offers, accepts and counteroffers show (see BeliefTracker.LearnsFromTrades)
        /// </summary>
        public bool LearnsFromTrades
        {
            get => _learnsFromTrades;
            init
            {
                _learnsFromTrades = value;
                _beliefs.LearnsFromTrades = value;
            }
        }

        private readonly bool _learnsFromTrades = true;

        /// <summary>
        /// Whether the next decision goes on with the subtree of the moves played since the last search (the MCTS agent's
        /// ReuseTree, as for the open-hand search) instead of a new tree
        /// </summary>
        public bool ReuseTree => _search.ReuseTree;

        /// <summary>
        /// Factor (above 0, at most 1) by which the statistics of a reused subtree are scaled before the next search (see
        /// ISNode.Decay): they come from worlds drawn from earlier beliefs, so they weigh less against the new iterations
        /// </summary>
        public double ReuseDecay
        {
            get => _reuseDecay;
            init => _reuseDecay = value is > 0 and <= 1 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "The decay of a reused tree is above 0 and at most 1.");
        }

        private readonly double _reuseDecay = 1;

        /// <summary>
        /// The tree of the last decision, for showing the search
        /// </summary>
        public ISMCTSTree? LastTree { get; private set; }

        /// <summary>
        /// The node the next search would go on with (for tests)
        /// </summary>
        internal ISNode? ReusableRoot => _reusable;

        public override Action Act(GameState state) => Act(state, CancellationToken.None);

        public override Action Act(GameState state, CancellationToken cancellation)
        {
            _searched = null;

            // Trades between players are not part of the search tree: answers, partner picks, counteroffers and offers are
            // decided here, before and outside the search (see MCTSAgent.Act)
            // An answer or pick on one world drawn from the view; a counteroffer instead of a rejection if one is worth it over
            // several worlds
            if (_search.DecideTrade(FullKnowledge ? state : _beliefs.Determinize(_beliefs.View(state), _random)) is Action trade)
            {
                return trade is RespondToTradeAction { Accept: false }
                    && _search.DecideCounter(state, count => FullKnowledge ? [state] : Worlds(state, count)) is CounterTradeAction counter ? counter : trade;
            }
            // On turn: an offer of its own, if one is worth it over the worlds
            if (_search.DecideOffer(state, count => FullKnowledge ? [state] : Worlds(state, count)) is OfferTradeAction offer) return offer;

            // A single legal move needs no search
            List<Action> legal = MCTSTree.GetValidActions(state);
            if (legal.Count == 1) return legal[0].Clone();

            // Go on with the subtree the game reached since the last search (if HandleAction could follow every move), its
            // statistics scaled down first if a decay is set
            ISNode? reused = ReuseTree ? _reusable : null;
            if (reused != null && ReuseDecay < 1) reused.Decay(ReuseDecay);
            ISMCTSTree tree = new ISMCTSTree(state, _beliefs, _search.ExplorationParameter, _random.Next(), FullKnowledge, reused)
            {
                FirstPlayReduction = _search.FirstPlayReduction,
                BranchHiddenOutcomes = BranchHiddenOutcomes,
                NormalizeValues = NormalizeValues
            };
            LastTree = tree;
            _searched = new GameState(state);
            _search.RunSearch(tree, cancellation);
            // Kept for the next decision; HandleAction walks it down with every move played
            _reusable = ReuseTree ? tree.RootNode : null;
            // Only among the moves legal now: a reused root keeps every child it ever had
            return tree.GetMostPromisingMove(legal);
        }

        // Worlds drawn from what the seat knows
        private List<GameState> Worlds(GameState state, int count)
        {
            GameState view = _beliefs.View(state);
            return Enumerable.Range(0, count).Select(_ => _beliefs.Determinize(view, _random)).ToList();
        }

        // The position of the last tree, for showing its search
        private GameState? _searched;

        public (GameState? state, IReadOnlyList<MCTSAgent.RootMove> moves) GetRootMoves() =>
            LastTree is ISMCTSTree tree && _searched is GameState state ? (state, tree.RootMoves()) : (null, []);

        public override void HandleAction(Action action)
        {
            // Called after the action was applied to the real game, so its outcome is in its history
            GameState before = new GameState(_mirror);
            Action replayed = action.Clone();
            if (!replayed.IsValidFor(_mirror))
            {
                ResetToState(_mirror);
                return;
            }
            replayed.Apply(_mirror);
            _beliefs.Observe(before, replayed, _mirror);
            // Down the tree only by what the seat saw of the move (its own moves and their outcomes in full)
            _reusable = _reusable?.Follow(replayed, PlayerIndex, BranchHiddenOutcomes);
        }

        /// <summary>
        /// Goes on from a position with what the seat knows there (see DeterminizedMCTSAgent.ResetTo)
        /// </summary>
        public void ResetTo(GameState state, BeliefTracker beliefs)
        {
            if (beliefs.Observer != PlayerIndex) throw new ArgumentException($"The beliefs are seat {beliefs.Observer}'s, not seat {PlayerIndex}'s.", nameof(beliefs));
            _mirror = new GameState(state);
            _beliefs = new BeliefTracker(beliefs) { LearnsFromTrades = _learnsFromTrades };
            _reusable = null;
        }

        public override void ResetToState(GameState state)
        {
            // Without the history the agent knows the hands of the given position (see BeliefTracker's constructor)
            _mirror = new GameState(state);
            _beliefs = new BeliefTracker(PlayerIndex, state) { LearnsFromTrades = _learnsFromTrades };
            _reusable = null;
        }
    }
}
