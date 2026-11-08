using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Agents
{
    /// <summary>
    /// How agents take part in trades between players: they answer offers addressed to them and, as the player on turn, pick a
    /// partner or let an offer go (their own offers and counteroffers come from TradeOffers). Trades between players are not
    /// part of the search tree: agents decide them here, before and outside any search (see MCTSTree). A trade is judged by
    /// comparing the position after it with the position without it, by a value function if the agent has one (the value
    /// net's utility) or else by how close the hand comes to the next buildings.
    /// </summary>
    public static class TradeDecisions
    {
        // A trade has to be better than this to be worth it, so agents don't trade for noise in the value
        internal const float ValueMargin = 0.002f, HandMargin = 0.15f;

        /// <summary>
        /// The share of the offerer's gain a value-net bot needs by default to accept an offer: as much as the offerer gains
        /// (it played best at a table with free trade and is far harder to exploit by lopsided offers)
        /// </summary>
        public const float DefaultFairness = 1f;

        // Victory points short of the win from which agents stop helping a player
        internal const int LeaderDistance = 2;

        /// <summary>
        /// The agent's move in an open trade, or null if there is no open trade it has to act in
        /// </summary>
        /// <param name="value">The agent's utility of a position for a player, or null for the hand heuristic</param>
        /// <param name="margin">How much better a trade has to be by the value function to accept it</param>
        /// <param name="fairness">With a value function, an offer is only accepted if it raises the agent's value by at least
        /// this share of what it raises the offerer's (0: any gain above the margin)</param>
        public static Action? Decide(GameState state, sbyte player, Func<GameState, int, float>? value = null, float margin = ValueMargin, float fairness = 0)
        {
            if (state.Turn.PendingTrade is not TradeOffer offer) return null;

            // The agent's answer is due (in seat order, or under JSettlers' flow whenever it hasn't answered yet)
            if (RespondToTradeAction.MayAnswer(state, player))
            {
                RespondToTradeAction reject = new RespondToTradeAction(player, false), accept = new RespondToTradeAction(player, true);
                // Without the asked cards it can only turn the offer down
                if (!accept.IsValidFor(state)) return reject;

                // No help for a player who is about to win
                if (state.Players[offer.Offerer].VictoryPoints.Total >= state.Settings.VictoryPoints - LeaderDistance) return reject;

                // The position if it accepts and the offerer picks it
                GameState traded = AfterTrade(state, offer.Offerer, player, offer.Give, offer.Get);
                if (!IsBetter(traded, state, player, value, margin)) return reject;
                // Fair answers: its own gain has to be at least that share of the offerer's, by the same value function
                if (value != null && fairness > 0
                    && value(traded, player) - value(state, player) < fairness * (value(traded, offer.Offerer) - value(state, offer.Offerer))) return reject;
                return accept;
            }

            // The agent's own offer and every answer is in (under JSettlers' flow: whenever it is asked): pick a partner or let
            // the offer go
            if (ConfirmTradeAction.MayPick(state, player))
            {
                // Letting it go keeps the position; a partner has to beat that, and the best one wins
                Action best = new CancelTradeAction(player);
                GameState bestState = state;
                // One confirm per player who accepted or countered (under JSettlers' flow only counteroffers stand)
                IEnumerable<Action> confirms = Enumerable.Range(0, state.Players.Length).Select(partner => (Action)new ConfirmTradeAction(player, (sbyte)partner))
                    .Where(confirm => confirm.IsValidFor(state));
                foreach (Action confirm in confirms)
                {
                    GameState after = new GameState(state);
                    confirm.Apply(after);
                    if (IsBetter(after, bestState, player, value))
                    {
                        best = confirm;
                        bestState = after;
                    }
                }
                return best;
            }

            return null;
        }

        /// <summary>
        /// The position after the offerer gives one set of cards to the partner and gets the other, with the offer closed
        /// </summary>
        internal static GameState AfterTrade(GameState state, int offerer, int partner, CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)
        {
            // Moves the cards directly as ConfirmTradeAction.Transfer does, without needing an open offer with an accepted partner
            GameState after = new GameState(state);
            after.Turn.PendingTrade = null;
            after.Players[offerer].ResourceCards.Remove(give);
            after.Players[partner].ResourceCards.Add(give);
            after.Players[partner].ResourceCards.Remove(get);
            after.Players[offerer].ResourceCards.Add(get);
            return after;
        }

        private static bool IsBetter(GameState candidate, GameState current, int player, Func<GameState, int, float>? value, float margin = ValueMargin)
        {
            if (value != null) return value(candidate, player) > value(current, player) + margin;
            return HandValue(candidate.Players[player].ResourceCards) > HandValue(current.Players[player].ResourceCards) + HandMargin;
        }

        private static readonly (float weight, (ResourceCardType type, uint amount)[] cost)[] Buildings =
        [
            (1f, [(ResourceCardType.Lumber, 1), (ResourceCardType.Brick, 1)]),
            (3f, [(ResourceCardType.Lumber, 1), (ResourceCardType.Brick, 1), (ResourceCardType.Wool, 1), (ResourceCardType.Grain, 1)]),
            (3f, [(ResourceCardType.Grain, 2), (ResourceCardType.Ore, 3)]),
            (1.5f, [(ResourceCardType.Wool, 1), (ResourceCardType.Grain, 1), (ResourceCardType.Ore, 1)]),
        ];

        /// <summary>
        /// How close a hand comes to the buildings: per building the share of its cost covered, squared so that completing one
        /// counts most, weighted by what the building is worth; a little for every card on top
        /// </summary>
        public static float HandValue(CardSet<ResourceCardType> hand)
        {
            float value = 0.02f * hand.Count();
            foreach ((float weight, (ResourceCardType type, uint amount)[] cost) in Buildings)
            {
                uint covered = 0, total = 0;
                foreach ((ResourceCardType type, uint amount) in cost)
                {
                    covered += Math.Min(hand.Get(type), amount);
                    total += amount;
                }
                float share = (float)covered / total;
                value += weight * share * share;
            }
            return value;
        }
    }
}
