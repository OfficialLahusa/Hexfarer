namespace Common.Actions
{
    /// <summary>
    /// An addressed player accepts or rejects the open offer, in turn. Accepting needs the cards asked for. When the last
    /// answer leaves nobody to trade with, the offer closes.
    /// </summary>
    public class RespondToTradeAction : Action, IReplayAction, IActionProvider
    {
        public bool Accept { get; init; }

        /// <summary>
        /// The offer this answer closed (when nobody is left to trade with), restored by Revert
        /// </summary>
        public TradeOffer? History { get; private set; }

        public RespondToTradeAction(sbyte playerIdx, bool accept)
            : base(playerIdx)
        {
            Accept = accept;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private RespondToTradeAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            TradeOffer offer = state.Turn.PendingTrade!;
            History = null;
            // JSettlers' flow: the first accept trades at once on the offer's terms and closes the offer (History holds it as
            // it stood, the answer still pending)
            if (Accept && state.Settings.TradeFlow == TradeFlow.FirstAccept)
            {
                History = new TradeOffer(offer);
                ConfirmTradeAction.Transfer(state, offer.Offerer, PlayerIndex, offer.Give, offer.Get);
                state.Turn.PendingTrade = null;
                return;
            }
            offer.Responses[PlayerIndex] = Accept ? TradeResponse.Accepted : TradeResponse.Rejected;
            // Everyone answered and nobody accepted or countered: the offer closes by itself
            if (offer.NextResponder == null && !offer.HasPartner)
            {
                History = new TradeOffer(offer);
                state.Turn.PendingTrade = null;
            }
        }

        public override void Revert(GameState state)
        {
            // An accept with a History traded at once (JSettlers' flow; under the rulebook an accept never closes the offer):
            // the cards go back and the offer stands again
            if (Accept && History != null)
            {
                TradeOffer offer = new TradeOffer(History);
                ConfirmTradeAction.Transfer(state, offer.Offerer, PlayerIndex, offer.Get, offer.Give);
                state.Turn.PendingTrade = offer;
                return;
            }
            if (History != null) state.Turn.PendingTrade = new TradeOffer(History);
            state.Turn.PendingTrade!.Responses[PlayerIndex] = TradeResponse.Pending;
        }

        public override bool IsValidFor(GameState state)
        {
            return MayAnswer(state, PlayerIndex)
                && (!Accept || state.Players[PlayerIndex].ResourceCards.Contains(state.Turn.PendingTrade!.Get));
        }

        /// <summary>
        /// Whether the player may answer the open offer now: the next addressed player in seat order, or under JSettlers'
        /// flow any addressed player who hasn't answered yet
        /// </summary>
        public static bool MayAnswer(GameState state, int playerIdx)
        {
            if (IsTurnValid(state.Turn, playerIdx)) return true;
            return state.Settings.TradeFlow == TradeFlow.FirstAccept && state.Turn.PendingTrade is TradeOffer offer
                && playerIdx >= 0 && playerIdx < offer.Responses.Length && offer.Responses[playerIdx] == TradeResponse.Pending;
        }

        /// <summary>
        /// Whether the player is the one to answer the open offer now
        /// </summary>
        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PendingTrade?.NextResponder == playerIdx;
        }

        public bool HasHistory() => History != null;

        public void ClearHistory() => History = null;

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            if (!IsTurnValid(state.Turn, playerIdx)) return [];
            List<Action> actions = [new RespondToTradeAction(playerIdx, false)];
            if (state.Players[playerIdx].ResourceCards.Contains(state.Turn.PendingTrade!.Get)) actions.Add(new RespondToTradeAction(playerIdx, true));
            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + (Accept ? ", accepts the offer" : ", rejects the offer");
        }
    }
}
