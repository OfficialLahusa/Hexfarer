using Common;
using Common.Actions;

namespace Agents
{
    /// <summary>
    /// How an agent on turn makes trade offers to the other players, and counteroffers to offers it turns down
    /// (TradeDecisions answers offers and picks the partner). Trades between players are not part of the search tree: offers
    /// and counteroffers are decided here, before and outside any search, by the value net alone (see MCTSTree). Before its
    /// move it looks at every trade of up to three of its cards for up to three cards of other types (MaxCards), addressed
    /// to every other player not about to win. With a
    /// value function it judges each offer over worlds of what the others may hold: in each, who holds the asked cards
    /// accepts if the trade raises their own value by the margin the agents answer with, and the agent takes the accepting
    /// partner best for itself if that beats the same margin; the offer is worth its mean gain over the worlds (all
    /// candidates in the first world, the Prune best of them in all), and the agent makes the best one worth more than
    /// Margin. Without a value function (the heuristic agents, which follow no beliefs) it judges only its own hand, looks
    /// at one or two cards a side and asks for no more cards than it gives. At most MaxPerTurn offers per turn, never the
    /// same terms twice in a turn. It reads from the true position only what its seat sees: its own hand, the turn and the
    /// public victory points. The defaults are those that played best at a table with free trade (see TODO.md).
    /// </summary>
    public sealed class TradeOffers
    {
        /// <summary>
        /// Whether the agent makes offers at all
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Mean gain in the agent's value an offer needs (more than for an answer: an offer shows cards and costs a round of answers)
        /// </summary>
        public float Margin { get; set; } = 0.005f;

        public int MaxPerTurn { get; set; } = 3;

        /// <summary>
        /// Worlds drawn from the agent's beliefs to judge an offer under hidden information
        /// </summary>
        public int Worlds { get; set; } = 8;

        /// <summary>
        /// Whether two cards for two are among the offers too
        /// </summary>
        public bool TwoForTwo { get; set; }

        /// <summary>
        /// Whether three cards for one and one for three are among the offers too
        /// </summary>
        public bool ThreeForOne { get; set; }

        /// <summary>
        /// Most cards on each side with a value function: 2 (the shapes above) or 3 (any trade up to three cards for three);
        /// the hand heuristic, which can't tell who would accept, keeps to the shapes of 2
        /// </summary>
        public int MaxCards { get; set; } = 3;

        /// <summary>
        /// If above 0 and there are more candidates: every candidate is judged in the first world only, and only this many of
        /// the best there in all worlds (the others count as worthless), which saves most of the net's work for many candidates
        /// </summary>
        public int Prune { get; set; } = 16;

        /// <summary>
        /// Whether the agent answers an offer it turns down with a counteroffer (only with a value function): the terms best
        /// for it among those the offerer would take, judged like its own offers but with the offerer as the only partner
        /// </summary>
        public bool Counters { get; set; } = true;

        /// <summary>
        /// The share of the agent's gain it expects a partner to need to accept an offer (0: any gain above the margin). Not
        /// the fairness the bots answer with by default: a bot expecting fair partners offered half as often and lost among
        /// bots that take any gain
        /// </summary>
        public float PartnerFairness { get; set; }

        // The terms offered this turn, forgotten when a turn starts without offers
        private readonly List<(CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)> _offered = [];

        /// <summary>
        /// The agent's offer now, or null
        /// </summary>
        /// <param name="worlds">Full positions the agent can't tell from the true one, as many as asked (only called if an
        /// offer is possible); the true position for an agent that sees every hand</param>
        /// <param name="utilities">Every player's value of each position (PlayerCount values per position, in one batch), or
        /// null for the hand heuristic</param>
        public OfferTradeAction? Decide(GameState state, sbyte player, Func<int, IReadOnlyList<GameState>>? worlds,
            Func<IReadOnlyList<GameState>, double[]>? utilities)
        {
            // No offer made yet this turn: the terms remembered are from an earlier turn
            if (state.Turn.TradeOffersThisTurn == 0) _offered.Clear();
            // Only the player on turn after the roll, and only up to MaxPerTurn offers
            if (!Enabled || !OfferTradeAction.IsTurnValid(state, player) || state.Turn.TradeOffersThisTurn >= MaxPerTurn) return null;
            // Offering gives up the roads of a Road Building card not placed yet
            if (state.Players[player].BuildingStock.FreeRoads > 0) return null;
            // Nobody helps a player about to win
            if (IsAboutToWin(state, player)) return null;

            // Everyone else not about to win; none left, no offer
            byte recipients = Recipients(state, player);
            if (recipients == 0) return null;

            // Every trade the own hand can pay (the heuristic only up to two cards a side), without terms already offered this turn
            CardSet<ResourceCardType> hand = state.Players[player].ResourceCards;
            List<(CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)> candidates = Candidates(hand, TwoForTwo, utilities == null, ThreeForOne, utilities == null ? 2 : MaxCards)
                .Where(terms => !_offered.Any(o => o.give.Equals(terms.give) && o.get.Equals(terms.get))).ToList();
            if (candidates.Count == 0) return null;

            // Each candidate's worth: the gain in hand value for the heuristic, else the mean gain over the worlds by the net
            double[] worth = utilities == null || worlds == null
                ? candidates.Select(c => (double)(TradeDecisions.HandValue(After(hand, c.give, c.get)) - TradeDecisions.HandValue(hand))).ToArray()
                : Pruned(worlds(Math.Max(1, Worlds)), candidates, Prune, (w, c) => ExpectedGains(w, player, recipients, c, utilities, PartnerFairness));
            // Hand values are on another scale than the net's utilities, so the heuristic has its own margin
            double margin = utilities == null || worlds == null ? TradeDecisions.HandMargin : Margin;

            // The best candidate above the margin (on ties the first)
            int best = -1;
            for (int c = 0; c < candidates.Count; c++)
                if (worth[c] > margin && (best < 0 || worth[c] > worth[best])) best = c;
            if (best < 0) return null;

            // Remembered so the same terms aren't offered again this turn
            _offered.Add(candidates[best]);
            return new OfferTradeAction(player, candidates[best].give, candidates[best].get, recipients);
        }

        /// <summary>
        /// The agent's counteroffer to the open offer it has to answer, or null: only if it would turn the offer down (the
        /// caller decides that), not to a player about to win, and worth more than Margin on average over the worlds
        /// </summary>
        public CounterTradeAction? DecideCounter(GameState state, sbyte player, Func<int, IReadOnlyList<GameState>> worlds,
            Func<IReadOnlyList<GameState>, double[]> utilities)
        {
            // Only the player whose answer is due
            if (!Counters || state.Turn.PendingTrade is not TradeOffer offer || offer.NextResponder != player) return null;
            // Nobody helps a player about to win
            if (IsAboutToWin(state, offer.Offerer)) return null;

            // In the offerer's terms: it gives what the agent asks for and gets cards of the agent's hand; the offer's own terms
            // are no counteroffer
            List<(CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)> candidates = Candidates(state.Players[player].ResourceCards, TwoForTwo, false, ThreeForOne, MaxCards)
                .Select(c => (give: c.get, get: c.give))
                .Where(c => !(c.give.Equals(offer.Give) && c.get.Equals(offer.Get))).ToList();
            if (candidates.Count == 0) return null;

            // Mean gain over the worlds, pruned as for offers; the best above the margin
            double[] worth = Pruned(worlds(Math.Max(1, Worlds)), candidates, Prune, (w, c) => CounterGains(w, offer.Offerer, player, c, utilities));
            int best = -1;
            for (int c = 0; c < candidates.Count; c++)
                if (worth[c] > Margin && (best < 0 || worth[c] > worth[best])) best = c;
            return best < 0 ? null : new CounterTradeAction(player, candidates[best].give, candidates[best].get);
        }

        /// <summary>
        /// Each candidate's worth by gains(worlds, candidates); with prune above 0 and more candidates, only the prune best in
        /// the first world are judged in all worlds, the others are worth -infinity
        /// </summary>
        internal static double[] Pruned(IReadOnlyList<GameState> worlds, List<(CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)> candidates, int prune,
            Func<IReadOnlyList<GameState>, IReadOnlyList<(CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)>, double[]> gains)
        {
            // Nothing to save: every candidate in every world
            if (prune <= 0 || worlds.Count < 2 || candidates.Count <= prune) return gains(worlds, candidates);

            // First stage: every candidate in the first world only
            double[] first = gains([worlds[0]], candidates);
            // The prune best there (ties by index, so the choice doesn't depend on the sort)
            int[] kept = Enumerable.Range(0, candidates.Count).OrderByDescending(c => first[c]).ThenBy(c => c).Take(prune).ToArray();
            // Second stage: only those in all worlds
            double[] keptWorth = gains(worlds, kept.Select(c => candidates[c]).ToList());
            // Back to the candidates' own indices; the dropped ones can't be picked
            double[] worth = Enumerable.Repeat(double.NegativeInfinity, candidates.Count).ToArray();
            for (int k = 0; k < kept.Length; k++) worth[kept[k]] = keptWorth[k];
            return worth;
        }

        /// <summary>
        /// The mean over the worlds of each counteroffer's gain for the agent: in a world where the offerer holds what it would
        /// give and would take the trade by the margin of a pick, the agent's gain over no trade; else 0
        /// </summary>
        internal static double[] CounterGains(IReadOnlyList<GameState> worlds, int offerer, int player,
            IReadOnlyList<(CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)> candidates, Func<IReadOnlyList<GameState>, double[]> utilities)
        {
            // All positions for one batch of the net: per world the position without a trade, then one after each counteroffer
            // the offerer can pay there; rows says which world and candidate each position after a trade belongs to
            List<GameState> states = [];
            List<(int world, int candidate, int state)> rows = [];
            int[] baseState = new int[worlds.Count];
            for (int w = 0; w < worlds.Count; w++)
            {
                // Without a trade the offer closes
                GameState none = new GameState(worlds[w]);
                none.Turn.PendingTrade = null;
                baseState[w] = states.Count;
                states.Add(none);
                for (int c = 0; c < candidates.Count; c++)
                {
                    // The offerer doesn't hold what it would give in this world: no trade, no gain there
                    if (!none.Players[offerer].ResourceCards.Contains(candidates[c].give)) continue;
                    rows.Add((w, c, states.Count));
                    states.Add(TradeDecisions.AfterTrade(none, offerer, player, candidates[c].give, candidates[c].get));
                }
            }

            double[] gains = new double[candidates.Count];
            if (rows.Count == 0) return gains;
            // PlayerCount values per position, in the order of states
            double[] values = utilities(states);
            int players = values.Length / states.Count;
            foreach ((int w, int c, int s) in rows)
            {
                // Offsets of the world without a trade and of the position after this counteroffer
                int before = baseState[w] * players, after = s * players;
                // The offerer takes it only if it gains by the margin of a pick; then the agent's gain counts, divided by
                // all worlds (a world without the trade adds 0)
                if (values[after + offerer] > values[before + offerer] + TradeDecisions.ValueMargin)
                    gains[c] += (values[after + player] - values[before + player]) / worlds.Count;
            }
            return gains;
        }

        /// <summary>
        /// The mean over the worlds of each offer's gain for the agent: in a world, the best gain among the partners who hold
        /// the asked cards and would accept, if it beats the margin of a pick; else 0 (the offer is cancelled)
        /// </summary>
        internal static double[] ExpectedGains(IReadOnlyList<GameState> worlds, int player, byte recipients,
            IReadOnlyList<(CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)> candidates, Func<IReadOnlyList<GameState>, double[]> utilities,
            double partnerFairness = 0)
        {
            // All positions for one batch of the net: per world the position as it is, then one after each candidate with each
            // partner who could take it; rows says which world, candidate and partner each position after a trade belongs to
            List<GameState> states = [];
            List<(int world, int candidate, int partner, int state)> rows = [];
            int[] baseState = new int[worlds.Count];
            for (int w = 0; w < worlds.Count; w++)
            {
                // No offer is open yet, so the world itself is the position without a trade
                GameState world = worlds[w];
                baseState[w] = states.Count;
                states.Add(world);
                for (int c = 0; c < candidates.Count; c++)
                {
                    for (int partner = 0; partner < world.Players.Length; partner++)
                    {
                        // Only recipients who hold the asked cards in this world can accept
                        if ((recipients & 1 << partner) == 0 || !world.Players[partner].ResourceCards.Contains(candidates[c].get)) continue;
                        rows.Add((w, c, partner, states.Count));
                        states.Add(TradeDecisions.AfterTrade(world, player, partner, candidates[c].give, candidates[c].get));
                    }
                }
            }

            double[] gains = new double[candidates.Count];
            if (rows.Count == 0) return gains;
            // PlayerCount values per position, in the order of states
            double[] values = utilities(states);
            int players = values.Length / states.Count;

            // Per world and candidate the agent's best gain among the partners who accept (0: nobody does, the offer is cancelled)
            double[,] best = new double[worlds.Count, candidates.Count];
            foreach ((int w, int c, int partner, int s) in rows)
            {
                // Offsets of the world without a trade and of the position after this trade with this partner
                int before = baseState[w] * players, after = s * players;
                // Both gains by the agent's own net: it models the partner as valuing positions like itself
                double gain = values[after + player] - values[before + player], partnerGain = values[after + partner] - values[before + partner];
                // The partner accepts any gain above the answer margin, with partnerFairness only if it is that share of the agent's
                bool accepts = partnerGain > TradeDecisions.ValueMargin && partnerGain >= partnerFairness * gain;
                // The agent picks the accepting partner best for itself, and only if that beats the margin of a pick
                if (accepts && gain > TradeDecisions.ValueMargin && gain > best[w, c]) best[w, c] = gain;
            }
            // An offer is worth its mean over the worlds
            for (int c = 0; c < candidates.Count; c++)
            {
                double sum = 0;
                for (int w = 0; w < worlds.Count; w++) sum += best[w, c];
                gains[c] = sum / worlds.Count;
            }
            return gains;
        }

        /// <summary>
        /// The offers to look at: one or two cards of the hand (one type twice or two types) for one or two cards of other
        /// types; two for two only if asked, three for one and one for three only if asked, any trade up to three cards for
        /// three with maxCards 3, and with askNoMore only offers asking for no more cards than they give
        /// </summary>
        internal static IEnumerable<(CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)> Candidates(CardSet<ResourceCardType> hand, bool twoForTwo, bool askNoMore,
            bool threeForOne = false, int maxCards = 2)
        {
            // Sets of three cards are needed for 3:1 and 1:3 and for any trade up to 3:3
            bool three = threeForOne || maxCards >= 3;
            // What it gives comes from its hand; what it asks for can be any set (up to three of a type)
            List<CardSet<ResourceCardType>> gives = Sets(type => hand.Get(type), three), gets = Sets(_ => 3, three);
            foreach (CardSet<ResourceCardType> give in gives)
            {
                foreach (CardSet<ResourceCardType> get in gets)
                {
                    uint given = give.Count(), asked = get.Count();
                    // Up to two cards a side: 2:2 only if asked, three cards only as 3:1 or 1:3
                    if (maxCards < 3)
                    {
                        if (given == 2 && asked == 2 && !twoForTwo) continue;
                        if (given == 3 && asked != 1 || asked == 3 && given != 1) continue;
                    }
                    if (askNoMore && asked > given) continue;
                    // The rules for offers: no gifts, no type on both sides
                    if (!TradeOffer.AreValidTerms(give, get)) continue;
                    yield return (give, get);
                }
            }
        }

        // One card of a type, two of a type, one each of two types, from what is available; with three also every set of
        // three cards
        private static List<CardSet<ResourceCardType>> Sets(Func<ResourceCardType, uint> available, bool three = false)
        {
            ResourceCardType[] types = CardTypeExtensions.ResourceTypes;
            List<CardSet<ResourceCardType>> sets = [];
            // One card of a type
            foreach (ResourceCardType type in types)
                if (available(type) >= 1) sets.Add(Of((type, 1)));
            // Two of a type
            foreach (ResourceCardType type in types)
                if (available(type) >= 2) sets.Add(Of((type, 2)));
            // One each of two types
            for (int i = 0; i < types.Length; i++)
                for (int j = i + 1; j < types.Length; j++)
                    if (available(types[i]) >= 1 && available(types[j]) >= 1) sets.Add(Of((types[i], 1), (types[j], 1)));
            if (!three) return sets;
            // Every set of three cards, types in ascending order (i <= j <= k) so each set comes once, if the cards are there
            for (int i = 0; i < types.Length; i++)
            {
                for (int j = i; j < types.Length; j++)
                {
                    for (int k = j; k < types.Length; k++)
                    {
                        CardSet<ResourceCardType> set = Of((types[i], 1));
                        set.Add(types[j], 1);
                        set.Add(types[k], 1);
                        if (types.All(type => set.Get(type) <= available(type))) sets.Add(set);
                    }
                }
            }
            return sets;
        }

        private static CardSet<ResourceCardType> Of(params (ResourceCardType type, uint amount)[] cards)
        {
            CardSet<ResourceCardType> set = new CardSet<ResourceCardType>();
            foreach ((ResourceCardType type, uint amount) in cards) set.Add(type, amount);
            return set;
        }

        private static CardSet<ResourceCardType> After(CardSet<ResourceCardType> hand, CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)
        {
            CardSet<ResourceCardType> after = new CardSet<ResourceCardType>(hand);
            after.Remove(give);
            after.Add(get);
            return after;
        }

        /// <summary>
        /// The other players the agent offers to: all but those about to win
        /// </summary>
        public static byte Recipients(GameState state, int player)
        {
            // One bit per player, as OfferTradeAction takes them
            byte recipients = 0;
            for (int other = 0; other < state.Players.Length; other++)
                if (other != player && !IsAboutToWin(state, other)) recipients |= (byte)(1 << other);
            return recipients;
        }

        // By the points everyone sees (without hidden victory point cards), so it reads nothing hidden from the true position;
        // answers (TradeDecisions.Decide) count the total of the position they judge instead, for ISMCTS a drawn world
        private static bool IsAboutToWin(GameState state, int player)
        {
            VictoryPointTracker points = state.Players[player].VictoryPoints;
            return points.Total - points.DevelopmentCardPoints >= state.Settings.VictoryPoints - TradeDecisions.LeaderDistance;
        }
    }
}
