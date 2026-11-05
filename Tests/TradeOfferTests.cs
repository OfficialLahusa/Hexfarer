using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Common.Actions;
using static Tests.FreeTradeTests;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Trade offers by agents (TradeOffers): the offers looked at, how an offer is judged over worlds, when agents offer and
    /// to whom, and what the other seats learn from offers and answers
    /// </summary>
    public class TradeOfferTests
    {
        private const ResourceCardType Lumber = ResourceCardType.Lumber, Brick = ResourceCardType.Brick, Wool = ResourceCardType.Wool,
            Grain = ResourceCardType.Grain, Ore = ResourceCardType.Ore;

        private static string Terms((CardSet<ResourceCardType> give, CardSet<ResourceCardType> get) terms) => $"{Describe(terms.give)}>{Describe(terms.get)}";

        private static string Describe(CardSet<ResourceCardType> cards) =>
            string.Concat(CardTypeExtensions.ResourceTypes.Select(type => $"{type.GetAbbreviation()}{cards.Get(type)}"));

        #region Candidates
        // Every set of one to three cards within the limits per type (by counting all draws of types in order), independently
        // of TradeOffers
        private static List<CardSet<ResourceCardType>> AllSets(Func<ResourceCardType, uint> available, int most)
        {
            HashSet<string> seen = [];
            List<CardSet<ResourceCardType>> sets = [];
            ResourceCardType[] types = CardTypeExtensions.ResourceTypes;
            IEnumerable<ResourceCardType[]> draws = types.Select(t => new[] { t });
            for (int size = 1; size <= most; size++)
            {
                foreach (ResourceCardType[] draw in draws)
                {
                    CardSet<ResourceCardType> set = new CardSet<ResourceCardType>();
                    foreach (ResourceCardType type in draw) set.Add(type, 1);
                    if (types.All(type => set.Get(type) <= available(type)) && seen.Add(Describe(set))) sets.Add(set);
                }
                draws = draws.SelectMany(draw => types.Select(t => draw.Append(t).ToArray())).ToList();
            }
            return sets;
        }

        [Theory]
        [InlineData(false, false, false)]
        [InlineData(true, false, false)]
        [InlineData(false, true, false)]
        [InlineData(false, false, true)]
        [InlineData(true, true, true)]
        public void TheOffersLookedAtAreAllSmallSetsOfTheHandForCardsOfOtherTypes(bool twoForTwo, bool askNoMore, bool threeForOne)
        {
            CardSet<ResourceCardType> hand = Cards((Lumber, 3), (Brick, 1), (Ore, 1));
            HashSet<string> expected = [];
            foreach (CardSet<ResourceCardType> give in AllSets(hand.Get, threeForOne ? 3 : 2))
            {
                foreach (CardSet<ResourceCardType> get in AllSets(_ => 3, threeForOne ? 3 : 2))
                {
                    uint given = give.Count(), asked = get.Count();
                    if (CardTypeExtensions.ResourceTypes.Any(type => give.Get(type) > 0 && get.Get(type) > 0)) continue;
                    if (given == 2 && asked == 2 && !twoForTwo) continue;
                    // Three cards only against one
                    if (given == 3 && asked != 1 || asked == 3 && given != 1) continue;
                    if (askNoMore && asked > given) continue;
                    expected.Add(Terms((give, get)));
                }
            }

            var candidates = TradeOffers.Candidates(hand, twoForTwo, askNoMore, threeForOne).ToList();

            Assert.Equal(expected.Order(), candidates.Select(Terms).Order());
            Assert.Equal(candidates.Count, candidates.Select(Terms).Distinct().Count());
            Assert.All(candidates, c => Assert.True(TradeOffer.AreValidTerms(c.give, c.get) && hand.Contains(c.give)));
            Assert.Equal(threeForOne, candidates.Any(c => c.give.Count() == 3));
            Assert.Equal(threeForOne && !askNoMore, candidates.Any(c => c.get.Count() == 3));
        }
        #endregion

        #region Judging an offer
        // A made-up value of each position: the offerer (seat 0) wants ore and dislikes seat 1 holding lumber, seat 1 and seat 3
        // want lumber, seat 3 wool too, seat 2 wants nothing; the offerer gains a hair from grain
        private static double Value(GameState state, int player)
        {
            uint Of(int seat, ResourceCardType type) => state.Players[seat].ResourceCards.Get(type);
            return player switch
            {
                0 => 0.05 * Of(0, Ore) - 0.03 * Of(1, Lumber) + 0.001 * Of(0, Grain),
                1 => 0.05 * Of(1, Lumber),
                3 => 0.05 * Of(3, Lumber) + 0.05 * Of(3, Wool),
                _ => 0,
            };
        }

        private static double[] Values(IReadOnlyList<GameState> states) =>
            states.SelectMany(state => Enumerable.Range(0, 4).Select(player => Value(state, player))).ToArray();

        private static GameState World(params CardSet<ResourceCardType>[] hands)
        {
            GameState state = new GameState(MapGenerator.GenerateRandomClassic(), 4);
            for (int seat = 0; seat < 4; seat++) state.Players[seat].ResourceCards = hands[seat];
            return state;
        }

        [Fact]
        public void AnOfferIsWorthTheMeanOverTheWorldsOfTheBestAcceptingPartnersGain()
        {
            CardSet<ResourceCardType> mine = Cards((Lumber, 1), (Wool, 1), (Ore, 1));
            List<GameState> worlds =
            [
                // Seats 1 and 3 accept lumber for ore, seat 2 holds ore but gains nothing; the offerer gains most with seat 3
                World(mine, Cards((Ore, 1)), Cards((Ore, 1)), Cards((Ore, 1), (Grain, 1))),
                // Only seat 1 can and will; it holds a lumber already, so each world starts from other values
                World(mine, Cards((Ore, 1), (Lumber, 1)), Cards((Ore, 1)), Cards((Grain, 1))),
                // Nobody holds ore
                World(mine, Cards((Brick, 1)), Cards((Brick, 1)), Cards((Grain, 1))),
            ];
            var lumberForOre = (give: Cards((Lumber, 1)), get: Cards((Ore, 1)));
            var oreForLumber = (give: Cards((Ore, 1)), get: Cards((Lumber, 1)));
            // Seat 3 accepts wool for grain, but the offerer's gain of 0.001 is below the margin of a pick
            var woolForGrain = (give: Cards((Wool, 1)), get: Cards((Grain, 1)));

            double[] gains = TradeOffers.ExpectedGains(worlds, 0, To(1, 2, 3), [lumberForOre, oreForLumber, woolForGrain], Values);

            Assert.Equal((0.05 + 0.02 + 0) / 3, gains[0], 9);
            Assert.Equal(0, gains[1], 9);
            Assert.Equal(0, gains[2], 9);

            // Without seat 3 among the recipients only seat 1's trades count
            double[] withoutThree = TradeOffers.ExpectedGains(worlds, 0, To(1, 2), [lumberForOre], Values);
            Assert.Equal((0.02 + 0.02 + 0) / 3, withoutThree[0], 9);
        }
        #endregion

        #region When agents offer
        /// <summary>
        /// The main phase after the roll with a hand one grain short of a settlement and of a development card
        /// </summary>
        private static (GameState state, sbyte active) GrainShort(bool freeTrade = true)
        {
            (GameState state, sbyte active) = MainPhase(freeTrade: freeTrade);
            state.Players[active].ResourceCards = Cards((Lumber, 1), (Brick, 1), (Wool, 1), (Ore, 2));
            return (state, active);
        }

        private static void RejectAll(GameState state)
        {
            while (state.Turn.PendingTrade is TradeOffer offer && offer.NextResponder is sbyte next)
                Play(state, new RespondToTradeAction(next, false));
            if (state.Turn.PendingTrade != null) Play(state, new CancelTradeAction((sbyte)state.Turn.PlayerIndex));
        }

        [Fact]
        public void AHeuristicAgentOffersWhatCompletesABuildingAndNeverTheSameTermsTwiceInATurn()
        {
            (GameState state, sbyte active) = GrainShort();
            BuilderAgent agent = new BuilderAgent(active);
            CardSet<ResourceCardType> hand = state.Players[active].ResourceCards;

            List<string> offered = [];
            while (agent.Act(new GameState(state)) is OfferTradeAction offer)
            {
                Assert.True(offer.IsValidFor(state), $"{offer} is not legal");
                Assert.True(offer.Get.Count() <= offer.Give.Count(), $"{offer} asks for more than it gives");
                CardSet<ResourceCardType> after = new CardSet<ResourceCardType>(hand);
                after.Remove(offer.Give);
                after.Add(offer.Get);
                Assert.True(TradeDecisions.HandValue(after) > TradeDecisions.HandValue(hand) + 0.15f, $"{offer} doesn't help the hand");
                Assert.DoesNotContain(Terms((offer.Give, offer.Get)), offered);
                offered.Add(Terms((offer.Give, offer.Get)));
                Play(state, offer);
                RejectAll(state);
                Assert.True(offered.Count <= 3);
            }
            // The best first: grain for ore; at most three offers a turn, and there are more than three helpful ones
            Assert.Equal("L0B0W0G0O1>L0B0W0G1O0", offered[0]);
            Assert.Equal(3, offered.Count);

            // A new turn starts without offers made: everything may be offered again
            GameState nextTurn = new GameState(state);
            nextTurn.Turn.TradeOffersThisTurn = 0;
            Assert.True(agent.Act(nextTurn) is OfferTradeAction { } again && Terms((again.Give, again.Get)) == offered[0]);
        }

        [Fact]
        public void AgentsOfferOnlyWhenTheRulesAllowItAndNotWhenAboutToWin()
        {
            (GameState off, sbyte a) = GrainShort(freeTrade: false);
            Assert.False(new BuilderAgent(a).Act(off) is OfferTradeAction);

            (GameState state, sbyte active) = GrainShort();
            Assert.True(new BuilderAgent(active).Act(new GameState(state)) is OfferTradeAction);

            GameState roll = new GameState(state);
            roll.Turn.MustRoll = true;
            Assert.False(new BuilderAgent(active).Act(roll) is OfferTradeAction);

            // An offer would give up the roads of a Road Building card
            GameState roads = new GameState(state);
            roads.Players[active].BuildingStock.FreeRoads = 2;
            Assert.False(new BuilderAgent(active).Act(roads) is OfferTradeAction);

            // Nobody would accept
            GameState winning = new GameState(state);
            winning.Players[active].VictoryPoints.SettlementPoints = 8;
            Assert.False(new BuilderAgent(active).Act(winning) is OfferTradeAction);

            BuilderAgent silent = new BuilderAgent(active);
            silent.Offers.Enabled = false;
            Assert.False(silent.Act(new GameState(state)) is OfferTradeAction);

            BuilderAgent once = new BuilderAgent(active);
            once.Offers.MaxPerTurn = 1;
            GameState afterOne = new GameState(state);
            Play(afterOne, (OfferTradeAction)once.Act(new GameState(afterOne)));
            RejectAll(afterOne);
            Assert.False(once.Act(afterOne) is OfferTradeAction);
        }

        [Fact]
        public void OffersGoToEveryoneButPlayersAboutToWinByTheirPublicPoints()
        {
            (GameState state, sbyte active) = GrainShort();
            sbyte leader = Seat(active, 1), hiding = Seat(active, 2), other = Seat(active, 3);
            state.Players[leader].VictoryPoints.SettlementPoints = 8;
            // Hidden victory point cards don't count: the agent can't see them
            state.Players[hiding].VictoryPoints.SettlementPoints = 6;
            state.Players[hiding].VictoryPoints.DevelopmentCardPoints = 3;

            OfferTradeAction offer = (OfferTradeAction)new BuilderAgent(active).Act(new GameState(state));

            Assert.Equal(To(hiding, other), offer.Recipients);
        }

        private static StateValueNet Net(string name) => new StateValueNet(TestUtil.FixturePath("ValueNet", name), provider: InferenceProvider.Cpu);

        [Fact]
        public void TheValueNetJudgesManyPositionsInOneBatchAsEachAlone()
        {
            (GameState a, sbyte active) = MainPhase();
            GameState b = new GameState(a);
            b.Players[active].ResourceCards = Cards((Ore, 5));
            b.Players[Seat(active, 2)].ResourceCards = Cards((Wool, 1));
            MCTSAgent agent = new MCTSAgent(active, a, valueNet: Net("sensitive_tiny.onnx"), iterationLimit: 16, leavesPerBatch: 8);

            double[] both = agent.Utilities([a, b]);
            double[] first = agent.Utilities([a]), second = agent.Utilities([b]);

            Assert.Equal(8, both.Length);
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(first[i], both[i], 6);
                Assert.Equal(second[i], both[4 + i], 6);
            }
            Assert.NotEqual(first[active], second[active], 6);
            Assert.Equal(1, both.Take(4).Sum(), 5);

            // More positions than one run of the net takes: b only at a few places, among them both ends of each run
            int count = MCTSAgent.UtilityBatch + 44;
            int[] atB = [0, MCTSAgent.UtilityBatch - 1, MCTSAgent.UtilityBatch, count - 1, 77];
            double[] many = agent.Utilities(Enumerable.Range(0, count).Select(i => atB.Contains(i) ? b : a).ToList());
            Assert.Equal(count * 4, many.Length);
            for (int s = 0; s < count; s++)
                for (int i = 0; i < 4; i++)
                    Assert.Equal((atB.Contains(s) ? second : first)[i], many[s * 4 + i], 6);
        }

        [Fact]
        public void ValueNetAgentsOfferOnlyLegalTermsToTheOthers()
        {
            int offers = 0;
            for (int seed = 1; seed <= 6; seed++)
            {
                (GameState state, sbyte active) = MainPhase(seed);
                state.Players[active].ResourceCards = Cards((Lumber, 2), (Ore, 3), (Wool, 1));
                MCTSAgent search = new MCTSAgent(active, state, valueNet: Net("plain_tiny.onnx"), iterationLimit: 16, leavesPerBatch: 8);
                Agent[] agents = [search, new InformationSetMCTSAgent(search, state, seed: 3), new DeterminizedMCTSAgent(search, state, 2, seed: 3)];
                foreach (Agent agent in agents)
                {
                    search.Offers.Margin = 0;
                    if (agent.Act(new GameState(state)) is not OfferTradeAction offer) continue;
                    offers++;
                    Assert.True(offer.IsValidFor(state), $"{offer} is not legal");
                    Assert.Equal(OfferTradeAction.Everyone(state, active), offer.Recipients);
                }
            }
            Assert.True(offers >= 6, $"only {offers} offers");
        }
        #endregion

        #region Counteroffers
        [Fact]
        public void ACounterofferIsWorthTheMeanGainOverTheWorldsWhereTheOffererWouldTakeIt()
        {
            List<GameState> worlds =
            [
                World(Cards((Lumber, 1), (Wool, 1), (Grain, 1)), Cards((Ore, 1), (Lumber, 1), (Wool, 1)), new(), new()),
                // The offerer holds no lumber here
                World(Cards((Wool, 1), (Grain, 1)), Cards((Ore, 1), (Lumber, 1), (Wool, 1)), new(), new()),
            ];
            // In the offerer's terms; seat 1 counters seat 0's offer
            var lumberForOre = (give: Cards((Lumber, 1)), get: Cards((Ore, 1)));
            // The offerer loses 0.001 by giving its grain: it says no
            var grainForWool = (give: Cards((Grain, 1)), get: Cards((Wool, 1)));
            // The offerer takes it (seat 1 holds a lumber less), but seat 1 loses
            var woolForLumber = (give: Cards((Wool, 1)), get: Cards((Lumber, 1)));

            double[] gains = TradeOffers.CounterGains(worlds, 0, 1, [lumberForOre, grainForWool, woolForLumber], Values);

            Assert.Equal((0.05 + 0) / 2, gains[0], 9);
            Assert.Equal(0, gains[1], 9);
            Assert.Equal(-0.05, gains[2], 9);
        }

        // The offerer wants ore and dislikes the responder holding lumber; the responder wants lumber, a little brick and
        // a little every card
        private static Func<IReadOnlyList<GameState>, double[]> CounterValues(int offerer, int responder) => states => states.SelectMany(state =>
        {
            uint Of(int seat, ResourceCardType type) => state.Players[seat].ResourceCards.Get(type);
            return Enumerable.Range(0, 4).Select(player =>
                player == offerer ? 0.05 * Of(offerer, Ore) - 0.03 * Of(responder, Lumber)
                : player == responder ? 0.05 * Of(responder, Lumber) + 0.002 * Of(responder, Brick) + 0.001 * state.Players[responder].ResourceCards.Count()
                : 0);
        }).ToArray();

        [Fact]
        public void ACounterofferIsTheBestTermsTheOffererWouldTakeAndOnlyWhenAnswering()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte responder = Seat(active, 1);
            Play(state, new OfferTradeAction(active, Cards((Wool, 1)), Cards((Grain, 1)), To(responder, Seat(active, 2))));
            TradeOffers offers = new TradeOffers { Counters = true };
            var values = CounterValues(active, responder);

            CounterTradeAction? counter = offers.DecideCounter(state, responder, _ => [state], values);

            Assert.NotNull(counter);
            Assert.True(counter.IsValidFor(state));
            // The offerer gives lumber and brick for one ore
            Assert.Equal("L1B1W0G0O0>L0B0W0G0O1", Terms((counter.Give, counter.Get)));

            Assert.Null(new TradeOffers().DecideCounter(state, responder, _ => [state], values));
            // The second addressee counters once it is its turn to answer, not before
            var second = CounterValues(active, Seat(active, 2));
            Assert.Null(offers.DecideCounter(state, Seat(active, 2), _ => [state], second));
            GameState later = new GameState(state);
            Play(later, new RespondToTradeAction(responder, false));
            Assert.NotNull(offers.DecideCounter(later, Seat(active, 2), _ => [later], second));
            GameState winning = new GameState(state);
            winning.Players[active].VictoryPoints.SettlementPoints = 8;
            Assert.Null(offers.DecideCounter(winning, responder, _ => [state], values));
        }

        [Fact]
        public void ValueNetAgentsCounterWithLegalTerms()
        {
            // Offers of one card for another in random games, answered by each kind of value-net agent with every gain counted
            int counters = 0, answers = 0, acceptsAll = 0, acceptsFair = 0;
            StateValueNet net = Net("plain_tiny.onnx");
            for (int seed = 1; seed <= 40 && counters < 12; seed++)
            {
                HiddenInformationTests.Followed game = new HiddenInformationTests.Followed(HiddenInformationTests.NewGame(seed, freeTrade: true));
                Agent[] random = TestUtil.RandomAgents();
                while (!game.Truth.HasEnded && game.Played.Count < 400)
                {
                    sbyte acting = TestUtil.ActingPlayer(game.Truth);
                    CardSet<ResourceCardType> hand = game.Truth.Players[acting].ResourceCards;
                    ResourceCardType? give = CardTypeExtensions.ResourceTypes.Cast<ResourceCardType?>().FirstOrDefault(t => hand.Get(t!.Value) > 0);
                    if (game.Played.Count < 60 || give is not ResourceCardType offered || !OfferTradeAction.IsTurnValid(game.Truth, acting))
                    {
                        game.Play(random[acting].Act(game.Truth));
                        continue;
                    }
                    ResourceCardType wanted = CardTypeExtensions.ResourceTypes.First(t => t != offered);
                    game.Play(new OfferTradeAction(acting, Cards((offered, 1)), Cards((wanted, 1)), OfferTradeAction.Everyone(game.Truth, acting)));
                    while (game.Truth.Turn.PendingTrade is TradeOffer { NextResponder: sbyte responder })
                    {
                        Agent[] Agents(bool withCounters, float answerMargin, float fairness = 0)
                        {
                            MCTSAgent search = new MCTSAgent(responder, game.Truth, valueNet: net, iterationLimit: 16, leavesPerBatch: 8) { AnswerMargin = answerMargin, AnswerFairness = fairness };
                            search.Offers.Counters = withCounters;
                            search.Offers.Margin = 0;
                            InformationSetMCTSAgent ismcts = new InformationSetMCTSAgent(search, game.Truth, seed: 3);
                            ismcts.ResetTo(game.Truth, game.Trackers[responder]);
                            DeterminizedMCTSAgent pimc = new DeterminizedMCTSAgent(new MCTSAgent(responder, game.Truth, valueNet: net, iterationLimit: 16, leavesPerBatch: 8)
                                { AnswerMargin = answerMargin, AnswerFairness = fairness, Offers = { Counters = withCounters, Margin = 0 } }, game.Truth, 2, seed: 3);
                            pimc.ResetTo(game.Truth, game.Trackers[responder]);
                            return [search, ismcts, pimc];
                        }
                        Agent[] with = Agents(true, TradeDecisions.ValueMargin), without = Agents(false, TradeDecisions.ValueMargin);
                        bool canAccept = new RespondToTradeAction(responder, true).IsValidFor(game.Truth);
                        for (int kind = 0; kind < 3; kind++)
                        {
                            Action answer = with[kind].Act(new GameState(game.Truth));
                            Assert.True(answer.IsValidFor(game.Truth), $"{answer} is not legal");
                            answers++;
                            if (answer is CounterTradeAction) counters++;
                            // A counteroffer only instead of turning the offer down
                            Action plain = without[kind].Act(new GameState(game.Truth));
                            if (plain is RespondToTradeAction { Accept: true }) Assert.True(answer is RespondToTradeAction { Accept: true }, $"{answer} instead of accepting");
                            else Assert.False(answer is RespondToTradeAction { Accept: true }, $"{answer} instead of turning down");

                            // The answer margin decides: any gain or none is enough
                            Assert.Equal(canAccept, Agents(false, -1)[kind].Act(new GameState(game.Truth)) is RespondToTradeAction { Accept: true });
                            Assert.True(Agents(false, 1)[kind].Act(new GameState(game.Truth)) is RespondToTradeAction { Accept: false });
                            // Asking for a thousand times the offerer's gain turns down what any gain would take
                            if (canAccept) acceptsAll++;
                            if (Agents(false, -1, 1000)[kind].Act(new GameState(game.Truth)) is RespondToTradeAction { Accept: true }) acceptsFair++;
                        }
                        game.Play(new RespondToTradeAction(responder, false));
                    }
                    if (game.Truth.Turn.PendingTrade != null) game.Play(new CancelTradeAction(acting));
                    game.Play(new EndTurnAction(acting));
                }
            }
            Assert.True(counters >= 12, $"only {counters} counteroffers in {answers} answers");
            Assert.True(acceptsFair < acceptsAll, $"fairness turned down nothing ({acceptsFair} of {acceptsAll})");
        }
        #endregion

        #region Answers
        [Fact]
        public void AFairAnswerNeedsAShareOfTheOfferersGain()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte responder = Seat(active, 1);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(responder)));
            // The responder gains 0.01 from a lumber, the offerer 0.05 from an ore
            float Value(GameState s, int player) => player == responder ? 0.01f * s.Players[responder].ResourceCards.Get(Lumber)
                : player == active ? 0.05f * s.Players[active].ResourceCards.Get(Ore) : 0;

            Assert.True(TradeDecisions.Decide(state, responder, Value) is RespondToTradeAction { Accept: true });
            Assert.True(TradeDecisions.Decide(state, responder, Value, fairness: 0.1f) is RespondToTradeAction { Accept: true });
            Assert.True(TradeDecisions.Decide(state, responder, Value, fairness: 0.5f) is RespondToTradeAction { Accept: false });
            Assert.True(TradeDecisions.Decide(state, responder, Value, margin: 0.02f) is RespondToTradeAction { Accept: false });
        }
        #endregion

        #region What the others learn
        [Fact]
        public void OffersAcceptsAndCountersShowCardsTheOthersDidNotKnowAbout()
        {
            int offers = 0, accepts = 0, counters = 0;
            for (int seed = 1; seed < 40 && (offers < 5 || accepts < 5 || counters < 5); seed++)
            {
                HiddenInformationTests.Followed game = new HiddenInformationTests.Followed(HiddenInformationTests.NewGame(seed, freeTrade: true));
                Agent[] agents = TestUtil.RandomAgents();
                Random random = new Random(seed);
                while (!game.Truth.HasEnded && game.Played.Count < 2000)
                {
                    sbyte acting = TestUtil.ActingPlayer(game.Truth);
                    int observer = (acting + 1) % 4;
                    BeliefTracker beliefs = game.Trackers[observer];
                    CardSet<ResourceCardType> hand = game.Truth.Players[acting].ResourceCards;
                    // A card of the acting player the observer doesn't know about
                    ResourceCardType? unknown = CardTypeExtensions.ResourceTypes.Cast<ResourceCardType?>()
                        .FirstOrDefault(type => hand.Get(type!.Value) > beliefs.Hand(acting).Known.Get(type!.Value));

                    if (unknown is ResourceCardType type && OfferTradeAction.IsTurnValid(game.Truth, acting) && random.NextDouble() < 0.5)
                    {
                        ResourceCardType wanted = CardTypeExtensions.ResourceTypes.First(t => t != type);
                        uint needed = hand.Get(type), knownBefore = beliefs.Hand(acting).Known.Get(type);
                        // An observer that ignores what trades show (an ablation) learns nothing from the offer
                        BeliefTracker deaf = new BeliefTracker(beliefs) { LearnsFromTrades = false };
                        GameState before = new GameState(game.Truth);
                        OfferTradeAction shown = new OfferTradeAction(acting, Cards((type, needed)), Cards((wanted, 1)), OfferTradeAction.Everyone(game.Truth, acting));
                        game.Play(shown);
                        deaf.Observe(before, shown, game.Truth);
                        Assert.Equal(needed, beliefs.Hand(acting).Known.Get(type));
                        Assert.Equal(knownBefore, deaf.Hand(acting).Known.Get(type));
                        Assert.False(new BeliefTracker(deaf).LearnsFromTrades);
                        offers++;
                        continue;
                    }

                    // The answers to an open offer: accept or counter with cards the offerer didn't know about
                    if (game.Truth.Turn.PendingTrade is TradeOffer pending && pending.NextResponder == acting)
                    {
                        CardSet<ResourceCardType> mine = game.Truth.Players[acting].ResourceCards;
                        BeliefTracker offerer = game.Trackers[pending.Offerer];
                        bool newsToOfferer = CardTypeExtensions.ResourceTypes.Any(t => pending.Get.Get(t) > offerer.Hand(acting).Known.Get(t));
                        if (mine.Contains(pending.Get) && newsToOfferer)
                        {
                            game.Play(new RespondToTradeAction(acting, true));
                            Assert.True(offerer.Hand(acting).Known.Contains(pending.Get), "accepting shows the asked cards");
                            accepts++;
                            continue;
                        }
                        ResourceCardType? spare = CardTypeExtensions.ResourceTypes.Cast<ResourceCardType?>()
                            .FirstOrDefault(t => mine.Get(t!.Value) > offerer.Hand(acting).Known.Get(t!.Value) && pending.Give.Get(t!.Value) == 0);
                        if (spare is ResourceCardType counterCard)
                        {
                            CardSet<ResourceCardType> theirs = Cards((CardTypeExtensions.ResourceTypes.First(t => t != counterCard), 1));
                            CounterTradeAction counter = new CounterTradeAction(acting, theirs, Cards((counterCard, mine.Get(counterCard))));
                            if (counter.IsValidFor(game.Truth))
                            {
                                game.Play(counter);
                                Assert.Equal(mine.Get(counterCard), offerer.Hand(acting).Known.Get(counterCard));
                                counters++;
                                continue;
                            }
                        }
                    }
                    game.Play(agents[acting].Act(game.Truth));
                }
            }
            Assert.True(offers >= 5 && accepts >= 5 && counters >= 5, $"offers {offers}, accepts {accepts}, counters {counters}");
        }
        #endregion
    }
}
