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
        // Every set of one or two cards (of one type or two) within the limits per type, counted independently of TradeOffers
        private static List<CardSet<ResourceCardType>> AllSets(Func<ResourceCardType, uint> available)
        {
            List<CardSet<ResourceCardType>> sets = [];
            ResourceCardType[] types = CardTypeExtensions.ResourceTypes;
            foreach (ResourceCardType a in types)
            {
                foreach (ResourceCardType b in types.Prepend(ResourceCardType.Unknown))
                {
                    CardSet<ResourceCardType> set = Cards((a, 1));
                    if (b != ResourceCardType.Unknown)
                    {
                        if (b < a) continue;
                        set.Add(b, 1);
                    }
                    if (types.All(type => set.Get(type) <= available(type))) sets.Add(set);
                }
            }
            return sets;
        }

        [Theory]
        [InlineData(false, false)]
        [InlineData(true, false)]
        [InlineData(false, true)]
        public void TheOffersLookedAtAreAllOneOrTwoCardsOfTheHandForOneOrTwoOthers(bool twoForTwo, bool askNoMore)
        {
            CardSet<ResourceCardType> hand = Cards((Lumber, 2), (Brick, 1), (Ore, 1));
            HashSet<string> expected = [];
            foreach (CardSet<ResourceCardType> give in AllSets(hand.Get))
            {
                foreach (CardSet<ResourceCardType> get in AllSets(_ => 2))
                {
                    if (CardTypeExtensions.ResourceTypes.Any(type => give.Get(type) > 0 && get.Get(type) > 0)) continue;
                    if (give.Count() == 2 && get.Count() == 2 && !twoForTwo) continue;
                    if (askNoMore && get.Count() > give.Count()) continue;
                    expected.Add(Terms((give, get)));
                }
            }

            var candidates = TradeOffers.Candidates(hand, twoForTwo, askNoMore).ToList();

            Assert.Equal(expected.Order(), candidates.Select(Terms).Order());
            Assert.Equal(candidates.Count, candidates.Select(Terms).Distinct().Count());
            Assert.All(candidates, c => Assert.True(TradeOffer.AreValidTerms(c.give, c.get) && hand.Contains(c.give)));
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
                        uint needed = hand.Get(type);
                        game.Play(new OfferTradeAction(acting, Cards((type, needed)), Cards((wanted, 1)), OfferTradeAction.Everyone(game.Truth, acting)));
                        Assert.Equal(needed, beliefs.Hand(acting).Known.Get(type));
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
