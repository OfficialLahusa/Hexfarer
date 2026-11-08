using System.Text.Json.Nodes;
using Agents;
using Common;
using Common.Actions;
using Common.OSP;
using Common.Serialization;
using OpenSeat;
using static Tests.FreeTradeTests;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Trades between players as JSettlers plays them (player_trading "first_accept", TradeFlow.FirstAccept): the addressed
    /// players may answer in any order, the first accept trades at once and closes the offer, and the offerer may take a
    /// standing counteroffer or withdraw at any time. The engine still has one player to act (the next addressed one in
    /// seat order, then the offerer), so the search and the legal lists keep the rulebook's order.
    /// </summary>
    public class FirstAcceptTradeTests
    {
        private const ResourceCardType Lumber = ResourceCardType.Lumber, Brick = ResourceCardType.Brick, Wool = ResourceCardType.Wool,
            Grain = ResourceCardType.Grain, Ore = ResourceCardType.Ore;

        // A game right after a roll with known hands (two of each resource for everyone), trading as JSettlers does
        private static (GameState state, sbyte active) FirstAcceptGame(int seed = 1)
        {
            (GameState state, sbyte active) = MainPhase(seed);
            state.Settings.TradeFlow = TradeFlow.FirstAccept;
            return (state, active);
        }

        // An offer of one lumber for one ore from the player on turn to everyone
        private static OfferTradeAction LumberForOre(GameState state, sbyte active) =>
            new(active, Cards((Lumber, 1)), Cards((Ore, 1)), OfferTradeAction.Everyone(state, active));

        [Fact]
        public void AnyAddressedPlayerMayAnswerFirst()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            Play(state, LumberForOre(state, active));
            // The last in seat order answers first, then the first; the one in between is still asked
            Play(state, new RespondToTradeAction(Seat(active, 3), false));
            Play(state, new RespondToTradeAction(Seat(active, 1), false));
            Assert.Equal(TradeResponse.Pending, state.Turn.PendingTrade!.Responses[Seat(active, 2)]);
            // Who answered can't answer again, and who wasn't asked can't answer at all
            Assert.False(new RespondToTradeAction(Seat(active, 3), true).IsValidFor(state));
            GameState partial = new GameState(state);
            Play(partial, new CancelTradeAction(active));
            Play(partial, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 1))));
            Assert.False(new RespondToTradeAction(Seat(active, 2), false).IsValidFor(partial));
        }

        [Fact]
        public void AnAcceptTradesAtOnceAndClosesTheOffer()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            GameState before = new GameState(state);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1), (Brick, 1)), Cards((Ore, 1)), OfferTradeAction.Everyone(state, active)));
            GameState offered = new GameState(state);
            sbyte partner = Seat(active, 2);
            RespondToTradeAction accept = new(partner, true);
            Play(state, accept);

            Assert.Null(state.Turn.PendingTrade);
            Assert.Equal(Cards((Lumber, 1), (Brick, 1), (Wool, 2), (Grain, 2), (Ore, 3)), state.Players[active].ResourceCards);
            Assert.Equal(Cards((Lumber, 3), (Brick, 3), (Wool, 2), (Grain, 2), (Ore, 1)), state.Players[partner].ResourceCards);
            Assert.Equal(before.Players[Seat(active, 1)].ResourceCards, state.Players[Seat(active, 1)].ResourceCards);
            Assert.Equal(before.Players[Seat(active, 3)].ResourceCards, state.Players[Seat(active, 3)].ResourceCards);

            // Undone, the offer stands again with the answer pending
            accept.Revert(state);
            Assert.Equal(offered, state);
            Assert.Equal(offered.GetHashCode(), state.GetHashCode());
        }

        [Fact]
        public void TheOffererTakesAStandingCounterofferAtAnyTime()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            Play(state, LumberForOre(state, active));
            sbyte counterer = Seat(active, 2);
            // In the offerer's terms: the offerer would give two lumber for one ore
            Play(state, new CounterTradeAction(counterer, Cards((Lumber, 2)), Cards((Ore, 1))));
            Assert.NotNull(state.Turn.PendingTrade!.NextResponder);
            GameState countered = new GameState(state);

            ConfirmTradeAction take = new(active, counterer);
            Play(state, take);
            Assert.Null(state.Turn.PendingTrade);
            Assert.Equal(Cards((Lumber, 0), (Brick, 2), (Wool, 2), (Grain, 2), (Ore, 3)), state.Players[active].ResourceCards);
            Assert.Equal(Cards((Lumber, 4), (Brick, 2), (Wool, 2), (Grain, 2), (Ore, 1)), state.Players[counterer].ResourceCards);
            take.Revert(state);
            Assert.Equal(countered, state);

            // A player who only declined has nothing to take
            Play(state, new RespondToTradeAction(Seat(active, 1), false));
            Assert.False(new ConfirmTradeAction(active, Seat(active, 1)).IsValidFor(state));
            Assert.False(new ConfirmTradeAction(active, Seat(active, 3)).IsValidFor(state));
        }

        [Fact]
        public void TheOffererWithdrawsAtAnyTime()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            GameState before = new GameState(state);
            Play(state, LumberForOre(state, active));
            CancelTradeAction withdraw = new(active);
            Play(state, withdraw);
            Assert.Null(state.Turn.PendingTrade);
            Assert.Equal(before.Players.Select(p => p.ResourceCards), state.Players.Select(p => p.ResourceCards));
            // Only the offerer withdraws
            GameState other = new GameState(before);
            Play(other, LumberForOre(other, active));
            Assert.False(new CancelTradeAction(Seat(active, 1)).IsValidFor(other));
        }

        [Fact]
        public void WhenEveryoneDeclinesInAnyOrderTheOfferCloses()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            Play(state, LumberForOre(state, active));
            foreach (int offset in new[] { 2, 3, 1 }) Play(state, new RespondToTradeAction(Seat(active, offset), false));
            Assert.Null(state.Turn.PendingTrade);
        }

        [Fact]
        public void AcceptingAndCounteringStillNeedTheCards()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            sbyte poor = Seat(active, 3);
            state.Players[poor].ResourceCards = Cards((Lumber, 2));
            Play(state, LumberForOre(state, active));
            Assert.False(new RespondToTradeAction(poor, true).IsValidFor(state));
            Assert.True(new RespondToTradeAction(poor, false).IsValidFor(state));
            // A counteroffer in the offerer's terms: the counterer gives the offerer's Get
            Assert.False(new CounterTradeAction(poor, Cards((Lumber, 1)), Cards((Ore, 1))).IsValidFor(state));
        }

        [Fact]
        public void TheEngineStillHasOnePlayerToAct()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            Play(state, LumberForOre(state, active));
            AssertOneToAct(state, Seat(active, 1));
            // Someone later in seat order answers first: the next in seat order still acts
            Play(state, new RespondToTradeAction(Seat(active, 3), false));
            AssertOneToAct(state, Seat(active, 1));
            Assert.Empty(LegalActionProvider.GetActionsForState(state, Seat(active, 2)));
            Play(state, new RespondToTradeAction(Seat(active, 1), false));
            AssertOneToAct(state, Seat(active, 2));
            // With a counteroffer standing and everyone answered, the offerer acts
            Play(state, new CounterTradeAction(Seat(active, 2), Cards((Lumber, 2)), Cards((Ore, 1))));
            AssertOneToAct(state, active);

            static void AssertOneToAct(GameState state, sbyte expected) =>
                Assert.Equal([expected], Enumerable.Range(0, 4).Where(state.CanPlayerAct).Select(p => (sbyte)p));
        }

        [Fact]
        public void UnderTheRulebookAnswersKeepTheirOrderAndTheOffererWaits()
        {
            (GameState state, sbyte active) = MainPhase();
            Play(state, LumberForOre(state, active));
            Assert.False(new RespondToTradeAction(Seat(active, 3), false).IsValidFor(state));
            Assert.False(new CancelTradeAction(active).IsValidFor(state));
            GameState accepted = new GameState(state);
            RespondToTradeAction accept = new(Seat(active, 1), true);
            Play(accepted, accept);
            // An accept trades nothing yet: the offerer picks once everyone answered
            Assert.Equal(state.Players.Select(p => p.ResourceCards), accepted.Players.Select(p => p.ResourceCards));
            Assert.NotNull(accepted.Turn.PendingTrade);
            Play(state, new CounterTradeAction(Seat(active, 1), Cards((Lumber, 2)), Cards((Ore, 1))));
            Assert.False(new ConfirmTradeAction(active, Seat(active, 1)).IsValidFor(state));
        }

        [Fact]
        public void TheFlowIsPartOfTheSettings()
        {
            GameSettings settings = new() { FreeTrade = true, TradeFlow = TradeFlow.FirstAccept };
            GameSettings copy = new(settings);
            Assert.Equal(settings, copy);
            Assert.Equal(settings.GetHashCode(), copy.GetHashCode());
            Assert.NotEqual(settings, new GameSettings { FreeTrade = true });

            (GameState state, _) = FirstAcceptGame();
            GameState loaded = SaveFileSerializer.Deserialize<GameState>(SaveFileSerializer.Serialize(state))!;
            Assert.Equal(TradeFlow.FirstAccept, loaded.Settings.TradeFlow);
            (GameState rulebook, _) = MainPhase();
            Assert.DoesNotContain("TradeFlow", SaveFileSerializer.Serialize(rulebook));
        }

        [Fact]
        public void TheRulesetNamesTheFlowAndTheReplicaPlaysIt()
        {
            GameSettings settings = new() { FreeTrade = true, TradeFlow = TradeFlow.FirstAccept };
            JsonObject ruleset = OSPCodec.Ruleset(settings);
            Assert.Equal("first_accept", ruleset["player_trading"]!.GetValue<string>());
            Assert.Equal("rulebook", OSPCodec.Ruleset(new GameSettings { FreeTrade = true })["player_trading"]?.GetValue<string>() ?? "rulebook");
            Assert.Null(Replica.Unsupported(JsonNode.Parse("""{"preset":"standard","player_trading":"first_accept","offer_limit":null}""")));

            Replica replica = new(Before(JsonNode.Parse("""{"preset":"standard","player_trading":"first_accept","offer_limit":null}""")!.AsObject(), out _));
            Assert.True(replica.State.Settings.FreeTrade);
            Assert.Equal(TradeFlow.FirstAccept, replica.State.Settings.TradeFlow);
            Replica rulebook = new(Before(JsonNode.Parse("""{"preset":"standard"}""")!.AsObject(), out _));
            Assert.Equal(TradeFlow.Rulebook, rulebook.State.Settings.TradeFlow);
        }

        [Fact]
        public void TheReplicaFollowsAnswersOutOfSeatOrder()
        {
            Replica replica = new(Before(JsonNode.Parse("""{"preset":"standard","player_trading":"first_accept","offer_limit":null}""")!.AsObject(), out OSPCodec codec));
            (GameState main, sbyte active) = FirstAcceptGame(seed: 3);
            replica.Rebuild(codec.State(main), new JsonArray(LegalActionProvider.GetActionsForState(main, active).Select(a => (JsonNode)codec.Move(a)).ToArray()));
            Assert.Equal(TradeFlow.FirstAccept, replica.State.Settings.TradeFlow);
            Assert.Equal(main.Players.Select(p => p.ResourceCards), replica.State.Players.Select(p => p.ResourceCards));
            Assert.Equal(active, replica.Acting());
            JsonObject Step(int n, int seat, JsonNode action) => new() { ["type"] = "step", ["n"] = n, ["seat"] = seat, ["action"] = action };

            Assert.Null(replica.Follow(Step(1, active, codec.Move(LumberForOre(main, active)))));
            // The last addressed player accepts before the others answered
            sbyte partner = Seat(active, 3);
            Assert.Null(replica.Follow(Step(2, partner, codec.Move(new RespondToTradeAction(partner, true)))));
            Assert.Null(replica.State.Turn.PendingTrade);
            Assert.Equal(3u, replica.State.Players[active].ResourceCards.Get(Ore));
            Assert.Equal(3u, replica.State.Players[partner].ResourceCards.Get(Lumber));
        }

        [Fact]
        public void APatchTakesOverTheRefereesOpenOffer()
        {
            Replica replica = new(Before(JsonNode.Parse("""{"preset":"standard","player_trading":"first_accept","offer_limit":null}""")!.AsObject(), out OSPCodec codec));
            (GameState main, sbyte active) = FirstAcceptGame(seed: 3);
            replica.Rebuild(codec.State(main), new JsonArray(LegalActionProvider.GetActionsForState(main, active).Select(a => (JsonNode)codec.Move(a)).ToArray()));
            GameState offered = new(main);
            Play(offered, LumberForOre(offered, active));
            Play(offered, new RespondToTradeAction(Seat(active, 2), false));

            // The copy has an offer the referee closed without a step (a late answer, a player moving on): the referee's stands
            Assert.Null(replica.Follow(new JsonObject { ["type"] = "step", ["n"] = 1, ["seat"] = (int)active, ["action"] = codec.Move(LumberForOre(main, active)) }));
            Assert.NotNull(replica.FirstDifference(codec.State(main), "closed"));
            Assert.True(replica.Patch(codec.State(main)));
            Assert.Null(replica.State.Turn.PendingTrade);
            Assert.Equal(active, replica.Acting());

            // The other way: an offer, with an answer, that a fresh copy (no offer made this turn) didn't see open
            replica = new(Before(JsonNode.Parse("""{"preset":"standard","player_trading":"first_accept","offer_limit":null}""")!.AsObject(), out _));
            replica.Rebuild(codec.State(main), new JsonArray(LegalActionProvider.GetActionsForState(main, active).Select(a => (JsonNode)codec.Move(a)).ToArray()));
            Assert.Equal(0, replica.State.Turn.TradeOffersThisTurn);
            Assert.NotNull(replica.FirstDifference(codec.State(offered), "opened"));
            Assert.True(replica.Patch(codec.State(offered)));
            Assert.Null(replica.FirstDifference(codec.State(offered), "opened"));
            Assert.Equal(offered.Turn.PendingTrade!.Give, replica.State.Turn.PendingTrade!.Give);
            Assert.Equal(TradeResponse.Rejected, replica.State.Turn.PendingTrade.Responses[Seat(active, 2)]);
            Assert.True(replica.State.Turn.TradeOffersThisTurn >= 1);
        }

        private static Agents.Inference.StateValueNet Net(string name) =>
            new(TestUtil.FixturePath("ValueNet", name), provider: Agents.Inference.InferenceProvider.CPU);

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void HexfarersAgentsAnswerWhenAskedOutOfSeatOrder(bool fair)
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            Play(state, LumberForOre(state, active));
            sbyte last = Seat(active, 3);
            Agents.MCTS.MCTSAgent search = new(last, state, valueNet: Net("sensitive_tiny.onnx"), iterationLimit: 16, leavesPerBatch: 8);
            Agent agent = fair ? new Agents.MCTS.InformationSetMCTSAgent(search, state, seed: 1) : search;
            Action answer = agent.Act(new GameState(state));
            Assert.True(answer is RespondToTradeAction or CounterTradeAction, $"answered with {answer}");
            Assert.Equal(last, answer.PlayerIndex);
            Assert.True(answer.IsValidFor(state), $"{answer} should be valid");
        }

        [Fact]
        public void HexfarersAgentPicksAStandingCounterofferOrWithdrawsWhenAsked()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            Play(state, LumberForOre(state, active));
            Play(state, new CounterTradeAction(Seat(active, 2), Cards((Lumber, 2)), Cards((Ore, 1))));
            Agents.MCTS.MCTSAgent agent = new(active, state, valueNet: Net("sensitive_tiny.onnx"), iterationLimit: 16, leavesPerBatch: 8);
            Action pick = agent.Act(new GameState(state));
            Assert.True(pick is ConfirmTradeAction { Partner: var partner } && partner == Seat(active, 2) || pick is CancelTradeAction, $"picked {pick}");
            Assert.True(pick.IsValidFor(state));
        }

        // Hexfarer's Open Seat Agent at a referee with JSettlers' flow, seated at the seat, after its handshake and before
        private static (SeatAgent agent, OSPCodec codec) SeatedAgent(GameState state, sbyte seat, bool declineTrades = false, string net = "sensitive_tiny.onnx")
        {
            SeatAgent agent = new(Net(net), iterations: 16, seconds: 0, new StringWriter(), leavesPerBatch: 8, declineTrades: declineTrades);
            OSPCodec codec = new(state.Board);
            agent.Handle(JsonNode.Parse("""{"type":"handshake","protocol":"osp","version":2}""")!);
            JsonNode ready = agent.Handle(new JsonObject
            {
                ["type"] = "before", ["game_id"] = "g1", ["seat"] = (int)seat, ["players"] = 4,
                ["ruleset"] = JsonNode.Parse("""{"preset":"standard","player_trading":"first_accept","offer_limit":null}"""),
                ["budget"] = new Agents.OpenSeat.OpenSeatBudget(1000).Json(), ["seed"] = 1, ["board"] = codec.BoardJson(), ["state"] = codec.State(state),
            })!;
            Assert.Equal("ready", ready["type"]!.GetValue<string>());
            return (agent, codec);
        }

        // A decision with the true state and the given legal moves and trades
        private static JsonObject Decision(OSPCodec codec, GameState state, sbyte seat, JsonArray legal, JsonNode? trades) => new()
        {
            ["type"] = "decide", ["game_id"] = "g1", ["n"] = 7, ["seat"] = (int)seat, ["hash"] = OSPCodec.Hash(codec.State(state)),
            ["state"] = codec.State(state), ["legal"] = legal, ["trades"] = trades,
        };

        [Fact]
        public void HexfarersOpenSeatAgentDecidesAnAnswerOutOfSeatOrder()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            Play(state, LumberForOre(state, active));
            sbyte last = Seat(active, 3);
            (SeatAgent agent, OSPCodec codec) = SeatedAgent(state, last);
            // No trade is worth that much: the agent has to turn the offer down, not play the first legal move
            agent.Search!.AnswerMargin = 100;
            JsonArray legal = [codec.Move(new RespondToTradeAction(last, true)), codec.Move(new RespondToTradeAction(last, false))];
            JsonNode answer = agent.Handle(Decision(codec, state, last, legal, JsonNode.Parse("""{"counter":{}}""")))!;
            Assert.Equal(OSPCodec.Canonical(codec.Move(new RespondToTradeAction(last, false))), OSPCodec.Canonical(answer["action"]));
        }

        [Fact]
        public void HexfarersOpenSeatAgentAcceptsOutOfSeatOrderWhenTheSeatBeforeItCannot()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            // The first addressee in seat order has no ore (back in the bank), so its legal moves hold no accept
            sbyte first = Seat(active, 1), last = Seat(active, 3);
            state.Players[first].ResourceCards.Remove(Cards((Ore, 2)));
            state.ResourceBank.Add(Cards((Ore, 2)));
            Play(state, LumberForOre(state, active));
            Assert.False(new RespondToTradeAction(first, true).IsValidFor(state));
            (SeatAgent agent, OSPCodec codec) = SeatedAgent(state, last);
            // Any trade seems worth it: the agent accepts, and the accept has to reach the referee, not the first legal move
            agent.Search!.AnswerMargin = -100;
            agent.Search.AnswerFairness = 0;
            JsonArray legal = [codec.Move(new RespondToTradeAction(last, false)), codec.Move(new RespondToTradeAction(last, true))];
            JsonNode answer = agent.Handle(Decision(codec, state, last, legal, JsonNode.Parse("""{"counter":{}}""")))!;
            Assert.Equal(OSPCodec.Canonical(codec.Move(new RespondToTradeAction(last, true))), OSPCodec.Canonical(answer["action"]));

            // A referee that doesn't allow the accept gets a move it allows, though the agent would accept
            (SeatAgent strict, OSPCodec strictCodec) = SeatedAgent(state, last);
            strict.Search!.AnswerMargin = -100;
            strict.Search.AnswerFairness = 0;
            JsonArray declineOnly = [strictCodec.Move(new RespondToTradeAction(last, false))];
            JsonNode declined = strict.Handle(Decision(strictCodec, state, last, declineOnly, JsonNode.Parse("""{"counter":{}}""")))!;
            Assert.Equal(OSPCodec.Canonical(declineOnly[0]), OSPCodec.Canonical(declined["action"]));
        }

        [Fact]
        public void HexfarersOpenSeatAgentPicksOnItsOfferBeforeEveryoneAnswered()
        {
            List<Action> decided = [];
            // Counteroffers in the offerer's terms (it gives, it gets): the plain test net puts some above the agent's margin and the
            // others below it, so the agent takes some and withdraws from others
            CardSet<ResourceCardType>[][] counters =
            [
                [Cards((Lumber, 1)), Cards((Ore, 2))], [Cards((Lumber, 2)), Cards((Ore, 1))], [Cards((Lumber, 1)), Cards((Grain, 1))],
                [Cards((Lumber, 1)), Cards((Wool, 2))], [Cards((Lumber, 2), (Brick, 2)), Cards((Ore, 1))], [Cards((Brick, 1)), Cards((Grain, 1))],
                [Cards((Lumber, 1)), Cards((Ore, 2), (Grain, 2), (Wool, 2), (Brick, 2))], [Cards((Wool, 2)), Cards((Ore, 1))],
            ];
            foreach (CardSet<ResourceCardType>[] terms in counters)
            {
                (GameState state, sbyte active) = FirstAcceptGame();
                Play(state, LumberForOre(state, active));
                sbyte counterer = Seat(active, 2);
                Play(state, new CounterTradeAction(counterer, terms[0], terms[1]));
                // The first addressee hasn't answered yet: in seat order it is the one to act, but the offerer may pick now
                Assert.Equal(Seat(active, 1), state.Turn.PendingTrade!.NextResponder);
                (SeatAgent agent, OSPCodec codec) = SeatedAgent(state, active, net: "plain_tiny.onnx");
                Action expected = agent.Search!.DecideTrade(new GameState(state))!;
                Assert.True(expected is ConfirmTradeAction or CancelTradeAction, $"decided {expected}");
                decided.Add(expected);
                Action other = expected is CancelTradeAction ? new ConfirmTradeAction(active, counterer) : new CancelTradeAction(active);
                // The other choice first: the agent's own has to reach the referee, not the first legal move
                JsonArray legal = [codec.Move(other), codec.Move(expected)];
                JsonNode answer = agent.Handle(Decision(codec, state, active, legal, null))!;
                Assert.Equal(OSPCodec.Canonical(codec.Move(expected)), OSPCodec.Canonical(answer["action"]));
            }
            // Both picks were made: taking the counteroffer and withdrawing
            Assert.Contains(decided, a => a is ConfirmTradeAction);
            Assert.Contains(decided, a => a is CancelTradeAction);
        }

        [Fact]
        public void HexfarersOpenSeatAgentDeclinesEveryTradeWhenToldTo()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            // On its turn with trades named: a legal move, no offer
            (SeatAgent onTurn, OSPCodec codec) = SeatedAgent(state, active, declineTrades: true);
            // Any offer would seem worth it: only the option keeps the agent from making one
            onTurn.Search!.Offers.Margin = -100;
            JsonArray main = new(LegalActionProvider.GetActionsForState(state, active).Select(a => (JsonNode)codec.Move(a)).ToArray());
            JsonObject offers = new() { ["offer"] = new JsonObject { ["to"] = new JsonArray((JsonNode)(int)Seat(active, 1), (int)Seat(active, 2), (int)Seat(active, 3)), ["left"] = null } };
            JsonNode move = onTurn.Handle(Decision(codec, state, active, main, offers))!;
            Assert.Contains(OSPCodec.Canonical(move["action"]), main.Select(OSPCodec.Canonical));

            // Asked about an offer (in or out of seat order): it declines
            Play(state, LumberForOre(state, active));
            foreach (sbyte seat in new[] { Seat(active, 1), Seat(active, 3) })
            {
                (SeatAgent agent, OSPCodec seatCodec) = SeatedAgent(state, seat, declineTrades: true);
                // Any trade would seem worth accepting: only the option keeps the agent from it
                agent.Search!.AnswerMargin = -100;
                JsonArray legal = [seatCodec.Move(new RespondToTradeAction(seat, true)), seatCodec.Move(new RespondToTradeAction(seat, false))];
                JsonNode answer = agent.Handle(Decision(seatCodec, state, seat, legal, JsonNode.Parse("""{"counter":{}}""")))!;
                Assert.Equal(OSPCodec.Canonical(seatCodec.Move(new RespondToTradeAction(seat, false))), OSPCodec.Canonical(answer["action"]));
            }
        }

        [Fact]
        public void AnOpenOfferRoundTripsThroughTheProtocolsState()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1), (Brick, 1)), Cards((Ore, 1)), To(Seat(active, 1), Seat(active, 3))));
            Play(state, new CounterTradeAction(Seat(active, 3), Cards((Lumber, 2)), Cards((Ore, 1))));
            OSPCodec codec = new(state.Board);
            JsonArray legal = new(LegalActionProvider.GetActionsForState(state, Seat(active, 1)).Select(a => (JsonNode)codec.Move(a)).ToArray());
            GameState rebuilt = codec.BuildState(codec.State(state), legal);
            Assert.Equal(state.Turn.PendingTrade, rebuilt.Turn.PendingTrade);
            Assert.Equal(OSPCodec.Canonical(codec.State(state)), OSPCodec.Canonical(codec.State(rebuilt)));
        }

        [Fact]
        public void TheProtocolsStateNamesWhoActsWhileAnOfferIsOpen()
        {
            (GameState state, sbyte active) = FirstAcceptGame();
            OSPCodec codec = new(state.Board);
            int Current() => codec.State(state)["current"]!.GetValue<int>();
            Assert.Equal(active, Current());
            Play(state, LumberForOre(state, active));
            // The next addressee in seat order, also when a later one answered first, then the offerer
            Assert.Equal(Seat(active, 1), Current());
            Play(state, new RespondToTradeAction(Seat(active, 3), false));
            Assert.Equal(Seat(active, 1), Current());
            Play(state, new RespondToTradeAction(Seat(active, 1), false));
            Play(state, new CounterTradeAction(Seat(active, 2), Cards((Lumber, 2)), Cards((Ore, 1))));
            Assert.Equal(active, Current());
        }

        [Fact]
        public void ARebuildWithinTheTurnKeepsTheOffersAlreadyMade()
        {
            Replica replica = new(Before(JsonNode.Parse("""{"preset":"standard","player_trading":"first_accept","offer_limit":null}""")!.AsObject(), out OSPCodec codec));
            (GameState main, sbyte active) = FirstAcceptGame(seed: 3);
            JsonArray Legal(GameState s) => new(LegalActionProvider.GetActionsForState(s, (sbyte)Enumerable.Range(0, 4).First(s.CanPlayerAct)).Select(a => (JsonNode)codec.Move(a)).ToArray());
            replica.Rebuild(codec.State(main), Legal(main));
            // Two offers this turn, both declined by everyone
            foreach (int round in new[] { 0, 1 })
            {
                OfferTradeAction offer = new(active, Cards((round == 0 ? Lumber : Brick, 1)), Cards((Ore, 1)), OfferTradeAction.Everyone(main, active));
                Play(main, offer);
                Play(replica.State, offer.Clone());
                foreach (int offset in new[] { 1, 2, 3 })
                {
                    Play(main, new RespondToTradeAction(Seat(active, offset), false));
                    Play(replica.State, new RespondToTradeAction(Seat(active, offset), false));
                }
            }
            Assert.Equal(2, replica.State.Turn.TradeOffersThisTurn);
            // The referee's state (which doesn't say how many offers came) taken over in the same turn: the count stays
            replica.Rebuild(codec.State(main), Legal(main));
            Assert.Equal(2, replica.State.Turn.TradeOffersThisTurn);
            // In the next turn it starts at 0
            Play(main, new EndTurnAction(active));
            replica.Rebuild(codec.State(main), Legal(main));
            Assert.Equal(0, replica.State.Turn.TradeOffersThisTurn);
        }

        [Fact]
        public void ARebuildFromTheRefereesStateKeepsTheRuleset()
        {
            Replica replica = new(Before(JsonNode.Parse("""{"preset":"standard","player_trading":"first_accept","offer_limit":4,"victory_points":12,"discard_limit":9}""")!.AsObject(), out OSPCodec codec));
            GameSettings expected = new(replica.State.Settings);
            (GameState main, sbyte active) = MainPhase(seed: 3);
            replica.Rebuild(codec.State(main), new JsonArray(LegalActionProvider.GetActionsForState(main, active).Select(a => (JsonNode)codec.Move(a)).ToArray()));
            Assert.Equal(expected, replica.State.Settings);
            Assert.Equal(4, replica.State.Settings.TradeOfferLimit);
            Assert.Equal(12, replica.State.Settings.VictoryPoints);
        }

        // A before for seat 0 of a fresh game, with the codec of its board
        private static JsonObject Before(JsonObject ruleset, out OSPCodec codec)
        {
            TestUtil.Seed(3);
            GameState game = TestUtil.NewGame();
            codec = new OSPCodec(game.Board);
            return new JsonObject
            {
                ["type"] = "before", ["game_id"] = "g3", ["seat"] = 0, ["players"] = 4, ["ruleset"] = ruleset,
                ["budget"] = new Agents.OpenSeat.OpenSeatBudget(1000).Json(), ["seed"] = 3, ["board"] = codec.BoardJson(), ["state"] = codec.State(game),
            };
        }

        [Fact]
        public void RandomGamesWithFirstAcceptTradesStayConsistentAndTracked()
        {
            Random random = new Random(7);
            int offers = 0, outOfOrder = 0, accepts = 0, early = 0;
            for (int game = 0; game < 20; game++)
            {
                GameState start = HiddenInformationTests.NewGame(300 + game, freeTrade: true);
                start.Settings.TradeFlow = TradeFlow.FirstAccept;
                HiddenInformationTests.Followed followed = new(start);
                GameState state = followed.Truth;
                for (int move = 0; move < 2500 && !state.HasEnded; move++)
                {
                    Action action = RandomFirstAcceptMove(state, random);
                    Assert.True(action.IsValidFor(state), $"{action} should be valid");

                    GameState before = new GameState(state);
                    Action probe = action.Clone();
                    probe.Apply(state);
                    probe.Revert(state);
                    Assert.Equal(before, state);
                    Assert.Equal(before.GetHashCode(), state.GetHashCode());

                    offers += action is OfferTradeAction ? 1 : 0;
                    accepts += action is RespondToTradeAction { Accept: true } ? 1 : 0;
                    outOfOrder += action is RespondToTradeAction or CounterTradeAction && state.Turn.PendingTrade?.NextResponder != action.PlayerIndex ? 1 : 0;
                    early += action is ConfirmTradeAction or CancelTradeAction && state.Turn.PendingTrade?.NextResponder != null ? 1 : 0;
                    followed.Play(action);

                    // Cards are neither made nor lost, and every observer still holds the true hands possible
                    foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                        Assert.Equal(19u, state.ResourceBank.Get(type) + (uint)state.Players.Sum(p => p.ResourceCards.Get(type)));
                    if (action is OfferTradeAction or RespondToTradeAction or CounterTradeAction or ConfirmTradeAction or CancelTradeAction)
                        foreach (BeliefTracker tracker in followed.Trackers.Values) HiddenInformationTests.AssertSupports(tracker, state);
                }
            }
            Assert.True(offers > 100 && outOfOrder > 30 && accepts > 10 && early > 10, $"{offers} offers, {outOfOrder} answers out of order, {accepts} accepts, {early} early picks or withdrawals");
        }

        // A random move of the game that uses the flow's freedoms: answers by any addressed player, the offerer taking a
        // counteroffer or withdrawing while others are still asked
        private static Action RandomFirstAcceptMove(GameState state, Random random)
        {
            if (state.Turn.PendingTrade is TradeOffer offer && random.NextDouble() < 0.5)
            {
                List<Action> free = [];
                for (sbyte player = 0; player < offer.Responses.Length; player++)
                {
                    if (offer.Responses[player] != TradeResponse.Pending) continue;
                    free.Add(new RespondToTradeAction(player, false));
                    free.Add(new RespondToTradeAction(player, true));
                    CardSet<ResourceCardType> hand = state.Players[player].ResourceCards;
                    ResourceCardType? wanted = CardTypeExtensions.ResourceTypes.Cast<ResourceCardType?>().FirstOrDefault(t => offer.Give.Get(t!.Value) == 0 && hand.Get(t.Value) > 0);
                    if (wanted is ResourceCardType theirs)
                        free.Add(new CounterTradeAction(player, offer.Give, Cards((theirs, 1))));
                }
                free.Add(new CancelTradeAction(offer.Offerer));
                foreach (TradeCounter counter in offer.Counters) free.Add(new ConfirmTradeAction(offer.Offerer, counter.Player));
                List<Action> valid = free.Where(a => a.IsValidFor(state)).ToList();
                if (valid.Count > 0) return valid[random.Next(valid.Count)];
            }
            sbyte acting = (sbyte)Enumerable.Range(0, 4).First(state.CanPlayerAct);
            List<Action> legal = LegalActionProvider.GetActionsForState(state, acting);
            if (OfferTradeAction.IsTurnValid(state, acting) && random.NextDouble() < 0.3)
            {
                CardSet<ResourceCardType> hand = state.Players[acting].ResourceCards;
                List<ResourceCardType> held = CardTypeExtensions.ResourceTypes.Where(t => hand.Get(t) > 0).ToList();
                if (held.Count > 0)
                {
                    ResourceCardType given = held[random.Next(held.Count)];
                    ResourceCardType asked = CardTypeExtensions.ResourceTypes.Where(t => t != given).ElementAt(random.Next(4));
                    OfferTradeAction offerAction = new(acting, Cards((given, 1)), Cards((asked, 1)), OfferTradeAction.Everyone(state, acting));
                    if (offerAction.IsValidFor(state)) return offerAction;
                }
            }
            return legal[random.Next(legal.Count)];
        }
    }
}
