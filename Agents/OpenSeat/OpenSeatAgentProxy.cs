using Common;
using Common.Actions;
using Common.OSP;
using System.Diagnostics;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace Agents.OpenSeat
{
    /// <summary>
    /// Plays a seat of a Lanzenbrecher game by asking an Open Seat Agent (see Bridges/PROTOCOL.md, version 2): Lanzenbrecher is the
    /// referee. The proxy shakes hands with the agent (it has to play the standard rules) and starts the game with before,
    /// which the agent takes (ready) or turns down; every decision of the seat goes out as decide with the seat's state (or
    /// only its hash, for a lean agent) and the legal moves, and the answer has to be one of them within the budget's time;
    /// every move goes out as step if the agent observes; the end of the game (or of the proxy) as after, with the game's
    /// result record. What the agent sees follows what it needs: its seat's view (another seat's hand, development cards
    /// and points with victory point cards are null, another seat's bought card and steal "unknown"), a world drawn from
    /// that view by the seat's BeliefTracker for every decision (marked "drawn"; steps then carry the view), or the true
    /// state (steps carry it too). The proxy keeps its own copy of the game, following the moves it is told, for the states
    /// before them and the seat's beliefs. A reset (an undo or an edit in the simulator) ends the agent's game, cut short,
    /// and starts a new one from the position. Whatever fails (an illegal or late answer, a malformed message, the agent's
    /// own failure, its process ending) is told to the agent where it still listens, recorded in the seat's record, and
    /// ends the seat with an OpenSeatException that carries the failure's code.
    /// </summary>
    public sealed class OpenSeatAgentProxy : Agent, IDisposable
    {
        private readonly IOpenSeatChannel _channel;
        private readonly string _baseId;
        private readonly OSPCodec _codec;
        private readonly OpenSeatBudget _budget;
        // The game as the moves told to the proxy left it, and what the seat knows of it (for worlds)
        private GameState _mirror;
        private BeliefTracker? _beliefs;
        private readonly Random _random;
        // The agent's current game (a new one after each reset), its moves and player turns
        private string _gameId;
        private int _moves, _turns, _resets;
        // Closed: the agent failed (why, if it did) or the seat ended
        private bool _closed;
        private string? _failure;

        /// <summary>
        /// The agent's handshake
        /// </summary>
        public OpenSeatHandshake Handshake { get; }

        /// <summary>
        /// The id of the agent's current game (a new one after each reset)
        /// </summary>
        public string GameId => _gameId;

        /// <summary>
        /// The seed the agent got for its randomness, which also draws its worlds
        /// </summary>
        public int Seed { get; }

        /// <summary>
        /// The seat's line in the game's result record: its decisions, their time, resyncs and failures so far
        /// </summary>
        public OpenSeatSeatRecord Record { get; }

        /// <summary>
        /// The moves and player turns of the agent's current game so far
        /// </summary>
        public int Moves => _moves;
        public int Turns => _turns;

        /// <summary>
        /// The game's result record for the agent's after, set by whoever runs the game once it is over (else the proxy
        /// writes one with only its own seat's line filled)
        /// </summary>
        public JsonObject? Result { get; set; }

        /// <summary>
        /// The channel to the agent, for tests
        /// </summary>
        internal IOpenSeatChannel Channel => _channel;

        /// <summary>
        /// Shakes hands with the agent on the channel and starts the game, which the agent takes or turns down (an
        /// OpenSeatException with the code unsupported); the channel is the proxy's from now on (disposed with it). seed is
        /// the agent's seed and draws its worlds (random if not given); beliefs are what the seat knows if the game is under
        /// way (else it starts); budget is the game's time budget (OpenSeatBudget.Default if not given); cancellation stops
        /// the wait for the agent's handshake and ready.
        /// </summary>
        public OpenSeatAgentProxy(sbyte seat, GameState start, IOpenSeatChannel channel, string gameId, int? seed = null, BeliefTracker? beliefs = null,
            CancellationToken cancellation = default, OpenSeatBudget? budget = null) : base(seat)
        {
            _channel = channel;
            _baseId = _gameId = gameId;
            _codec = new OSPCodec(start.Board);
            _mirror = new GameState(start);
            _budget = budget ?? OpenSeatBudget.Default;
            // The agent's seed for this game: the one given (so a run repeats) or a fresh one; the same seed makes a seeded
            // agent play the same game the same way
            Seed = seed ?? Random.Shared.Next();
            _random = new Random(Seed);
            try
            {
                Handshake = OpenSeatHandshake.Exchange(channel, cancellation);
            }
            catch
            {
                channel.Dispose();
                throw;
            }
            Record = new OpenSeatSeatRecord(seat, Handshake);
            // Lanzenbrecher's referee plays the rulebook; its options (no trades between players, other limits) go with the game
            // No game has started yet, so the failure names none
            if (!Handshake.Rulesets.Contains("standard"))
                Fail(OpenSeatFailure.Unsupported, $"The Open Seat Agent {Handshake.Name} doesn't play the standard rules (it plays {string.Join(", ", Handshake.Rulesets)}).",
                    detail: new JsonObject { ["preset"] = "standard" }, inGame: false);
            if (Handshake.Needs == OpenSeatNeeds.World) _beliefs = beliefs != null ? new BeliefTracker(beliefs) : new BeliefTracker(seat, start);

            // The agent takes the game or turns it down (or has ended since its handshake)
            JsonNode answer;
            try
            {
                SendBefore(start);
                answer = OpenSeatHandshake.ReceiveWithin(channel, OpenSeatHandshake.AnswerTime, cancellation);
            }
            catch (OpenSeatException e) when (e.Code == OpenSeatFailure.Timeout)
            {
                Fail(OpenSeatFailure.Timeout, $"The Open Seat Agent {Handshake.Name} didn't take the game within {OpenSeatHandshake.AnswerTime.TotalSeconds:0} s.");
                throw;
            }
            catch (OpenSeatException e)
            {
                Lost(e, null);
                throw new OpenSeatException(e.Message, e.Code, e.FromAgent, e) { Handshake = Handshake };
            }
            catch (OperationCanceledException)
            {
                Close(null);
                throw;
            }
            if (OpenSeatHandshake.Text(answer["type"]) == "failure") throw AgentFailed(answer, null);
            if (OpenSeatHandshake.Text(answer["type"]) != "ready" || OpenSeatHandshake.Text(answer["game_id"]) != _gameId)
                Fail(OpenSeatFailure.Malformed, $"The Open Seat Agent {Handshake.Name} answered before with {answer.ToJsonString()}.");
        }

        private void SendBefore(GameState start) => Send(new JsonObject
        {
            ["type"] = "before",
            ["game_id"] = _gameId,
            ["seat"] = (int)PlayerIndex,
            ["players"] = start.Players.Length,
            ["ruleset"] = OSPCodec.Ruleset(start.Settings),
            ["budget"] = _budget.Json(),
            ["seed"] = Seed,
            ["board"] = _codec.BoardJson(),
            ["state"] = StateFor(start),
        });

        private void SendAfter() => _channel.Send(new JsonObject
        {
            ["type"] = "after",
            ["game_id"] = _gameId,
            ["winner"] = _mirror.HasEnded ? _mirror.Turn.PlayerIndex : null,
            ["vp"] = new JsonArray(_mirror.Players.Select(p => (JsonNode)p.VictoryPoints.Total).ToArray()),
            ["truncated"] = !_mirror.HasEnded,
            ["result"] = Result?.DeepClone() ?? OpenSeatSeatRecord.Result(Seed, _moves, _turns,
                Enumerable.Range(0, _mirror.Players.Length).Select(s => s == PlayerIndex ? Record : new OpenSeatSeatRecord(s))),
        });

        /// <summary>
        /// The game jumped (an undo or an edit): the agent's game ends, cut short, and a new one starts from the position,
        /// with what the seat knows there (beliefs, for an agent that gets worlds). The agent's ready comes with its next
        /// decision's answer, so the reset doesn't wait for it.
        /// </summary>
        public void ResetTo(GameState state, BeliefTracker? beliefs)
        {
            if (_closed) return;
            SendAfter();
            _gameId = $"{_baseId}-{++_resets}";
            _mirror = new GameState(state);
            _moves = _turns = 0;
            Result = null;
            if (Handshake.Needs == OpenSeatNeeds.World) _beliefs = beliefs != null ? new BeliefTracker(beliefs) : new BeliefTracker(PlayerIndex, state);
            SendBefore(state);
        }

        public override void ResetToState(GameState state) => ResetTo(state, null);

        // The state the agent gets for a decision: its view, a world drawn from it, or the truth
        private JsonObject StateFor(GameState state)
        {
            switch (Handshake.Needs)
            {
                case OpenSeatNeeds.Truth:
                    return _codec.State(state);
                case OpenSeatNeeds.World:
                    JsonObject world = _codec.State(_beliefs!.Determinize(_beliefs.View(state), _random));
                    world["drawn"] = true;
                    return world;
                default:
                    return _codec.View(state, PlayerIndex);
            }
        }

        public override Action Act(GameState state) => Act(state, CancellationToken.None);

        public override Action Act(GameState state, CancellationToken cancellation)
        {
            List<Action> legal = LegalActionProvider.GetActionsForState(state, PlayerIndex);
            JsonArray moves = new(legal.Select(action => (JsonNode)_codec.Move(action)).ToArray());
            JsonObject? trades = Trades(state);
            JsonObject seatState = StateFor(state);
            string hash = OSPCodec.Hash(seatState);
            // A lean agent gets only the hash, unless it asks for the state (resync)
            JsonObject Decide(bool withState)
            {
                JsonObject decide = new() { ["type"] = "decide", ["game_id"] = _gameId, ["n"] = _moves, ["seat"] = (int)PlayerIndex, ["hash"] = hash };
                if (withState || !Handshake.Lean) decide["state"] = seatState.DeepClone();
                decide["legal"] = moves.DeepClone();
                if (trades != null) decide["trades"] = trades.DeepClone();
                return decide;
            }
            Stopwatch answering = Stopwatch.StartNew();
            Send(Decide(false));
            while (true)
            {
                JsonNode answer;
                try
                {
                    TimeSpan left = TimeSpan.FromMilliseconds(_budget.DecideMs) - answering.Elapsed;
                    if (left <= TimeSpan.Zero) throw new OpenSeatException("no time left", OpenSeatFailure.Timeout);
                    answer = OpenSeatHandshake.ReceiveWithin(_channel, left, cancellation);
                }
                catch (OperationCanceledException)
                {
                    // The simulation stopped waiting (a pause, an undo) and throws the move away; the agent's answer comes later
                    return legal[0];
                }
                catch (OpenSeatException e) when (e.Code == OpenSeatFailure.Timeout)
                {
                    Fail(OpenSeatFailure.Timeout, $"The Open Seat Agent {Handshake.Name} gave no answer within {_budget.DecideMs} ms.", _moves);
                    throw;
                }
                catch (OpenSeatException e)
                {
                    // Its process ended (or the seat closed it): nothing to tell it
                    Lost(e, _moves);
                    throw;
                }

                string? type = OpenSeatHandshake.Text(answer["type"]);
                string? id = OpenSeatHandshake.Text(answer["game_id"]);
                int? n = answer["n"] is JsonValue at && at.TryGetValue(out int asked) ? asked : null;
                // A reset's ready, and answers to earlier decisions (one the simulation stopped waiting for, or of a game
                // reset since), are skipped
                if (type == "ready") continue;
                if (type == "decide" && id != null && (id != _gameId || n < _moves)) continue;
                if (type == "failure") throw AgentFailed(answer, _moves);
                if (type == "resync" && id == _gameId && n == _moves)
                {
                    Record.Resyncs++;
                    Send(Decide(true));
                    continue;
                }
                if (type != "decide" || answer["action"] is not JsonNode chosen)
                    Fail(OpenSeatFailure.Malformed, $"The Open Seat Agent {Handshake.Name} answered a decision with {answer.ToJsonString()}.", _moves);
                if (n != _moves)
                    Fail(OpenSeatFailure.Illegal, $"The Open Seat Agent {Handshake.Name} answered decision {answer["n"]?.ToJsonString()} when asked for {_moves}.", _moves);

                // The answer has to be one of the listed moves, as written there
                string wanted = OSPCodec.Canonical(answer["action"]);
                for (int i = 0; i < moves.Count; i++)
                {
                    if (OSPCodec.Canonical(moves[i]) != wanted) continue;
                    Record.Decided(answering.ElapsedMilliseconds);
                    return legal[i];
                }
                // Or a trade the decision named, with terms of the agent's own that the rules allow
                if (TradeOf(answer["action"], trades, state) is Action trade)
                {
                    Record.Decided(answering.ElapsedMilliseconds);
                    return trade;
                }
                Fail(OpenSeatFailure.Illegal, $"The Open Seat Agent {Handshake.Name} answered {wanted}, which is neither one of the legal moves nor a trade with valid terms.", _moves);
            }
        }

        /// <summary>
        /// The trades the seat may propose now with terms of its own, which no legal list can hold (see PROTOCOL.md): an
        /// offer on its turn after the roll (the other seats, the offers left this turn), a counteroffer while it answers an
        /// offer; null if none
        /// </summary>
        internal static JsonObject? Trades(GameState state, sbyte seat)
        {
            JsonObject trades = [];
            if (OfferTradeAction.IsTurnValid(state, seat))
                trades["offer"] = new JsonObject
                {
                    ["to"] = new JsonArray(Enumerable.Range(0, state.Players.Length).Where(other => other != seat).Select(other => (JsonNode)other).ToArray()),
                    ["left"] = state.Settings.TradeOfferLimit - state.Turn.TradeOffersThisTurn,
                };
            if (RespondToTradeAction.IsTurnValid(state.Turn, seat))
                trades["counter"] = new JsonObject();
            return trades.Count > 0 ? trades : null;
        }

        private JsonObject? Trades(GameState state) => Trades(state, PlayerIndex);

        // The agent's trade if the decision named its kind (an offer only to seats it named) and the rules allow its terms
        private Action? TradeOf(JsonNode? move, JsonObject? trades, GameState state)
        {
            if (trades == null || OSPCodec.TradeAction(move, PlayerIndex) is not Action trade) return null;
            bool named = trade switch
            {
                OfferTradeAction offer => trades["offer"]?["to"] is JsonArray to
                    && (offer.Recipients & ~to.Aggregate(0, (seats, seat) => seats | 1 << seat!.GetValue<int>())) == 0,
                CounterTradeAction => trades["counter"] != null,
                _ => false,
            };
            return named && trade.IsValidFor(state) ? trade : null;
        }

        public override void HandleAction(Action action)
        {
            // A closed agent gets no more moves; its seat fails when it is asked next
            if (_closed) return;
            // Called after the move was played, so its history holds its result; the copy is still before it
            if (Handshake.Observe)
            {
                JsonObject step = new()
                {
                    ["type"] = "step",
                    ["game_id"] = _gameId,
                    ["n"] = _moves,
                    ["seat"] = (int)action.PlayerIndex,
                    ["action"] = _codec.Move(action),
                    // An open-hand agent sees every result in full, the others as their seat may
                    ["result"] = OSPCodec.Result(action, Handshake.Needs == OpenSeatNeeds.Truth ? null : PlayerIndex),
                };
                // The state before the move: the truth for an open-hand agent, the view for one that gets worlds
                if (Handshake.Needs == OpenSeatNeeds.Truth) step["state"] = _codec.State(_mirror);
                else if (Handshake.Needs == OpenSeatNeeds.World) step["state"] = _codec.View(_mirror, PlayerIndex);
                try
                {
                    Send(step);
                }
                catch (OpenSeatException e)
                {
                    // The agent ended or broke: the move stands, the seat fails at its next decision with why
                    Lost(e, _moves);
                    return;
                }
            }

            GameState? before = _beliefs != null ? new GameState(_mirror) : null;
            Action replay = action.Clone();
            replay.Apply(_mirror);
            _beliefs?.Observe(before!, replay, _mirror);
            _moves++;
            if (action is EndTurnAction) _turns++;
        }

        private void Send(JsonNode message)
        {
            if (_closed) throw new OpenSeatException($"The Open Seat Agent {Handshake?.Name} was already closed{(_failure != null ? $": {_failure}" : ".")}", OpenSeatFailure.Ended);
            _channel.Send(message);
        }

        // Closes the seat; a failure's message stays for the next use
        private void Close(string? failure)
        {
            _closed = true;
            _failure = failure;
            _channel.Dispose();
        }

        // The referee's failure: told to the agent (if it still listens; for its game, if one started), recorded, and the
        // seat ends with it
        private void Fail(string code, string message, int? n = null, JsonNode? detail = null, bool inGame = true)
        {
            try { _channel.Send(OpenSeatFailure.Message(inGame ? _gameId : null, n, code, message, detail)); }
            catch (OpenSeatException) { }
            Record?.Failed(n, code, byAgent: false, message);
            Close(message);
            throw new OpenSeatException(message, code) { Handshake = Handshake };
        }

        // The agent can't be told any more (its process ended): recorded, and the seat ends
        private void Lost(OpenSeatException e, int? n)
        {
            if (_closed) return;
            Record.Failed(n, e.Code, byAgent: false, e.Message);
            Close(e.Message);
        }

        // The agent's own failure: recorded, and the seat ends without an answer
        private OpenSeatException AgentFailed(JsonNode failure, int? n)
        {
            OpenSeatException e = OpenSeatHandshake.AgentFailure(failure, Handshake);
            Record.Failed(n, e.Code, byAgent: true, e.Message);
            Close(e.Message);
            return e;
        }

        /// <summary>
        /// Ends the game for the agent (after: the winner, everyone's points, whether the game was cut short, the result
        /// record) and closes it
        /// </summary>
        public void Dispose()
        {
            if (_closed) return;
            _closed = true;
            try
            {
                SendAfter();
            }
            catch (OpenSeatException) { }
            finally
            {
                _channel.Dispose();
            }
        }
    }
}
