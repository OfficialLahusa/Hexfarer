using Common;
using Common.Actions;
using Common.Osp;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace Agents.OpenSeat
{
    /// <summary>
    /// Plays a seat of a Hexfarer game by asking an Open Seat Agent (see Bridges/PROTOCOL.md): Hexfarer is the referee. The
    /// proxy greets the agent (it has to play the standard rules) and starts the game with before; every decision of the
    /// seat goes out as decide with the seat's state and the legal moves, and the answer has to be one of them; every move
    /// goes out as step if the agent observes; the end of the game (or of the proxy) as after. What the agent sees follows
    /// what it needs: its seat's view (another seat's hand, development cards and points with victory point cards are
    /// null, another seat's bought card and steal "unknown"), a world drawn from that view by the seat's BeliefTracker for
    /// every decision (marked "drawn"; steps then carry the view), or the true state (steps carry it too). The proxy keeps
    /// its own copy of the game, following the moves it is told, for the states before them and the seat's beliefs. A
    /// reset (an undo or an edit in the simulator) ends the agent's game, cut short, and starts a new one from the position.
    /// </summary>
    public sealed class OpenSeatAgentProxy : Agent, IDisposable
    {
        private readonly IOpenSeatChannel _channel;
        private readonly string _baseId;
        private readonly OspCodec _codec;
        // The game as the moves told to the proxy left it, and what the seat knows of it (for worlds)
        private GameState _mirror;
        private BeliefTracker? _beliefs;
        private readonly Random _random;
        // The agent's current game (a new one after each reset) and its moves
        private string _gameId;
        private int _moves, _resets;
        private bool _closed;

        /// <summary>
        /// The agent's answer to hello
        /// </summary>
        public OpenSeatHello Hello { get; }

        /// <summary>
        /// Greets the agent on the channel and starts the game; the channel is the proxy's from now on (disposed with it).
        /// seed draws the worlds reproducibly; beliefs are what the seat knows if the game is under way (else it starts).
        /// </summary>
        public OpenSeatAgentProxy(sbyte seat, GameState start, IOpenSeatChannel channel, string gameId, int? seed = null, BeliefTracker? beliefs = null) : base(seat)
        {
            _channel = channel;
            _baseId = _gameId = gameId;
            _codec = new OspCodec(start.Board);
            _mirror = new GameState(start);
            _random = seed is int s ? new Random(s) : new Random();
            try
            {
                Hello = OpenSeatHello.Exchange(channel);
            }
            catch
            {
                channel.Dispose();
                throw;
            }
            // Hexfarer's referee plays the rulebook; its options (no trades between players, other limits) go with the game
            if (!Hello.Rulesets.Contains("standard"))
                Fail($"The Open Seat Agent {Hello.Name} doesn't play the standard rules (it plays {string.Join(", ", Hello.Rulesets)}).");
            if (Hello.Needs == OpenSeatNeeds.World) _beliefs = beliefs != null ? new BeliefTracker(beliefs) : new BeliefTracker(seat, start);
            SendBefore(start);
        }

        private void SendBefore(GameState start) => Send(new JsonObject
        {
            ["type"] = "before",
            ["game_id"] = _gameId,
            ["seat"] = (int)PlayerIndex,
            ["players"] = start.Players.Length,
            ["ruleset"] = OspCodec.Ruleset(start.Settings),
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
        });

        /// <summary>
        /// The game jumped (an undo or an edit): the agent's game ends, cut short, and a new one starts from the position,
        /// with what the seat knows there (beliefs, for an agent that gets worlds)
        /// </summary>
        public void ResetTo(GameState state, BeliefTracker? beliefs)
        {
            if (_closed) return;
            SendAfter();
            _gameId = $"{_baseId}-{++_resets}";
            _mirror = new GameState(state);
            _moves = 0;
            if (Hello.Needs == OpenSeatNeeds.World) _beliefs = beliefs != null ? new BeliefTracker(beliefs) : new BeliefTracker(PlayerIndex, state);
            SendBefore(state);
        }

        public override void ResetToState(GameState state) => ResetTo(state, null);

        // The state the agent gets for a decision: its view, a world drawn from it, or the truth
        private JsonObject StateFor(GameState state)
        {
            switch (Hello.Needs)
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
            Send(new JsonObject
            {
                ["type"] = "decide",
                ["game_id"] = _gameId,
                ["n"] = _moves,
                ["seat"] = (int)PlayerIndex,
                ["state"] = StateFor(state),
                ["legal"] = moves,
            });
            JsonNode answer;
            try
            {
                // An answer to an earlier decision (one the simulation stopped waiting for, or of a game reset since) is skipped
                do answer = _channel.Receive(cancellation);
                while (answer["type"]?.ToJsonString() == "\"decide\"" && answer["game_id"]?.ToJsonString() is string id
                    && (id != $"\"{_gameId}\"" || answer["n"] is JsonNode at && int.TryParse(at.ToJsonString(), out int asked) && asked < _moves));
            }
            catch (OperationCanceledException)
            {
                // The simulation stopped waiting (a pause, an undo) and throws the move away; the agent's answer comes later
                return legal[0];
            }
            if (answer["type"]?.ToJsonString() != "\"decide\"" || answer["action"] is not JsonNode chosen)
                Fail($"The Open Seat Agent {Hello.Name} answered a decision with {answer.ToJsonString()}.");
            if (answer["n"] is JsonNode n && n.ToJsonString() != _moves.ToString())
                Fail($"The Open Seat Agent {Hello.Name} answered decision {n.ToJsonString()} when asked for {_moves}.");

            // The answer has to be one of the listed moves, as written there
            string wanted = OspCodec.Canonical(answer["action"]);
            for (int i = 0; i < moves.Count; i++)
                if (OspCodec.Canonical(moves[i]) == wanted) return legal[i];
            Fail($"The Open Seat Agent {Hello.Name} answered {wanted}, which is not one of the legal moves.");
            return null!;
        }

        public override void HandleAction(Action action)
        {
            // Called after the move was played, so its history holds its result; the copy is still before it
            if (Hello.Observe)
            {
                JsonObject step = new()
                {
                    ["type"] = "step",
                    ["game_id"] = _gameId,
                    ["n"] = _moves,
                    ["seat"] = (int)action.PlayerIndex,
                    ["action"] = _codec.Move(action),
                    // An open-hand agent sees every result in full, the others as their seat may
                    ["result"] = OspCodec.Result(action, Hello.Needs == OpenSeatNeeds.Truth ? null : PlayerIndex),
                };
                // The state before the move: the truth for an open-hand agent, the view for one that gets worlds
                if (Hello.Needs == OpenSeatNeeds.Truth) step["state"] = _codec.State(_mirror);
                else if (Hello.Needs == OpenSeatNeeds.World) step["state"] = _codec.View(_mirror, PlayerIndex);
                Send(step);
            }

            GameState? before = _beliefs != null ? new GameState(_mirror) : null;
            Action replay = action.Clone();
            replay.Apply(_mirror);
            _beliefs?.Observe(before!, replay, _mirror);
            _moves++;
        }

        private void Send(JsonNode message)
        {
            if (_closed) throw new OpenSeatException($"The Open Seat Agent {Hello?.Name} was already closed.");
            _channel.Send(message);
        }

        // Ends the seat: tells the agent why (if it still listens) and closes it
        private void Fail(string message)
        {
            try { _channel.Send(new JsonObject { ["type"] = "error", ["message"] = message }); }
            catch (OpenSeatException) { }
            _closed = true;
            _channel.Dispose();
            throw new OpenSeatException(message);
        }

        /// <summary>
        /// Ends the game for the agent (after: the winner, everyone's points, whether the game was cut short) and closes it
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
