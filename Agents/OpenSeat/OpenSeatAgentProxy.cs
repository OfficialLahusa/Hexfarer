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
    /// its own copy of the game, following the moves it is told, for the states before them and the seat's beliefs.
    /// </summary>
    public sealed class OpenSeatAgentProxy : Agent, IDisposable
    {
        private readonly IOpenSeatChannel _channel;
        private readonly string _gameId;
        private readonly OspCodec _codec;
        // The game as the moves told to the proxy left it, and what the seat knows of it (for worlds)
        private readonly GameState _mirror;
        private readonly BeliefTracker? _beliefs;
        private readonly Random _random;
        private int _moves;
        private bool _closed;

        /// <summary>
        /// The agent's answer to hello
        /// </summary>
        public OpenSeatHello Hello { get; }

        /// <summary>
        /// Greets the agent on the channel and starts the game; the channel is the proxy's from now on (disposed with it).
        /// seed draws the worlds reproducibly.
        /// </summary>
        public OpenSeatAgentProxy(sbyte seat, GameState start, IOpenSeatChannel channel, string gameId, int? seed = null) : base(seat)
        {
            _channel = channel;
            _gameId = gameId;
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
            if (Hello.Needs == OpenSeatNeeds.World) _beliefs = new BeliefTracker(seat, start);

            Send(new JsonObject
            {
                ["type"] = "before",
                ["game_id"] = gameId,
                ["seat"] = (int)seat,
                ["players"] = start.Players.Length,
                ["ruleset"] = OspCodec.Ruleset(start.Settings),
                ["board"] = _codec.BoardJson(),
                ["state"] = StateFor(start),
            });
        }

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
            JsonNode answer = _channel.Receive(cancellation);
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
                _channel.Send(new JsonObject
                {
                    ["type"] = "after",
                    ["game_id"] = _gameId,
                    ["winner"] = _mirror.HasEnded ? _mirror.Turn.PlayerIndex : null,
                    ["vp"] = new JsonArray(_mirror.Players.Select(p => (JsonNode)p.VictoryPoints.Total).ToArray()),
                    ["truncated"] = !_mirror.HasEnded,
                });
            }
            catch (OpenSeatException) { }
            finally
            {
                _channel.Dispose();
            }
        }
    }
}
