using Agents.Inference;
using Agents.MCTS;
using Common.Actions;
using Common.Osp;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace OpenSeat
{
    /// <summary>
    /// Hexfarer's agents as an Open Seat Agent: answers a referee's Open Seat Protocol messages (one JSON object per line
    /// on standard input and output) for one seat, with Hexfarer's open-hand MCTS and the value net. It keeps a copy of
    /// the game (Replica) that follows every step; before each decision it compares the copy with the referee's state and
    /// takes over what the referee's known rule differences change (Replica.Patch), since the referee decides the game.
    /// The search plans by Hexfarer's rules; a move the referee doesn't allow is replaced by the most visited root move it
    /// allows. It needs the full state ("truth"): every hand is the true one, as for the open-hand MCTS in Hexfarer.
    /// </summary>
    internal sealed class SeatAgent(StateValueNet net, long iterations, double seconds, TextWriter log)
    {
        private Replica? _replica;
        private MCTSAgent? _agent;
        private string _gameId = "";
        private sbyte _seat;
        // Per game: decisions, those searched, patched states, replaced moves, whether the copy was lost (it then plays the
        // referee's first legal move)
        private int _patches, _fallbacks, _decisions, _searched;
        private bool _lost;

        public string Name => $"hexfarer-mcts-open-{(iterations > 0 ? $"{iterations}it" : $"{seconds}s")}";

        /// <summary>
        /// Answers the referee's messages until its input ends
        /// </summary>
        public void Run(TextReader input, TextWriter output)
        {
            while (input.ReadLine() is string line)
            {
                if (line.Length == 0) continue;
                JsonNode message = JsonNode.Parse(line)!;
                JsonNode? answer = Handle(message);
                if (answer != null)
                {
                    output.WriteLine(answer.ToJsonString());
                    output.Flush();
                }
            }
        }

        /// <summary>
        /// The answer to one message (hello and decide), or null
        /// </summary>
        public JsonNode? Handle(JsonNode message)
        {
            switch (message["type"]!.GetValue<string>())
            {
                case "hello":
                    return new JsonObject { ["type"] = "hello", ["protocol"] = "osp", ["version"] = 1, ["name"] = Name,
                        ["rulesets"] = new JsonArray("standard", "catanatron"), ["observe"] = true, ["needs"] = "truth" };
                case "before":
                    Before(message);
                    return null;
                case "step":
                    Step(message);
                    return null;
                case "decide":
                    return Decide(message);
                case "after":
                    log.WriteLine(new JsonObject { ["type"] = "stats", ["game_id"] = _gameId, ["seat"] = _seat, ["decisions"] = _decisions, ["searched"] = _searched,
                        ["patches"] = _patches, ["fallbacks"] = _fallbacks, ["lost"] = _lost }.ToJsonString());
                    log.Flush();
                    return null;
                default:
                    return null;
            }
        }

        private void Before(JsonNode before)
        {
            _gameId = before["game_id"]!.GetValue<string>();
            _seat = (sbyte)before["seat"]!.GetValue<int>();
            _replica = new Replica(before);
            _patches = _fallbacks = _decisions = _searched = 0;
            _lost = !_replica.BoardMatches(before);
            if (_lost) log.WriteLine($"{_gameId}: the board doesn't write back as sent");
            _agent = new MCTSAgent(_seat, _replica.State, timeLimitSeconds: seconds, valueNet: net, iterationLimit: iterations);
            // No trades between players under these rules
            _agent.Offers.Enabled = false;
        }

        private void Step(JsonNode step)
        {
            if (_replica == null || _lost) return;
            if (_replica.Follow(step) is (string kind, string detail))
            {
                // The copy can't follow (a rule difference that changes the board): it's no longer the referee's game
                _lost = true;
                log.WriteLine($"{_gameId}: lost the game: {kind}: {detail}");
                return;
            }
            _agent!.HandleAction(_replica.Applied!);
        }

        private JsonNode Decide(JsonNode decide)
        {
            _decisions++;
            JsonArray legal = decide["legal"]!.AsArray();
            HashSet<string> allowed = legal.Select(OspCodec.Canonical).ToHashSet();
            JsonNode Answer(JsonNode move) => new JsonObject { ["type"] = "decide", ["game_id"] = decide["game_id"]!.DeepClone(), ["n"] = decide["n"]!.DeepClone(), ["action"] = move.DeepClone() };

            if (!_lost && _replica!.FirstDifference(decide["state"]!, $"move {decide["n"]}") is (string kind, string detail))
            {
                // The referee decides: take over its state where the known differences changed it
                _patches++;
                if (!_replica.Patch(decide["state"]!))
                {
                    _lost = true;
                    log.WriteLine($"{_gameId}: lost the game: can't take over {kind}: {detail}");
                }
                else _agent!.ResetToState(_replica.State);
            }
            if (_lost || _replica!.Acting() != _seat) return Answer(legal[0]!);

            // A single legal move needs no search
            if (legal.Count == 1) return Answer(legal[0]!);

            Action chosen = _agent!.Act(_replica.State);
            _searched++;
            if (Allowed(chosen) is JsonNode move) return Answer(move);

            // The referee doesn't allow the move (its rules are narrower): the most visited root move it allows
            _fallbacks++;
            (_, IReadOnlyList<MCTSAgent.RootMove> roots) = _agent.GetRootMoves();
            foreach (MCTSAgent.RootMove root in roots.OrderByDescending(r => r.Visits))
                if (Allowed(root.Move) is JsonNode fallback) return Answer(fallback);
            return Answer(legal[0]!);

            // The protocol's move of one of Hexfarer's actions, as the referee's legal moves write it, or null if it isn't one
            JsonNode? Allowed(Action action)
            {
                string wanted = OspCodec.Canonical(_replica!.Codec.Move(action));
                foreach ((Action legalAction, JsonArray encoded) in _replica.Legal())
                {
                    // Same decision, as the folded legal moves write it (a robber move against a victim without cards steals from nobody)
                    if (OspCodec.Canonical(_replica.Codec.Move(legalAction)) != wanted) continue;
                    string canonical = OspCodec.Canonical(encoded);
                    if (allowed.Contains(canonical)) return encoded;
                }
                return null;
            }
        }
    }
}
