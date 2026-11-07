using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
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
    /// allows. Where the copy can't follow (no steps came, as from JSettlers' robot, or a rule difference changed the board),
    /// the open-hand agent rebuilds it from the decision's state (Replica.Rebuild); the fair agent can't without losing
    /// what its seat knows, and plays the referee's first legal move from then on. It needs the full state ("truth"): every hand is the true one, as for the open-hand MCTS in Hexfarer. The
    /// fair agent (fair: true) gets the same messages but plays like Hexfarer's fair agents in the Arena: Information Set
    /// MCTS over worlds drawn from what its seat knows (BeliefTracker), which takes in every move as the seat sees it.
    /// </summary>
    internal sealed class SeatAgent(StateValueNet net, long iterations, double seconds, TextWriter log, int leavesPerBatch = MCTSAgent.DefaultLeavesPerBatch,
        bool parallelSelection = true, bool fair = false)
    {
        private Replica? _replica;
        private MCTSAgent? _agent;
        // The fair agent around the search (null for the open-hand agent)
        private InformationSetMCTSAgent? _fair;
        // Times the fair agent's own copy of the game had drifted from the replica and was set back to it (should stay 0)
        private int _resyncs;
        private string _gameId = "";
        private sbyte _seat;
        // Per game: decisions, those searched, patched states, replaced moves, whether the copy was lost (it then plays the
        // referee's first legal move)
        private int _patches, _fallbacks, _decisions, _searched, _rebuilds;
        private bool _lost;
        // The open-hand agent's copy missed a move: rebuilt from the next decision's state
        private bool _stale;
        // Per game, the search's work: searches that ran (a decision with an immediate win may skip it), leaf evaluations
        // (whole batches) and the distinct positions among them, for comparisons with other engines at equal work
        private long _searches, _iterations, _distinctLeaves;
        private double _searchSeconds;
        // The process's CPU time at the game's start (every thread: the search and the value net's), for comparisons with
        // other engines at equal compute
        private TimeSpan _cpuAtStart;

        public string Name => $"hexfarer-{(fair ? "ismcts-fair" : "mcts-open")}-{(iterations > 0 ? $"{iterations}it" : $"{seconds.ToString(System.Globalization.CultureInfo.InvariantCulture)}s")}"
            + (leavesPerBatch != MCTSAgent.DefaultLeavesPerBatch ? $"-b{leavesPerBatch}" : "") + (parallelSelection ? "" : "-serial");

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
                        ["patches"] = _patches, ["fallbacks"] = _fallbacks, ["rebuilds"] = _rebuilds, ["lost"] = _lost,
                        ["searches"] = _searches, ["iterations"] = _iterations, ["distinct_leaves"] = _distinctLeaves, ["search_seconds"] = Math.Round(_searchSeconds, 3),
                        ["fair"] = fair, ["resyncs"] = _resyncs,
                        ["cpu_seconds"] = Math.Round((System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime - _cpuAtStart).TotalSeconds, 3) }.ToJsonString());
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
            _patches = _fallbacks = _decisions = _searched = _rebuilds = 0;
            _searches = _iterations = _distinctLeaves = 0;
            _searchSeconds = 0;
            _cpuAtStart = System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime;
            _resyncs = 0;
            _stale = false;
            _lost = !_replica.BoardMatches(before);
            if (_lost) log.WriteLine($"{_gameId}: the board doesn't write back as sent");
            _agent = new MCTSAgent(_seat, _replica.State, timeLimitSeconds: seconds, valueNet: net, iterationLimit: iterations, leavesPerBatch: leavesPerBatch,
                parallelSelection: parallelSelection);
            _fair = fair ? new InformationSetMCTSAgent(_agent, _replica.State) : null;
            // No trades between players under these rules
            _agent.Offers.Enabled = false;
        }

        private void Step(JsonNode step)
        {
            if (_replica == null || _lost || _stale) return;
            // With the full state the step carries the referee's state before the move: the copy takes over where the
            // rules differ, so it can follow moves that depend on it (a discard of cards Hexfarer's bank wouldn't have
            // given, any move after Hexfarer's rules ended the game)
            if (step["state"] is JsonNode before && _replica.FirstDifference(before, $"move {step["n"]}") is (string differs, string why))
            {
                _patches++;
                GameState untouched = new(_replica.State);
                if (!_replica.Patch(before))
                {
                    // The open-hand agent waits for the next decision's state to rebuild from
                    if (fair) _lost = true;
                    else _stale = true;
                    if (fair) log.WriteLine($"{_gameId}: lost the game: can't take over {differs}: {why}");
                    return;
                }
                TookOver(untouched);
            }
            // The fair agent's copy must be the replica's position before the move, or its beliefs would follow another game
            if (_fair != null && !_fair.Mirror.Equals(_replica.State))
            {
                _resyncs++;
                _fair.ResetTo(_replica.State, _fair.Beliefs);
            }
            if (_replica.Follow(step) is (string kind, string detail))
            {
                // The copy can't follow (a rule difference that changes the board): it's no longer the referee's game; the
                // open-hand agent rebuilds it at its next decision
                if (fair)
                {
                    _lost = true;
                    log.WriteLine($"{_gameId}: lost the game: {kind}: {detail}");
                }
                else _stale = true;
                return;
            }
            Player.HandleAction(_replica.Applied!);
        }

        /// <summary>
        /// The fair agent (for tests), null for the open-hand agent
        /// </summary>
        internal InformationSetMCTSAgent? Fair => _fair;

        // The agent that plays the seat: the fair one around the search, or the open-hand search itself
        private Agent Player => (Agent?)_fair ?? _agent!;

        // After the replica took over the referee's state: the open-hand search starts from it; the fair agent takes the
        // change in as an edit everyone sees (a bank, hands after a payout by the referee's rules, the Longest Road holder,
        // a game going on), so its beliefs keep what it doesn't know (a reset would make every hand known)
        private void TookOver(GameState untouched)
        {
            if (_fair == null)
            {
                _agent!.ResetToState(_replica!.State);
                return;
            }
            BeliefTracker beliefs = new(_fair.Beliefs);
            beliefs.Observe(untouched, null, _replica!.State);
            _fair.ResetTo(_replica.State, beliefs);
        }

        private JsonNode Decide(JsonNode decide)
        {
            _decisions++;
            JsonArray legal = decide["legal"]!.AsArray();
            HashSet<string> allowed = legal.Select(OspCodec.Canonical).ToHashSet();
            JsonNode Answer(JsonNode move) => new JsonObject { ["type"] = "decide", ["game_id"] = decide["game_id"]!.DeepClone(), ["n"] = decide["n"]!.DeepClone(), ["action"] = move.DeepClone() };

            if (!_lost && (_stale || _replica!.FirstDifference(decide["state"]!, $"move {decide["n"]}") is not null))
            {
                // The referee decides: take over its state where the known differences changed it, or rebuild from it
                (string kind, string detail)? difference = _replica!.FirstDifference(decide["state"]!, $"move {decide["n"]}");
                GameState untouched = new(_replica.State);
                if (difference != null) _patches++;
                if (_stale || difference != null && !_replica.Patch(decide["state"]!))
                {
                    if (fair)
                    {
                        _lost = true;
                        log.WriteLine($"{_gameId}: lost the game: can't take over {difference?.kind}: {difference?.detail}");
                    }
                    else
                    {
                        _replica.Rebuild(decide["state"]!, legal);
                        _rebuilds++;
                        _stale = false;
                        _agent!.ResetToState(_replica.State);
                    }
                }
                else if (difference != null) TookOver(untouched);
            }
            if (_lost || _replica!.Acting() != _seat) return Answer(legal[0]!);

            // A single legal move needs no search
            if (legal.Count == 1) return Answer(legal[0]!);
            // A copy without a legal move (it would have to be lost) can't search
            if (!_replica.Legal().Any())
            {
                _lost = true;
                log.WriteLine($"{_gameId}: lost the game: no legal move in Hexfarer's copy at move {decide["n"]}");
                return Answer(legal[0]!);
            }

            // The fair agent searches worlds drawn from its beliefs, not the true state it is given
            SearchStatistics? previous = _agent!.LastSearch;
            Action chosen = Player.Act(_replica.State);
            _searched++;
            // A new statistics object means a search ran for this decision (the last one stays when Act needs none)
            if (_agent.LastSearch is SearchStatistics search && !ReferenceEquals(search, previous))
            {
                _searches++;
                _iterations += search.Iterations;
                _distinctLeaves += search.DistinctLeaves;
                _searchSeconds += search.Seconds;
            }
            if (Allowed(chosen) is JsonNode move) return Answer(move);

            // The referee doesn't allow the move (its rules are narrower): the most visited root move it allows
            _fallbacks++;
            (_, IReadOnlyList<MCTSAgent.RootMove> roots) = _fair?.GetRootMoves() ?? _agent.GetRootMoves();
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
