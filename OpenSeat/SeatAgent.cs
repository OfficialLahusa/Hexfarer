using Agents;
using Agents.Inference;
using Agents.MCTS;
using Agents.OpenSeat;
using Common;
using Common.Actions;
using Common.OSP;
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
        bool parallelSelection = true, bool fair = false, bool lean = false, int threads = 0)
    {
        private Replica? _replica;
        private MCTSAgent? _agent;
        // The fair agent around the search (null for the open-hand agent)
        private InformationSetMCTSAgent? _fair;
        // Times the fair agent's own copy of the game had drifted from the replica and was set back to it (should stay 0),
        // and the decisions a lean agent asked the referee's state for
        private int _resyncs, _resyncsAsked;
        // The copy's state at the last lean decision whose hash it didn't match, to name what differed once the state comes;
        // the diagnostic lines a game has written (at most 5: what differed at a resync, a move the copy couldn't follow)
        private JsonObject? _unmatched;
        private int _unmatchedAt = -1, _diagnosed;
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
        /// Answers the referee's messages until its input ends; a message it can't handle is answered with a failure
        /// (internal) for its game, and a line it can't read with a failure without a game (malformed), which ends the
        /// connection
        /// </summary>
        public void Run(TextReader input, TextWriter output)
        {
            void Write(JsonNode answer)
            {
                output.WriteLine(answer.ToJsonString());
                output.Flush();
            }
            while (input.ReadLine() is string line)
            {
                if (line.Length == 0) continue;
                if (OpenSeatAgentProcess.AnswerOf(line) is not JsonObject message)
                {
                    log.WriteLine($"{_gameId}: can't read the referee's line: {line}");
                    Write(OpenSeatFailure.Message(null, null, OpenSeatFailure.Malformed, $"This agent can't read the line {line[..Math.Min(line.Length, 200)]}"));
                    return;
                }
                JsonNode? answer;
                try
                {
                    answer = Handle(message);
                }
                catch (Exception e)
                {
                    log.WriteLine($"{_gameId}: failed: {e}");
                    answer = Failure(message, OpenSeatFailure.Internal, e.Message);
                    _lost = true;
                }
                if (answer != null) Write(answer);
            }
        }

        // The failure for a message's game and decision
        private static JsonObject Failure(JsonNode message, string code, string why, JsonNode? detail = null) =>
            OpenSeatFailure.Message(OpenSeatHandshake.Text(message["game_id"]), message["n"] is JsonValue n && n.TryGetValue(out int at) ? at : null, code, why, detail);

        /// <summary>
        /// What of a game's budget this agent can't keep, as a failure's detail, or null: no GPU while its net runs on one,
        /// or fewer CPU threads than it uses (the net's threads, all cores when not set, or the search's: one when serial,
        /// all cores otherwise); cores is the machine's (Environment.ProcessorCount if not given)
        /// </summary>
        internal static JsonObject? UnkeptBudget(JsonNode? budget, InferenceProvider provider, int threads, bool parallelSelection, int cores = 0)
        {
            if (budget?["gpu"] is JsonValue gpu && gpu.TryGetValue(out bool allowed) && !allowed && provider != InferenceProvider.CPU)
                return new JsonObject { ["budget"] = "gpu", ["value"] = false };
            if (cores <= 0) cores = Environment.ProcessorCount;
            int used = Math.Max(parallelSelection ? cores : 1, threads > 0 ? threads : cores);
            if (budget?["threads"] is JsonValue limit && limit.TryGetValue(out int most) && used > most)
                return new JsonObject { ["budget"] = "threads", ["value"] = most };
            return null;
        }

        /// <summary>
        /// The answer to one message (handshake, before and decide), or null
        /// </summary>
        public JsonNode? Handle(JsonNode message)
        {
            switch (message["type"]!.GetValue<string>())
            {
                case "handshake":
                    if (message["protocol"]?.ToJsonString() != "\"osp\"" || message["version"]?.ToJsonString() != "2")
                        return OpenSeatFailure.Message(null, null, OpenSeatFailure.Version, $"This agent speaks protocol osp version 2, not {message.ToJsonString()}.");
                    return new JsonObject { ["type"] = "handshake", ["protocol"] = "osp", ["version"] = 2, ["name"] = Name,
                        ["rulesets"] = new JsonArray("standard", "catanatron"), ["observe"] = true, ["needs"] = "truth", ["lean"] = lean };
                case "before":
                    // A ruleset the copy can't play, or a budget the agent can't keep, is turned down
                    if (Replica.Unsupported(message["ruleset"]) is JsonObject unsupported)
                        return Failure(message, OpenSeatFailure.Unsupported, $"This agent can't play {unsupported.ToJsonString()}.", unsupported);
                    if (UnkeptBudget(message["budget"], net.Provider, threads, parallelSelection) is JsonObject unkept)
                        return Failure(message, OpenSeatFailure.Unsupported, $"This agent can't keep the budget {unkept.ToJsonString()}.", unkept);
                    Before(message);
                    return new JsonObject { ["type"] = "ready", ["game_id"] = message["game_id"]!.DeepClone() };
                case "failure":
                    // The game is over for this agent: it plays no more of it
                    log.WriteLine($"{_gameId}: the referee reported a failure: {message.ToJsonString()}");
                    _lost = true;
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
                        ["fair"] = fair, ["resyncs"] = _resyncs, ["resyncs_asked"] = _resyncsAsked,
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
            _resyncs = _resyncsAsked = 0;
            _unmatched = null;
            _unmatchedAt = -1;
            _diagnosed = 0;
            _stale = false;
            _lost = !_replica.BoardMatches(before);
            if (_lost) log.WriteLine($"{_gameId}: the board doesn't write back as sent");
            // The game's seed seeds the agent's randomness, its search included: with an iteration budget and selection on one
            // thread (--serial) it plays the same game the same way; on all cores or with a time budget the timing decides too
            int? gameSeed = before["seed"] is JsonValue seed && seed.TryGetValue(out int value) ? value : null;
            if (gameSeed is int seeded) Utils.Random.SeedCurrentThread(seeded);
            // A time-based search keeps within the budget's time per decision, with a fifth of it to spare
            double budgetSeconds = before["budget"]?["decide_ms"] is JsonValue ms && ms.TryGetValue(out int decideMs) ? decideMs * 0.8 / 1000 : double.MaxValue;
            _agent = new MCTSAgent(_seat, _replica.State, timeLimitSeconds: seconds > 0 ? Math.Min(seconds, budgetSeconds) : seconds, valueNet: net,
                iterationLimit: iterations, leavesPerBatch: leavesPerBatch, parallelSelection: parallelSelection, seed: gameSeed);
            // The fair agent's worlds and trees draw from a stream of their own, also from the game's seed
            _fair = fair ? new InformationSetMCTSAgent(_agent, _replica.State, seed: gameSeed is int fairSeed ? fairSeed ^ 0x2545F491 : null) : null;
            // Offers and counteroffers only where a decision names them (in trades, since their cards can be any)
            _agent.Offers.Enabled = false;
            _agent.Offers.Counters = false;
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
                else
                {
                    _stale = true;
                    if (_diagnosed++ < 5) log.WriteLine($"{_gameId}: can't follow move {step["n"]} ({kind}), rebuilds at the next decision: {detail}");
                }
                return;
            }
            Player.HandleAction(_replica.Applied!);
        }

        /// <summary>
        /// The fair agent (for tests), null for the open-hand agent
        /// </summary>
        internal InformationSetMCTSAgent? Fair => _fair;

        /// <summary>
        /// The search of the game (for tests)
        /// </summary>
        internal MCTSAgent? Search => _agent;

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
            // A lean decision carries only the state's hash: the copy has to be the referee's game, else the agent asks for
            // the state (resync), and the referee sends the decision again with it
            int n = decide["n"] is JsonValue at && at.TryGetValue(out int number) ? number : -1;
            if (decide["state"] == null)
            {
                JsonObject? mine = _replica == null || _lost || _stale ? null : _replica.Codec.State(_replica.State);
                if (mine == null || OSPCodec.Hash(mine) != decide["hash"]?.GetValue<string>())
                {
                    _resyncsAsked++;
                    (_unmatched, _unmatchedAt) = (mine, n);
                    return new JsonObject { ["type"] = "resync", ["game_id"] = decide["game_id"]!.DeepClone(), ["n"] = decide["n"]!.DeepClone() };
                }
                return Choose(decide);
            }
            // The state after a resync: what the copy had differently goes to the log (the first few of a game)
            if (_unmatched != null && n == _unmatchedAt && _diagnosed++ < 5)
                log.WriteLine($"{_gameId}: resync at move {n}: {string.Join("; ", Differences(_unmatched, decide["state"]!, "", 4))}");
            _unmatched = null;
            return Choose(decide);
        }

        /// <summary>
        /// Which trades a decision names (see PROTOCOL.md): an offer (with offers left this turn, or no limit), a
        /// counteroffer
        /// </summary>
        internal static (bool offers, bool counters) TradesNamed(JsonNode? trades)
        {
            bool offers = trades?["offer"] is JsonObject offer
                && (offer["left"] is not JsonValue left || !left.TryGetValue(out int count) || count > 0);
            return (offers, trades?["counter"] is JsonObject);
        }

        /// <summary>
        /// Whether a trade of the agent's is one the decision names: an offer only to seats it named, a counteroffer
        /// </summary>
        internal static bool Proposable(Action trade, JsonNode? trades) => trade switch
        {
            OfferTradeAction offer => TradesNamed(trades).offers && trades!["offer"]!["to"] is JsonArray to && (offer.Recipients & ~SeatsOf(to)) == 0,
            CounterTradeAction => TradesNamed(trades).counters,
            _ => false,
        };

        // The seats a decision's trades name, as a bit set (anything that isn't a seat number left out)
        private static int SeatsOf(JsonArray to) =>
            to.Aggregate(0, (seats, seat) => seat is JsonValue value && value.TryGetValue(out int number) && number is >= 0 and < 8 ? seats | 1 << number : seats);

        /// <summary>
        /// Where two states differ, as paths with both values (ours, then the referee's), at most max of them
        /// </summary>
        internal static List<string> Differences(JsonNode? ours, JsonNode? theirs, string path, int max)
        {
            List<string> found = [];
            void Walk(JsonNode? a, JsonNode? b, string at)
            {
                if (found.Count >= max) return;
                if (a is JsonObject x && b is JsonObject y)
                {
                    // A key one side leaves out differs from a null on the other (the hash tells them apart)
                    foreach (string key in x.Select(p => p.Key).Union(y.Select(p => p.Key)).Order(StringComparer.Ordinal))
                        if (found.Count >= max) return;
                        else if (x.ContainsKey(key) != y.ContainsKey(key))
                            found.Add($"{at}.{key} {(x.ContainsKey(key) ? Shorten(OSPCodec.Canonical(x[key])) : "absent")} vs {(y.ContainsKey(key) ? Shorten(OSPCodec.Canonical(y[key])) : "absent")}");
                        else Walk(x[key], y[key], $"{at}.{key}");
                    return;
                }
                if (a is JsonArray u && b is JsonArray v && u.Count == v.Count)
                {
                    for (int i = 0; i < u.Count; i++) Walk(u[i], v[i], $"{at}[{i}]");
                    return;
                }
                string mine = OSPCodec.Canonical(a), theirs = OSPCodec.Canonical(b);
                if (mine != theirs) found.Add($"{(at.Length > 0 ? at : ".")} {Shorten(mine)} vs {Shorten(theirs)}");
            }
            static string Shorten(string text) => text.Length > 120 ? text[..120] + "..." : text;
            Walk(ours, theirs, path);
            return found;
        }

        private JsonNode Choose(JsonNode decide)
        {
            _decisions++;
            JsonArray legal = decide["legal"]!.AsArray();
            HashSet<string> allowed = legal.Select(OSPCodec.Canonical).ToHashSet();
            JsonNode Answer(JsonNode move) => new JsonObject { ["type"] = "decide", ["game_id"] = decide["game_id"]!.DeepClone(), ["n"] = decide["n"]!.DeepClone(), ["action"] = move.DeepClone() };

            if (!_lost && decide["state"] is JsonNode referees && (_stale || _replica!.FirstDifference(referees, $"move {decide["n"]}") is not null))
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

            // A single legal move needs no search, unless the decision names trades the agent may propose instead
            JsonNode? trades = decide["trades"];
            if (legal.Count == 1 && trades == null) return Answer(legal[0]!);
            // A copy without a legal move (it would have to be lost) can't search
            if (!_replica.Legal().Any())
            {
                _lost = true;
                log.WriteLine($"{_gameId}: lost the game: no legal move in Hexfarer's copy at move {decide["n"]}");
                return Answer(legal[0]!);
            }

            // The fair agent searches worlds drawn from its beliefs, not the true state it is given
            SearchStatistics? previous = _agent!.LastSearch;
            (_agent.Offers.Enabled, _agent.Offers.Counters) = TradesNamed(trades);
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

            // The protocol's move of one of Hexfarer's actions, as the referee's legal moves write it (or a trade the decision
            // named, with its terms), or null if it isn't one
            JsonNode? Allowed(Action action)
            {
                if (action is OfferTradeAction or CounterTradeAction)
                    return Proposable(action, trades) ? _replica!.Codec.Move(action) : null;
                string wanted = OSPCodec.Canonical(_replica!.Codec.Move(action));
                foreach ((Action legalAction, JsonArray encoded) in _replica.Legal())
                {
                    // Same decision, as the folded legal moves write it (a robber move against a victim without cards steals from nobody)
                    if (OSPCodec.Canonical(_replica.Codec.Move(legalAction)) != wanted) continue;
                    string canonical = OSPCodec.Canonical(encoded);
                    if (allowed.Contains(canonical)) return encoded;
                }
                return null;
            }
        }
    }
}
