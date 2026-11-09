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
    /// allows. Where the copy can't follow (no steps came, as from JSettlers' robot, a step went missing, or a rule
    /// difference changed the board), it waits for the next decision and is rebuilt from its state (Replica.Rebuild); the
    /// fair agent's beliefs take the change since the state the seat last saw in as an edit everyone sees, so they keep what
    /// the seat doesn't know. Only a copy that can't become the referee's state even so is lost: the agent then answers its
    /// next decision with a failure (internal), so the referee records it, instead of playing on blind. A board it can't
    /// hold it turns down. It needs the full state ("truth"): every hand is the true one, as for the open-hand MCTS in
    /// Hexfarer. The fair agent (fair: true) gets the same messages but plays like Hexfarer's fair agents in the Arena:
    /// Information Set MCTS over worlds drawn from what its seat knows (BeliefTracker), which takes in every move as the seat
    /// sees it.
    /// </summary>
    internal sealed class SeatAgent(StateValueNet net, long iterations, double seconds, TextWriter log, int? leavesPerBatch = null,
        bool parallelSelection = true, bool fair = false, bool lean = false, int threads = 0, bool declineTrades = false)
    {
        /// <summary>
        /// Leaves per batch of the search: as given, else as in the Arena, MCTSAgent.LeavesFor an iteration budget (100 tree
        /// updates per decision; 5000 iterations: 32) and MCTSAgent.DefaultLeavesPerBatch for a time budget
        /// </summary>
        public int LeavesPerBatch { get; } = leavesPerBatch ?? (iterations > 0 ? MCTSAgent.LeavesFor(iterations) : MCTSAgent.DefaultLeavesPerBatch);

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
        // Per game: decisions, those searched, patched states, replaced moves, rebuilt copies, and whether the copy was lost
        // (it can't become the referee's state; the agent then answers with a failure) and why
        private int _patches, _fallbacks, _decisions, _searched, _rebuilds;
        private bool _lost;
        private string? _lostWhy;
        // The copy missed a move or can't take the referee's state over: rebuilt from the next decision's state. Until then
        // it follows no step, and _staleFrom holds the state the seat last saw (the fair agent's beliefs take the change
        // from there in as an edit)
        private bool _stale;
        private GameState? _staleFrom;
        // Per game, the search's work: searches that ran (a decision with an immediate win may skip it), leaf evaluations
        // (whole batches) and the distinct positions among them, for comparisons with other engines at equal work
        private long _searches, _iterations, _distinctLeaves;
        private double _searchSeconds;
        // The process's CPU time at the game's start (every thread: the search and the value net's), for comparisons with
        // other engines at equal compute
        private TimeSpan _cpuAtStart;

        public string Name => $"hexfarer-{(fair ? "ismcts-fair" : "mcts-open")}-{(iterations > 0 ? $"{iterations}it" : $"{seconds.ToString(System.Globalization.CultureInfo.InvariantCulture)}s")}"
            + (LeavesPerBatch != MCTSAgent.DefaultLeavesPerBatch ? $"-b{LeavesPerBatch}" : "") + (parallelSelection ? "" : "-serial");

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
                    _lostWhy ??= $"failed: {e.Message}";
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
                    // A board the copy doesn't write back as sent isn't the referee's: the agent couldn't follow the game
                    if (!_replica!.BoardMatches(message))
                    {
                        _lost = true;
                        _lostWhy = "the board doesn't write back as sent";
                        log.WriteLine($"{_gameId}: the board doesn't write back as sent; game turned down");
                        return Failure(message, OpenSeatFailure.Unsupported, "This agent can't hold the board as sent: it writes it back differently.");
                    }
                    return new JsonObject { ["type"] = "ready", ["game_id"] = message["game_id"]!.DeepClone() };
                case "failure":
                    // The game is over for this agent: it plays no more of it
                    log.WriteLine($"{_gameId}: the referee reported a failure: {message.ToJsonString()}");
                    _lost = true;
                    _lostWhy ??= $"the referee reported a failure ({OpenSeatHandshake.Text(message["code"])})";
                    return null;
                case "step":
                    Step(message);
                    return null;
                case "decide":
                    return Decide(message);
                case "after":
                    log.WriteLine(new JsonObject { ["type"] = "stats", ["game_id"] = _gameId, ["seat"] = _seat, ["decisions"] = _decisions, ["searched"] = _searched,
                        ["patches"] = _patches, ["fallbacks"] = _fallbacks, ["rebuilds"] = _rebuilds, ["lost"] = _lost, ["lost_why"] = _lostWhy,
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
            _staleFrom = null;
            _lost = false;
            _lostWhy = null;
            // The game's seed seeds the agent's randomness, its search included. SeedCurrentThread covers only what this thread
            // draws from Utils.Random; the search runs on pool threads and draws from its trees' sources, so it gets the seed
            // itself (without it the trees draw from Random.Shared, which the seed doesn't reach)
            // A game replays exactly only where nothing else decides: an iteration budget (a time budget depends on the speed),
            // selection on one thread (--serial; on all cores the threads' order decides and the search ignores the seed), the
            // same build, net file and execution provider (their float results can differ in the last digits), and a referee
            // whose own game repeats (JSettlers rolls its own dice, so there the seed fixes only this side)
            // The same seed plays the same game the same way, so the variety between games comes from the referee's seeds
            int? gameSeed = before["seed"] is JsonValue seed && seed.TryGetValue(out int value) ? value : null;
            if (gameSeed is int seeded) Utils.Random.SeedCurrentThread(seeded);
            // A time-based search keeps within the budget's time per decision, with a fifth of it to spare; an iteration budget
            // doesn't watch it, so for a replay the referee has to allow the time
            double budgetSeconds = before["budget"]?["decide_ms"] is JsonValue ms && ms.TryGetValue(out int decideMs) ? decideMs * 0.8 / 1000 : double.MaxValue;
            _agent = new MCTSAgent(_seat, _replica.State, timeLimitSeconds: seconds > 0 ? Math.Min(seconds, budgetSeconds) : seconds, valueNet: net,
                iterationLimit: iterations, leavesPerBatch: LeavesPerBatch, parallelSelection: parallelSelection, seed: gameSeed);
            // The fair agent's worlds and trees draw from a stream of their own, also from the game's seed: mixed, since two
            // sources from one seed draw the same numbers (its worlds would follow the MCTS agent's tree seeds)
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
                    // The copy waits for the next decision's state to rebuild from, from the state the seat last saw (the
                    // failed takeover may have changed some of it, which the beliefs never took in)
                    Stale(untouched, $"can't take over {differs} at move {step["n"]}, rebuilds at the next decision: {why}");
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
                // The copy can't follow (a rule difference that changes the board): it's no longer the referee's game until it
                // is rebuilt at the next decision (a failed follow leaves the copy as it was)
                Stale(new GameState(_replica.State), $"can't follow move {step["n"]} ({kind}), rebuilds at the next decision: {detail}");
                return;
            }
            Player.HandleAction(_replica.Applied!);
        }

        // The copy follows no more steps until the next decision rebuilds it; seen is the state the seat last saw
        private void Stale(GameState seen, string why)
        {
            _stale = true;
            _staleFrom = seen;
            if (_diagnosed++ < 5) log.WriteLine($"{_gameId}: {why}");
        }

        // The copy can't become the referee's game: the agent answers its next decision with a failure
        private void Lose(string why)
        {
            _lost = true;
            _lostWhy ??= why;
            log.WriteLine($"{_gameId}: lost the game: {why}");
        }

        // The answer to a decision once the game is lost: a failure the referee records, with what went wrong
        private JsonObject LostFailure(JsonNode decide) =>
            Failure(decide, OpenSeatFailure.Internal, $"Hexfarer's copy of the game is lost: {_lostWhy}");

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

        // After the replica took over the referee's state or was rebuilt from it: the open-hand search starts from it; the fair
        // agent takes the change since untouched (the state its seat last saw) in as an edit everyone sees (a bank, hands
        // after a payout by the referee's rules, the Longest Road holder, a game going on, the moves of a gap), so its beliefs
        // keep what it doesn't know (a reset would make every hand known); only a steal between two others inside a gap shows
        // its card's type that way
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
            // A lost game gets no move from the agent: the referee is told with a failure and records it
            if (_lost) return LostFailure(decide);
            if (decide["state"] == null)
            {
                JsonObject? mine = _replica == null || _stale ? null : _replica.Codec.State(_replica.State);
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

        // Whether the seat has a decision in the state: it is the one to act, or it may answer the open offer or pick on it now
        private static bool MayDecide(GameState state, sbyte seat) =>
            Enumerable.Range(0, state.Players.Length).FirstOrDefault(state.CanPlayerAct, -1) == seat
            || RespondToTradeAction.MayAnswer(state, seat) || ConfirmTradeAction.MayPick(state, seat);

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

            if (decide["state"] is JsonNode referees && (_stale || _replica!.FirstDifference(referees, $"move {decide["n"]}") is not null))
            {
                // The referee decides: take over its state where the known differences changed it, or rebuild from it
                (string kind, string detail)? difference = _replica!.FirstDifference(referees, $"move {decide["n"]}");
                // What the seat saw last: the copy now, or where it stopped following
                GameState untouched = _staleFrom ?? new(_replica.State);
                if (difference != null) _patches++;
                if (_stale || difference != null && !_replica.Patch(referees))
                {
                    _replica.Rebuild(referees, legal);
                    _rebuilds++;
                    _stale = false;
                    _staleFrom = null;
                    // A copy that isn't the referee's state even rebuilt can't play the game
                    if (_replica.FirstDifference(referees, $"move {decide["n"]}") is (string kind, string detail))
                    {
                        Lose($"rebuilt from the referee's state, the copy still differs in {kind}: {detail}");
                        return LostFailure(decide);
                    }
                    if (_diagnosed++ < 5) log.WriteLine($"{_gameId}: rebuilt the copy at move {decide["n"]}{(difference is (string what, _) ? $" ({what})" : "")}");
                    TookOver(untouched);
                }
                else if (difference != null) TookOver(untouched);
            }
            // The copy has to say the seat decides: it acts in seat order, or (under JSettlers' flow) answers an offer or picks on
            // its own before the others are through; where it doesn't, the referee's first legal move stands in (counted)
            if (!MayDecide(_replica!.State, _seat))
            {
                _fallbacks++;
                return Answer(legal[0]!);
            }

            // A single legal move needs no search, unless the decision names trades the agent may propose instead
            JsonNode? trades = decide["trades"];
            // Told not to trade (a comparison's seat without trades at a table with them): every offer declined, a standing
            // offer of its own withdrawn, none made
            if (declineTrades)
            {
                string decline = OSPCodec.Canonical(_replica.Codec.Move(new RespondToTradeAction(_seat, false)));
                string withdraw = OSPCodec.Canonical(_replica.Codec.Move(new CancelTradeAction(_seat)));
                if (legal.FirstOrDefault(m => OSPCodec.Canonical(m) is string move && (move == decline || move == withdraw)) is JsonNode no) return Answer(no);
                trades = null;
            }
            if (legal.Count == 1 && trades == null) return Answer(legal[0]!);
            // A copy without a legal move (it would have to be lost) can't search
            if (!_replica.Legal().Any())
            {
                Lose($"no legal move in Hexfarer's copy at move {decide["n"]}");
                return LostFailure(decide);
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
                // An answer, pick or withdrawal names no seat, and under JSettlers' flow the seat may make it before its turn in
                // seat order, where the copy's legal list is another seat's: checked by the rules for this seat instead
                if (action is RespondToTradeAction or ConfirmTradeAction or CancelTradeAction)
                {
                    JsonArray answer = _replica!.Codec.Move(action);
                    return action.IsValidFor(_replica.State) && allowed.Contains(OSPCodec.Canonical(answer)) ? answer : null;
                }
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
