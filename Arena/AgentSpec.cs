using Agents;
using Agents.Inference;
using Agents.MCTS;
using Agents.OpenSeat;
using Common;
using System.Globalization;
using System.Text.Json.Nodes;

namespace Arena
{
    /// <summary>
    /// An MCTS agent of a duel, given as "NAME=MODEL.onnx[=OPTION...]". Options: an opponent model (MaxN, Paranoid, Random;
    /// MaxN if none), cX exploration constant, bN leaves per batch (default from the iteration budget, 16 to 256, or 256
    /// with a time budget, see BatchFor), fX first play reduction, pX placement weight, u PUCT
    /// with uniform priors instead of the policy head's, ptX the priors at temperature X (softmax of the logits / X, above 0;
    /// default 1, see MCTSSearch.PriorTemperature), fqX plays the top of the improved policy at scale X after each search
    /// instead of the most visited move (above 0, open-hand MCTS only, see MCTSTree.FinalMoveScale), dN hidden information: only its seat's view, searched as N
    /// determinizations (PIMC, see DeterminizedMCTSAgent), is hidden information searched as one tree over the seat's
    /// information sets (see InformationSetMCTSAgent), isfull the same search on the true position (an ablation); without
    /// these the agent sees every hand. Ablations of the search: norm or nonorm selects with or without values scaled to
    /// the range seen in the tree (MCTS normalizes, ISMCTS not), ob lets ISMCTS branch on outcomes its seat can't see,
    /// noreuse starts MCTS and ISMCTS on a new tree every decision, noleaf lets MCTS evaluate one of a new leaf's
    /// moves instead of the leaf itself (the old search; leaf is the default), searchforced lets MCTS and PIMC search a
    /// decision with a single legal move instead of playing it at once (the old search), gX scales the statistics of the
    /// subtree ISMCTS goes on with by X (above 0, at most 1; 1, full weight, if not given), suX gives the initial placements
    /// X times the search budget of the other decisions (default 1, see MCTSAgent.SetupBudgetFactor), itX searches X times the duel's
    /// budget per decision (iterations or seconds, above 0; default 1), e.g. to measure what more search is worth. Trade offers (with free trade, see
    /// TradeOffers): nooffers makes neither offers nor counteroffers, omX needs a mean gain above X (default 0.005), onN makes
    /// at most N per turn (default 3), owN judges them over N worlds (default 8), oc2 keeps to one or two cards a side (oc3,
    /// any trade up to three for three, is the default), with oc2 o22 looks at two cards for two too and o31 at three for
    /// one and one for three, opN judges all in one world and only the N best there in all (default 16, op0 judges all in
    /// all); pfX expects partners to need X times its gain (default 0, any gain); nolearn (ISMCTS) ignores the cards others' offers
    /// and answers show. Answers: afX accepts an offer only if it gains at least X times what the offerer gains (default 1,
    /// af0 any gain above the margin), amX accepts an offer from a gain above X (default 0.002), noco answers
    /// an offer it turns down without a counteroffer (co with one, the default).
    /// </summary>
    internal sealed record AgentSpec(string Name, string ModelPath, OpponentModel OpponentModel, double Exploration, int? LeavesPerBatch,
        double FirstPlayReduction, double PlacementWeight, bool UniformPriors, int Determinizations = 0, bool InformationSets = false,
        bool FullKnowledge = false, bool? NormalizeValues = null, bool BranchHiddenOutcomes = false, bool ReuseTree = true,
        bool EvaluateNewLeaves = true, bool PlayForcedMovesAtOnce = true, double ReuseDecay = 1, bool MakeOffers = true,
        double OfferMargin = 0.005, int MaxOffers = 3, int OfferWorlds = 8, bool TwoForTwo = false, double AnswerMargin = 0.002, bool Counters = true,
        bool ThreeForOne = false, double AnswerFairness = 1, bool LearnsFromTrades = true, int MaxCards = 3, int Prune = 16, double PartnerFairness = 0,
        double SetupBudget = 1, string? Command = null, double PriorTemperature = 1, double BudgetFactor = 1, double FinalMoveScale = 0)
    {
        /// <summary>
        /// Whether an Open Seat Agent plays the side: a process started from Command (see OpenSeatAgentProxy), given as
        /// "NAME=command:COMMAND LINE"
        /// </summary>
        public bool IsOpenSeatAgent => Command != null;

        /// <summary>
        /// The time budget of an Open Seat Agent's games in a duel: 5 minutes per decision (OpenSeatBudget.Default)
        /// </summary>
        public static readonly OpenSeatBudget OpenSeatAgentBudget = OpenSeatBudget.Default;

        /// <summary>
        /// The leaves per batch of the search: bN if given, else MCTSAgent.LeavesFor its iterations with an iteration budget
        /// (PIMC's determinizations share the budget, so each searches a tree of its share) and MCTSAgent.DefaultLeavesPerBatch
        /// with a time budget (where the throughput of large batches counts: 32 against 256 gained nothing at 0.1 s, on
        /// DirectML and on the CPU), the same on every machine
        /// </summary>
        public int BatchFor(long iterations) => LeavesPerBatch
            ?? (iterations > 0 ? MCTSAgent.LeavesFor(Budget(iterations) / Math.Max(1, Determinizations)) : MCTSAgent.DefaultLeavesPerBatch);

        /// <summary>
        /// The iterations per decision this side searches in a duel of the given iterations: BudgetFactor times them,
        /// rounded (0, a time budget, stays 0)
        /// </summary>
        public long Budget(long iterations) => (long)Math.Round(iterations * BudgetFactor);

        /// <summary>
        /// Leaf rows per net file that the games played at once keep in flight by default: throughput follows games times
        /// leaves per batch, the rows waiting to be merged (800 iterations in batches of 16, one file, games per hour on CUDA
        /// and DirectML: 16 games 3137 and 2835, 38% and 39% of the rows run used; 32 games 4325 and 4988; 64 games 5248 and
        /// 6249, 91% and 92%, more than batches of 256 unmerged, 4413 and 3760). With two files each side's evaluator gets
        /// about half the rows: 64 games 4276 (54%), 128 games 5113 (86%); at 5000 iterations in batches of 32 on CUDA one
        /// file and 32 games 716 (88%), two files and 32 games 597 (54%), 64 games 696 (82%), batches of 256 unmerged 731
        /// </summary>
        public const int RowsInFlight = 1024;

        /// <summary>
        /// Bounds of the games played at once by default per net file: at least as many as before the merging, and at most so
        /// many that the games cancelled when the SPRT decides cost little
        /// </summary>
        public const int MinGamesAtOnce = 16, MaxGamesAtOnce = 64;

        /// <summary>
        /// Games played at once by default with an iteration budget for searches in batches of this size, with the sides'
        /// nets in one or two files: enough for RowsInFlight leaf rows per file, between MinGamesAtOnce and MaxGamesAtOnce per
        /// file (one file, batches of 16: 64 games, 32: 32, from 64: 16; two files twice that). More games don't change the
        /// search, only the throughput
        /// </summary>
        public static int GamesFor(int leavesPerBatch, int files = 1) =>
            files * Math.Clamp(RowsInFlight / leavesPerBatch, MinGamesAtOnce, MaxGamesAtOnce);

        /// <summary>
        /// Rows per net call at most when the leaf batches of games played at once are merged
        /// </summary>
        public const int MergedRows = 256;

        /// <summary>
        /// Whether a side's searches send their leaf batches to a merging evaluator: always on DirectML and the CPU (on
        /// DirectML even in batches of 256: its two nets ran 1.2x the games per hour of one net shared by all games), on CUDA
        /// only if a batch doesn't fill a merged call by itself (there the games run their calls on one net side by side,
        /// and batches of 256 merged played 0.88x and 0.94x the games per hour at 800 and 5000 iterations)
        /// </summary>
        public static bool Merges(InferenceProvider provider, long iterations, params AgentSpec[] sides) =>
            provider != InferenceProvider.CUDA || sides.Min(side => side.BatchFor(iterations)) < MergedRows;

        /// <summary>
        /// Nets per merging evaluator: one runs a call while the next one is collected
        /// </summary>
        public const int MergingNets = 2;

        /// <summary>
        /// The nets of a duel's two sides. With an iteration budget and games played at once, each side's searches send their
        /// leaf batches to one BatchingEvaluator (where Merges), which merges those waiting into calls of up to MergedRows rows
        /// (the side's batch size if larger): small batches searched alone left the GPU waiting on many small calls (16 games
        /// in batches of 16 played 0.24x the games per hour of 256 on DirectML, 0.35x on CUDA; merged 3.1x and 2.0x that).
        /// Both sides share it if they use the same file. Otherwise every search runs its batches on its side's net, whose
        /// DirectML shape is the side's batch size; shared with the same file and batch size.
        /// </summary>
        public static (SideNets challenger, SideNets baseline) Nets(AgentSpec challenger, AgentSpec baseline, InferenceProvider provider, long iterations, int parallel)
        {
            // An Open Seat Agent brings its own: only a side with a net gets one, as if it played itself
            if (challenger.IsOpenSeatAgent || baseline.IsOpenSeatAgent)
            {
                SideNets none = new(null, null);
                if (challenger.IsOpenSeatAgent && baseline.IsOpenSeatAgent) return (none, none);
                AgentSpec withNet = challenger.IsOpenSeatAgent ? baseline : challenger;
                SideNets nets = Nets(withNet, withNet, provider, iterations, parallel).challenger;
                return challenger.IsOpenSeatAgent ? (none, nets) : (nets, none);
            }
            bool sameFile = baseline.ModelPath == challenger.ModelPath;
            if (iterations > 0 && parallel > 1)
            {
                SideNets Merged(params AgentSpec[] sides)
                {
                    int rows = Math.Max(MergedRows, sides.Max(side => side.BatchFor(iterations)));
                    StateValueNet first = new(sides[0].ModelPath, provider: provider, fixedBatchSize: rows);
                    // The provider is known once the net is made (Auto, or a fallback to the CPU): on CUDA, batches that fill a
                    // call by themselves run on the net side by side, unmerged
                    if (!Merges(first.Provider, iterations, sides))
                        return new SideNets(first, null);
                    StateValueNet[] nets = [first, .. Enumerable.Range(1, MergingNets - 1).Select(_ => new StateValueNet(sides[0].ModelPath, provider: provider, fixedBatchSize: rows))];
                    BatchingEvaluator leaves = new(nets, rows);
                    // The agents' own runs (encoding, trade judgments, single positions) go to the first worker's net
                    return new SideNets(nets[0], leaves);
                }
                SideNets merged = sameFile ? Merged(challenger, baseline) : Merged(challenger);
                return (merged, sameFile ? merged : Merged(baseline));
            }

            StateValueNet first = new(challenger.ModelPath, provider: provider, fixedBatchSize: challenger.BatchFor(iterations));
            bool shared = sameFile && baseline.BatchFor(iterations) == challenger.BatchFor(iterations);
            return (new SideNets(first, null), new SideNets(shared ? first : new StateValueNet(baseline.ModelPath, provider: provider, fixedBatchSize: baseline.BatchFor(iterations)), null));
        }

        public static AgentSpec Parse(string spec)
        {
            string[] parts = spec.Split('=');
            if (parts.Length < 2 || parts[0].Length == 0 || parts[1].Length == 0)
                throw new ArgumentException($"Agent \"{spec}\" is not NAME=MODEL.onnx[=OPTION...].");
            // An Open Seat Agent: the rest of the spec is its command line, equals signs and all
            const string command = "command:";
            if (parts[1].StartsWith(command, StringComparison.Ordinal))
            {
                string line = spec[(parts[0].Length + 1 + command.Length)..].Trim();
                if (line.Length == 0) throw new ArgumentException($"Agent \"{spec}\" names no command.");
                return new AgentSpec(parts[0], "", OpponentModel.MaxN, 1.414, null, 0.1, 0, false, Command: line);
            }

            var result = new AgentSpec(parts[0], parts[1], OpponentModel.MaxN, 1.414, null, 0.1, 0, false);
            foreach (string option in parts.Skip(2))
            {
                double Number() => double.Parse(option[1..], CultureInfo.InvariantCulture);
                result = option switch
                {
                    "u" => result with { UniformPriors = true },
                    "is" => result with { InformationSets = true },
                    "isfull" => result with { InformationSets = true, FullKnowledge = true },
                    "norm" => result with { NormalizeValues = true },
                    "nonorm" => result with { NormalizeValues = false },
                    "ob" => result with { BranchHiddenOutcomes = true },
                    "noreuse" => result with { ReuseTree = false },
                    "leaf" => result with { EvaluateNewLeaves = true },
                    "noleaf" => result with { EvaluateNewLeaves = false },
                    "searchforced" => result with { PlayForcedMovesAtOnce = false },
                    // Trade options (see the summary); no offers means no counteroffers either
                    "nooffers" => result with { MakeOffers = false, Counters = false },
                    "o22" => result with { TwoForTwo = true },
                    "co" => result with { Counters = true },
                    "noco" => result with { Counters = false },
                    "o31" => result with { ThreeForOne = true },
                    "nolearn" => result with { LearnsFromTrades = false },
                    // Options with a number after a two-letter prefix, before the one-letter ones (pf would otherwise be read as p)
                    ['o', 'c', ..] => int.Parse(option[2..], CultureInfo.InvariantCulture) is int most and (2 or 3) ? result with { MaxCards = most }
                        : throw new ArgumentException($"Option \"{option}\" in agent \"{spec}\" allows 2 or 3 cards a side."),
                    ['o', 'p', ..] => result with { Prune = int.Parse(option[2..], CultureInfo.InvariantCulture) },
                    ['a', 'f', ..] => result with { AnswerFairness = double.Parse(option[2..], CultureInfo.InvariantCulture) },
                    ['p', 'f', ..] => result with { PartnerFairness = double.Parse(option[2..], CultureInfo.InvariantCulture) },
                    ['a', 'm', ..] => result with { AnswerMargin = double.Parse(option[2..], CultureInfo.InvariantCulture) },
                    ['o', 'm', ..] => result with { OfferMargin = double.Parse(option[2..], CultureInfo.InvariantCulture) },
                    ['o', 'n', ..] => result with { MaxOffers = int.Parse(option[2..], CultureInfo.InvariantCulture) },
                    ['o', 'w', ..] => int.Parse(option[2..], CultureInfo.InvariantCulture) is int worlds and > 0 ? result with { OfferWorlds = worlds }
                        : throw new ArgumentException($"Option \"{option}\" in agent \"{spec}\" needs at least one world."),
                    ['s', 'u', ..] => double.Parse(option[2..], CultureInfo.InvariantCulture) is double factor and > 0 ? result with { SetupBudget = factor }
                        : throw new ArgumentException($"The setup budget factor \"{option}\" in agent \"{spec}\" is not above 0."),
                    ['p', 't', ..] => double.Parse(option[2..], CultureInfo.InvariantCulture) is double temperature and > 0 ? result with { PriorTemperature = temperature }
                        : throw new ArgumentException($"The prior temperature \"{option}\" in agent \"{spec}\" is not above 0."),
                    ['i', 't', ..] => double.Parse(option[2..], CultureInfo.InvariantCulture) is double budget and > 0 ? result with { BudgetFactor = budget }
                        : throw new ArgumentException($"The budget factor \"{option}\" in agent \"{spec}\" is not above 0."),
                    ['f', 'q', ..] => double.Parse(option[2..], CultureInfo.InvariantCulture) is double scale and > 0 ? result with { FinalMoveScale = scale }
                        : throw new ArgumentException($"The final move scale \"{option}\" in agent \"{spec}\" is not above 0."),
                    _ when Enum.TryParse(option, out OpponentModel model) => result with { OpponentModel = model },
                    ['c', ..] => result with { Exploration = Number() },
                    ['b', ..] => result with { LeavesPerBatch = int.Parse(option[1..], CultureInfo.InvariantCulture) },
                    ['f', ..] => result with { FirstPlayReduction = Number() },
                    ['p', ..] => result with { PlacementWeight = Number() },
                    ['d', ..] => result with { Determinizations = int.Parse(option[1..], CultureInfo.InvariantCulture) },
                    ['g', ..] => Number() is > 0 and <= 1 ? result with { ReuseDecay = Number() } : throw new ArgumentException($"The decay \"{option}\" in agent \"{spec}\" is not above 0 and at most 1."),
                    _ => throw new ArgumentException($"Unknown option \"{option}\" in agent \"{spec}\"."),
                };
            }
            return result;
        }

        /// <summary>
        /// The name, the model's file name and the options that differ from the defaults
        /// </summary>
        public override string ToString()
        {
            if (IsOpenSeatAgent) return $"{Name} Open Seat Agent: {Command}";
            List<string> options = [OpponentModel.ToString()];
            if (Exploration != 1.414) options.Add($"c{Exploration.ToString(CultureInfo.InvariantCulture)}");
            if (LeavesPerBatch is int batch) options.Add($"b{batch}");
            if (FirstPlayReduction != 0.1) options.Add($"f{FirstPlayReduction.ToString(CultureInfo.InvariantCulture)}");
            if (FinalMoveScale != 0) options.Add($"fq{FinalMoveScale.ToString(CultureInfo.InvariantCulture)}");
            if (PlacementWeight != 0) options.Add($"p{PlacementWeight.ToString(CultureInfo.InvariantCulture)}");
            if (UniformPriors) options.Add("u");
            if (PriorTemperature != 1) options.Add($"pt{PriorTemperature.ToString(CultureInfo.InvariantCulture)}");
            if (Determinizations > 0) options.Add($"d{Determinizations}");
            if (InformationSets) options.Add(FullKnowledge ? "isfull" : "is");
            if (NormalizeValues is bool normalize) options.Add(normalize ? "norm" : "nonorm");
            if (BranchHiddenOutcomes) options.Add("ob");
            if (!ReuseTree) options.Add("noreuse");
            if (!EvaluateNewLeaves) options.Add("noleaf");
            if (!PlayForcedMovesAtOnce) options.Add("searchforced");
            if (ReuseDecay != 1) options.Add($"g{ReuseDecay.ToString(CultureInfo.InvariantCulture)}");
            if (SetupBudget != 1) options.Add($"su{SetupBudget.ToString(CultureInfo.InvariantCulture)}");
            if (BudgetFactor != 1) options.Add($"it{BudgetFactor.ToString(CultureInfo.InvariantCulture)}");
            // Trade options only where they differ from the defaults
            if (!MakeOffers) options.Add("nooffers");
            if (OfferMargin != 0.005) options.Add($"om{OfferMargin.ToString(CultureInfo.InvariantCulture)}");
            if (MaxOffers != 3) options.Add($"on{MaxOffers}");
            if (OfferWorlds != 8) options.Add($"ow{OfferWorlds}");
            if (TwoForTwo) options.Add("o22");
            if (ThreeForOne) options.Add("o31");
            if (AnswerMargin != 0.002) options.Add($"am{AnswerMargin.ToString(CultureInfo.InvariantCulture)}");
            if (!Counters) options.Add("noco");
            if (AnswerFairness != 1) options.Add($"af{AnswerFairness.ToString(CultureInfo.InvariantCulture)}");
            if (PartnerFairness != 0) options.Add($"pf{PartnerFairness.ToString(CultureInfo.InvariantCulture)}");
            if (!LearnsFromTrades) options.Add("nolearn");
            if (MaxCards != 3) options.Add($"oc{MaxCards}");
            if (Prune != 16) options.Add($"op{Prune}");
            return $"{Name} {Path.GetFileName(ModelPath)} {string.Join(" ", options)}";
        }

        /// <summary>
        /// The agent for a seat, searching with the given net and budget (iterations, or seconds if 0); an Open Seat Agent
        /// gets the game's seed
        /// </summary>
        public Agent Create(sbyte seat, GameState start, StateValueNet? net, long iterations, double seconds, ILeafEvaluator? leaves = null, string gameId = "game",
            int? seed = null)
        {
            // An Open Seat Agent plays the seat as a process of its own, one per game and seat; the proxy keeps to its budget
            if (IsOpenSeatAgent)
                return new OpenSeatAgentProxy(seat, start, OpenSeatAgentProcess.Start(Command!, Timeout.InfiniteTimeSpan), gameId, seed: seed, budget: OpenSeatAgentBudget);
            if (net == null) throw new ArgumentNullException(nameof(net), $"Agent {Name} needs a net.");

            // The agent picks the encoding of its net (see Vectorizers): a belief-aware net sees its seat's view; its search's
            // leaf batches go to leaves if given (shared with the other games), else to the net
            // It searches its multiple of the duel's budget (BudgetFactor), in the batches of that budget
            MCTSAgent search = new MCTSAgent(seat, start, explorationParameter: Exploration, timeLimitSeconds: seconds * BudgetFactor,
                valueNet: net, placementWeight: PlacementWeight, opponentModel: OpponentModel,
                iterationLimit: Budget(iterations), leavesPerBatch: BatchFor(iterations), evaluator: leaves)
            {
                UniformPriors = UniformPriors, FirstPlayReduction = FirstPlayReduction, NormalizeValues = NormalizeValues ?? true, ReuseTree = ReuseTree,
                EvaluateNewLeaves = EvaluateNewLeaves, PlayForcedMovesAtOnce = PlayForcedMovesAtOnce, SetupBudgetFactor = SetupBudget,
                PriorTemperature = PriorTemperature, FinalMoveScale = FinalMoveScale
            };
            // Offers and counteroffers (TradeOffers), then the answers (MCTSAgent)
            search.Offers.Enabled = MakeOffers;
            search.Offers.Margin = (float)OfferMargin;
            search.Offers.MaxPerTurn = MaxOffers;
            search.Offers.Worlds = OfferWorlds;
            search.Offers.TwoForTwo = TwoForTwo;
            search.Offers.Counters = Counters;
            search.Offers.ThreeForOne = ThreeForOne;
            search.Offers.MaxCards = MaxCards;
            search.Offers.Prune = Prune;
            search.AnswerMargin = (float)AnswerMargin;
            search.AnswerFairness = (float)AnswerFairness;
            search.Offers.PartnerFairness = (float)PartnerFairness;

            // The final move by the improved policy is the open-hand tree's; the hidden information agents pick theirs otherwise
            if (FinalMoveScale > 0 && (InformationSets || Determinizations > 0))
                throw new ArgumentException($"Agent {Name}: the final move scale (fq) applies to open-hand MCTS only, not under hidden information.");

            // Under hidden information the agent sees only its seat's view: one tree over its information sets, or the
            // iterations split between its determinizations
            if (InformationSets)
            {
                return new InformationSetMCTSAgent(search, start)
                {
                    FullKnowledge = FullKnowledge,
                    BranchHiddenOutcomes = BranchHiddenOutcomes,
                    NormalizeValues = NormalizeValues ?? false,
                    ReuseDecay = ReuseDecay,
                    LearnsFromTrades = LearnsFromTrades
                };
            }
            return Determinizations > 0 ? new DeterminizedMCTSAgent(search, start, Determinizations) : search;
        }
    }

    /// <summary>
    /// A duel side's nets: Net for what its agents run themselves (their encoding, trade judgments, single positions), and
    /// Leaves for their searches' leaf batches, merged across the games played at once (null: each search runs them on Net)
    /// </summary>
    internal sealed record SideNets(StateValueNet? Net, BatchingEvaluator? Leaves);

    /// <summary>
    /// A game's agents: created seat by seat (if one fails, those already made are closed), closed after the game (an
    /// Open Seat Agent hears the end with the game's result record and its process stops), and the Open Seat Agents named
    /// once per side in the log
    /// </summary>
    internal sealed class Seating
    {
        private readonly HashSet<string> _named = [];

        /// <summary>
        /// The game's agents seat by seat; if an Open Seat Agent fails while being seated (its handshake refused, the game
        /// turned down), the game's result record names the failure in the log (seed: the seed its agents get), and the
        /// agents made are closed
        /// </summary>
        public static Agent[] Create(int seats, Func<int, Agent> create, int seed, Action<string> write)
        {
            List<Agent> agents = [];
            try
            {
                for (int seat = 0; seat < seats; seat++) agents.Add(create(seat));
            }
            catch (Exception e)
            {
                if (e is OpenSeatException failure)
                {
                    int failed = agents.Count;
                    OpenSeatSeatRecord record = new(failed, failure.Handshake);
                    record.Failed(null, failure.Code, failure.FromAgent, failure.Message);
                    write($"result {OpenSeatSeatRecord.Result(seed, 0, 0, Enumerable.Range(0, seats).Select(seat =>
                        seat == failed ? record : agents.ElementAtOrDefault(seat) is OpenSeatAgentProxy proxy ? proxy.Record : new OpenSeatSeatRecord(seat))).ToJsonString()}");
                }
                Close(agents);
                throw;
            }
            return agents.ToArray();
        }

        public static void Close(IEnumerable<Agent> agents)
        {
            foreach (Agent agent in agents) (agent as IDisposable)?.Dispose();
        }

        /// <summary>
        /// Closes a game's agents; with an Open Seat Agent among them, the game's result record (see Bridges/PROTOCOL.md)
        /// goes to each of them in their after and into the log as one line
        /// </summary>
        public static void Close(Agent[] agents, int seed, Action<string> write)
        {
            OpenSeatAgentProxy[] proxies = agents.OfType<OpenSeatAgentProxy>().ToArray();
            if (proxies.Length > 0)
            {
                JsonObject result = OpenSeatSeatRecord.Result(seed, proxies.Max(p => p.Moves), proxies.Max(p => p.Turns),
                    agents.Select((agent, seat) => agent is OpenSeatAgentProxy proxy ? proxy.Record : new OpenSeatSeatRecord(seat)));
                foreach (OpenSeatAgentProxy proxy in proxies) proxy.Result = result;
                write($"result {result.ToJsonString()}");
            }
            Close(agents);
        }

        /// <summary>
        /// The line naming a side's Open Seat Agent (its name, what it needs, its rulesets, whether it observes), the first
        /// time the side's agent answered, else null
        /// </summary>
        public string? Name(string side, Agent agent)
        {
            if (agent is not OpenSeatAgentProxy proxy) return null;
            lock (_named)
                if (!_named.Add(side)) return null;
            OpenSeatHandshake handshake = proxy.Handshake;
            return $"Open Seat Agent {side}: {handshake.Name}, needs {handshake.Needs.ToString().ToLowerInvariant()}, rulesets {string.Join(" ", handshake.Rulesets)}"
                + (handshake.Observe ? ", observes every move" : "") + (handshake.Lean ? ", lean" : "");
        }
    }

    /// <summary>
    /// The net calls of a duel's merged leaf batches: calls, the leaves they evaluated and the rows they ran (padding included)
    /// </summary>
    internal readonly record struct MergedLeaves(long Calls, long Rows, long RunRows)
    {
        /// <summary>
        /// For the duel's first line: how the leaf batches reach the net
        /// </summary>
        public static string Describe(SideNets challenger, SideNets baseline)
        {
            BatchingEvaluator[] merging = new[] { challenger.Leaves, baseline.Leaves }.OfType<BatchingEvaluator>().Distinct().ToArray();
            if (merging.Length == 0) return "";
            // Whose batches are merged, if not both sides' through their own evaluators
            string whose = challenger.Leaves == baseline.Leaves ? " (shared by both sides)"
                : challenger.Leaves == null ? " (the baseline's only)" : baseline.Leaves == null ? " (the challenger's only)" : "";
            return $", merged across games into net calls of up to {merging.Max(leaves => leaves.MaxRows)} rows{whose}";
        }

        /// <summary>
        /// Stops the sides' merging evaluators (each once) and sums their calls
        /// </summary>
        public static MergedLeaves Close(SideNets challenger, SideNets baseline)
        {
            MergedLeaves total = default;
            foreach (BatchingEvaluator leaves in new[] { challenger.Leaves, baseline.Leaves }.OfType<BatchingEvaluator>().Distinct())
            {
                leaves.Dispose();
                total = new MergedLeaves(total.Calls + leaves.Batches, total.Rows + leaves.Rows, total.RunRows + leaves.RunRows);
            }
            return total;
        }

        public override string ToString() => Calls == 0 ? "none"
            : FormattableString.Invariant($"{Calls} net calls, {(double)Rows / Calls:F1} leaves per call ({(double)Rows / RunRows:P0} of the rows run)");
    }
}
