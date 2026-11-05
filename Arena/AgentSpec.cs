using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using System.Globalization;

namespace Arena
{
    /// <summary>
    /// An MCTS agent of a duel, given as "NAME=MODEL.onnx[=OPTION...]". Options: an opponent model (MaxN, Paranoid, Random;
    /// MaxN if none), cX exploration constant, bN leaves per batch, fX first play reduction, pX placement weight, u PUCT
    /// with uniform priors instead of the policy head's, dN hidden information: only its seat's view, searched as N
    /// determinizations (PIMC, see DeterminizedMCTSAgent), is hidden information searched as one tree over the seat's
    /// information sets (see InformationSetMCTSAgent), isfull the same search on the true position (an ablation); without
    /// these the agent sees every hand. Ablations of the search: norm or nonorm selects with or without values scaled to
    /// the range seen in the tree (MCTS normalizes, ISMCTS not), ob lets ISMCTS branch on outcomes its seat can't see,
    /// noreuse starts MCTS and ISMCTS on a new tree every decision, noleaf lets MCTS evaluate one of a new leaf's
    /// moves instead of the leaf itself (the old search; leaf is the default), searchforced lets MCTS and PIMC search a
    /// decision with a single legal move instead of playing it at once (the old search), gX scales the statistics of the
    /// subtree ISMCTS goes on with by X (above 0, at most 1; 1, full weight, if not given), suX gives the initial placements
    /// X times the search budget of the other decisions (default 1, see MCTSAgent.SetupBudgetFactor). Trade offers (with free trade, see
    /// TradeOffers): nooffers makes neither offers nor counteroffers, omX needs a mean gain above X (default 0.005), onN makes
    /// at most N per turn (default 3), owN judges them over N worlds (default 8), oc2 keeps to one or two cards a side (oc3,
    /// any trade up to three for three, is the default), with oc2 o22 looks at two cards for two too and o31 at three for
    /// one and one for three, opN judges all in one world and only the N best there in all (default 16, op0 judges all in
    /// all); pfX expects partners to need X times its gain (default 0, any gain); nolearn (ISMCTS) ignores the cards others' offers
    /// and answers show. Answers: afX accepts an offer only if it gains at least X times what the offerer gains (default 1,
    /// af0 any gain above the margin), amX accepts an offer from a gain above X (default 0.002), noco answers
    /// an offer it turns down without a counteroffer (co with one, the default).
    /// </summary>
    internal sealed record AgentSpec(string Name, string ModelPath, OpponentModel OpponentModel, double Exploration, int LeavesPerBatch,
        double FirstPlayReduction, double PlacementWeight, bool UniformPriors, int Determinizations = 0, bool InformationSets = false,
        bool FullKnowledge = false, bool? NormalizeValues = null, bool BranchHiddenOutcomes = false, bool ReuseTree = true,
        bool EvaluateNewLeaves = true, bool PlayForcedMovesAtOnce = true, double ReuseDecay = 1, bool MakeOffers = true,
        double OfferMargin = 0.005, int MaxOffers = 3, int OfferWorlds = 8, bool TwoForTwo = false, double AnswerMargin = 0.002, bool Counters = true,
        bool ThreeForOne = false, double AnswerFairness = 1, bool LearnsFromTrades = true, int MaxCards = 3, int Prune = 16, double PartnerFairness = 0,
        double SetupBudget = 1)
    {
        public static AgentSpec Parse(string spec)
        {
            string[] parts = spec.Split('=');
            if (parts.Length < 2 || parts[0].Length == 0 || parts[1].Length == 0)
                throw new ArgumentException($"Agent \"{spec}\" is not NAME=MODEL.onnx[=OPTION...].");

            var result = new AgentSpec(parts[0], parts[1], OpponentModel.MaxN, 1.414, MCTSAgent.DefaultLeavesPerBatch, 0.1, 0, false);
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
            List<string> options = [OpponentModel.ToString()];
            if (Exploration != 1.414) options.Add($"c{Exploration.ToString(CultureInfo.InvariantCulture)}");
            if (LeavesPerBatch != MCTSAgent.DefaultLeavesPerBatch) options.Add($"b{LeavesPerBatch}");
            if (FirstPlayReduction != 0.1) options.Add($"f{FirstPlayReduction.ToString(CultureInfo.InvariantCulture)}");
            if (PlacementWeight != 0) options.Add($"p{PlacementWeight.ToString(CultureInfo.InvariantCulture)}");
            if (UniformPriors) options.Add("u");
            if (Determinizations > 0) options.Add($"d{Determinizations}");
            if (InformationSets) options.Add(FullKnowledge ? "isfull" : "is");
            if (NormalizeValues is bool normalize) options.Add(normalize ? "norm" : "nonorm");
            if (BranchHiddenOutcomes) options.Add("ob");
            if (!ReuseTree) options.Add("noreuse");
            if (!EvaluateNewLeaves) options.Add("noleaf");
            if (!PlayForcedMovesAtOnce) options.Add("searchforced");
            if (ReuseDecay != 1) options.Add($"g{ReuseDecay.ToString(CultureInfo.InvariantCulture)}");
            if (SetupBudget != 1) options.Add($"su{SetupBudget.ToString(CultureInfo.InvariantCulture)}");
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
        /// The agent for a seat, searching with the given net and budget (iterations, or seconds if 0)
        /// </summary>
        public Agent Create(sbyte seat, GameState start, StateValueNet net, long iterations, double seconds)
        {
            // The agent picks the encoding of its net (see Vectorizers): a belief-aware net sees its seat's view
            MCTSAgent search = new MCTSAgent(seat, start, explorationParameter: Exploration, timeLimitSeconds: seconds,
                valueNet: net, placementWeight: PlacementWeight, opponentModel: OpponentModel,
                iterationLimit: iterations, leavesPerBatch: LeavesPerBatch)
            {
                UniformPriors = UniformPriors, FirstPlayReduction = FirstPlayReduction, NormalizeValues = NormalizeValues ?? true, ReuseTree = ReuseTree,
                EvaluateNewLeaves = EvaluateNewLeaves, PlayForcedMovesAtOnce = PlayForcedMovesAtOnce, SetupBudgetFactor = SetupBudget
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
}
