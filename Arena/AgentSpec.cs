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
    /// subtree ISMCTS goes on with by X (above 0, at most 1; 1, full weight, if not given). Trade offers (with free trade, see
    /// TradeOffers): nooffers makes none, omX needs a mean gain above X (default 0.005), onN makes at most N per turn
    /// (default 3), owN judges them over N worlds (default 8), o22 looks at two cards for two too. Answers: amX accepts an
    /// offer from a gain above X (default 0.002), co answers an offer it turns down with a counteroffer.
    /// </summary>
    internal sealed record AgentSpec(string Name, string ModelPath, OpponentModel OpponentModel, double Exploration, int LeavesPerBatch,
        double FirstPlayReduction, double PlacementWeight, bool UniformPriors, int Determinizations = 0, bool InformationSets = false,
        bool FullKnowledge = false, bool? NormalizeValues = null, bool BranchHiddenOutcomes = false, bool ReuseTree = true,
        bool EvaluateNewLeaves = true, bool PlayForcedMovesAtOnce = true, double ReuseDecay = 1, bool MakeOffers = true,
        double OfferMargin = 0.005, int MaxOffers = 3, int OfferWorlds = 8, bool TwoForTwo = false, double AnswerMargin = 0.002, bool Counters = false)
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
                    "nooffers" => result with { MakeOffers = false },
                    "o22" => result with { TwoForTwo = true },
                    "co" => result with { Counters = true },
                    ['a', 'm', ..] => result with { AnswerMargin = double.Parse(option[2..], CultureInfo.InvariantCulture) },
                    ['o', 'm', ..] => result with { OfferMargin = double.Parse(option[2..], CultureInfo.InvariantCulture) },
                    ['o', 'n', ..] => result with { MaxOffers = int.Parse(option[2..], CultureInfo.InvariantCulture) },
                    ['o', 'w', ..] => int.Parse(option[2..], CultureInfo.InvariantCulture) is int worlds and > 0 ? result with { OfferWorlds = worlds }
                        : throw new ArgumentException($"Option \"{option}\" in agent \"{spec}\" needs at least one world."),
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
            if (!MakeOffers) options.Add("nooffers");
            if (OfferMargin != 0.005) options.Add($"om{OfferMargin.ToString(CultureInfo.InvariantCulture)}");
            if (MaxOffers != 3) options.Add($"on{MaxOffers}");
            if (OfferWorlds != 8) options.Add($"ow{OfferWorlds}");
            if (TwoForTwo) options.Add("o22");
            if (AnswerMargin != 0.002) options.Add($"am{AnswerMargin.ToString(CultureInfo.InvariantCulture)}");
            if (Counters) options.Add("co");
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
                EvaluateNewLeaves = EvaluateNewLeaves, PlayForcedMovesAtOnce = PlayForcedMovesAtOnce
            };
            search.Offers.Enabled = MakeOffers;
            search.Offers.Margin = (float)OfferMargin;
            search.Offers.MaxPerTurn = MaxOffers;
            search.Offers.Worlds = OfferWorlds;
            search.Offers.TwoForTwo = TwoForTwo;
            search.Offers.Counters = Counters;
            search.AnswerMargin = (float)AnswerMargin;

            // Under hidden information the agent sees only its seat's view: one tree over its information sets, or the
            // iterations split between its determinizations
            if (InformationSets)
            {
                return new InformationSetMCTSAgent(search, start)
                {
                    FullKnowledge = FullKnowledge,
                    BranchHiddenOutcomes = BranchHiddenOutcomes,
                    NormalizeValues = NormalizeValues ?? false,
                    ReuseDecay = ReuseDecay
                };
            }
            return Determinizations > 0 ? new DeterminizedMCTSAgent(search, start, Determinizations) : search;
        }
    }
}
