using YamlDotNet.Serialization;

namespace Common
{
    /// <summary>
    /// How trades between players go (with FreeTrade on)
    /// </summary>
    public enum TradeFlow
    {
        /// <summary>
        /// The rulebook: the addressed players answer one after another in seat order (accept, decline or counter), then
        /// the offerer trades with one who accepted or countered, or lets the offer go
        /// </summary>
        Rulebook,
        /// <summary>
        /// As JSettlers plays it: the addressed players answer in any order, the first accept trades at once and closes the
        /// offer, and the offerer may take a standing counteroffer or withdraw at any time. The engine still has one player
        /// to act (the next addressed one in seat order, then the offerer); the freedom is in what is valid.
        /// </summary>
        FirstAccept,
    }

    public class GameSettings
    {
        // Maximum allowed number of resource cards before discarding is required
        public int RobberCardLimit { get; set; }
        // Required number of victory points to win the match
        public int VictoryPoints { get; set; }

        /// <summary>
        /// Whether players trade with each other (offers, counteroffers) besides with the bank. Off for agent games, self-play
        /// and datasets, whose agents and nets don't trade; on for games with humans.
        /// </summary>
        [YamlMember(DefaultValuesHandling = DefaultValuesHandling.OmitDefaults)]
        public bool FreeTrade { get; set; }

        /// <summary>
        /// Offers a player may make per turn, 0 for the default: the rulebook sets no limit, but endless offers would stall the game
        /// </summary>
        [YamlMember(DefaultValuesHandling = DefaultValuesHandling.OmitDefaults)]
        public int MaxTradeOffersPerTurn { get; set; }

        /// <summary>
        /// How trades between players go (see TradeFlow); the rulebook unless the referee plays JSettlers' flow
        /// </summary>
        [YamlMember(DefaultValuesHandling = DefaultValuesHandling.OmitDefaults)]
        public TradeFlow TradeFlow { get; set; }

        public const int DefaultMaxTradeOffersPerTurn = 10;

        /// <summary>
        /// Whether the players' screens show the other hands as each player can know them (the cards confirmed by what
        /// happened in public, see BeliefTracker) instead of only their size. A table rule of the display, not of the game:
        /// all of it is public, a player could count along.
        /// </summary>
        [YamlMember(DefaultValuesHandling = DefaultValuesHandling.OmitDefaults)]
        public bool ShowTrackedCards { get; set; }

        [YamlIgnore]
        public int TradeOfferLimit => MaxTradeOffersPerTurn > 0 ? MaxTradeOffersPerTurn : DefaultMaxTradeOffersPerTurn;

        public GameSettings()
        {
            RobberCardLimit = 7;
            VictoryPoints = 10;
        }

        /// <summary>
        /// Deep copy constructor
        /// </summary>
        /// <param name="copy">Instance to copy</param>
        public GameSettings(GameSettings copy)
        {
            RobberCardLimit = copy.RobberCardLimit;
            VictoryPoints = copy.VictoryPoints;
            FreeTrade = copy.FreeTrade;
            MaxTradeOffersPerTurn = copy.MaxTradeOffersPerTurn;
            TradeFlow = copy.TradeFlow;
            ShowTrackedCards = copy.ShowTrackedCards;
        }

        public override bool Equals(object? obj)
        {
            return obj is GameSettings settings &&
                   RobberCardLimit == settings.RobberCardLimit &&
                   VictoryPoints == settings.VictoryPoints &&
                   FreeTrade == settings.FreeTrade &&
                   MaxTradeOffersPerTurn == settings.MaxTradeOffersPerTurn &&
                   TradeFlow == settings.TradeFlow &&
                   ShowTrackedCards == settings.ShowTrackedCards;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(RobberCardLimit, VictoryPoints, FreeTrade, MaxTradeOffersPerTurn, TradeFlow, ShowTrackedCards);
        }
    }
}