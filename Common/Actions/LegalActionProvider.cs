using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class LegalActionProvider : IActionProvider
    {
        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            // One list for every move type, each adding its moves in this order
            List<Action> actions = [];
            EndTurnAction.AddActionsForState(state, playerIdx, actions);

            RollAction.AddActionsForState(state, playerIdx, actions);
            DiscardAction.AddActionsForState(state, playerIdx, actions);
            RobberAction.AddActionsForState(state, playerIdx, actions);

            FirstInitialSettlementAction.AddActionsForState(state, playerIdx, actions);
            FirstInitialRoadAction.AddActionsForState(state, playerIdx, actions);
            SecondInitialSettlementAction.AddActionsForState(state, playerIdx, actions);
            SecondInitialRoadAction.AddActionsForState(state, playerIdx, actions);

            RoadAction.AddActionsForState(state, playerIdx, actions);
            SettlementAction.AddActionsForState(state, playerIdx, actions);
            CityAction.AddActionsForState(state, playerIdx, actions);
            BuyDevelopmentCardAction.AddActionsForState(state, playerIdx, actions);

            KnightAction.AddActionsForState(state, playerIdx, actions);
            MonopolyAction.AddActionsForState(state, playerIdx, actions);
            RoadBuildingAction.AddActionsForState(state, playerIdx, actions);
            YearOfPlentyAction.AddActionsForState(state, playerIdx, actions);

            FourToOneTradeAction.AddActionsForState(state, playerIdx, actions);
            ThreeToOneTradeAction.AddActionsForState(state, playerIdx, actions);
            TwoToOneTradeAction.AddActionsForState(state, playerIdx, actions);

            // Answers to an open offer and the offerer's pick; offers and counteroffers themselves are any card
            // combination and come from the client or an agent
            actions.AddRange(RespondToTradeAction.GetActionsForState(state, playerIdx));
            actions.AddRange(ConfirmTradeAction.GetActionsForState(state, playerIdx));
            actions.AddRange(CancelTradeAction.GetActionsForState(state, playerIdx));

            return actions;
        }
    }
}
