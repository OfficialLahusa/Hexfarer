using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class ThreeToOneTradeAction : Action, IActionProvider
    {
        public ResourceCardType InputType { get; init; }
        public ResourceCardType OutputType { get; init; }

        public ThreeToOneTradeAction(sbyte playerIdx, ResourceCardType inputType, ResourceCardType outputType)
            : base(playerIdx)
        {
            InputType = inputType;
            OutputType = outputType;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private ThreeToOneTradeAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Road Building roads that were not placed yet are given up
            ForfeitFreeRoads(state);

            // Remove input from hand
            state.Players[PlayerIndex].ResourceCards.Remove(InputType, 3);

            // Remove output from bank
            state.ResourceBank.Remove(OutputType, 1);

            // Add input to bank
            state.ResourceBank.Add(InputType, 3);

            // Add output to hand
            state.Players[PlayerIndex].ResourceCards.Add(OutputType, 1);
        }

        public override void Revert(GameState state)
        {
            // Return input to hand
            state.Players[PlayerIndex].ResourceCards.Add(InputType, 3);

            // Return output to bank
            state.ResourceBank.Add(OutputType, 1);

            // Remove input from bank
            state.ResourceBank.Remove(InputType, 3);

            // Remove output from hand
            state.Players[PlayerIndex].ResourceCards.Remove(OutputType, 1);

            ReturnForfeitedFreeRoads(state);
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex) && IsBoardValid(state);
        }

        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PlayerIndex == playerIdx
                && turn.TypeOfRound == TurnState.RoundType.Normal
                && turn.PendingTrade == null
                && !turn.MustRoll
                && !turn.MustDiscard
                && !turn.MustMoveRobber;
        }

        public bool IsBoardValid(GameState state)
        {
            bool cardTypesDiffer = InputType != OutputType;
            bool playerHasInput = state.Players[PlayerIndex].ResourceCards.Contains(InputType, 3);
            bool bankHasOutput = state.ResourceBank.Contains(OutputType, 1);

            // Check port privilege
            bool hasPort = state.Players[PlayerIndex].PortPrivileges.HasFlag(PortPrivileges.GenericThreeToOne);

            return cardTypesDiffer && playerHasInput && bankHasOutput && hasPort;
        }

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = [];
            AddActionsForState(state, playerIdx, actions);
            return actions;
        }

        /// <summary>
        /// Adds the moves GetActionsForState lists to the given list, so the legal moves of all move types go into one list
        /// (see MCTSTree.GetValidActions)
        /// </summary>
        public static void AddActionsForState(GameState state, sbyte playerIdx, List<Action> actions)
        {
            if (!IsTurnValid(state.Turn, playerIdx)) return;

            if (!state.Players[playerIdx].PortPrivileges.HasFlag(PortPrivileges.GenericThreeToOne)) return;

            // The trades IsBoardValid accepts, without creating the rejected ones
            CardSet<ResourceCardType> hand = state.Players[playerIdx].ResourceCards;
            foreach (ResourceCardType inputType in CardTypeExtensions.ResourceTypes)
            {
                if (!hand.Contains(inputType, 3)) continue;

                foreach (ResourceCardType outputType in CardTypeExtensions.ResourceTypes)
                {
                    if (outputType != inputType && state.ResourceBank.Contains(outputType, 1))
                        actions.Add(new ThreeToOneTradeAction(playerIdx, inputType, outputType));
                }
            }
        }

        public override string ToString()
        {
            return base.ToString() + $", {InputType.GetAbbreviation()}{InputType.GetAbbreviation()}{InputType.GetAbbreviation()} -> {OutputType.GetAbbreviation()}";
        }
    }
}
