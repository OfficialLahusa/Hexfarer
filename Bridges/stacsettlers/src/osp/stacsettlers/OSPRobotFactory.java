package osp.stacsettlers;

import soc.game.SOCGame;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.robot.SOCRobotFactory;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * The robot factory of the agent kind "osp" in a simulation config (OSPSimulation): robots whose decisions an Open Seat
 * Agent makes (OSPRobotBrain). It takes no type flags: the agent's command is the system property osp.agent.
 */
public final class OSPRobotFactory implements SOCRobotFactory
{
    @Override
    public SOCRobotBrain getRobot(SOCRobotClient client, SOCRobotParameters params, SOCGame game, CappedQueue queue)
    {
        return new OSPRobotBrain(client, params, game, queue);
    }

    @Override
    public boolean isType(String type)
    {
        return false;
    }

    @Override
    public void setTypeFlag(String type)
    {
        throw new UnsupportedOperationException("The osp robots take no type flags");
    }

    @Override
    public void setTypeFlag(String type, String param)
    {
        throw new UnsupportedOperationException("The osp robots take no type flags");
    }
}
