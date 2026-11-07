package osp.jsettlers;

import soc.baseclient.ServerConnectInfo;
import soc.game.SOCGame;
import soc.message.SOCMessage;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.util.CappedQueue;
import soc.util.SOCFeatureSet;
import soc.util.SOCRobotParameters;

/**
 * A JSettlers robot whose decisions an Open Seat Agent makes (see OspRobotBrain). JSettlers' server starts it like its
 * other third-party robots (jsettlers.bots.start3p); it plays classic 4-player games only.
 */
public class OspRobotClient extends SOCRobotClient
{
    public OspRobotClient(final ServerConnectInfo sci, final String nn, final String pw)
    {
        super(sci, nn, pw);
        rbclass = OspRobotClient.class.getName();
    }

    @Override
    protected SOCFeatureSet buildClientFeats()
    {
        // Neither 6 players nor the sea board
        return new SOCFeatureSet(false, false);
    }

    @Override
    public SOCRobotBrain createBrain(final SOCRobotParameters params, final SOCGame ga, final CappedQueue<SOCMessage> mq)
    {
        return new OspRobotBrain(this, params, ga, mq);
    }
}
