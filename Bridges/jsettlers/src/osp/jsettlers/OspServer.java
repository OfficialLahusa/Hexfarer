package osp.jsettlers;

import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import soc.game.SOCGame;
import soc.server.SOCServer;

/**
 * JSettlers' server with Open Seat Agents among its robots: it starts JSettlers' server (SOCServer) with the given
 * properties and keeps it, so the Open Seat Agents' robots (OspRobotClient, started by the server through
 * jsettlers.bots.start3p) can read the true game they play in. They run inside the server's process; a robot on another
 * machine would only see its seat's view.
 *
 * Usage: java -cp build:JSETTLERS_JAR -Dosp.agent="COMMAND" osp.jsettlers.OspServer [-Dname=value ...] [port]
 * e.g. robot-only games with one robot of Hexfarer's: -Djsettlers.startrobots=7
 * -Djsettlers.bots.start3p=2,osp.jsettlers.OspRobotClient -Djsettlers.bots.percent3p=25 -Djsettlers.bots.botgames.total=100
 * -Djsettlers.bots.botgames.shutdown=Y -Djsettlers.bots.fast_pause_percent=10 -Djsettlers.gameopt.NT=t
 */
public final class OspServer
{
    private static volatile SOCServer server;
    // The robots' brains in games, so a shutdown lets them tell their agents the game's end first
    static final Set<OspRobotBrain> BRAINS = ConcurrentHashMap.newKeySet();

    private OspServer() {}

    /** The true game of that name on this process's server, or null */
    public static SOCGame game(String name)
    {
        SOCServer s = server;
        return s == null ? null : s.getGame(name);
    }

    /** JSettlers' server, letting the robots' brains finish a game before it is removed */
    private static final class Server extends SOCServer
    {
        Server(int port, Properties props) throws Exception
        {
            super(port, props);
        }

        @Override
        public void destroyGame(String name)
        {
            for (OspRobotBrain brain : BRAINS)
                brain.gameEnding(name);
            super.destroyGame(name);
        }
    }

    public static void main(String[] args) throws Exception
    {
        Properties props = SOCServer.parseCmdline_DashedArgs(args);
        if (props == null)
            System.exit(1);
        int port = Integer.parseInt(props.getProperty(SOCServer.PROP_JSETTLERS_PORT, Integer.toString(SOCServer.SOC_PORT_DEFAULT)));
        // JSettlers' server exits right after its last robot-only game: the brains finish theirs first
        Runtime.getRuntime().addShutdownHook(new Thread(() ->
        {
            for (OspRobotBrain brain : BRAINS)
                brain.finishOnShutdown();
        }, "osp-shutdown"));
        server = new Server(port, props);
        server.setPriority(5);
        server.start();
    }
}
