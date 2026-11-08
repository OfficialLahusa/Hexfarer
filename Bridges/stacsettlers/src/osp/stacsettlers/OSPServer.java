package osp.stacsettlers;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

import soc.game.SOCGame;
import soc.server.SOCServer;
import soc.server.database.DBHelper;
import soc.server.genericServer.StringConnection;
import soc.server.logger.SOCLogger;

/**
 * StacSettlers' server (SOCServer) as OSPSimulation runs it, made reachable for the Open Seat Agents' robots
 * (OSPRobotBrain), which run in the same process: they read the true game they play in (game), under the lock every
 * client request is handled with (read), so they never see a request half handled; a game's record (GameRecord) keeps
 * its result record, and before the server removes a game its robots tell their agents its end. Nothing else of the
 * server changes.
 */
public final class OSPServer extends SOCServer
{
    private static volatile OSPServer server;
    // The lock of the server's one request thread: held while it handles a request, and by a robot reading a game
    private static final Object REQUESTS = new Object();
    // The robots' brains in games, so a game's end reaches their agents before the server removes it
    static final Set<OSPRobotBrain> BRAINS = ConcurrentHashMap.newKeySet();
    // Each game's record, from the moment a robot of ours first looked at it
    private static final Map<String, GameRecord> RECORDS = new ConcurrentHashMap<>();

    /** StacSettlers' server on a local string port, with the arguments StacSettlers' Simulation gives its own */
    public OSPServer(String port, int maxConnections, DBHelper db, String databaseUser, String databasePassword, SOCLogger logger, boolean useParser)
    {
        super(port, maxConnections, db, databaseUser, databasePassword, logger, useParser);
        server = this;
    }

    /** The true game of that name on this process's server, or null */
    static SOCGame game(String name)
    {
        OSPServer s = server;
        return s == null ? null : s.gameList.getGameData(name);
    }

    /** The record of a started game, begun when one of our robots first looks at it (holding the read locks) */
    static GameRecord recordOf(SOCGame game)
    {
        return RECORDS.computeIfAbsent(game.getName(), name -> new GameRecord(game));
    }

    /**
     * Reads the server's game under its locks: the request lock, so no request is half handled, and the game's own
     * monitor, which StacSettlers takes while it changes the game (also outside requests). The caller must hold neither
     * (StacSettlers' monitor isn't reentrant).
     */
    static <T> T read(SOCGame game, Supplier<T> reader)
    {
        synchronized (REQUESTS)
        {
            game.takeMonitor();
            try
            {
                return reader.get();
            }
            finally
            {
                game.releaseMonitor();
            }
        }
    }

    @Override
    public void processCommand(String command, StringConnection connection)
    {
        synchronized (REQUESTS)
        {
            super.processCommand(command, connection);
        }
    }

    @Override
    public void destroyGame(String name, boolean haveMonitor)
    {
        // The game's end first, while the server still has it: its record (if it ended with a winner), then the robots
        // hand it to their agents. The thread that removes the game holds it, so it is read as it is.
        SOCGame game = gameList.getGameData(name);
        GameRecord record = RECORDS.get(name);
        if (record != null && game != null && game.getPlayer(0) != null)
            record.ended(game);
        for (OSPRobotBrain brain : BRAINS)
            brain.gameEnding(name);
        RECORDS.remove(name);
        super.destroyGame(name, haveMonitor);
    }
}
