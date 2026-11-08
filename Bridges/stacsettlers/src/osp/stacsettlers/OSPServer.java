package osp.stacsettlers;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.message.SOCMessage;
import soc.message.SOCMessageForGame;
import soc.server.SOCServer;
import soc.server.database.DBHelper;
import soc.server.genericServer.StringConnection;
import soc.server.logger.SOCLogger;

/**
 * StacSettlers' server (SOCServer) as OSPSimulation runs it, made reachable for the Open Seat Agents' robots
 * (OSPRobotBrain), which run in the same process: they read the true game they play in (game), under the lock every
 * client request is handled with (read), so they never see a request half handled; a game's record (GameRecord) keeps
 * its result record, and before the server removes a game its robots tell their agents its end. Every request of a
 * started game passes through the game's move log (MoveLog), which turns it into protocol steps for the agents that
 * observe and, with the system property osp.record=DIR, writes the game as a referee log of every seat's moves. Nothing
 * else of the server changes.
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

    // Where each game is written as a referee log while it runs (osp.record), or null; each started game's move log
    private static final String RECORD = System.getProperty("osp.record");
    private static final Map<String, MoveLog> LOGS = new ConcurrentHashMap<>();

    @Override
    public void processCommand(String command, StringConnection connection)
    {
        synchronized (REQUESTS)
        {
            SOCMessage message = SOCMessage.toMsg(command);
            MoveLog log = message == null ? null : logOf(message);
            if (log == null)
            {
                super.processCommand(command, connection);
                return;
            }
            // The request's seat: the player of the connection's name, if it sits in the game
            SOCPlayer player = connection == null || connection.getData() == null ? null : log.game.getPlayer((String) connection.getData());
            int seat = player == null ? -1 : log.osp.seatOf(player.getPlayerNumber());
            log.dispatch(message, seat, () -> super.processCommand(command, connection));
        }
    }

    // The move log of the game a request is for, begun at its first request once the game started, or null
    private MoveLog logOf(SOCMessage message)
    {
        if (! (message instanceof SOCMessageForGame) || ((SOCMessageForGame) message).getGame() == null)
            return null;
        String name = ((SOCMessageForGame) message).getGame();
        MoveLog log = LOGS.get(name);
        if (log != null)
            return log;
        SOCGame game = gameList.getGameData(name);
        if (game == null || game.getGameState() < SOCGame.START1A || game.getGameState() >= SOCGame.OVER)
            return null;
        game.takeMonitor();
        try
        {
            return logIn(game);
        }
        finally
        {
            game.releaseMonitor();
        }
    }

    /**
     * The move log of a started game, begun now if the game has none yet: at its first request, or when a robot of ours
     * first looks at it (the caller holds the request lock and the game's monitor)
     */
    static MoveLog logIn(SOCGame game)
    {
        return LOGS.computeIfAbsent(game.getName(), name -> new MoveLog(game, RECORD));
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
        // Its record closed between two requests (the request thread writes it)
        MoveLog log = LOGS.remove(name);
        if (log != null)
            synchronized (REQUESTS)
            {
                log.close();
            }
        super.destroyGame(name, haveMonitor);
    }
}
