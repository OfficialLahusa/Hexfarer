package osp.jsettlers;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CopyOnWriteArrayList;

import soc.game.ResourceSet;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.message.SOCBankTrade;
import soc.message.SOCChoosePlayer;
import soc.message.SOCDiscard;
import soc.message.SOCMessage;
import soc.message.SOCMessageForGame;
import soc.message.SOCMoveRobber;
import soc.message.SOCPickResourceType;
import soc.message.SOCPickResources;
import soc.message.SOCPlayDevCardRequest;
import soc.message.SOCPutPiece;
import soc.server.GameMessageHandler;
import soc.server.genericServer.Connection;

/**
 * Every move of a JSettlers game on this process's server as Open Seat Protocol steps (see ../../PROTOCOL.md), with its
 * result as the referee knows it and the true state before it. JSettlers' server hands each player's request to its
 * message handler on one thread; OSPServer's handler passes it through dispatch here, which reads the game before and
 * after JSettlers handles it and turns the change into a step: one protocol move where JSettlers takes two requests (a
 * bought piece and its place, a card and its resources, a robber move and its victim), and the discards after a 7, which
 * JSettlers' players make at the same time, one after another from the lowest seat, each with the state the protocol's
 * order gives it. A change no request made (the server ending a slow player's turn, say) counts as a gap: the next step
 * then starts from another state than the last one ended in.
 *
 * The log's monitor is the lock for reading the game: a robot's brain that reads under it (OSPServer.read) sees no move
 * half handled. With the system property osp.record set, each game is also written there as a referee log while it
 * runs (GAME.jsonl: before, a decide and a step per move, after), the input of Hexfarer's replay harness and relay. At its
 * end each game writes one line to standard error: the players by seat, the winner, and per seat its moves, those with a
 * choice, and the CPU time of its robot's brain thread in this game.
 */
final class MoveLog
{
    final String gameId;
    final SOCGame game;
    final OSPGame osp;

    // The steps so far: read without the log's lock by a brain finishing its game (which may hold its own lock)
    private final List<Map<String, Object>> steps = new CopyOnWriteArrayList<>();
    // The state before the first move, and after the last request handled
    private final Map<String, Object> start;
    private Map<String, Object> last;
    // The first request of a two-step move: the move's kind and value so far, the seat, the state before it, its legal moves
    private String pendingKind;
    private Object pendingValue;
    private int pendingSeat;
    private Map<String, Object> pendingBefore;
    private List<Object> pendingLegal;
    // After a 7: the state before the first discard, the seats that have to discard (lowest first), the discards come in
    // (seat: card set and legal moves) and how many of the seats are written as steps
    private Map<String, Object> discardStart;
    private List<Integer> discarders;
    private final Map<Integer, Object[]> discarded = new TreeMap<>();
    private int discardsDone;
    private int gaps;
    private boolean ended;
    private final int[] moves, choices;
    private Writer record;

    MoveLog(SOCGame game, Map<String, Object> start)
    {
        this.game = game;
        this.gameId = game.getName();
        this.osp = new OSPGame(game);
        this.start = start;
        this.last = start;
        this.moves = new int[game.maxPlayers];
        this.choices = new int[game.maxPlayers];
        String dir = System.getProperty("osp.record");
        if (dir != null)
        {
            try
            {
                new File(dir).mkdirs();
                String name = gameId.replaceAll("[^A-Za-z0-9._-]", "_");
                record = new OutputStreamWriter(new FileOutputStream(new File(dir, name + ".jsonl")), StandardCharsets.UTF_8);
                write(Json.object("type", "before", "game_id", gameId, "players", (long) game.maxPlayers, "jsettlers_first", (long) osp.first,
                    "ruleset", OSPRobotBrain.RULESET, "board", osp.board.boardJson(), "state", start));
            }
            catch (IOException e)
            {
                System.err.println("osp: can't record " + gameId + ": " + e);
                record = null;
            }
        }
    }

    /** The state before the game's first move */
    Map<String, Object> start()
    {
        return start;
    }

    /** How many moves the game has had so far (the next one's n) */
    int size()
    {
        return steps.size();
    }

    /** The steps from the index on */
    List<Map<String, Object>> stepsFrom(int index)
    {
        Object[] all = steps.toArray();
        List<Map<String, Object>> from = new ArrayList<>();
        for (int i = index; i < all.length; i++)
        {
            @SuppressWarnings("unchecked")
            Map<String, Object> step = (Map<String, Object>) all[i];
            from.add(step);
        }
        return from;
    }

    int gaps()
    {
        return gaps;
    }

    /**
     * The state the next move starts from, in the protocol's order: the true state, or while discards go on, the state
     * before the first discard with the discards before the next seat's done (call under the log's lock and the game's
     * monitor)
     */
    Map<String, Object> state()
    {
        if (discardStart != null)
            return discardState(discardsDone);
        return osp.state(-1);
    }

    /**
     * JSettlers' handling of one request, with the step it makes: the game is read before and after (under its
     * monitor, which JSettlers' handler takes itself in between). A robot-only game's last move ends inside it: JSettlers
     * destroys the game before the handler returns, and OSPServer lets the log read the end first (gameDestroyed).
     */
    static boolean dispatch(GameMessageHandler inner, SOCGame game, SOCMessageForGame message, Connection connection) throws Exception
    {
        MoveLog log = game.getGameState() >= SOCGame.START1A ? OSPServer.logOf(game) : null;
        if (log == null)
            return inner.dispatch(game, message, connection);
        SOCPlayer player = connection == null ? null : game.getPlayer(connection.getData());
        return dispatchAs(log, inner, game, message, connection, player == null ? -1 : log.osp.seatOf(player.getPlayerNumber()));
    }

    /** JSettlers' handling of a request from a seat (-1: from no player), with the step it makes */
    static boolean dispatchAs(MoveLog log, GameMessageHandler inner, SOCGame game, SOCMessageForGame message, Connection connection, int seat) throws Exception
    {
        synchronized (log)
        {
            if (log.ended)
                return inner.dispatch(game, message, connection);
            game.takeMonitor();
            try
            {
                Map<String, Object> before = log.osp.state(-1);
                int acting = log.osp.acting();
                List<Object> legal = message.getType() == SOCMessage.DISCARD && seat >= 0 ? log.osp.discardMoves(seat) : acting >= 0 ? log.osp.legal(acting) : new ArrayList<>();
                List<Integer> mustDiscard = new ArrayList<>();
                for (int s = 0; s < game.maxPlayers; s++)
                    if (game.getPlayer(log.osp.playerOf(s)).getNeedToDiscard())
                        mustDiscard.add(s);
                log.inFlight = new InFlight(message, seat, before, legal, mustDiscard);
            }
            finally
            {
                game.releaseMonitor();
            }
            try
            {
                return inner.dispatch(game, message, connection);
            }
            finally
            {
                if (log.inFlight != null && log.alive())
                {
                    game.takeMonitor();
                    try
                    {
                        log.afterRequest();
                    }
                    finally
                    {
                        game.releaseMonitor();
                    }
                }
                log.inFlight = null;
            }
        }
    }

    /** A request JSettlers is handling: who sent it, and the game before it */
    private static final class InFlight
    {
        final SOCMessageForGame message;
        final int seat;
        final Map<String, Object> before;
        final List<Object> legal;
        final List<Integer> mustDiscard;

        InFlight(SOCMessageForGame message, int seat, Map<String, Object> before, List<Object> legal, List<Integer> mustDiscard)
        {
            this.message = message;
            this.seat = seat;
            this.before = before;
            this.legal = legal;
            this.mustDiscard = mustDiscard;
        }
    }

    private InFlight inFlight;

    // Whether JSettlers still has the game (it empties a destroyed one)
    private boolean alive()
    {
        return game.getPlayer(0) != null && game.getBoard() != null;
    }

    // The request handled: its step, and the game's end if it ended (a failure here never stops JSettlers' handling: it
    // counts as a gap)
    private void afterRequest()
    {
        InFlight request = inFlight;
        inFlight = null;
        try
        {
            Map<String, Object> after = osp.state(-1);
            if (request.seat >= 0)
                handled(request.message, request.seat, request.before, after, request.legal, request.mustDiscard);
            last = after;
            if (osp.isOver() && game.getPlayerWithWin() != null)
                end(after);
        }
        catch (RuntimeException e)
        {
            gaps++;
            System.err.println("osp: move log of " + gameId + " failed: " + e);
            e.printStackTrace();
        }
    }

    /**
     * JSettlers is destroying the game (OSPServer, on its message thread while it still handles the game's last request,
     * whose monitor it may hold): the last move and the end are read first, then the record is closed
     */
    synchronized void gameDestroyed()
    {
        if (inFlight != null && alive())
            afterRequest();
        close();
    }

    // Turns a handled request into its step, if it made one
    private void handled(SOCMessageForGame message, int seat, Map<String, Object> before, Map<String, Object> after, List<Object> legal,
        List<Integer> mustDiscard)
    {
        boolean changed = ! Json.write(before, true).equals(Json.write(after, true));
        if (! changed)
            return;
        // Something changed the game since the last request: the server's own move (a turn it ended); what was pending is lost
        if (! Json.write(before, true).equals(Json.write(last, true)))
        {
            gaps++;
            pendingKind = null;
            discardStart = null;
            discarded.clear();
        }
        switch (message.getType())
        {
        case SOCMessage.ROLLDICE:
        {
            int total = game.getCurrentDice();
            int a = Math.min(6, total - 1);
            step(seat, Json.array("roll"), Json.object("dice", Json.array((long) a, (long) (total - a))), before, legal);
            break;
        }
        case SOCMessage.PUTPIECE:
        {
            SOCPutPiece put = (SOCPutPiece) message;
            String kind = put.getPieceType() == SOCPlayingPiece.ROAD ? "build_road" : put.getPieceType() == SOCPlayingPiece.SETTLEMENT ? "build_settlement" : "build_city";
            Object place = kind.equals("build_road") ? osp.board.path(put.getCoordinates()) : osp.board.intersection(put.getCoordinates());
            if (pendingKind != null && pendingKind.equals(kind))
                step(seat, Json.array(kind, place), null, pendingBefore, pendingLegal);
            else
                step(seat, Json.array(kind, place), null, before, legal);
            break;
        }
        case SOCMessage.BUILDREQUEST:
        {
            int type = ((soc.message.SOCBuildRequest) message).getPieceType();
            pending(type == SOCPlayingPiece.ROAD ? "build_road" : type == SOCPlayingPiece.SETTLEMENT ? "build_settlement" : "build_city", null, seat, before, legal);
            break;
        }
        case SOCMessage.CANCELBUILDREQUEST:
            pendingKind = null;
            break;
        case SOCMessage.BUYDEVCARDREQUEST:
        {
            String card = null;
            Map<String, Object> had = development(before, seat), has = development(after, seat);
            for (String name : has.keySet())
                if (Json.integer(has.get(name)) > Json.integer(had.get(name)))
                    card = name;
            step(seat, Json.array("buy_development"), Json.object("card", card), before, legal);
            break;
        }
        case SOCMessage.PLAYDEVCARDREQUEST:
            switch (((SOCPlayDevCardRequest) message).getDevCard())
            {
            case SOCDevCardConstants.KNIGHT: step(seat, Json.array("play_knight"), null, before, legal); break;
            case SOCDevCardConstants.ROADS: step(seat, Json.array("play_road_building"), null, before, legal); break;
            case SOCDevCardConstants.DISC: pending("play_year_of_plenty", null, seat, before, legal); break;
            case SOCDevCardConstants.MONO: pending("play_monopoly", null, seat, before, legal); break;
            default: gaps++;
            }
            break;
        case SOCMessage.PICKRESOURCES:
            if ("play_year_of_plenty".equals(pendingKind))
            {
                ResourceSet picked = ((SOCPickResources) message).getResources();
                List<Object> names = new ArrayList<>();
                for (String name : new String[] {"brick", "lumber", "ore", "grain", "wool"})
                    for (int i = 0; i < picked.getAmount(OSPGame.resourceType(name)); i++)
                        names.add(name);
                names.sort(null);
                step(seat, Json.array("play_year_of_plenty", names), null, pendingBefore, pendingLegal);
            }
            break;
        case SOCMessage.PICKRESOURCETYPE:
            if ("play_monopoly".equals(pendingKind))
            {
                String resource = OSPGame.resourceName(((SOCPickResourceType) message).getResourceType());
                Map<String, Object> taken = new LinkedHashMap<>();
                for (int other = 0; other < game.maxPlayers; other++)
                    if (other != seat)
                        taken.put(Integer.toString(other), (long) (Json.integer(hand(pendingBefore, other).get(resource)) - Json.integer(hand(after, other).get(resource))));
                step(seat, Json.array("play_monopoly", resource), Json.object("taken", taken), pendingBefore, pendingLegal);
            }
            break;
        case SOCMessage.MOVEROBBER:
        {
            List<Object> hex = osp.board.hex(((SOCMoveRobber) message).getCoordinates());
            if (game.getGameState() == SOCGame.WAITING_FOR_ROB_CHOOSE_PLAYER)
                pending("move_robber", hex, seat, before, legal);
            else
                robbery(seat, hex, before, after, legal);
            break;
        }
        case SOCMessage.CHOOSEPLAYER:
            if ("move_robber".equals(pendingKind))
                robbery(seat, pendingValue, pendingBefore, after, pendingLegal);
            break;
        case SOCMessage.DISCARD:
        {
            if (discardStart == null)
            {
                discardStart = before;
                discarders = mustDiscard;
                discardsDone = 0;
                discarded.clear();
            }
            ResourceSet set = ((SOCDiscard) message).getResources();
            int[] counts = new int[soc.game.SOCResourceConstants.WOOD + 1];
            for (String name : new String[] {"brick", "lumber", "ore", "grain", "wool"})
                counts[OSPGame.resourceType(name)] = set.getAmount(OSPGame.resourceType(name));
            discarded.put(seat, new Object[] {OSPGame.cardSet(counts), legal});
            // In the protocol's order, as far as the seats before have discarded
            while (discardsDone < discarders.size() && discarded.containsKey(discarders.get(discardsDone)))
            {
                int next = discarders.get(discardsDone);
                Object[] discard = discarded.get(next);
                @SuppressWarnings("unchecked")
                List<Object> moves = (List<Object>) discard[1];
                step(next, Json.array("discard", discard[0]), null, discardState(discardsDone), moves);
                discardsDone++;
            }
            if (discardsDone == discarders.size())
            {
                discardStart = null;
                discarded.clear();
            }
            break;
        }
        case SOCMessage.BANKTRADE:
        {
            SOCBankTrade trade = (SOCBankTrade) message;
            step(seat, Json.array("bank_trade", Json.object("give", cards(trade.getGiveSet()), "get", cards(trade.getGetSet()))), null, before, legal);
            break;
        }
        case SOCMessage.ENDTURN:
            step(seat, Json.array("end_turn"), null, before, legal);
            break;
        default:
            // A request no protocol move makes changed the game
            gaps++;
        }
    }

    private void pending(String kind, Object value, int seat, Map<String, Object> before, List<Object> legal)
    {
        pendingKind = kind;
        pendingValue = value;
        pendingSeat = seat;
        pendingBefore = before;
        pendingLegal = legal;
    }

    // A robber move: the victim is who lost a card, the card what the thief got
    private void robbery(int seat, Object hex, Map<String, Object> before, Map<String, Object> after, List<Object> legal)
    {
        Object victim = null;
        String stolen = null;
        for (int other = 0; other < game.maxPlayers; other++)
            if (other != seat && handSize(after, other) < handSize(before, other))
                victim = (long) other;
        Map<String, Object> had = hand(before, seat), has = hand(after, seat);
        for (String name : has.keySet())
            if (Json.integer(has.get(name)) > Json.integer(had.get(name)))
                stolen = name;
        step(seat, Json.array("move_robber", Json.object("hex", hex, "victim", victim)), victim == null ? null : Json.object("stolen", stolen), before, legal);
    }

    private static Map<String, Object> cards(soc.game.SOCResourceSet set)
    {
        int[] counts = new int[soc.game.SOCResourceConstants.WOOD + 1];
        for (String name : new String[] {"brick", "lumber", "ore", "grain", "wool"})
            counts[OSPGame.resourceType(name)] = set.getAmount(OSPGame.resourceType(name));
        return OSPGame.cardSet(counts);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> player(Map<String, Object> state, int seat)
    {
        return (Map<String, Object>) ((List<Object>) state.get("players")).get(seat);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> hand(Map<String, Object> state, int seat)
    {
        return (Map<String, Object>) player(state, seat).get("hand");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> development(Map<String, Object> state, int seat)
    {
        return (Map<String, Object>) player(state, seat).get("development");
    }

    private static int handSize(Map<String, Object> state, int seat)
    {
        return Json.integer(player(state, seat).get("hand_size"));
    }

    // The state before the first discard with the first done discards made, the next discarding seat to act
    @SuppressWarnings("unchecked")
    private Map<String, Object> discardState(int done)
    {
        Map<String, Object> state = OSPGame.copy(discardStart);
        Map<String, Object> bank = (Map<String, Object>) state.get("bank");
        for (int i = 0; i < done; i++)
        {
            int seat = discarders.get(i);
            Map<String, Object> set = (Map<String, Object>) discarded.get(seat)[0];
            Map<String, Object> player = player(state, seat), hand = (Map<String, Object>) player.get("hand");
            for (Map.Entry<String, Object> card : set.entrySet())
            {
                int count = Json.integer(card.getValue());
                hand.put(card.getKey(), (long) (Json.integer(hand.get(card.getKey())) - count));
                bank.put(card.getKey(), (long) (Json.integer(bank.get(card.getKey())) + count));
                player.put("hand_size", (long) (Json.integer(player.get("hand_size")) - count));
            }
        }
        if (done < discarders.size())
            state.put("current", (long) (int) discarders.get(done));
        return state;
    }

    // A move made: its step, and its decision and step in the record
    private void step(int seat, List<Object> action, Map<String, Object> result, Map<String, Object> before, List<Object> legal)
    {
        pendingKind = null;
        long n = steps.size();
        steps.add(Json.object("type", "step", "game_id", gameId, "n", n, "seat", (long) seat, "action", action, "result", result, "state", before));
        moves[seat]++;
        if (legal != null && legal.size() > 1)
            choices[seat]++;
        write(Json.object("type", "decide", "game_id", gameId, "n", n, "seat", (long) seat, "state", before, "legal", legal));
        write(Json.object("type", "step", "game_id", gameId, "n", n, "seat", (long) seat, "action", action, "result", result));
    }

    // The game is over and its winner named: the record's end and the game's statistics line
    private void end(Map<String, Object> state)
    {
        if (ended)
            return;
        ended = true;
        List<Object> vp = Json.array();
        List<Object> players = Json.array();
        for (int seat = 0; seat < game.maxPlayers; seat++)
        {
            SOCPlayer player = game.getPlayer(osp.playerOf(seat));
            vp.add((long) player.getTotalVP());
            players.add(player.getName());
        }
        SOCPlayer won = game.getPlayerWithWin();
        Object winner = won == null ? null : (Object) (long) osp.seatOf(won.getPlayerNumber());
        write(Json.object("type", "after", "game_id", gameId, "winner", winner, "vp", vp, "truncated", winner == null,
            "rounds", (long) game.getRoundCount(), "state", state));
        List<Object> moveCounts = Json.array(), choiceCounts = Json.array(), cpu = Json.array();
        Map<String, Long> millis = brainMillis();
        for (int seat = 0; seat < game.maxPlayers; seat++)
        {
            moveCounts.add((long) moves[seat]);
            choiceCounts.add((long) choices[seat]);
            cpu.add(millis.get((String) players.get(seat)));
        }
        System.err.println(Json.write(Json.object("type", "osp-game", "game_id", gameId, "players", players, "winner", winner, "vp", vp,
            "rounds", (long) game.getRoundCount(), "moves", moveCounts, "choices", choiceCounts, "brain_cpu_ms", cpu, "gaps", (long) gaps)));
        close();
    }

    // The CPU time of each robot brain's thread in this game so far, by robot name (JSettlers' brains and ours name their
    // threads robotBrain- or ospBrain- with the robot and the game)
    private Map<String, Long> brainMillis()
    {
        Map<String, Long> millis = new LinkedHashMap<>();
        ThreadMXBean threads = ManagementFactory.getThreadMXBean();
        for (Thread thread : Thread.getAllStackTraces().keySet())
        {
            String name = thread.getName();
            String suffix = "-" + gameId;
            for (String prefix : new String[] {"robotBrain-", "ospBrain-"})
                if (name.startsWith(prefix) && name.endsWith(suffix))
                {
                    long nanos = threads.getThreadCpuTime(thread.getId());
                    if (nanos >= 0)
                        millis.put(name.substring(prefix.length(), name.length() - suffix.length()), nanos / 1_000_000);
                }
        }
        return millis;
    }

    private void write(Map<String, Object> message)
    {
        if (record == null)
            return;
        try
        {
            record.write(Json.write(message));
            record.write('\n');
            record.flush();
        }
        catch (IOException e)
        {
            System.err.println("osp: can't record " + gameId + ": " + e);
            close();
        }
    }

    /** The game is gone: a record of a game that ended without a winner gets its end */
    synchronized void close()
    {
        if (record == null)
            return;
        try
        {
            if (! ended)
            {
                ended = true;
                record.write(Json.write(Json.object("type", "after", "game_id", gameId, "winner", null, "vp", null, "truncated", true)));
                record.write('\n');
            }
            record.close();
        }
        catch (IOException e) {}
        record = null;
    }
}
