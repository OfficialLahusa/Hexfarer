package osp.stacsettlers;

import java.io.File;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import soc.game.SOCCity;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;
import soc.game.SOCRoad;
import soc.game.SOCSettlement;
import soc.dialogue.StacDialogueManager;
import soc.dialogue.StacTradeMessage;
import soc.message.SOCGameTextMsg;
import soc.message.SOCMessage;
import soc.robot.SOCRobotBrainImpl;
import soc.robot.stac.StacRobotDialogueManager;
import soc.robot.SOCRobotClient;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * The brain of a StacSettlers robot whose decisions an Open Seat Agent makes (see ../../PROTOCOL.md). It runs inside
 * StacSettlers' simulation (OSPSimulation, whose server is OSPServer) and reads the true game there, under the lock the
 * server handles every request with; whenever the game waits for its seat, it asks the agent with the state and the
 * legal moves as OSPGame writes them (the true state for an open-hand agent, the seat's view otherwise) and plays the
 * answer through the robot client's requests, a protocol move that takes StacSettlers two steps (a bought piece and its
 * place, a card and its resources, a robber move and its victim) one after the other. It keeps none of StacSettlers' own
 * robot logic or game tracking, and it takes no part in the trades the other robots negotiate in chat: it never announces
 * that it participates (so Stac agents don't wait for its answers) and declines every offer made to it (Dobre's MCTS
 * agents wait for the answer).
 *
 * It speaks version 2: it serves view and truth (no worlds), steps and lean decisions. An agent that observes gets every
 * move of the game as a step from the game's start (the server's move log, MoveLog), before each decision and before the
 * game's end, as the referee knows it for an agent that needs the truth (with the state before each move), else as the
 * seat sees it (a card bought by another seat unnamed, a card stolen between two others unknown); one that doesn't has to
 * take its game from each decision's state, and decisions are numbered by the game's moves either way. A lean agent gets
 * the full state again when it asks, and on its own after a change no step explains. The agent takes each game (ready) or
 * turns it down; a decision has the budget's time (120 s). Whatever fails (a game turned down, an illegal, late or
 * malformed answer, the agent's own failure, its process ending) is told to the agent where it still listens and
 * recorded in the seat's line of the game's result record (GameRecord), and the robot plays the first legal move for the
 * rest of the game (fallbacks). A move the server doesn't take within STALL_MILLIS (a rule the robot got wrong) counts as
 * a stall: the robot leaves that state's move out and decides again, or cancels a half-made move, so the game goes on.
 * The agent's command comes from the system property osp.agent, its standard error goes to the directory osp.log if set;
 * osp.debug prints each move the robot plays.
 */
public class OSPRobotBrain extends SOCRobotBrainImpl
{
    // The game's budget: how long the agent may think about one decision; the handshake and ready get a minute
    private static final long DECIDE_MILLIS = 120_000, HANDSHAKE_MILLIS = 60_000;
    // Any threads and a GPU allowed, as at every referee unless the game says otherwise
    static final Map<String, Object> BUDGET = Json.object("decide_ms", DECIDE_MILLIS, "game_ms", null, "threads", null, "gpu", true);
    // How long a move may wait for the server before it counts as not taken
    private static final long STALL_MILLIS = Long.getLong("osp.stall_ms", 20_000L);

    private OSPGame osp;
    private GameRecord record;
    private AgentProcess agent;
    private boolean agentFailed, truth, lean, observe, after;
    // The game's move log, the steps the agent has been sent, and the log's gaps at the last decision (a change no step
    // explains sends a lean agent the full state)
    private MoveLog log;
    private int sent;
    private long gapsAtDecision;
    private String agentName = "?";
    // The game as it was when the seat last sent a request (its state and position): the next decision waits until the
    // server took that request and the game went on
    private String actedOn;
    private long actedAt;
    // The second step of a two-step move, and the game state its first step was sent in
    private List<Object> pending;
    private int pendingFrom;
    // The moves the server didn't take, by the position they were tried in
    private final Set<String> refused = new HashSet<>();
    private boolean pinged;

    public OSPRobotBrain(SOCRobotClient rc, SOCRobotParameters params, SOCGame ga, CappedQueue mq)
    {
        super(rc, params, ga, mq);
    }

    /** Before the server removes the game: its end for the agent, while the record still has it */
    synchronized void gameEnding(String name)
    {
        if (osp != null && game != null && name.equals(game.getName()))
            finish();
    }

    @Override
    public void run()
    {
        OSPServer.BRAINS.add(this);
        final String name = game.getName();
        try
        {
            Thread.currentThread().setName("ospBrain-" + client.getNickname() + "-" + name);
        }
        catch (Throwable th) {}
        // StacSettlers' pinger sends the brain a message once a second, so it looks at the game again even when no game
        // message comes (after it read the game while the server changed it, say)
        if (pinger != null)
            pinger.start();
        try
        {
            while (alive)
            {
                Object message = gameEventQ.get();
                if (message == null)
                    break;
                int type = ((SOCMessage) message).getType();
                if (type == SOCMessage.ROBOTDISMISS)
                {
                    client.leaveGame(game, "dismiss msg", false);
                    alive = false;
                    break;
                }
                if (type == SOCMessage.GAMETEXTMSG)
                {
                    if (! pinged)
                    {
                        pinged = true;
                        debug("pinged", null);
                    }
                    declineOffer((SOCGameTextMsg) message);
                }
                try
                {
                    // Read outside the brain's lock: the server's thread may wait for it (gameEnding) while it holds a lock
                    Seen seen = look();
                    if (seen != null)
                        synchronized (this)
                        {
                            act(seen);
                        }
                }
                catch (Exception e)
                {
                    System.err.println(client.getNickname() + " in " + name + ": " + e);
                    e.printStackTrace();
                }
            }
        }
        catch (Exception e)
        {
            System.err.println(client.getNickname() + " in " + name + " stopped: " + e);
        }
        finally
        {
            synchronized (this)
            {
                if (osp != null)
                    finish();
                close();
            }
            if (pinger != null)
                pinger.stopPinger();
            OSPServer.BRAINS.remove(this);
        }
    }

    /**
     * Declines a trade offer made to this seat in the game's chat (StacSettlers' agents negotiate there): the seat takes
     * no part in trades between players. Stac agents offer to every player but wait only for those that announced they
     * participate, which this robot never does; Dobre's MCTS agents wait for the answer, so it answers with
     * StacSettlers' own reject message, as their agents that don't trade do.
     */
    private void declineOffer(SOCGameTextMsg chat)
    {
        String text = chat.getText();
        if (text == null || ! text.startsWith(StacTradeMessage.TRADE) || client.getNickname().equals(chat.getNickname()) || ourPlayerData == null)
            return;
        try
        {
            StacTradeMessage trade = StacTradeMessage.parse(StacDialogueManager.fromMessage(text));
            int us = ourPlayerData.getPlayerNumber();
            if (trade.getOffer() == null || trade.getSenderInt() == us || ! java.util.Arrays.asList(trade.getReceivers().split(",")).contains(Integer.toString(us)))
                return;
            client.sendText(game, StacRobotDialogueManager.makeRejectMessage(us, trade.getSenderInt(), "No, thanks."));
            if (record != null)
                record.seat(osp.seatOf(us)).declined();
            debug("declined", Json.object("from", (long) trade.getSenderInt(), "offer", text));
        }
        catch (RuntimeException e)
        {
            System.err.println(client.getNickname() + " in " + game.getName() + ": can't read the trade message " + text + ": " + e);
        }
    }

    /** What the brain read of the true game at one moment */
    private static final class Seen
    {
        int state, acting;
        boolean over, deciding;
        // The game's state and position, which changes with every move the server takes
        String key;
        // The state the next move starts from (in the protocol's order of discards)
        Map<String, Object> position;
        // The game's moves so far, its gaps, and the steps the agent hasn't been sent
        long moves, gaps;
        List<Map<String, Object>> steps;
        List<Object> legal;
    }

    // Reads the true game under the server's locks, or null if there's no game to play yet
    private Seen look()
    {
        final SOCGame truthGame = OSPServer.game(game.getName());
        if (truthGame == null || truthGame.getGameState() < SOCGame.START1A || ourPlayerData == null)
            return null;
        return OSPServer.read(truthGame, () ->
        {
            // StacSettlers empties a game it destroys
            if (truthGame.getPlayer(0) == null || truthGame.getBoard() == null)
                return null;
            if (osp == null)
            {
                osp = new OSPGame(truthGame);
                record = OSPServer.recordOf(truthGame);
                log = OSPServer.logIn(truthGame);
            }
            Seen seen = new Seen();
            seen.over = osp.isOver();
            if (seen.over)
                record.ended(truthGame);
            seen.state = truthGame.getGameState();
            seen.position = log.state();
            seen.moves = log.size();
            seen.gaps = log.gaps();
            seen.steps = log.stepsFrom(sent);
            seen.key = seen.state + "/" + OSPGame.hash(seen.position);
            seen.acting = osp.acting();
            seen.deciding = osp.deciding();
            int seat = osp.seatOf(ourPlayerNumber());
            seen.legal = seen.acting == seat && ! seen.over ? osp.legal(seat) : null;
            return seen;
        });
    }

    private int ourPlayerNumber()
    {
        return ourPlayerData.getPlayerNumber();
    }

    // Plays the seat's move if the true game waits for it
    private void act(Seen seen) throws Exception
    {
        int seat = osp.seatOf(ourPlayerNumber());
        if (seen.over)
        {
            if (record.end() != null)
                finish();
            return;
        }
        if (seen.acting != seat)
        {
            pending = null;
            return;
        }
        int state = seen.state;
        // The second step of a two-step move, once the server asks for it; until the server took the first step the game
        // stays in the state the first step was sent in
        if (pending != null)
        {
            if (secondStep(pending, state))
            {
                pending = null;
                acted(seen);
                return;
            }
            if (state == pendingFrom)
            {
                stalled(seen);
                return;
            }
            // The server went elsewhere (the move needed no second step): choose anew
            pending = null;
        }
        // Until the server took the last request the game stays as it was then
        if (seen.key.equals(actedOn) || ! seen.deciding)
        {
            stalled(seen);
            return;
        }
        List<Object> legal = seen.legal;
        if (! refused.isEmpty())
            legal.removeIf(move -> refused.contains(seen.key + Json.write(move, true)));
        if (legal.isEmpty())
            return;
        record.seat(seat).robotDecided();
        @SuppressWarnings("unchecked")
        List<Object> move = (List<Object>) (legal.size() == 1 ? legal.get(0) : ask(seat, legal, seen));
        pendingFrom = state;
        acted(seen);
        decided = seen.key + Json.write(move, true);
        firstStep(move, state);
    }

    // The last move decided, with the position it was decided in: a move the server stopped halfway isn't tried again there
    private String decided;

    // The seat sent a request in this position: the next decision waits until the game went on
    private void acted(Seen seen)
    {
        actedOn = seen.key;
        actedAt = System.currentTimeMillis();
        lastMove = null;
    }

    // The last move sent, until the game went on (for a stall)
    private List<Object> lastMove;

    // The game hasn't gone on since the seat's last request: past STALL_MILLIS the server didn't take it, which is
    // counted, and the robot gets the game going again
    private void stalled(Seen seen)
    {
        if (actedOn == null || System.currentTimeMillis() - actedAt < STALL_MILLIS)
            return;
        record.seat(osp.seatOf(ourPlayerNumber())).stalled();
        System.err.println(client.getNickname() + " in " + game.getName() + ": the server didn't take " + Json.write(lastMove)
            + " in game state " + seen.state + " within " + STALL_MILLIS + " ms");
        debug("stall", Json.object("state", (long) seen.state, "move", lastMove));
        actedOn = null;
        pending = null;
        if (seen.deciding)
        {
            // Decided again, without that move if it was a whole one (a second step leaves nothing to rule out)
            if (lastMove != null)
                refused.add(seen.key + Json.write(lastMove, true));
            return;
        }
        // Halfway through a move: its first step undone (the move left out when the robot decides there again), or its
        // second made with the first choice there is
        if (decided != null)
            refused.add(decided);
        switch (seen.state)
        {
        case SOCGame.PLACING_ROAD: client.cancelBuildRequest(game, SOCPlayingPiece.ROAD); break;
        case SOCGame.PLACING_SETTLEMENT: client.cancelBuildRequest(game, SOCPlayingPiece.SETTLEMENT); break;
        case SOCGame.PLACING_CITY: client.cancelBuildRequest(game, SOCPlayingPiece.CITY); break;
        case SOCGame.WAITING_FOR_DISCOVERY: client.discoveryPick(game, new SOCResourceSet(1, 1, 0, 0, 0, 0)); break;
        case SOCGame.WAITING_FOR_MONOPOLY: client.monopolyPick(game, SOCResourceConstants.CLAY); break;
        case SOCGame.WAITING_FOR_CHOICE:
        {
            Object victim = osp.game.getPossibleVictims().isEmpty() ? null : osp.game.getPossibleVictims().get(0);
            if (victim != null)
                client.choosePlayer(game, ((soc.game.SOCPlayer) victim).getPlayerNumber());
            break;
        }
        default:
            break;
        }
        acted(seen);
    }

    /** A failure of the agent's seat: its code (see PROTOCOL.md) and whether the agent reported it itself */
    private static final class AgentFailure extends Exception
    {
        final String code;
        final boolean byAgent;

        AgentFailure(String code, String message, boolean byAgent)
        {
            super(message);
            this.code = code;
            this.byAgent = byAgent;
        }
    }

    // Tells the agent what failed (if it still listens) and returns the failure
    private AgentFailure fail(Object n, String code, String message, Map<String, Object> detail)
    {
        try
        {
            Map<String, Object> failure = Json.object("type", "failure", "game_id", game.getName(), "n", n, "code", code, "message", message);
            if (detail != null)
                failure.put("detail", detail);
            agent.send(failure);
        }
        catch (Exception e) {}
        return new AgentFailure(code, message, false);
    }

    // The agent's own failure, as the exception for it
    private static AgentFailure reported(Map<String, Object> failure)
    {
        Object code = failure.get("code");
        return new AgentFailure(code instanceof String ? (String) code : "internal", "the agent reported a failure: " + failure.get("message"), true);
    }

    // Asks the agent; the first legal move once it failed
    private Object ask(int seat, List<Object> legal, Seen seen)
    {
        long started = System.currentTimeMillis();
        Long n = seen.moves;
        try
        {
            if (agent == null && ! agentFailed)
                start(seat, seen);
            if (agent == null)
                return fallback(legal);
            sendSteps(seat, seen.steps);
            Map<String, Object> state = truth ? seen.position : OSPGame.hide(seen.position, seat);
            // A lean agent gets only the hash, unless a change no step explains came since its last decision
            boolean force = seen.gaps != gapsAtDecision;
            gapsAtDecision = seen.gaps;
            Map<String, Object> decide = Json.object("type", "decide", "game_id", game.getName(), "n", n, "seat", (long) seat, "hash", OSPGame.hash(state));
            if (! lean || force)
                decide.put("state", state);
            decide.put("legal", legal);
            agent.send(decide);
            while (true)
            {
                long left = DECIDE_MILLIS - (System.currentTimeMillis() - started);
                Map<String, Object> answer;
                try
                {
                    if (left <= 0)
                        throw new AgentProcess.Timeout("no time left");
                    answer = agent.receive(left);
                }
                catch (AgentProcess.Timeout e)
                {
                    throw fail(n, "timeout", "the agent gave no answer within " + DECIDE_MILLIS + " ms", null);
                }
                Object type = answer.get("type");
                if ("failure".equals(type))
                    throw reported(answer);
                if ("resync".equals(type) && n.equals(answer.get("n")))
                {
                    record.seat(seat).resynced();
                    Map<String, Object> full = new LinkedHashMap<>(decide);
                    full.put("state", state);
                    agent.send(full);
                    continue;
                }
                if (! "decide".equals(type) || ! answer.containsKey("action"))
                    throw fail(n, "malformed", "the agent answered a decision with " + Json.write(answer), null);
                if (! n.equals(answer.get("n")))
                    throw fail(n, "illegal", "the agent answered decision " + answer.get("n") + " when asked for " + n, null);
                String chosen = Json.write(answer.get("action"), true);
                for (Object move : legal)
                    if (Json.write(move, true).equals(chosen))
                    {
                        record.seat(seat).decided(System.currentTimeMillis() - started);
                        return move;
                    }
                throw fail(n, "illegal", "the agent answered " + chosen + ", not a legal move", null);
            }
        }
        catch (AgentFailure e)
        {
            return failed(seat, n, e.code, e.byAgent, e.getMessage(), legal);
        }
        catch (Exception e)
        {
            // Its process ended (or broke off): nothing to tell it
            return failed(seat, n, "ended", false, e.getMessage(), legal);
        }
    }

    // The agent failed: recorded, closed, and the robot plays on with the first legal move
    private Object failed(int seat, Long n, String code, boolean byAgent, String message, List<Object> legal)
    {
        System.err.println(client.getNickname() + " in " + game.getName() + ": the agent failed (" + code + "): " + message);
        record.seat(seat).failed(n, code, byAgent, message);
        agentFailed = true;
        close();
        return fallback(legal);
    }

    private Object fallback(List<Object> legal)
    {
        record.seat(osp.seatOf(ourPlayerNumber())).fellBack();
        return legal.get(0);
    }

    // Starts the agent and the game for it: the handshake (what it plays and wants), before, and its ready
    private void start(int seat, Seen seen) throws Exception
    {
        String command = System.getProperty("osp.agent");
        if (command == null)
            throw new IllegalStateException("No agent: set the system property osp.agent");
        String logs = System.getProperty("osp.log");
        File errors = logs == null ? null : new File(logs, game.getName() + "-" + client.getNickname() + ".log");
        agent = new AgentProcess(command, errors);
        agent.send(Json.object("type", "handshake", "protocol", "osp", "version", 2L));
        Map<String, Object> handshake = receiveOrTimeout(HANDSHAKE_MILLIS, null);
        if ("failure".equals(handshake.get("type")))
            throw reported(handshake);
        if (! "handshake".equals(handshake.get("type")) || ! "osp".equals(handshake.get("protocol")) || ! Long.valueOf(2).equals(handshake.get("version")))
            throw fail(null, "version", "the agent didn't answer the handshake in protocol osp version 2: " + Json.write(handshake), null);
        agentName = String.valueOf(handshake.get("name"));
        Object needs = handshake.containsKey("needs") ? handshake.get("needs") : "view";
        Object observing = handshake.containsKey("observe") ? handshake.get("observe") : false, leaning = handshake.containsKey("lean") ? handshake.get("lean") : false;
        Object rulesets = handshake.get("rulesets");
        record.seat(seat).agent(agentName, needs instanceof String ? (String) needs : null, Boolean.TRUE.equals(observing), Boolean.TRUE.equals(leaning));
        if (! (observing instanceof Boolean) || ! (leaning instanceof Boolean) || ! (rulesets == null || rulesets instanceof List))
            throw fail(null, "malformed", "the agent's handshake has a field of the wrong type: " + Json.write(handshake), null);
        // What this referee can't serve, it refuses instead of sending something else
        if (! "view".equals(needs) && ! "truth".equals(needs))
            throw fail(null, "unsupported", "this robot serves view and truth, not " + Json.write(needs) + ": it draws no worlds", null);
        if (rulesets == null || ! ((List<?>) rulesets).contains("standard"))
            throw fail(null, "unsupported", "StacSettlers plays the standard preset with options, the agent plays " + Json.write(rulesets), Json.object("preset", "standard"));
        truth = "truth".equals(needs);
        lean = (Boolean) leaning;
        if (lean && ! (Boolean) observing)
            throw fail(null, "unsupported", "a lean agent has to observe: it keeps its own game from the steps", null);
        // An observing agent follows the game from its start; one that doesn't starts from now
        observe = (Boolean) observing;
        gapsAtDecision = seen.gaps;
        Map<String, Object> from = observe ? log.start() : seen.position;
        if (! observe)
            sent = (int) seen.moves;
        agent.send(Json.object("type", "before", "game_id", game.getName(), "seat", (long) seat, "players", (long) game.maxPlayers,
            "ruleset", OSPGame.ruleset(), "budget", BUDGET, "seed", (long) record.seed, "board", osp.board.boardJson(),
            "state", truth ? from : OSPGame.hide(from, seat)));
        // The agent takes the game or turns it down
        Map<String, Object> ready = receiveOrTimeout(HANDSHAKE_MILLIS, null);
        if ("failure".equals(ready.get("type")))
            throw reported(ready);
        if (! "ready".equals(ready.get("type")) || ! game.getName().equals(ready.get("game_id")))
            throw fail(null, "malformed", "the agent answered before with " + Json.write(ready), null);
    }

    // The agent's next message within the time, or a timeout failure
    private Map<String, Object> receiveOrTimeout(long millis, Object n) throws Exception
    {
        try
        {
            return agent.receive(millis);
        }
        catch (AgentProcess.Timeout e)
        {
            throw fail(n, "timeout", "the agent gave no answer within " + millis + " ms", null);
        }
    }

    private static final boolean DEBUG = System.getProperty("osp.debug") != null;

    // A debug line: [osp ROBOT GAME] WHAT {details}
    private void debug(String what, Map<String, Object> details)
    {
        if (DEBUG)
            System.err.println("[osp " + client.getNickname() + " " + game.getName() + "] " + what + (details == null ? "" : " " + Json.write(details)));
    }

    // Plays a protocol move, or its first step
    @SuppressWarnings("unchecked")
    private void firstStep(List<Object> move, int state)
    {
        debug("move", Json.object("state", (long) state, "move", move));
        lastMove = move;
        String type = (String) move.get(0);
        Object value = move.size() > 1 ? move.get(1) : null;
        boolean placingFree = state < SOCGame.PLAY || state == SOCGame.PLACING_FREE_ROAD1 || state == SOCGame.PLACING_FREE_ROAD2;
        switch (type)
        {
        case "roll": client.rollDice(game); break;
        case "end_turn": client.endTurn(game); break;
        case "build_settlement":
            if (placingFree)
                client.putPiece(game, new SOCSettlement(ourPlayerData, osp.board.node(value), game.getBoard()));
            else
            {
                pending = move;
                client.buildRequest(game, SOCPlayingPiece.SETTLEMENT);
            }
            break;
        case "build_road":
            if (placingFree)
                client.putPiece(game, new SOCRoad(ourPlayerData, osp.board.edge(value), game.getBoard()));
            else
            {
                pending = move;
                client.buildRequest(game, SOCPlayingPiece.ROAD);
            }
            break;
        case "build_city":
            pending = move;
            client.buildRequest(game, SOCPlayingPiece.CITY);
            break;
        case "buy_development": client.buyDevCard(game); break;
        // StacSettlers' card numbers (JSettlers 1.1.16's, not JSettlers 2's)
        case "play_knight": client.playDevCard(game, SOCDevCardConstants.KNIGHT); break;
        case "play_road_building": client.playDevCard(game, SOCDevCardConstants.ROADS); break;
        case "play_year_of_plenty":
            pending = move;
            client.playDevCard(game, SOCDevCardConstants.DISC);
            break;
        case "play_monopoly":
            pending = move;
            client.playDevCard(game, SOCDevCardConstants.MONO);
            break;
        case "move_robber":
        {
            Map<String, Object> robbery = (Map<String, Object>) value;
            if (robbery.get("victim") != null)
                pending = move;
            client.moveRobber(game, ourPlayerData, osp.board.hexCoord((List<?>) robbery.get("hex")));
            break;
        }
        case "discard": client.discard(game, OSPGame.resourceSet(value)); break;
        case "bank_trade":
        {
            Map<String, Object> trade = (Map<String, Object>) value;
            client.bankTrade(game, OSPGame.resourceSet(trade.get("give")), OSPGame.resourceSet(trade.get("get")));
            break;
        }
        default:
            System.err.println(client.getNickname() + " in " + game.getName() + ": no StacSettlers request for " + type);
        }
    }

    // Plays the second step of a pending move if the server waits for it; returns whether it did
    @SuppressWarnings("unchecked")
    private boolean secondStep(List<Object> move, int state)
    {
        String type = (String) move.get(0);
        Object value = move.get(1);
        switch (type)
        {
        case "build_settlement":
            if (state != SOCGame.PLACING_SETTLEMENT) return false;
            client.putPiece(game, new SOCSettlement(ourPlayerData, osp.board.node(value), game.getBoard()));
            break;
        case "build_road":
            if (state != SOCGame.PLACING_ROAD) return false;
            client.putPiece(game, new SOCRoad(ourPlayerData, osp.board.edge(value), game.getBoard()));
            break;
        case "build_city":
            if (state != SOCGame.PLACING_CITY) return false;
            client.putPiece(game, new SOCCity(ourPlayerData, osp.board.node(value), game.getBoard()));
            break;
        case "play_year_of_plenty":
        {
            if (state != SOCGame.WAITING_FOR_DISCOVERY) return false;
            int[] counts = new int[SOCResourceConstants.WOOD + 1];
            for (Object name : (List<Object>) value)
                counts[OSPGame.resourceType((String) name)]++;
            client.discoveryPick(game, OSPGame.resourceSet(OSPGame.cardSet(counts)));
            break;
        }
        case "play_monopoly":
            if (state != SOCGame.WAITING_FOR_MONOPOLY) return false;
            client.monopolyPick(game, OSPGame.resourceType((String) value));
            break;
        case "move_robber":
            if (state != SOCGame.WAITING_FOR_CHOICE) return false;
            client.choosePlayer(game, osp.playerOf(Json.integer(((Map<String, Object>) value).get("victim"))));
            break;
        default:
            return false;
        }
        debug("second", Json.object("state", (long) state, "move", move));
        return true;
    }

    // The moves the agent hasn't been sent, if it observes: as the referee knows them for an agent that needs the truth
    // (with the state before each move), else as the seat sees them (PROTOCOL.md, moves)
    @SuppressWarnings("unchecked")
    private void sendSteps(int seat, List<Map<String, Object>> steps) throws Exception
    {
        if (steps == null)
            return;
        for (Map<String, Object> step : steps)
        {
            sent++;
            if (! observe)
                continue;
            Map<String, Object> message = OSPGame.copy(step);
            if (! truth)
            {
                message.remove("state");
                int mover = Json.integer(message.get("seat"));
                List<Object> action = (List<Object>) message.get("action");
                Map<String, Object> result = (Map<String, Object>) message.get("result");
                if (action.get(0).equals("buy_development") && mover != seat)
                    result.put("card", null);
                if (action.get(0).equals("move_robber") && result != null && mover != seat
                    && Json.integer(((Map<String, Object>) action.get(1)).get("victim")) != seat)
                    result.put("stolen", "unknown");
            }
            agent.send(message);
        }
    }

    // Tells the agent the game's end: with the record's result once the game ended with a winner, else cut short
    private void finish()
    {
        if (after)
            return;
        Map<String, Object> end = record == null ? null : record.end();
        after = true;
        try
        {
            if (agent != null)
            {
                // The moves since the last decision, up to the end (the log's steps are read without the server's lock)
                if (log != null)
                    sendSteps(osp.seatOf(ourPlayerData.getPlayerNumber()), log.stepsFrom(sent));
                Map<String, Object> message = Json.object("type", "after", "game_id", game.getName());
                if (end != null)
                    message.putAll(end);
                else
                    message.putAll(Json.object("winner", null, "vp", null, "truncated", true));
                agent.send(message);
            }
        }
        catch (Exception e) {}
        close();
    }

    private void close()
    {
        if (agent != null)
        {
            agent.close();
            agent = null;
        }
    }
}
