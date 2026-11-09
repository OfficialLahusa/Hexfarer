package osp.stacsettlers;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
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
import soc.game.StacTradeOffer;
import soc.dialogue.StacDialogueManager;
import soc.dialogue.StacTradeMessage;
import soc.message.SOCGameTextMsg;
import soc.message.SOCMessage;
import soc.robot.SOCRobotBrainImpl;
import soc.robot.stac.Persuasion;
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
 * robot logic or game tracking. By default it takes no part in the trades the other robots negotiate in chat: it never
 * announces that it participates (so Stac agents don't wait for its answers) and declines every offer made to it (Dobre's
 * MCTS agents wait for the answer).
 *
 * With osp.trade_in_chat it takes part, in StacSettlers' chat format: it announces that it participates (PARTICIPATE),
 * so the Stac agents wait for its answer to every offer, with no time limit; it owes them NULL:ST once a turn and an
 * answer to every offer (NO_RESPONSE to one not made to it), which it gives itself, also once its agent failed. An offer
 * made to its seat becomes a decision (prompt answer_offer, with the offer in the state): accept (ACC; the offerer then
 * picks among its accepts and confirms one, which the server carries out), decline (REJ) or a counteroffer (an offer of
 * the seat's to the offerer, whose accept the seat confirms, having written the terms). On its own turn after the roll
 * its agent may offer (trades in the decision, at most OFFERS_PER_TURN a turn): the first accept trades at once (the seat
 * confirms it, player_trading first_accept), counteroffers wait until every addressee answered (OFFER_WAIT_MILLIS at
 * most), then the agent takes one (prompt pick_partner) or withdraws. A trade that happens is the move log's offer and
 * accept on what changed hands; the negotiation around it makes no step.
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

    // Taking part in the Stac agents' chat trades (see above)
    static final boolean TRADE_IN_CHAT = Boolean.getBoolean("osp.trade_in_chat");
    // How long the seat's own offer waits for its addressees' answers (missing ones then count as declines), and for the
    // server to carry out a trade the seat confirmed
    private static final long OFFER_WAIT_MILLIS = Long.getLong("osp.offer_wait_ms", 20_000L), TRADE_WAIT_MILLIS = 5_000L;
    // Offers the seat may make in one turn (the ruleset's offer_limit; its agent keeps its own, lower limit)
    static final int OFFERS_PER_TURN = 10;
    // The turn (round and player on turn) the robot last sent NULL:ST for, the turn its offers so far were made in and
    // their number
    private String stSentFor, offersTurn;
    private int offersThisTurn;
    // The seat's own offer while it stands, and the StacSettlers player a counteroffer of the seat's went to (-1: none)
    private OwnOffer own;
    private int counterTo = -1;

    /** The seat's own offer in chat: its terms and addressees (seats), their answers and counteroffers as they come */
    private static final class OwnOffer
    {
        final Map<String, Object> give, get;
        final List<Object> to;
        // By seat (as a string): accept, decline or counter; the counteroffers in the offerer's terms
        final Map<String, Object> answers = new LinkedHashMap<>(), counters = new LinkedHashMap<>();
        // The position it was made in, and when
        final String key;
        final long at = System.currentTimeMillis();
        // The seat the seat's confirming accept went to (-1 before), and when
        int partner = -1;
        long confirmedAt;

        OwnOffer(Map<String, Object> give, Map<String, Object> get, List<Object> to, String key)
        {
            this.give = give;
            this.get = get;
            this.to = to;
            this.key = key;
        }

        boolean addressed(int seat)
        {
            return to.contains((long) seat);
        }

        boolean answered()
        {
            return answers.size() >= to.size();
        }
    }

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
                SOCGameTextMsg chat = null;
                if (type == SOCMessage.GAMETEXTMSG)
                {
                    if (! pinged)
                    {
                        pinged = true;
                        debug("pinged", null);
                    }
                    if (TRADE_IN_CHAT)
                        chat = (SOCGameTextMsg) message;
                    else
                        declineOffer((SOCGameTextMsg) message);
                }
                try
                {
                    // Read outside the brain's lock: the server's thread may wait for it (gameEnding) while it holds a lock
                    Seen seen = look();
                    synchronized (this)
                    {
                        // A trade message is answered even without a game to look at (the robot owes every offer one)
                        if (chat != null)
                            tradeMessage(chat, seen);
                        if (seen != null)
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

    /** At the game's start, taking part in the chat trades: the Stac agents then wait for the robot's answers */
    @Override
    public void startGameChat()
    {
        if (TRADE_IN_CHAT)
            send(StacRobotDialogueManager.ANN_PARTICIPATION);
    }

    // A chat line to the game
    private void send(String text)
    {
        debug("chat", Json.object("text", text));
        client.sendText(game, text);
    }

    private static String accept(int sender, int receiver)
    {
        return StacRobotDialogueManager.makeAcceptMessage(sender, receiver, "Deal.");
    }

    private static String reject(int sender, int receiver)
    {
        return StacRobotDialogueManager.makeRejectMessage(sender, receiver, "No, thanks.");
    }

    private static String noResponse(int sender, int receiver)
    {
        return new StacTradeMessage(Integer.toString(sender), Integer.toString(receiver), false, false, false, false, null, true, "Nothing from me.").toMessage();
    }

    // An offer from the player to those named, giving give and getting get (StacSettlers' resources)
    private String offerMessage(int from, boolean[] to, SOCResourceSet give, SOCResourceSet get)
    {
        StacTradeOffer offer = new StacTradeOffer(game.getName(), from, to, give, false, get, false);
        return new StacTradeMessage(Integer.toString(from), StacTradeMessage.getToAsString(to), offer, false, new Persuasion(), "Would you trade?").toMessage();
    }

    // Once a turn, at its first PLAY1 (after the roll and the robber), NULL:ST, which every Stac agent waits for from each
    // participant before the player on turn goes on; a new turn also starts the seat's count of offers afresh
    private void turnDuties(Seen seen)
    {
        if (! seen.turn.equals(offersTurn))
        {
            offersTurn = seen.turn;
            offersThisTurn = 0;
            counterTo = -1;
        }
        if (seen.state == SOCGame.PLAY1 && ! seen.turn.equals(stSentFor))
        {
            stSentFor = seen.turn;
            send("NULL:ST");
        }
    }

    // The seat's offer made: in chat to the seats it names
    private void offer(int seat, Map<String, Object> terms, Seen seen)
    {
        @SuppressWarnings("unchecked")
        List<Object> seats = (List<Object>) terms.get("to");
        boolean[] to = new boolean[game.maxPlayers];
        for (Object s : seats)
            to[osp.playerOf(Json.integer(s))] = true;
        @SuppressWarnings("unchecked")
        Map<String, Object> give = (Map<String, Object>) terms.get("give"), get = (Map<String, Object>) terms.get("get");
        send(offerMessage(ourPlayerNumber(), to, OSPGame.resourceSet(give), OSPGame.resourceSet(get)));
        own = new OwnOffer(give, get, new ArrayList<>(seats), seen.key);
        offersThisTurn++;
        record.seat(seat).offered();
    }

    // The seat's own offer while it stands: done once the server carried out the trade the seat confirmed (or gave up on
    // it); once every addressee answered (or the time is up), the agent takes a counteroffer or withdraws, else it is over
    private void standingOffer(int seat, Seen seen)
    {
        long now = System.currentTimeMillis();
        if (own.partner >= 0)
        {
            if (! seen.key.equals(own.key) || now - own.confirmedAt > TRADE_WAIT_MILLIS)
                own = null;
            return;
        }
        if (! own.answered() && now - own.at < OFFER_WAIT_MILLIS)
            return;
        for (Object s : own.to)
            own.answers.putIfAbsent(String.valueOf(s), "decline");
        if (own.counters.isEmpty())
        {
            own = null;
            return;
        }
        // Withdrawing first: the move a failed agent's robot plays
        List<Object> legal = new ArrayList<>();
        legal.add(Json.array("cancel"));
        for (String s : own.counters.keySet())
            legal.add(Json.array("confirm", Json.object("partner", Long.valueOf(s))));
        Map<String, Object> state = OSPGame.copy(seen.position);
        state.put("prompt", "pick_partner");
        state.put("current", (long) seat);
        state.put("offer", Json.object("from", (long) seat, "give", own.give, "get", own.get, "to", own.to, "answers", own.answers, "counters", own.counters));
        @SuppressWarnings("unchecked")
        List<Object> move = (List<Object>) ask(seat, legal, seen, state, null);
        if ("confirm".equals(move.get(0)))
        {
            @SuppressWarnings("unchecked")
            Map<String, Object> pick = (Map<String, Object>) move.get(1);
            confirm(Json.integer(pick.get("partner")));
            return;
        }
        int us = ourPlayerNumber();
        for (String s : own.counters.keySet())
            send(reject(us, osp.playerOf(Integer.parseInt(s))));
        own = null;
    }

    // The seat confirms a trade on its offer: an accept of it (the seat wrote the terms, so its accept carries the trade
    // out), or a counteroffer it takes (its accept, which the counterer then confirms); the other counteroffers declined
    private void confirm(int partner)
    {
        int us = ourPlayerNumber();
        send(accept(us, osp.playerOf(partner)));
        own.partner = partner;
        own.confirmedAt = System.currentTimeMillis();
        for (String s : own.counters.keySet())
            if (Integer.parseInt(s) != partner)
                send(reject(us, osp.playerOf(Integer.parseInt(s))));
    }

    /**
     * A trade line in the game's chat, taking part in the chat trades (TRADE_IN_CHAT): an offer not made to the seat gets
     * NO_RESPONSE; on the seat's own turn an offer to it is a counteroffer to its own offer (kept for the agent's pick, or
     * declined once that offer is through); on another's turn its agent answers it (answerOffer). An accept of the seat's
     * offer is confirmed if it is the first (first_accept), an accept of its counteroffer always; rejects count as
     * declines. Without a game to look at, every offer to the seat is declined.
     */
    private void tradeMessage(SOCGameTextMsg message, Seen seen)
    {
        String text = message.getText();
        if (text == null || ! text.startsWith(StacTradeMessage.TRADE) || client.getNickname().equals(message.getNickname()) || ourPlayerData == null)
            return;
        StacTradeMessage trade;
        try
        {
            trade = StacTradeMessage.parse(StacDialogueManager.fromMessage(text));
        }
        catch (RuntimeException e)
        {
            System.err.println(client.getNickname() + " in " + game.getName() + ": can't read the trade message " + text + ": " + e);
            return;
        }
        int us = ourPlayerNumber(), from = trade.getSenderInt();
        if (from == us || from < 0 || from >= game.maxPlayers)
            return;
        boolean toUs = Arrays.asList(trade.getReceivers().split(",")).contains(Integer.toString(us));
        StacTradeOffer offer = trade.getOffer();
        if (offer != null)
        {
            if (! toUs)
            {
                send(noResponse(us, from));
                return;
            }
            int fromSeat = osp == null ? -1 : osp.seatOf(from);
            if (own != null && own.partner < 0 && own.addressed(fromSeat) && ! own.answers.containsKey(String.valueOf(fromSeat)))
            {
                // A counteroffer to the seat's offer, in the offerer's terms: what the counterer gets, the seat gives
                own.answers.put(String.valueOf(fromSeat), "counter");
                own.counters.put(String.valueOf(fromSeat), Json.object("give", OSPGame.cardSet(offer.getGetSet()), "get", OSPGame.cardSet(offer.getGiveSet())));
                return;
            }
            boolean ourTurn = game.getCurrentPlayerNumber() == us;
            if (seen == null || osp == null || ourTurn || agentFailed)
            {
                send(reject(us, from));
                // An offer on another's turn declined (a late counteroffer to the seat's own offer is none)
                if (record != null && ! ourTurn)
                    record.seat(osp.seatOf(us)).declined();
                return;
            }
            answerOffer(offer, from, seen);
            return;
        }
        if (! toUs || osp == null)
            return;
        int fromSeat = osp.seatOf(from);
        boolean ownAddressee = own != null && own.addressed(fromSeat) && ! own.answers.containsKey(String.valueOf(fromSeat));
        if (trade.isAccept())
        {
            if (ownAddressee)
            {
                own.answers.put(String.valueOf(fromSeat), "accept");
                if (own.partner < 0)
                    confirm(fromSeat);
            }
            else if (counterTo == from)
            {
                // The offerer took the seat's counteroffer: the seat wrote the terms, so it confirms them
                send(accept(us, from));
                counterTo = -1;
            }
            // Else the offerer confirmed the seat's accept, which the server carries out
            return;
        }
        if (trade.isReject() || trade.isNoResponse())
        {
            if (ownAddressee)
                own.answers.put(String.valueOf(fromSeat), "decline");
            if (counterTo == from)
                counterTo = -1;
        }
    }

    // An offer of another seat's to this one, answered by the agent: accept (only with the cards it asks for), decline, or
    // a counteroffer in the offerer's terms; an offer the protocol can't hold (one side empty, either side "one of") is
    // declined
    private void answerOffer(StacTradeOffer offer, int from, Seen seen)
    {
        int us = ourPlayerNumber(), seat = osp.seatOf(us);
        Map<String, Object> give = OSPGame.cardSet(offer.getGiveSet()), get = OSPGame.cardSet(offer.getGetSet());
        if (give.isEmpty() || get.isEmpty() || offer.hasDisjunctiveGiveSet() || offer.hasDisjunctiveGetSet())
        {
            send(reject(us, from));
            record.seat(seat).declined();
            return;
        }
        List<Object> to = new ArrayList<>();
        boolean[] receivers = offer.getTo();
        for (int s = 0; s < game.maxPlayers; s++)
            if (receivers[osp.playerOf(s)])
                to.add((long) s);
        // Declining first: the move a failed agent's robot plays
        List<Object> legal = new ArrayList<>();
        legal.add(Json.array("respond", Json.object("accept", false)));
        if (holds(seen.position, seat, get))
            legal.add(Json.array("respond", Json.object("accept", true)));
        Map<String, Object> state = OSPGame.copy(seen.position);
        state.put("prompt", "answer_offer");
        // The next to answer as the rules name it: the first addressee in seat order from the offerer's left (under
        // first_accept the seat may answer out of that order all the same)
        int offerer = osp.seatOf(from);
        long current = seat;
        for (int k = 1; k < game.maxPlayers; k++)
            if (to.contains((long) ((offerer + k) % game.maxPlayers)))
            {
                current = (offerer + k) % game.maxPlayers;
                break;
            }
        state.put("current", current);
        state.put("offer", Json.object("from", (long) offerer, "give", give, "get", get, "to", to, "answers", new LinkedHashMap<>()));
        record.seat(seat).answered();
        @SuppressWarnings("unchecked")
        List<Object> move = (List<Object>) ask(seat, legal, seen, state, Json.object("counter", Json.object()));
        if ("counter".equals(move.get(0)))
        {
            // The seat's own offer to the offerer: it gives what the counteroffer has the offerer get, and back
            @SuppressWarnings("unchecked")
            Map<String, Object> terms = (Map<String, Object>) move.get(1);
            boolean[] back = new boolean[game.maxPlayers];
            back[from] = true;
            send(offerMessage(us, back, OSPGame.resourceSet(terms.get("get")), OSPGame.resourceSet(terms.get("give"))));
            counterTo = from;
            return;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> answer = (Map<String, Object>) move.get(1);
        if (Boolean.TRUE.equals(answer.get("accept")))
            send(accept(us, from));
        else
        {
            send(reject(us, from));
            record.seat(seat).declined();
        }
    }

    // Whether the seat's hand in the state holds the cards
    @SuppressWarnings("unchecked")
    private static boolean holds(Map<String, Object> state, int seat, Map<String, Object> cards)
    {
        Map<String, Object> hand = (Map<String, Object>) ((Map<String, Object>) ((List<Object>) state.get("players")).get(seat)).get("hand");
        for (Map.Entry<String, Object> card : cards.entrySet())
            if (hand == null || ! hand.containsKey(card.getKey()) || Json.integer(hand.get(card.getKey())) < Json.integer(card.getValue()))
                return false;
        return true;
    }

    // Why a trade the agent proposed is invalid (PROTOCOL.md, trades between players), or null if it is valid: card sets of
    // resources with at least one card each, no resource on both sides, an offer's give and a counteroffer's get (what the
    // offerer would get) within the seat's hand, an offer to seats the decision named, each once
    @SuppressWarnings("unchecked")
    static String tradeError(Object answer, Map<String, Object> trades, int seat, Map<String, Object> state)
    {
        if (! (answer instanceof List) || ((List<Object>) answer).size() != 2 || ! (((List<Object>) answer).get(1) instanceof Map))
            return "not a legal move or a trade";
        List<Object> move = (List<Object>) answer;
        String kind = String.valueOf(move.get(0));
        if (! ("offer".equals(kind) || "counter".equals(kind)) || ! trades.containsKey(kind))
            return "not a legal move or a trade the decision names";
        Map<String, Object> terms = (Map<String, Object>) move.get(1);
        if (! cardSet(terms.get("give")) || ! cardSet(terms.get("get")))
            return "give and get have to be card sets of resources with at least one card";
        Map<String, Object> give = (Map<String, Object>) terms.get("give"), get = (Map<String, Object>) terms.get("get");
        for (String name : give.keySet())
            if (get.containsKey(name))
                return name + " on both sides";
        if (! holds(state, seat, "offer".equals(kind) ? give : get))
            return "more cards than the seat holds";
        if ("counter".equals(kind))
            return terms.containsKey("to") ? "a counteroffer names no seats" : null;
        Object to = terms.get("to");
        List<Object> named = (List<Object>) ((Map<String, Object>) trades.get("offer")).get("to");
        if (! (to instanceof List) || ((List<Object>) to).isEmpty())
            return "an offer names at least one seat";
        Set<Object> once = new HashSet<>();
        for (Object s : (List<Object>) to)
            if (! (s instanceof Long) || ! named.contains(s) || ! once.add(s))
                return "the seat " + Json.write(s) + " isn't one to name, or is named twice";
        return null;
    }

    // Whether the value is a card set of resources: names to positive counts, at least one card
    @SuppressWarnings("unchecked")
    private static boolean cardSet(Object value)
    {
        if (! (value instanceof Map) || ((Map<String, Object>) value).isEmpty())
            return false;
        for (Map.Entry<String, Object> card : ((Map<String, Object>) value).entrySet())
        {
            if (! (card.getValue() instanceof Long) || (Long) card.getValue() < 1)
                return false;
            try
            {
                OSPGame.resourceType(card.getKey());
            }
            catch (RuntimeException e)
            {
                return false;
            }
        }
        return true;
    }

    /** What the brain read of the true game at one moment */
    private static final class Seen
    {
        int state, acting;
        boolean over, deciding;
        // The game's state and position, which changes with every move the server takes
        String key;
        // The turn: the round and the player on turn
        String turn;
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
            seen.turn = truthGame.getRoundCount() + "/" + truthGame.getCurrentPlayerNumber();
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
        if (TRADE_IN_CHAT)
        {
            turnDuties(seen);
            // While the seat's own offer stands nothing else is decided
            if (own != null)
            {
                standingOffer(seat, seen);
                return;
            }
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
        // On its own turn after the roll the seat may offer a trade, to every other seat
        Map<String, Object> trades = null;
        if (TRADE_IN_CHAT && state == SOCGame.PLAY1 && offersThisTurn < OFFERS_PER_TURN && ! agentFailed)
        {
            List<Object> others = new ArrayList<>();
            for (int s = 0; s < game.maxPlayers; s++)
                if (s != seat)
                    others.add((long) s);
            trades = Json.object("offer", Json.object("to", others, "left", (long) (OFFERS_PER_TURN - offersThisTurn)));
        }
        @SuppressWarnings("unchecked")
        List<Object> move = (List<Object>) (legal.size() == 1 && trades == null ? legal.get(0) : ask(seat, legal, seen, seen.position, trades));
        if ("offer".equals(move.get(0)))
        {
            // A chat message, no move: the game stays where it is until the offer is through
            @SuppressWarnings("unchecked")
            Map<String, Object> terms = (Map<String, Object>) move.get(1);
            offer(seat, terms, seen);
            return;
        }
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

    // Asks the agent about the true state given (the game's, or with an offer to answer or pick from), with the trades it
    // may propose (null: none); the first legal move once it failed
    private Object ask(int seat, List<Object> legal, Seen seen, Map<String, Object> full, Map<String, Object> trades)
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
            // The steps are sent: a further decision before the next look doesn't send them again
            seen.steps = null;
            Map<String, Object> state = truth ? full : OSPGame.hide(full, seat);
            // A lean agent gets only the hash, unless a change no step explains came since its last decision
            boolean force = seen.gaps != gapsAtDecision;
            gapsAtDecision = seen.gaps;
            Map<String, Object> decide = Json.object("type", "decide", "game_id", game.getName(), "n", n, "seat", (long) seat, "hash", OSPGame.hash(state));
            if (! lean || force)
                decide.put("state", state);
            decide.put("legal", legal);
            if (trades != null)
                decide.put("trades", trades);
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
                    Map<String, Object> resent = new LinkedHashMap<>(decide);
                    resent.put("state", state);
                    agent.send(resent);
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
                // A trade the decision names, on terms of the agent's (PROTOCOL.md, trades between players)
                String wrong = trades == null ? "not a legal move" : tradeError(answer.get("action"), trades, seat, full);
                if (wrong == null)
                {
                    record.seat(seat).decided(System.currentTimeMillis() - started);
                    return answer.get("action");
                }
                throw fail(n, "illegal", "the agent answered " + chosen + ": " + wrong, null);
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
        // An observing agent follows the game from its start; one that doesn't starts from now (the steps so far count
        // as sent with the first decision's)
        observe = (Boolean) observing;
        gapsAtDecision = seen.gaps;
        Map<String, Object> from = observe ? log.start() : seen.position;
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
        // StacSettlers' card numbers (JSettlers 1.1.13's, not JSettlers 2's)
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
