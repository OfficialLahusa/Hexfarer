package osp.stacsettlers;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.message.SOCBankTrade;
import soc.message.SOCBuildRequest;
import soc.message.SOCChoosePlayer;
import soc.message.SOCDiscard;
import soc.message.SOCDiscoveryPick;
import soc.message.SOCMessage;
import soc.message.SOCMonopolyPick;
import soc.message.SOCMoveRobber;
import soc.message.SOCPlayDevCardRequest;
import soc.message.SOCPutPiece;

/**
 * Every move of a StacSettlers game on OSPServer as Open Seat Protocol steps (see ../../PROTOCOL.md), written as a
 * referee log while the game runs (-Dosp.record=DIR: GAME.jsonl, or GAME-2.jsonl and so on if an earlier run wrote that
 * game's name), the input of Hexfarer's replay harness (OpenSeat replay): before, a decide (the true state, the legal
 * moves by StacSettlers' rules) and a step per move of every seat, after with the winner. StacSettlers' server handles
 * each player's request on its one request thread; OSPServer passes it through dispatch here, which reads the game
 * before and after StacSettlers handles it and turns the change into a step: one protocol move where StacSettlers takes
 * two requests (a bought piece and its place, a card and its resources, a robber move and its victim), and the discards
 * after a 7, which StacSettlers' players make at the same time, one after another from the lowest seat, each with the
 * state the protocol's order gives it. A change no protocol move makes counts as a gap: a trade between players
 * (StacSettlers' agents negotiate them in chat and the server executes them), or a request the protocol has no move for;
 * the next decide then starts from another state than the last step ended in, where a replay stops.
 */
final class MoveLog
{
    // The requests that make (a step of) a protocol move
    private static final java.util.Set<Integer> MOVES = new java.util.HashSet<>(java.util.Arrays.asList(SOCMessage.ROLLDICE,
        SOCMessage.PUTPIECE, SOCMessage.BUILDREQUEST, SOCMessage.CANCELBUILDREQUEST, SOCMessage.BUYCARDREQUEST, SOCMessage.PLAYDEVCARDREQUEST,
        SOCMessage.DISCOVERYPICK, SOCMessage.MONOPOLYPICK, SOCMessage.MOVEROBBER, SOCMessage.CHOOSEPLAYER, SOCMessage.DISCARD,
        SOCMessage.BANKTRADE, SOCMessage.ENDTURN));

    final String gameId;
    final SOCGame game;
    final OSPGame osp;

    private Map<String, Object> last;
    // The first request of a two-step move: the move's kind and value so far, the state before it, its legal moves
    private String pendingKind;
    private Object pendingValue;
    private Map<String, Object> pendingBefore;
    private List<Object> pendingLegal;
    // After a 7: the state before the first discard, the seats that have to discard (lowest first), the discards come in
    // (seat: card set and legal moves) and how many of the seats are written as steps
    private Map<String, Object> discardStart;
    private List<Integer> discarders;
    private final Map<Integer, Object[]> discarded = new TreeMap<>();
    private int discardsDone;
    private int gaps;
    private long steps;
    private boolean ended;
    private Writer record;

    MoveLog(SOCGame game, String directory)
    {
        this.game = game;
        this.gameId = game.getName();
        this.osp = new OSPGame(game);
        this.last = osp.state(-1);
        try
        {
            new File(directory).mkdirs();
            // StacSettlers numbers its games afresh in each run: a name an earlier run recorded gets a number
            String name = gameId.replaceAll("[^A-Za-z0-9._-]", "_");
            File file = new File(directory, name + ".jsonl");
            for (int copy = 2; ! file.createNewFile(); copy++)
                file = new File(directory, name + "-" + copy + ".jsonl");
            record = new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8);
            List<Object> players = Json.array();
            for (int seat = 0; seat < game.maxPlayers; seat++)
                players.add(game.getPlayer(osp.playerOf(seat)).getName());
            write(Json.object("type", "before", "game_id", gameId, "players", (long) game.maxPlayers, "stacsettlers_first", (long) osp.first,
                "names", players, "ruleset", OSPGame.RULESET, "board", osp.board.boardJson(), "state", last));
        }
        catch (IOException e)
        {
            System.err.println("osp: can't record " + gameId + ": " + e);
            record = null;
        }
    }

    int gaps()
    {
        return gaps;
    }

    /** A request StacSettlers is handling: who sent it, and the game before it */
    private static final class InFlight
    {
        final SOCMessage message;
        final int seat;
        final Map<String, Object> before;
        final List<Object> legal;
        final List<Integer> mustDiscard;
        // StacSettlers' game state before the request
        final int gameState;

        InFlight(SOCMessage message, int seat, Map<String, Object> before, List<Object> legal, List<Integer> mustDiscard, int gameState)
        {
            this.message = message;
            this.seat = seat;
            this.before = before;
            this.legal = legal;
            this.mustDiscard = mustDiscard;
            this.gameState = gameState;
        }
    }

    /**
     * StacSettlers' handling of one request from a player (seat -1: from no player), with the step it makes: the game is
     * read before and after (under its monitor, which StacSettlers' handler takes itself in between). The caller holds
     * the server's request lock.
     */
    void dispatch(SOCMessage message, int seat, Runnable handler)
    {
        if (ended || game.getPlayer(0) == null)
        {
            handler.run();
            return;
        }
        InFlight request;
        game.takeMonitor();
        try
        {
            Map<String, Object> before = osp.state(-1);
            int acting = osp.acting();
            // The legal moves only for the requests that make moves (not for the chat of the agents' negotiations)
            List<Object> legal = ! MOVES.contains(message.getType()) ? null
                : message.getType() == SOCMessage.DISCARD && seat >= 0 ? osp.discardMoves(seat) : acting >= 0 ? osp.legal(acting) : new ArrayList<>();
            List<Integer> mustDiscard = new ArrayList<>();
            for (int s = 0; s < game.maxPlayers; s++)
                if (game.getPlayer(osp.playerOf(s)).getNeedToDiscard())
                    mustDiscard.add(s);
            request = new InFlight(message, seat, before, legal, mustDiscard, game.getGameState());
        }
        finally
        {
            game.releaseMonitor();
        }
        try
        {
            handler.run();
        }
        finally
        {
            // StacSettlers empties a game it destroys
            if (game.getPlayer(0) != null && game.getBoard() != null)
            {
                game.takeMonitor();
                try
                {
                    afterRequest(request);
                }
                finally
                {
                    game.releaseMonitor();
                }
            }
        }
    }

    // The request handled: its step, and the game's end if it ended (a failure here never stops StacSettlers' handling:
    // it counts as a gap)
    private void afterRequest(InFlight request)
    {
        try
        {
            Map<String, Object> after = osp.state(-1);
            // Something changed the game since the last request, which no request made: a gap, which loses what was pending
            if (! request.before.equals(last))
                gap();
            if (request.seat >= 0)
                handled(request.message, request.seat, request.before, after, request.legal, request.mustDiscard, request.gameState);
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

    private void gap()
    {
        gaps++;
        pendingKind = null;
        discardStart = null;
        discarded.clear();
    }

    // Turns a handled request into its step, if it made one
    private void handled(SOCMessage message, int seat, Map<String, Object> before, Map<String, Object> after, List<Object> legal,
        List<Integer> mustDiscard, int stateBefore)
    {
        // The states are maps, lists and values as OSPGame writes them, so they compare as their JSON would; StacSettlers'
        // game state too, since a second step can change nothing else (a Monopoly that finds no cards)
        if (before.equals(after) && stateBefore == game.getGameState())
            return;
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
            int type = ((SOCBuildRequest) message).getPieceType();
            pending(type == SOCPlayingPiece.ROAD ? "build_road" : type == SOCPlayingPiece.SETTLEMENT ? "build_settlement" : "build_city", null, before, legal);
            break;
        }
        case SOCMessage.CANCELBUILDREQUEST:
            pendingKind = null;
            break;
        case SOCMessage.BUYCARDREQUEST:
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
            case SOCDevCardConstants.DISC: pending("play_year_of_plenty", null, before, legal); break;
            case SOCDevCardConstants.MONO: pending("play_monopoly", null, before, legal); break;
            default: gap();
            }
            break;
        case SOCMessage.DISCOVERYPICK:
            if ("play_year_of_plenty".equals(pendingKind))
            {
                List<Object> names = new ArrayList<>();
                for (Map.Entry<String, Object> picked : OSPGame.cardSet(((SOCDiscoveryPick) message).getResources()).entrySet())
                    for (int i = 0; i < Json.integer(picked.getValue()); i++)
                        names.add(picked.getKey());
                names.sort(null);
                step(seat, Json.array("play_year_of_plenty", names), null, pendingBefore, pendingLegal);
            }
            break;
        case SOCMessage.MONOPOLYPICK:
            if ("play_monopoly".equals(pendingKind))
            {
                String resource = OSPGame.resourceName(((SOCMonopolyPick) message).getResource());
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
            if (game.getGameState() == SOCGame.WAITING_FOR_CHOICE)
                pending("move_robber", hex, before, legal);
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
            discarded.put(seat, new Object[] {OSPGame.cardSet(((SOCDiscard) message).getResources()), legal});
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
            step(seat, Json.array("bank_trade", Json.object("give", OSPGame.cardSet(trade.getGiveSet()), "get", OSPGame.cardSet(trade.getGetSet()))), null, before, legal);
            break;
        }
        case SOCMessage.ENDTURN:
            step(seat, Json.array("end_turn"), null, before, legal);
            break;
        default:
            // A request no protocol move makes changed the game (a trade between players, executed after their chat)
            gap();
        }
    }

    private void pending(String kind, Object value, Map<String, Object> before, List<Object> legal)
    {
        pendingKind = kind;
        pendingValue = value;
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

    // A move made: its decision and step in the record
    private void step(int seat, List<Object> action, Map<String, Object> result, Map<String, Object> before, List<Object> legal)
    {
        pendingKind = null;
        long n = steps++;
        write(Json.object("type", "decide", "game_id", gameId, "n", n, "seat", (long) seat, "state", before, "legal", legal));
        write(Json.object("type", "step", "game_id", gameId, "n", n, "seat", (long) seat, "action", action, "result", result));
    }

    // The game is over and its winner named: the record's end
    private void end(Map<String, Object> state)
    {
        if (ended)
            return;
        ended = true;
        List<Object> vp = Json.array();
        for (int seat = 0; seat < game.maxPlayers; seat++)
            vp.add((long) game.getPlayer(osp.playerOf(seat)).getTotalVP());
        SOCPlayer won = game.getPlayerWithWin();
        Object winner = won == null ? null : (Object) (long) osp.seatOf(won.getPlayerNumber());
        write(Json.object("type", "after", "game_id", gameId, "winner", winner, "vp", vp, "truncated", winner == null,
            "rounds", (long) game.getRoundCount(), "gaps", (long) gaps, "state", state));
        close();
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
    void close()
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
