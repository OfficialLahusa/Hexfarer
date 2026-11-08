package osp.stacsettlers;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import soc.game.SOCGame;
import soc.game.SOCPlayer;

/**
 * A StacSettlers game's result record (see ../../PROTOCOL.md), for the games an Open Seat Agent's robot plays in: the
 * seed every agent of the game gets, a line per seat that the seat's robot brain fills in (the agent, what it received,
 * its decisions and their time, resyncs, fallbacks, failures, the robot's own moves and its stalls; StacSettlers' own
 * robots leave theirs empty), and at the game's end the winner and the points. It is written once, as an osp-result line
 * on standard error with the players by seat and the CPU time of every robot brain's thread in the game. Moves are
 * counted for our seats only: without a recorder of the server's requests the referee sees no other seat's moves, so the
 * record's total is null.
 */
final class GameRecord
{
    final String gameId;
    final OSPGame osp;
    // A fresh seed per game: StacSettlers' game draws its board, first player, dice and development cards from an unseeded
    // Random of its own, so a game doesn't repeat, and the seed fixes only the agents' side
    final int seed = new java.util.Random().nextInt(Integer.MAX_VALUE);
    private final SeatRecord[] seats;
    // The end once the game ended with a winner: the after message's fields, and the result record
    private Map<String, Object> end;

    GameRecord(SOCGame game)
    {
        gameId = game.getName();
        osp = new OSPGame(game);
        seats = new SeatRecord[game.maxPlayers];
        for (int seat = 0; seat < seats.length; seat++)
            seats[seat] = new SeatRecord();
    }

    /** A seat's line */
    SeatRecord seat(int seat)
    {
        return seats[seat];
    }

    /** A seat's line in the result record, as its robot brain fills it in */
    static final class SeatRecord
    {
        private String agent, needs;
        private Boolean observe, lean;
        private int decisions, resyncs, fallbacks, robotDecisions, stalls;
        private long timeMs, maxMs;
        private final List<Object> failures = new ArrayList<>();

        synchronized void agent(String name, String needs, boolean observe, boolean lean)
        {
            this.agent = name;
            this.needs = needs;
            this.observe = observe;
            this.lean = lean;
        }

        synchronized void decided(long ms)
        {
            decisions++;
            timeMs += ms;
            maxMs = Math.max(maxMs, ms);
        }

        synchronized void resynced()
        {
            resyncs++;
        }

        synchronized void fellBack()
        {
            fallbacks++;
        }

        synchronized void robotDecided()
        {
            robotDecisions++;
        }

        synchronized void stalled()
        {
            stalls++;
        }

        synchronized void failed(Long n, String code, boolean byAgent, String message)
        {
            failures.add(Json.object("n", n, "code", code, "by", byAgent ? "agent" : "referee", "message", message));
        }

        synchronized Map<String, Object> json(int seat)
        {
            return Json.object("seat", (long) seat, "agent", agent, "needs", needs, "observe", observe, "lean", lean,
                "decisions", (long) decisions, "time_ms", timeMs, "max_ms", maxMs, "resyncs", (long) resyncs, "fallbacks", (long) fallbacks,
                "failures", new ArrayList<>(failures), "robot_decisions", (long) robotDecisions, "stalls", (long) stalls,
                "moves", agent == null ? null : (Object) (long) robotDecisions);
        }
    }

    /** The game's end once it ended with a winner (winner, vp, truncated, result), else null */
    synchronized Map<String, Object> end()
    {
        return end;
    }

    /**
     * The game is over (or about to be removed): if it ended with a winner, its end is kept and the osp-result line
     * written, once (the caller holds the read locks, or removes the game)
     */
    synchronized void ended(SOCGame game)
    {
        if (end != null || ! osp.isOver() || game.getPlayerWithWin() == null)
            return;
        List<Object> vp = Json.array(), players = Json.array();
        for (int seat = 0; seat < game.maxPlayers; seat++)
        {
            SOCPlayer player = game.getPlayer(osp.playerOf(seat));
            vp.add((long) player.getTotalVP());
            players.add(player.getName());
        }
        Object winner = (long) osp.seatOf(game.getPlayerWithWin().getPlayerNumber());
        List<Object> lines = Json.array();
        Map<String, Long> millis = brainMillis();
        for (int seat = 0; seat < seats.length; seat++)
        {
            Map<String, Object> line = seats[seat].json(seat);
            line.put("brain_cpu_ms", millis.get((String) players.get(seat)));
            lines.add(line);
        }
        Map<String, Object> result = Json.object("seed", (long) seed, "moves", null, "turns", (long) game.getTurnCount(), "seats", lines);
        end = Json.object("winner", winner, "vp", vp, "truncated", false, "result", result);
        System.err.println(Json.write(Json.object("type", "osp-result", "game_id", gameId, "players", players, "winner", winner, "vp", vp,
            "rounds", (long) game.getRoundCount(), "result", result)));
    }

    // The CPU time of each robot brain's thread in this game so far, by robot name (StacSettlers' brains name their threads
    // robotBrain- and ours ospBrain-, with the robot and the game; Thread.getId, since threadId needs Java 19)
    @SuppressWarnings("deprecation")
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
}
