"""
Games of one Open Seat Agent against StacSettlers' agents in StacSettlers' headless simulation (OSPSimulation), with a
summary: the agent's wins, its decisions and their time, fallbacks, failures, stalled moves, trade offers its robot
declined and turns the server ended. A game whose seat failed (a failure, or moves the referee made for the seat) says
nothing about strength: it stays out of the wins and the win rate, is counted beside them (failed_games), and is played
again in a further simulation until N games count. A run stops once more games failed than --max-failed allows (default
3, or one in a hundred of N; the simulation stops itself there too, osp.max_failed_games), when a simulation finished
no game or timed out, and with a negative --max-failed (any number) when every game of a simulation failed; the summary
then says why (stopped) and the exit code is 1.

Usage: python play.py --agent "COMMAND" --opponents stac|ss|pomcp|jsettlers|random|LINE --games N --out DIR
                      [--name NAME] [--fully-observable] [--no-chat] [--no-trades] [--trade-in-chat] [--max-failed K]
                      [--build]
The opponents fill the other three seats: a preset (stac: the Stac agent with the parameters its README calls best and
at most 3 offers without building; ss: SmartSettlers; pomcp: Dobre's typed POMCP agent at 10 000 iterations on 4
threads, as in his thesis; jsettlers: JSettlers 1.1's robot; random) or an agent line's value as Simulation reads it
(COUNT,NAME,KIND[,FLAGS]). SmartSettlers and Dobre's MCTS agents without a belief model need --fully-observable (the
pomcp preset keeps a factored belief and plays without it; StacSettlers then shows every hand, discard, robbery and
bought card to everyone). The games run with ChatNegotiation=true (the Stac agents only trade that way), Trades=true
(--no-trades: false) and ForceEndTurns=false (the server ends no turn on its own, however long the agent thinks). With
--trade-in-chat our seat takes part in the chat trades (its agent answers the offers made to it and makes its own;
without it, it declines every offer and makes none). DIR gets the simulation's working directory (config.txt, results/),
the agents' standard error (agents/), the output (simulation.log) and summary.json; a further simulation's go to
simulation-2 and simulation-2.log, and so on.
Needs a JDK and StacSettlers' jar in STACSETTLERS_JAR.
"""

import argparse
import collections
import json
import os
import statistics
import sys

import stacsettlers

PRESETS = {
    "stac": "3,Stac,stac,TRY_N_BEST_BUILD_PLANS:0|FAVOUR_DEV_CARDS:-5|NP_MAX_NUMBER_OF_OFFERS_WITHOUT_BUILDING:3",
    "ss": "3,SmartSettlers,originalSS",
    "pomcp": "3,POMCPTSCR,mcts,MCTS_FACTORED_BELIEF|MCTS_POMCP|MCTS_LIST_POMS|MCTS_PUCT|MCTS_Cp:4|MCTS_HUMAN_ACTION_TYPE_PDF_CONDITIONED_SEEDING"
             "|MCTS_HUMAN_ACTION_TYPE_PDF_SEEDING_TEMPERATURE:7|MCTS_TYPED_ROLLOUTS|MCTS_HUMAN_ACTION_TYPE_CONDITIONED_DIST_ROLLOUTS"
             "|MCTS_ITERATIONS:10000|MCTS_THREADS:4|MCTS_OFFERS_LIMIT:3",
    "jsettlers": "3,JSettlers,jsettlers",
    "random": "3,Random,random",
}


def default_max_failed(games):
    """The failed games a run of that many allows by default: three, or one in a hundred"""
    return max(3, games // 100)


def failed(seat):
    """Whether the Open Seat Agent's seat failed in a game: a failure, or moves the referee made for it (fallbacks)"""
    return bool(seat["failures"]) or seat["fallbacks"] > 0


def summary(output, our_names=("Lanzenbrecher", "Hexfarer")):
    """The games' result lines and what they say about the Open Seat Agent's seat (found by its agent, or by its player's
    name, the project's current one or the one older records have): games counts every game, failed_games those where the
    seat failed, counted_games the rest; wins, win_rate and mean_vp are of the counted games only"""
    lines = [json.loads(line[line.index("{"):]) for line in output.splitlines() if '"osp-result"' in line]
    games = []
    for line in lines:
        seat = next(s for s in line["result"]["seats"] if s["agent"] is not None or line["players"][s["seat"]] in our_names)
        games.append({"game": line["game_id"], "seat": seat["seat"], "winner": line["winner"], "won": line["winner"] == seat["seat"],
                      "failed": failed(seat), "vp": line["vp"], "turns": line["result"]["turns"], "decisions": seat["decisions"],
                      "time_ms": seat["time_ms"], "max_ms": seat["max_ms"], "robot_decisions": seat["robot_decisions"], "fallbacks": seat["fallbacks"],
                      "failures": seat["failures"], "stalls": seat["stalls"], "declined": seat.get("declined"), "agent": seat["agent"],
                      # Trading in chat (--trade-in-chat): offers made, offers its agent answered, trades it was part of
                      "offers": seat.get("offers", 0), "answers": seat.get("answers", 0), "trades": seat.get("trades", 0),
                      # What the seat asked for: the true state or its view, steps or none, lean decisions
                      "needs": seat.get("needs"), "observe": seat.get("observe"), "lean": seat.get("lean")})
    decisions = sum(g["decisions"] for g in games)
    # A failed game says nothing about strength: out of the wins, the win rate and the points
    counted = [g for g in games if not g["failed"]]
    return {
        "games": len(games),
        "failed_games": len(games) - len(counted),
        "counted_games": len(counted),
        "wins": sum(g["won"] for g in counted),
        "win_rate": round(sum(g["won"] for g in counted) / len(counted), 3) if counted else None,
        "mean_vp": round(statistics.mean(g["vp"][g["seat"]] for g in counted), 2) if counted else None,
        "decisions": decisions,
        "mean_decision_ms": round(sum(g["time_ms"] for g in games) / decisions, 1) if decisions else None,
        "max_decision_ms": max((g["max_ms"] for g in games), default=None),
        "fallbacks": sum(g["fallbacks"] for g in games),
        "failures": [f for g in games for f in g["failures"]],
        "stalls": sum(g["stalls"] for g in games),
        "declined": sum(g["declined"] or 0 for g in games),
        "offers": sum(g["offers"] for g in games),
        "answers": sum(g["answers"] for g in games),
        "trades": sum(g["trades"] for g in games),
        "forced_turn_ends": output.count("force end turn"),
        "asked": dict(collections.Counter(f'needs {g["needs"]}, {"observes" if g["observe"] else "no steps"}{", lean" if g["lean"] else ""}' for g in games)),
        "timed_out": stacsettlers.TIMED_OUT.split("{")[0] in output,
        "per_game": games,
    }


def run_games(simulate, games, max_failed):
    """Plays until that many games count: the first simulation plays them all, each further one the games that failed so
    far. simulate(round_number, games, allowance) runs one simulation of that many games, which stops itself past
    allowance more failed games, and returns its output. Stops once more games failed than max_failed allows, when a
    simulation finished no game or timed out, and with a negative max_failed (any number) when every game of a simulation
    failed. Returns the simulations' outputs and why the run stopped (None if enough games count)"""
    outputs = []
    while True:
        before = summary("\n".join(outputs))
        missing = games - before["counted_games"]
        if missing <= 0:
            return outputs, None
        output = simulate(len(outputs) + 1, missing, max_failed - before["failed_games"])
        outputs.append(output)
        after = summary("\n".join(outputs))
        # Past the limit (here, or in the simulation's own count): the setup is broken, not unlucky (a negative limit has none)
        if 0 <= max_failed < after["failed_games"] or stacsettlers.STOPPED in output:
            return outputs, f"stopped: {after['failed_games']} of {after['games']} games failed, at most {max_failed} allowed (--max-failed)"
        timed_out = stacsettlers.TIMED_OUT.split("{")[0] in output
        if after["games"] == before["games"]:
            return outputs, "stopped: the simulation finished no game" + (" (it timed out)" if timed_out else "")
        # A negative limit has no count to stop at, so a setup that fails every game would be played again forever: a
        # simulation whose games all failed ends the run (a limit ends it by its count)
        if max_failed < 0 and after["failed_games"] - before["failed_games"] == after["games"] - before["games"]:
            return outputs, (f"stopped: every game of simulation {len(outputs)} failed ({after['failed_games']} of {after['games']} games failed, "
                             "any allowed, --max-failed)")
        if timed_out:
            return outputs, f"stopped: simulation {len(outputs)} timed out"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="the Open Seat Agent's command line")
    parser.add_argument("--opponents", required=True, help="a preset (" + ", ".join(PRESETS) + ") or COUNT,NAME,KIND[,FLAGS]")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="Lanzenbrecher_vs")
    parser.add_argument("--fully-observable", action="store_true")
    parser.add_argument("--no-chat", action="store_true", help="ChatNegotiation=false (the Stac agents then hang)")
    parser.add_argument("--no-trades", action="store_true", help="Trades=false: no trades between players at all")
    parser.add_argument("--trade-in-chat", action="store_true",
                        help="our seat takes part in the Stac agents' chat trades (osp.trade_in_chat): its agent answers their offers and makes its own")
    parser.add_argument("--max-failed", type=int, help="failed games the run allows before it stops (default: 3, or 1 in 100 games; negative: any, "
                                                       "but a simulation whose games all failed ends the run)")
    parser.add_argument("--build", action="store_true", help="compile the adapter first")
    parser.add_argument("--timeout", type=int, default=24 * 3600, help="seconds per simulation")
    args = parser.parse_args()
    if not stacsettlers.available():
        sys.exit("needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
    if args.build:
        stacsettlers.build()
    out = os.path.abspath(args.out)
    agents = os.path.join(out, "agents")
    os.makedirs(agents, exist_ok=True)
    opponents = PRESETS.get(args.opponents, args.opponents).split(",", 3)
    max_failed = args.max_failed if args.max_failed is not None else default_max_failed(args.games)
    configs = []

    def simulation(round_number, games, allowance):
        # A further simulation gets a directory, a log and game names of its own (the agents' logs are named by game)
        suffix = "" if round_number == 1 else f"-{round_number}"
        config = stacsettlers.config(args.name + suffix.replace("-", "_r"), games, [(1, "Lanzenbrecher", "osp"), opponents], chat=not args.no_chat,
                                     fully_observable=args.fully_observable, trades=not args.no_trades)
        configs.append(config)
        output = stacsettlers.simulate(os.path.join(out, "simulation" + suffix), config, args.agent,
                                       {"osp.log": agents, "osp.max_failed_games": allowance, **({"osp.trade_in_chat": "true"} if args.trade_in_chat else {})},
                                       timeout=args.timeout)
        with open(os.path.join(out, f"simulation{suffix}.log"), "w", encoding="utf-8") as file:
            file.write(output)
        return output

    outputs, stopped = run_games(simulation, args.games, max_failed)
    result = summary("\n".join(outputs))
    result["stopped"] = stopped
    result["max_failed"] = max_failed
    result["simulations"] = len(outputs)
    result["config"] = configs[0]
    result["agent_command"] = args.agent
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(result, file, indent=1)
    shown = {k: v for k, v in result.items() if k not in ("per_game", "config")}
    print(json.dumps(shown, indent=1))
    for game in result["per_game"]:
        print(f'{game["game"]}: seat {game["seat"]}, winner {game["winner"]}, vp {game["vp"]}, {game["decisions"]} decisions, '
              f'max {game["max_ms"]} ms, fallbacks {game["fallbacks"]}, stalls {game["stalls"]}, failures {len(game["failures"])}'
              + (" (failed, not counted)" if game["failed"] else ""))
    if stopped:
        print(stopped)
    # A run counts as failed where it stopped (too many failed games, a simulation that hung) or a move stalled; failed games
    # it played again are in the summary
    sys.exit(0 if result["counted_games"] >= args.games and not stopped and not result["stalls"] else 1)


if __name__ == "__main__":
    main()
