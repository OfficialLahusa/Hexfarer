"""
Games of one Open Seat Agent against StacSettlers' agents in StacSettlers' headless simulation (OSPSimulation), with a
summary: the agent's wins, its decisions and their time, fallbacks, failures, stalled moves and turns the server ended.

Usage: python play.py --agent "COMMAND" --opponents stac|ss|pomcp|jsettlers|random|LINE --games N --out DIR
                      [--name NAME] [--fully-observable] [--no-chat] [--build]
The opponents fill the other three seats: a preset (stac: the Stac agent with the parameters its README calls best and
at most 3 offers without building; ss: SmartSettlers; pomcp: Dobre's typed POMCP agent at 10 000 iterations on 4
threads, as in his thesis; jsettlers: JSettlers 1.1's robot; random) or an agent line's value as Simulation reads it
(COUNT,NAME,KIND[,FLAGS]). SmartSettlers and the typed MCTS agents need --fully-observable (StacSettlers then shows
every hand, discard, robbery and bought card to everyone). The games run with ChatNegotiation=true (the Stac agents only
trade that way), Trades=true and ForceEndTurns=false (the server ends no turn on its own, however long the agent
thinks). DIR gets the simulation's working directory (config.txt, results/), the agents' standard error (agents/), the
output (simulation.log) and summary.json. Needs a JDK and StacSettlers' jar in STACSETTLERS_JAR.
"""

import argparse
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


def summary(output, our_name="Hexfarer"):
    """The games' result lines and what they say about the Open Seat Agent's seat"""
    lines = [json.loads(line[line.index("{"):]) for line in output.splitlines() if '"osp-result"' in line]
    games = []
    for line in lines:
        seat = next(s for s in line["result"]["seats"] if s["agent"] is not None or line["players"][s["seat"]] == our_name)
        games.append({"game": line["game_id"], "seat": seat["seat"], "winner": line["winner"], "won": line["winner"] == seat["seat"],
                      "vp": line["vp"], "turns": line["result"]["turns"], "decisions": seat["decisions"], "time_ms": seat["time_ms"],
                      "max_ms": seat["max_ms"], "robot_decisions": seat["robot_decisions"], "fallbacks": seat["fallbacks"],
                      "failures": seat["failures"], "stalls": seat["stalls"], "agent": seat["agent"]})
    decisions = sum(g["decisions"] for g in games)
    return {
        "games": len(games),
        "wins": sum(g["won"] for g in games),
        "win_rate": round(sum(g["won"] for g in games) / len(games), 3) if games else None,
        "mean_vp": round(statistics.mean(g["vp"][g["seat"]] for g in games), 2) if games else None,
        "decisions": decisions,
        "mean_decision_ms": round(sum(g["time_ms"] for g in games) / decisions, 1) if decisions else None,
        "max_decision_ms": max((g["max_ms"] for g in games), default=None),
        "fallbacks": sum(g["fallbacks"] for g in games),
        "failures": [f for g in games for f in g["failures"]],
        "stalls": sum(g["stalls"] for g in games),
        "forced_turn_ends": output.count("force end turn"),
        "per_game": games,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="the Open Seat Agent's command line")
    parser.add_argument("--opponents", required=True, help="a preset (" + ", ".join(PRESETS) + ") or COUNT,NAME,KIND[,FLAGS]")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="Hexfarer_vs")
    parser.add_argument("--fully-observable", action="store_true")
    parser.add_argument("--no-chat", action="store_true", help="ChatNegotiation=false (the Stac agents then hang)")
    parser.add_argument("--build", action="store_true", help="compile the adapter first")
    parser.add_argument("--timeout", type=int, default=24 * 3600)
    args = parser.parse_args()
    if not stacsettlers.available():
        sys.exit("needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
    if args.build:
        stacsettlers.build()
    out = os.path.abspath(args.out)
    agents = os.path.join(out, "agents")
    os.makedirs(agents, exist_ok=True)
    opponents = PRESETS.get(args.opponents, args.opponents).split(",", 3)
    config = stacsettlers.config(args.name, args.games, [(1, "Hexfarer", "osp"), opponents], chat=not args.no_chat,
                                 fully_observable=args.fully_observable)
    output = stacsettlers.simulate(os.path.join(out, "simulation"), config, args.agent, {"osp.log": agents}, timeout=args.timeout)
    with open(os.path.join(out, "simulation.log"), "w", encoding="utf-8") as file:
        file.write(output)
    result = summary(output)
    result["config"] = config
    result["agent_command"] = args.agent
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(result, file, indent=1)
    shown = {k: v for k, v in result.items() if k not in ("per_game", "config")}
    print(json.dumps(shown, indent=1))
    for game in result["per_game"]:
        print(f'{game["game"]}: seat {game["seat"]}, winner {game["winner"]}, vp {game["vp"]}, {game["decisions"]} decisions, '
              f'max {game["max_ms"]} ms, fallbacks {game["fallbacks"]}, stalls {game["stalls"]}, failures {len(game["failures"])}')
    # A run counts as failed where a game didn't end with a result or the seat failed
    sys.exit(0 if result["games"] == args.games and not result["failures"] and not result["fallbacks"] and not result["stalls"] else 1)


if __name__ == "__main__":
    main()
