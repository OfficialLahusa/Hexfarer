"""
Records StacSettlers' own games as referee logs (OSPSimulation with -Dosp.record, MoveLog.java), the input of Hexfarer's
replay harness: OpenSeat replay DIR/records/*.jsonl plays them through Hexfarer's rules and lists where the rules differ.

Usage: python record.py --games N --out DIR [--agents LINE] [--trades]
The seats are four Stac agents (the parameters its README calls best), or the agent line's value as Simulation reads it
(COUNT,NAME,KIND[,FLAGS]); without --trades they don't trade with each other (Trades=false), since a trade between
players is no protocol move and stops a replay. DIR gets the logs (records/), the simulation's working directory
(simulation/) and its output (simulation.log). Needs a JDK and StacSettlers' jar in STACSETTLERS_JAR.
"""

import argparse
import os
import sys

import stacsettlers

STAC = "4,Stac,stac,TRY_N_BEST_BUILD_PLANS:0|FAVOUR_DEV_CARDS:-5|NP_MAX_NUMBER_OF_OFFERS_WITHOUT_BUILDING:3"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--agents", default=STAC)
    parser.add_argument("--trades", action="store_true")
    parser.add_argument("--name", default="StacRecord")
    parser.add_argument("--fully-observable", action="store_true")
    parser.add_argument("--build", action="store_true", help="compile the adapter first")
    args = parser.parse_args()
    if not stacsettlers.available():
        sys.exit("needs a JDK and StacSettlers' jar in STACSETTLERS_JAR")
    if args.build:
        stacsettlers.build()
    out = os.path.abspath(args.out)
    records = os.path.join(out, "records")
    config = stacsettlers.config(args.name, args.games, [args.agents.split(",", 3)], fully_observable=args.fully_observable)
    if not args.trades:
        config = config.replace("Trades=true", "Trades=false")
    output = stacsettlers.simulate(os.path.join(out, "simulation"), config, None, {"osp.record": records}, timeout=24 * 3600)
    with open(os.path.join(out, "simulation.log"), "w", encoding="utf-8") as file:
        file.write(output)
    logs = sorted(os.listdir(records)) if os.path.isdir(records) else []
    print(f"{len(logs)} of {args.games} games recorded in {records}")
    sys.exit(0 if len(logs) == args.games else 1)


if __name__ == "__main__":
    main()
