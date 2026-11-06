using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Common.Actions;
using Common.Osp;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;

namespace OpenSeat
{
    /// <summary>
    /// Hexfarer's side of the Open Seat Protocol (see Bridges/PROTOCOL.md): the replay harness, Hexfarer's agents as an
    /// Open Seat Agent, the boards of the Arena's map seeds, and a speed benchmark of the simulator.
    /// </summary>
    internal static class Program
    {
        public const string Usage = """
            Usage: OpenSeat replay LOG [LOG ...] [--max-games N] [--examples K]
                   OpenSeat agent [--net MODEL.onnx] [--iterations N | --seconds S] [--provider auto|cpu|dml|cuda] [--leaves-per-batch B]
                                  [--threads T] [--serial]
                   OpenSeat boards --seed-base S --count N --out FILE
                   OpenSeat bench [--games N] [--seed S]

            replay: plays the games of Open Seat Protocol referee logs (one JSON object per line, .gz allowed; e.g. from
            Bridges/catanatron/record.py) through Hexfarer's rules with the recorded results, and compares the state and the
            legal moves at every move. Prints the differences by kind with up to K examples each (default 3). Exit code 0 if
            no game diverged, 1 otherwise.

            agent: plays one seat for a referee over the protocol on standard input and output, with open-hand MCTS and the
            value net (default: the deployed net, 5000 iterations per decision; budgets round up to whole batches of B leaves,
            default 256). Needs the full state ("truth"). Writes one stats line per game to standard error, with the
            search's leaf evaluations and the distinct positions among them. --threads sets the value net's threads per
            evaluation (default half the logical cores), --serial selects each batch on one thread instead of all cores:
            both for an agent pinned to few cores, e.g. at equal thinking time on one core against another engine's bot.

            boards: writes the boards of the map seeds S to S+N-1 as the Arena generates them, one protocol board per line.

            bench: plays N random games (default 200) with random agents on one thread and reports moves, legal move lists
            and state copies per second.
            """;

        public static int Main(string[] args)
        {
            if (args.Length == 0 || args.Contains("--help"))
            {
                Console.WriteLine(Usage);
                return args.Contains("--help") ? 0 : 2;
            }
            string Option(string name, string fallback)
            {
                int i = Array.IndexOf(args, name);
                return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
            }

            switch (args[0])
            {
                case "replay":
                    {
                        List<string> logs = [];
                        for (int i = 1; i < args.Length; i++)
                        {
                            if (args[i] is "--max-games" or "--examples") i++;
                            else logs.Add(args[i]);
                        }
                        Replay replay = new(int.Parse(Option("--examples", "3")));
                        foreach (string log in logs) replay.Run(Replay.Read(log), int.Parse(Option("--max-games", int.MaxValue.ToString())));
                        Console.WriteLine(replay.Result);
                        return replay.Result.Diverged == 0 ? 0 : 1;
                    }
                case "agent":
                    {
                        InferenceProvider provider = Option("--provider", "auto") switch
                        {
                            "cpu" => InferenceProvider.Cpu, "dml" => InferenceProvider.DirectML, "cuda" => InferenceProvider.Cuda, _ => InferenceProvider.Auto,
                        };
                        double seconds = double.Parse(Option("--seconds", "0"), CultureInfo.InvariantCulture);
                        long iterations = seconds > 0 ? 0 : long.Parse(Option("--iterations", "5000"));
                        // Threads of the value net inside one evaluation (0: half the logical cores)
                        int threads = int.Parse(Option("--threads", "0"));
                        StateValueNet net = new(Option("--net", StateValueNet.DefaultModelPath), intraOpThreads: threads, provider: provider);
                        // The agents' console output would mix with the protocol
                        TextWriter protocol = Console.Out;
                        Console.SetOut(TextWriter.Null);
                        int leavesPerBatch = int.Parse(Option("--leaves-per-batch", MCTSAgent.DefaultLeavesPerBatch.ToString()));
                        // Select each batch on the calling thread instead of on all cores (e.g. for a process pinned to one core)
                        bool serial = args.Contains("--serial");
                        new SeatAgent(net, iterations, seconds, Console.Error, leavesPerBatch, parallelSelection: !serial).Run(Console.In, protocol);
                        return 0;
                    }
                case "boards":
                    {
                        int seedBase = int.Parse(Option("--seed-base", "30000")), count = int.Parse(Option("--count", "100"));
                        using StreamWriter output = new(Option("--out", "boards.jsonl"));
                        for (int i = 0; i < count; i++)
                        {
                            // As the Arena draws a map from its seed
                            Utils.Random.SeedCurrentThread(seedBase + i);
                            Board board = MapGenerator.GenerateRandomClassic();
                            output.WriteLine(new JsonObject { ["seed"] = seedBase + i, ["board"] = new OspCodec(board).BoardJson() }.ToJsonString());
                        }
                        return 0;
                    }
                case "bench":
                    Bench(int.Parse(Option("--games", "200")), int.Parse(Option("--seed", "1")));
                    return 0;
                default:
                    Console.WriteLine(Usage);
                    return 2;
            }
        }

        /// <summary>
        /// Random games on one thread: the simulator's own speed, without search (moves applied, legal move lists, copies)
        /// </summary>
        private static void Bench(int games, int seed)
        {
            Utils.Random = new ThreadSafeRandom(seed);
            long moves = 0, lists = 0, copies = 0;
            double playSeconds = 0, listSeconds = 0, copySeconds = 0;
            int ended = 0;
            Stopwatch clock = new();
            for (int g = 0; g < games; g++)
            {
                GameState state = new(MapGenerator.GenerateRandomClassic(), 4);
                Agent[] agents = Enumerable.Range(0, 4).Select(i => (Agent)new RandomAgent((sbyte)i)).ToArray();
                // Whole games as the agents play them: choosing (from the legal moves) and applying
                clock.Restart();
                int turns = 0;
                while (!state.HasEnded && state.Turn.RoundCounter < 1000)
                {
                    sbyte acting = 0;
                    while (!state.CanPlayerAct(acting)) acting++;
                    agents[acting].Act(state).Apply(state);
                    moves++;
                    turns++;
                }
                playSeconds += clock.Elapsed.TotalSeconds;
                if (state.HasEnded) ended++;

                // Legal move lists and copies of the final position, as a search needs them
                clock.Restart();
                for (int i = 0; i < 2000; i++) { LegalActionProvider.GetActionsForState(state, (sbyte)state.Turn.PlayerIndex); lists++; }
                listSeconds += clock.Elapsed.TotalSeconds;
                clock.Restart();
                for (int i = 0; i < 2000; i++) { _ = new GameState(state); copies++; }
                copySeconds += clock.Elapsed.TotalSeconds;
            }
            Console.WriteLine(new JsonObject
            {
                ["engine"] = "hexfarer", ["games"] = games, ["ended"] = ended, ["moves"] = moves,
                ["games_per_s"] = Math.Round(games / playSeconds, 2), ["moves_per_s"] = Math.Round(moves / playSeconds),
                ["legal_lists_per_s"] = Math.Round(lists / listSeconds), ["copies_per_s"] = Math.Round(copies / copySeconds),
            }.ToJsonString());
        }
    }
}
