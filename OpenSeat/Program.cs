namespace OpenSeat
{
    /// <summary>
    /// Hexfarer's side of the Open Seat Protocol (see Bridges/PROTOCOL.md). For now the replay harness: plays another
    /// referee's recorded games through Hexfarer's rules and reports every difference.
    /// </summary>
    internal static class Program
    {
        public const string Usage = """
            Usage: OpenSeat replay LOG [LOG ...] [--max-games N] [--examples K]

            Plays the games of Open Seat Protocol referee logs (one JSON object per line, .gz allowed; e.g. from
            Bridges/catanatron/record.py) through Hexfarer's rules with the recorded results, and compares the state and
            the legal moves at every move. Prints the differences by kind with up to K examples each (default 3).
            Exit code 0 if no game diverged, 1 otherwise.
            """;

        public static int Main(string[] args)
        {
            if (args.Length == 0 || args[0] != "replay" || args.Contains("--help"))
            {
                Console.WriteLine(Usage);
                return args.Contains("--help") ? 0 : 2;
            }

            List<string> logs = [];
            int maxGames = int.MaxValue, examples = 3;
            for (int i = 1; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--max-games": maxGames = int.Parse(args[++i]); break;
                    case "--examples": examples = int.Parse(args[++i]); break;
                    default: logs.Add(args[i]); break;
                }
            }

            Replay replay = new(examples);
            foreach (string log in logs) replay.Run(Replay.Read(log), maxGames);
            Console.WriteLine(replay.Result);
            return replay.Result.Diverged == 0 ? 0 : 1;
        }
    }
}
