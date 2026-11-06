using Common.Osp;
using System.IO.Compression;
using System.Text.Json.Nodes;

namespace OpenSeat
{
    /// <summary>
    /// The replay harness: plays the games another engine's referee recorded as Open Seat Protocol messages (e.g.
    /// Catanatron's, by Bridges/catanatron/record.py) through Hexfarer's rules, with the recorded dice, cards and steals,
    /// and compares at every move what both engines hold: the full state and the legal moves. A game's replay stops at the
    /// first state that differs or the first recorded move Hexfarer doesn't allow (the games no longer agree after it);
    /// differences in the legal moves alone are counted and the replay goes on. Known differences of the referee's
    /// ruleset are folded before comparing (see Normalize), so every remaining difference is a rule difference not
    /// described yet, a mapping bug or a bug in either engine.
    /// </summary>
    internal sealed class Replay
    {
        /// <summary>
        /// Counts and first examples by kind of difference, over all games
        /// </summary>
        public sealed class Report
        {
            public int Games, Moves, Diverged;
            public Dictionary<string, int> Counts { get; } = [];
            public Dictionary<string, List<string>> Examples { get; } = [];

            public void Add(string kind, string example, int maxExamples)
            {
                Counts[kind] = Counts.GetValueOrDefault(kind) + 1;
                if (!Examples.TryGetValue(kind, out List<string>? examples)) Examples[kind] = examples = [];
                if (examples.Count < maxExamples) examples.Add(example);
            }

            public override string ToString()
            {
                List<string> lines = [$"{Games} games, {Moves} moves replayed, {Diverged} games diverged"];
                foreach ((string kind, int count) in Counts.OrderByDescending(c => c.Value))
                {
                    lines.Add($"{count,8}  {kind}");
                    foreach (string example in Examples[kind]) lines.Add($"            {example}");
                }
                return string.Join(Environment.NewLine, lines);
            }
        }

        private readonly int _maxExamples;

        public Report Result { get; } = new();

        public Replay(int maxExamples = 3) => _maxExamples = maxExamples;

        /// <summary>
        /// The messages of a log: one JSON object per line, gzipped if the name ends in .gz
        /// </summary>
        public static IEnumerable<JsonNode> Read(string path)
        {
            using Stream file = File.OpenRead(path);
            using Stream input = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? new GZipStream(file, CompressionMode.Decompress) : file;
            using StreamReader reader = new(input);
            while (reader.ReadLine() is string line)
                if (line.Length > 0) yield return JsonNode.Parse(line)!;
        }

        /// <summary>
        /// Replays every game of the messages
        /// </summary>
        public Report Run(IEnumerable<JsonNode> messages, int maxGames = int.MaxValue)
        {
            Game? game = null;
            foreach (JsonNode message in messages)
            {
                string type = message["type"]!.GetValue<string>();
                if (type == "before")
                {
                    if (Result.Games >= maxGames) break;
                    game = new Game(this, message);
                    Result.Games++;
                    continue;
                }
                if (game == null || game.Stopped) continue;
                switch (type)
                {
                    case "decide": game.Decide(message); break;
                    case "step": game.Step(message); Result.Moves++; break;
                    case "after": game.After(message); break;
                }
            }
            return Result;
        }

        /// <summary>
        /// One game's replay: Hexfarer's copy follows the referee and stops at the first difference in a state
        /// </summary>
        private sealed class Game
        {
            private readonly Replay _replay;
            private readonly string _id;
            private readonly Replica _replica;

            public bool Stopped { get; private set; }

            public Game(Replay replay, JsonNode before)
            {
                _replay = replay;
                _id = before["game_id"]!.GetValue<string>();
                _replica = new Replica(before);
                // The board as Hexfarer holds it is the board the referee sent
                if (!_replica.BoardMatches(before)) Diverge("board", "the board built from the message writes back differently");
                else Compare(before["state"]!, "before");
            }

            public void Decide(JsonNode decide)
            {
                string at = $"move {decide["n"]}";
                if (!Compare(decide["state"]!, at)) return;
                sbyte seat = _replica.Acting();
                if (seat != decide["seat"]!.GetValue<int>()) { Diverge("seat", $"{at}: seat {decide["seat"]} acts, Hexfarer expects {seat}"); return; }

                // The legal moves of both, as sets of protocol moves
                HashSet<string> ours = _replica.Legal().Select(m => OspCodec.Canonical(m.move)).ToHashSet();
                HashSet<string> theirs = decide["legal"]!.AsArray().Select(OspCodec.Canonical).ToHashSet();
                string prompt = decide["state"]!["prompt"]!.GetValue<string>();
                foreach (string missing in theirs.Except(ours))
                    Note($"legal: only the referee allows {Replica.MoveType(missing)} ({prompt})", $"{at}: {missing}");
                foreach (string extra in ours.Except(theirs))
                    Note($"legal: only Hexfarer allows {Replica.MoveType(extra)} ({prompt})", $"{at}: {extra}");
            }

            public void Step(JsonNode step)
            {
                if (_replica.Follow(step) is (string kind, string detail)) Diverge(kind, detail);
            }

            public void After(JsonNode after)
            {
                if (_replica.FirstDifference(after["state"]!, "after", goesOn: false) is (string kind, string detail)) { Diverge(kind, detail); return; }
                int? winner = after["winner"] is JsonNode w ? w.GetValue<int>() : null;
                int? ours = _replica.State.HasEnded ? _replica.State.Turn.PlayerIndex : null;
                if (winner == ours) return;
                // Explained if the referee's winner isn't the player on turn: Catanatron lets anyone win at any time
                bool offTurn = winner != null && ours == null && winner != int.Parse(after["state"]!["current"]!.ToJsonString());
                Diverge(offTurn ? "winner (known: Catanatron counts a win outside the winner's own turn)" : "winner",
                    $"the referee's winner is seat {winner?.ToString() ?? "none"}, Hexfarer's {ours?.ToString() ?? "none"}");
            }

            // Compares the referee's state with Hexfarer's; at the first difference the game stops
            private bool Compare(JsonNode theirs, string at) =>
                _replica.FirstDifference(theirs, at) is not (string kind, string detail) || Diverge(kind, detail);

            private void Note(string kind, string example) => _replay.Result.Add(kind, $"{_id} {example}", _replay._maxExamples);

            private bool Diverge(string kind, string example)
            {
                Note(kind, example);
                _replay.Result.Diverged++;
                Stopped = true;
                return false;
            }
        }
    }
}
