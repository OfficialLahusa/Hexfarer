using Common;
using Common.Serialization;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace OpenSeat
{
    /// <summary>
    /// An outside referee's games as Hexfarer save files, for the client to show (OpenSeat relay). Each game plays through
    /// Hexfarer's rules with the referee's results, as in the replay harness (Replica). Where the rules part (a state
    /// Hexfarer holds differently, a move it doesn't allow, a result it can't reach), the part ends, and a new one starts
    /// from the referee's state at its next decision (Replica.Rebuild). A save file holds where its part ended and the
    /// moves since it began, so the client steps back through them to the part's start.
    /// </summary>
    internal sealed class Relay
    {
        /// <summary>
        /// A stretch of a game that Hexfarer's rules followed: its number in the game (from 1), the move it starts at, the
        /// states at its start and end, its moves, and why it ended before the game did (a difference's kind), or null
        /// </summary>
        public sealed record Part(string GameId, int Number, int FirstMove, GameState Start, GameState End, List<Action> Actions, string? Cut);

        /// <summary>
        /// One line per part started from the referee's state: the game, the move and the difference
        /// </summary>
        public List<string> Log { get; } = [];

        private static int Int(JsonNode? number) => int.Parse(number!.ToJsonString());

        /// <summary>
        /// The parts of every game of the messages (up to maxGames games)
        /// </summary>
        public IEnumerable<Part> Run(IEnumerable<JsonNode> messages, int maxGames = int.MaxValue)
        {
            Replica? replica = null;
            string id = "";
            int games = 0, number = 0, first = 0;
            GameState start = null!;
            List<Action> actions = [];
            // Whether a part is under way (not before a start that already differs), and why it can't go on: the next
            // decision then starts a new one from the referee's state
            bool started = false;
            string? broken = null;

            foreach (JsonNode message in messages)
            {
                switch (message["type"]!.GetValue<string>())
                {
                    case "before":
                        if (games++ >= maxGames) yield break;
                        id = message["game_id"]!.GetValue<string>();
                        replica = new Replica(message);
                        (number, first, actions) = (0, 0, []);
                        start = new GameState(replica.State);
                        broken = !replica.BoardMatches(message) ? "board" : replica.FirstDifference(message["state"]!, "before") is (string atStart, _) ? atStart : null;
                        started = broken == null;
                        break;

                    case "decide" when replica != null:
                        {
                            int n = Int(message["n"]);
                            string? difference = broken ?? (replica.FirstDifference(message["state"]!, $"move {n}") is (string found, _) ? found : null);
                            if (difference == null && replica.Acting() != Int(message["seat"])) difference = "seat";
                            if (difference == null) break;
                            if (started) yield return new Part(id, ++number, first, start, new GameState(replica.State), actions, difference);
                            Log.Add($"{id} move {n}: {difference}; part {number + 1} starts from the referee's state");
                            replica.Rebuild(message["state"]!, message["legal"]!.AsArray());
                            (start, first, actions, broken, started) = (new GameState(replica.State), n, [], null, true);
                            break;
                        }

                    case "step" when replica != null && broken == null && started:
                        if (replica.Follow(message) is (string failed, _)) broken = failed;
                        else actions.Add(replica.Applied!);
                        break;

                    case "after" when replica != null:
                        {
                            string? difference = broken ?? (replica.FirstDifference(message["state"]!, "after", goesOn: false) is (string atEnd, _) ? atEnd : null);
                            if (started) yield return new Part(id, ++number, first, start, new GameState(replica.State), actions, difference);
                            replica = null;
                            break;
                        }
                }
            }
        }

        /// <summary>
        /// Runs the messages and writes each part as a save file into the directory
        /// </summary>
        public IEnumerable<Part> Write(IEnumerable<JsonNode> messages, string directory, int maxGames = int.MaxValue)
        {
            Directory.CreateDirectory(directory);
            foreach (Part part in Run(messages, maxGames))
            {
                File.WriteAllText(Path.Combine(directory, FileName(part)), SaveFileSerializer.Serialize(Save(part)));
                yield return part;
            }
        }

        /// <summary>
        /// The save file of a part: where it ended, and its moves to step back through
        /// </summary>
        public static SaveFile Save(Part part) => new(new GameState(part.End), [.. part.Actions], []);

        /// <summary>
        /// A game's first part is named after the game, the later ones with their number
        /// </summary>
        public static string FileName(Part part)
        {
            string name = string.Concat(part.GameId.Select(c => Path.GetInvalidFileNameChars().Contains(c) ? '_' : c));
            return part.Number == 1 ? $"{name}.yaml" : $"{name}-part{part.Number}.yaml";
        }
    }
}
