using Common;
using Common.Serialization;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace OpenSeat
{
    /// <summary>
    /// An outside referee's games as Lanzenbrecher save files, for the client to show (OpenSeat relay), or move by move into a
    /// watched match (LiveRelay). Each game plays through Lanzenbrecher's rules with the referee's results, as in the replay
    /// harness (Replica). Where the rules part (a state Lanzenbrecher holds differently, a move it doesn't allow, a result it
    /// can't reach), the part ends, and a new one starts from the referee's state at its next decision (Replica.Rebuild).
    /// A save file holds where its part ended and the moves since it began, so the client steps back through them to the
    /// part's start. With patch (for a match, which keeps edits in its history), what the known rule differences change
    /// (Replica.Patch) is taken over as an edit instead, and the part goes on.
    /// </summary>
    internal sealed class Relay(bool patch = false)
    {
        /// <summary>
        /// A stretch of a game that Lanzenbrecher's rules followed: its number in the game (from 1), the move it starts at, the
        /// states at its start and end, its moves, and why it ended before the game did (a difference's kind), or null
        /// </summary>
        public sealed record Part(string GameId, int Number, int FirstMove, GameState Start, GameState End, List<Action> Actions, string? Cut);

        /// <summary>
        /// What following a message gives: a part starts (from this state), a move was followed (with its outcome), the
        /// referee's state was taken over at a decision (the seat that decides, before and after, the difference), or a part
        /// ended
        /// </summary>
        public abstract record Event;
        public sealed record Started(string GameId, int Number, int FirstMove, GameState State) : Event;
        public sealed record Moved(string GameId, Action Action) : Event;
        public sealed record Patched(string GameId, int Seat, GameState Before, GameState After, string Difference) : Event;
        public sealed record Ended(Part Part) : Event;

        /// <summary>
        /// One line per part started from the referee's state: the game, the move and the difference
        /// </summary>
        public List<string> Log { get; } = [];

        private static int Int(JsonNode? number) => int.Parse(number!.ToJsonString());

        // The game being followed
        private Replica? _replica;
        private string _id = "";
        private int _number, _first;
        private GameState _start = null!;
        private List<Action> _actions = [];
        // Whether a part is under way (not before a start that already differs), and why it can't go on: the next decision
        // then starts a new one from the referee's state
        private bool _started;
        private string? _broken;

        /// <summary>
        /// The games of the messages played since their before, so far
        /// </summary>
        public int Games { get; private set; }

        /// <summary>
        /// Follows one message of a referee's log
        /// </summary>
        public List<Event> Feed(JsonNode message)
        {
            List<Event> events = [];
            switch (message["type"]!.GetValue<string>())
            {
                case "before":
                    Games++;
                    _id = message["game_id"]!.GetValue<string>();
                    _replica = new Replica(message);
                    (_number, _first, _actions) = (0, 0, []);
                    _start = new GameState(_replica.State);
                    _broken = !_replica.BoardMatches(message) ? "board" : _replica.FirstDifference(message["state"]!, "before") is (string atStart, _) ? atStart : null;
                    _started = _broken == null;
                    if (_started) events.Add(new Started(_id, ++_number, 0, new GameState(_start)));
                    break;

                case "decide" when _replica != null:
                    {
                        int n = Int(message["n"]);
                        string? difference = _broken ?? (_replica.FirstDifference(message["state"]!, $"move {n}") is (string found, _) ? found : null);
                        if (difference == null && _replica.Acting() != Int(message["seat"])) difference = "seat";
                        if (difference == null) break;
                        GameState end = new(_replica.State);
                        if (patch && _started && _broken == null && _replica.Patch(message["state"]!) && _replica.Acting() == Int(message["seat"]))
                        {
                            Log.Add($"{_id} move {n}: {difference}; taken over from the referee's state");
                            events.Add(new Patched(_id, Int(message["seat"]), end, new GameState(_replica.State), difference));
                            break;
                        }
                        if (_started) events.Add(new Ended(new Part(_id, _number, _first, _start, end, _actions, difference)));
                        Log.Add($"{_id} move {n}: {difference}; part {_number + 1} starts from the referee's state");
                        _replica.Rebuild(message["state"]!, message["legal"]!.AsArray());
                        (_start, _first, _actions, _broken, _started) = (new GameState(_replica.State), n, [], null, true);
                        events.Add(new Started(_id, ++_number, n, new GameState(_start)));
                        break;
                    }

                case "step" when _replica != null && _broken == null && _started:
                    if (_replica.Follow(message) is (string failed, _)) _broken = failed;
                    else
                    {
                        _actions.Add(_replica.Applied!);
                        events.Add(new Moved(_id, _replica.Applied!.Clone()));
                    }
                    break;

                case "after" when _replica != null:
                    {
                        string? difference = _broken ?? (message["state"] is JsonNode final && _replica.FirstDifference(final, "after", goesOn: false) is (string atEnd, _) ? atEnd : null);
                        if (_started) events.Add(new Ended(new Part(_id, _number, _first, _start, new GameState(_replica.State), _actions, difference)));
                        _replica = null;
                        break;
                    }
            }
            return events;
        }

        /// <summary>
        /// The parts of every game of the messages (up to maxGames games)
        /// </summary>
        public IEnumerable<Part> Run(IEnumerable<JsonNode> messages, int maxGames = int.MaxValue)
        {
            int before = Games;
            foreach (JsonNode message in messages)
            {
                if (message["type"]!.GetValue<string>() == "before" && Games - before >= maxGames) yield break;
                foreach (Event e in Feed(message))
                    if (e is Ended ended) yield return ended.Part;
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
