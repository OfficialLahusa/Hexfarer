using System.Text.Json;
using Simulation;

namespace Client
{
    /// <summary>
    /// The user's settings from the main menu, kept in %AppData%\Hexfarer\settings.json. A missing or unreadable file gives the
    /// defaults, so the client always starts. Open Seat Agents are set up only in the file (no menu page): each a name for
    /// the seat pickers and the command that starts it (see Bridges/PROTOCOL.md).
    /// </summary>
    public sealed class ClientSettings
    {
        public bool Sounds { get; set; } = true;

        // Scales the volume of every sound effect, 0 to 1
        public float Volume { get; set; } = 1f;

        // Cards flying between the board, the hands and the bank
        public bool Particles { get; set; } = true;

        // Pause between two agent moves in a local game, in seconds, so they can be followed
        public float BotPace { get; set; } = 0.6f;

        // Outside agents the local games can seat, e.g. { "Name": "Catanatron AB:2 (fair)", "Command": "python open_seat_agent.py --bot AB:2 --needs world" }
        public List<ExternalAgent> OpenSeatAgents { get; set; } = [];

        private static string FilePath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Hexfarer", "settings.json");

        private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

        public static ClientSettings Load()
        {
            try
            {
                if (File.Exists(FilePath) && JsonSerializer.Deserialize<ClientSettings>(File.ReadAllText(FilePath), Options) is ClientSettings settings)
                {
                    settings.Volume = Math.Clamp(settings.Volume, 0, 1);
                    settings.BotPace = Math.Clamp(settings.BotPace, 0, 3);
                    // Entries without a name or command are dropped, and a name counts once
                    settings.OpenSeatAgents = (settings.OpenSeatAgents ?? []).Where(a => a != null && !string.IsNullOrWhiteSpace(a.Name) && !string.IsNullOrWhiteSpace(a.Command))
                        .DistinctBy(a => a.Name).ToList();
                    return settings;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"Could not read the settings, using the defaults: {e.Message}");
            }
            return new ClientSettings();
        }

        /// <summary>
        /// Writes the settings; why that failed, or null
        /// </summary>
        public string? Save()
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
                File.WriteAllText(FilePath, JsonSerializer.Serialize(this, Options));
                return null;
            }
            catch (Exception e)
            {
                Console.WriteLine($"Could not save the settings: {e.Message}");
                return e.Message;
            }
        }
    }
}
