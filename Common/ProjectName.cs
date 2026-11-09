namespace Common
{
    /// <summary>
    /// The project's name and its former ones (Lanzenbrecher, before that Hexfarer, before that CatanSim), for what users
    /// keep across a rename: an environment variable counts under any of the names, the current one first, and a file in
    /// the user's application folder kept under a former name is copied over to the current folder the first time it is
    /// asked for
    /// </summary>
    public static class ProjectName
    {
        /// <summary>
        /// The name the project goes by
        /// </summary>
        public const string Current = "Lanzenbrecher";

        /// <summary>
        /// The current name, then the former ones, newest first
        /// </summary>
        public static readonly string[] All = [Current, "Hexfarer", "CatanSim"];

        /// <summary>
        /// The environment variable NAME_SUFFIX under the first of the names that has it set (e.g. LANZENBRECHER_INFERENCE,
        /// then HEXFARER_INFERENCE, then CATANSIM_INFERENCE), or null; read is the lookup (the process environment if null)
        /// </summary>
        public static string? Variable(string suffix, Func<string, string?>? read = null)
        {
            read ??= Environment.GetEnvironmentVariable;
            foreach (string name in All)
                if (read($"{name.ToUpperInvariant()}_{suffix}") is string value)
                    return value;
            return null;
        }

        /// <summary>
        /// The path of a file in the user's application folder under the current name. If only a folder under a former name
        /// holds the file, that file (the newest name's) is copied over first, so settings and the host certificate's
        /// fingerprint survive the rename; a file already in the current folder is never replaced. appData is the
        /// application data folder (the user's if null).
        /// </summary>
        public static string AppDataFile(string fileName, string? appData = null)
        {
            appData ??= Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string path = Path.Combine(appData, Current, fileName);
            if (File.Exists(path)) return path;
            foreach (string former in All.Skip(1))
            {
                string old = Path.Combine(appData, former, fileName);
                if (!File.Exists(old)) continue;
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                File.Copy(old, path);
                break;
            }
            return path;
        }
    }
}
