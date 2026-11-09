using Common;

namespace Tests
{
    /// <summary>
    /// What users keep across the project's renames (Lanzenbrecher, formerly Hexfarer, before that CatanSim): an
    /// environment variable under any of the names counts, the current name first; a file in the application folder under a
    /// former name is copied over to the current one, the newest former name first, and never over a file already there
    /// </summary>
    public class ProjectNameTests
    {
        private static Func<string, string?> Environment(params (string name, string value)[] set) =>
            name => set.FirstOrDefault(v => v.name == name).value;

        [Fact]
        public void TheCurrentNameComesFirstThenTheFormerOnesNewestFirst()
        {
            Assert.Equal("Lanzenbrecher", ProjectName.Current);
            Assert.Equal(["Lanzenbrecher", "Hexfarer", "CatanSim"], ProjectName.All);
        }

        [Fact]
        public void AnEnvironmentVariableCountsUnderAnyNameTheCurrentFirst()
        {
            Assert.Equal("cuda", ProjectName.Variable("INFERENCE", Environment(("LANZENBRECHER_INFERENCE", "cuda"), ("HEXFARER_INFERENCE", "dml"), ("CATANSIM_INFERENCE", "cpu"))));
            Assert.Equal("dml", ProjectName.Variable("INFERENCE", Environment(("HEXFARER_INFERENCE", "dml"), ("CATANSIM_INFERENCE", "cpu"))));
            Assert.Equal("cpu", ProjectName.Variable("INFERENCE", Environment(("CATANSIM_INFERENCE", "cpu"))));
            Assert.Null(ProjectName.Variable("INFERENCE", Environment(("LANZENBRECHER_TOKEN", "t"))));
        }

        [Fact]
        public void AFileUnderAFormerNameIsCopiedOverOnceAndNeverOverAFileAlreadyThere()
        {
            string appData = Path.Combine(Path.GetTempPath(), "project-name-test-" + Guid.NewGuid().ToString("N"));
            try
            {
                void Write(string folder, string text)
                {
                    Directory.CreateDirectory(Path.Combine(appData, folder));
                    File.WriteAllText(Path.Combine(appData, folder, "host.pfx"), text);
                }

                // Nothing anywhere: the current folder's path, nothing created
                string current = ProjectName.AppDataFile("host.pfx", appData);
                Assert.Equal(Path.Combine(appData, "Lanzenbrecher", "host.pfx"), current);
                Assert.False(File.Exists(current));

                // Under both former names: the newer one's file is copied, the old ones stay
                Write("CatanSim", "oldest");
                Write("Hexfarer", "former");
                Assert.Equal("former", File.ReadAllText(ProjectName.AppDataFile("host.pfx", appData)));
                Assert.Equal("former", File.ReadAllText(Path.Combine(appData, "Hexfarer", "host.pfx")));
                Assert.Equal("oldest", File.ReadAllText(Path.Combine(appData, "CatanSim", "host.pfx")));

                // Once there, the current file stays, whatever the former folders hold
                File.WriteAllText(current, "current");
                Write("Hexfarer", "changed later");
                Assert.Equal("current", File.ReadAllText(ProjectName.AppDataFile("host.pfx", appData)));

                // Only under the oldest name: that one is copied
                File.Delete(current);
                File.Delete(Path.Combine(appData, "Hexfarer", "host.pfx"));
                Assert.Equal("oldest", File.ReadAllText(ProjectName.AppDataFile("host.pfx", appData)));
            }
            finally
            {
                if (Directory.Exists(appData)) Directory.Delete(appData, true);
            }
        }
    }
}
