using Common;
using Common.Actions;
using System.Text.Json;
using static Common.BoardGeometry;

namespace Tests
{
    /// <summary>
    /// The board geometry for other engines (BoardGeometry): tiles by cube coordinates, intersections by their three tiles,
    /// edges by their intersections, ports by the intersections they serve. Pinned to the map's own neighbor shifts, the
    /// corner directions, the edges' sides and the port privileges of the rules, so that a mirrored, rotated or shifted key
    /// can't pass
    /// </summary>
    public class BoardGeometryTests
    {
        private static IEnumerable<Board> Boards()
        {
            for (int seed = 1; seed <= 5; seed++)
            {
                TestUtil.Seed(seed);
                yield return MapGenerator.GenerateRandomClassic();
            }
        }

        // Hex distance between two tiles
        private static int Steps(Cube a, Cube b) => Math.Max(Math.Abs(a.Q - b.Q), Math.Max(Math.Abs(a.R - b.R), Math.Abs(a.S - b.S)));

        private static Cube CubeAt(Board board, Tile tile) => CubeOf(board, tile.Y * (int)board.Map.Width + tile.X);

        [Fact]
        public void TilesAreTheLandAndTheWaterRingAndFollowTheMapsNeighbors()
        {
            foreach (Board board in Boards())
            {
                Export export = Of(board);
                Assert.Equal(37, export.Tiles.Length);
                Assert.Equal(19, export.Tiles.Count(t => t.Land));
                Assert.Equal(37, export.Tiles.Select(t => (t.Cube[0], t.Cube[1], t.Cube[2])).Distinct().Count());
                foreach (TileEntry tile in export.Tiles)
                {
                    Cube cube = new Cube(tile.Cube[0], tile.Cube[1], tile.Cube[2]);
                    Assert.Equal(0, cube.Q + cube.R + cube.S);
                    // Land within two steps of the center, the water ring at three
                    Assert.Equal(tile.Land, cube.Distance <= 2);
                    Assert.Equal(tile.Land, board.Map.Values[tile.Index].IsLandTile());
                    Assert.Equal(tile.Index, TileIndexOf(board, cube));
                }
                Assert.Equal(new Cube(0, 0, 0), CubeOf(board, 3 * (int)board.Map.Width + 3));

                // A step in a direction on the map is the same step in cube coordinates (as Direction's offsets: east is
                // (1, 0, -1), south-east (0, 1, -1))
                foreach (Tile tile in board.Map.Values.Where(t => t.IsLandTile()))
                {
                    foreach (Direction.Tile direction in Enum.GetValues<Direction.Tile>())
                    {
                        Tile neighbor = board.Adjacency.GetTile(tile, direction)!;
                        Assert.Equal(CubeAt(board, tile).Shift(direction), CubeAt(board, neighbor));
                    }
                }
                Assert.Equal(new Cube(1, 0, -1), CubeAt(board, board.Adjacency.GetTile(board.Map.GetTile(3, 3), Direction.Tile.East)!));
                Assert.Equal(new Cube(0, 1, -1), CubeAt(board, board.Adjacency.GetTile(board.Map.GetTile(3, 3), Direction.Tile.SouthEast)!));
            }
        }

        [Fact]
        public void IntersectionsAreTheCornersOfThreeNeighboringTiles()
        {
            foreach (Board board in Boards())
            {
                Export export = Of(board);
                Assert.Equal(54, export.Intersections.Length);
                HashSet<string> keys = [];
                for (int i = 0; i < export.Intersections.Length; i++)
                {
                    Cube[] key = export.Intersections[i].Select(c => new Cube(c[0], c[1], c[2])).ToArray();
                    Assert.Equal(3, key.Length);
                    Assert.True(key.All(cube => cube.Distance <= 3), $"intersection {i} touches a tile off the board");
                    Assert.True(key.Any(cube => cube.Distance <= 2), $"intersection {i} touches no land");
                    // Three tiles meet at a corner only if each is next to the other two
                    Assert.True(Steps(key[0], key[1]) == 1 && Steps(key[0], key[2]) == 1 && Steps(key[1], key[2]) == 1, $"intersection {i}: {string.Join(" ", key)}");
                    Assert.True(keys.Add(string.Join(" ", key)), $"intersection {i} has the key of another");
                }

                // The land tiles the rules link to an intersection are in its key, the other tiles of the key are water
                Cube[][] all = IntersectionKeys(board);
                for (int i = 0; i < all.Length; i++)
                {
                    Cube[] land = board.Adjacency.GetTiles(board.Intersections[i]).Select(t => CubeAt(board, t)).Order().ToArray();
                    Assert.Equal(land, all[i].Where(cube => cube.Distance <= 2).ToArray());
                }

                // The corner in a direction lies between the tile and its neighbors on either side of it (the north corner
                // between the tile, its north-west and its north-east neighbor)
                foreach (Tile tile in board.Map.Values.Where(t => t.IsLandTile()))
                {
                    Cube center = CubeAt(board, tile);
                    foreach (Direction.Corner corner in Enum.GetValues<Direction.Corner>())
                    {
                        (Direction.Tile left, Direction.Tile right) = corner.GetAdjacentTiles();
                        Cube[] expected = new[] { center, center.Shift(left), center.Shift(right) }.Order().ToArray();
                        Intersection intersection = board.Adjacency.GetIntersection(tile, corner)!;
                        Assert.Equal(expected, IntersectionKey(board, intersection.Index));
                    }
                }
            }
        }

        [Fact]
        public void EdgesJoinTwoIntersectionsBetweenTheirSides()
        {
            foreach (Board board in Boards())
            {
                Export export = Of(board);
                Assert.Equal(72, export.Edges.Length);
                Assert.Equal(72, export.Edges.Select(e => (e[0], e[1])).Distinct().Count());
                for (int e = 0; e < export.Edges.Length; e++)
                {
                    (int a, int b) = (export.Edges[e][0], export.Edges[e][1]);
                    Assert.True(a < b, $"edge {e}: {a} {b}");
                }

                // The ends of an edge share exactly the two tiles on its sides: the land tile the rules link it to in a
                // direction and the neighbor in that direction
                int checkedEdges = 0;
                foreach (Tile tile in board.Map.Values.Where(t => t.IsLandTile()))
                {
                    foreach (Direction.Tile direction in Enum.GetValues<Direction.Tile>())
                    {
                        Edge edge = board.Adjacency.GetEdge(tile, direction)!;
                        (int a, int b) = (export.Edges[edge.Index][0], export.Edges[edge.Index][1]);
                        Cube[] shared = IntersectionKey(board, a).Intersect(IntersectionKey(board, b)).Order().ToArray();
                        Cube[] sides = new[] { CubeAt(board, tile), CubeAt(board, tile).Shift(direction) }.Order().ToArray();
                        Assert.Equal(sides, shared);
                        checkedEdges++;
                    }
                }
                Assert.Equal(19 * 6, checkedEdges);
            }
        }

        [Fact]
        public void PortsServeTheIntersectionsThatGiveTheirPrivileges()
        {
            foreach (Board board in Boards())
            {
                Export export = Of(board);
                Assert.Equal(9, export.Ports.Length);
                Assert.Equal(4, export.Ports.Count(p => p.Type == nameof(Port.TradeType.Generic)));
                Assert.Equal(18, export.Ports.SelectMany(p => p.Intersections).Distinct().Count());
                for (int p = 0; p < export.Ports.Length; p++)
                {
                    PortEntry entry = export.Ports[p];
                    Cube water = new Cube(entry.Water[0], entry.Water[1], entry.Water[2]);
                    Cube land = water.Shift(Enum.Parse<Direction.Tile>(entry.Facing));
                    Assert.Equal(3, water.Distance);
                    Assert.True(land.Distance <= 2, $"port {p} faces {land}, no land");
                    foreach (int intersection in entry.Intersections)
                    {
                        Cube[] key = IntersectionKey(board, intersection);
                        Assert.Contains(water, key);
                        Assert.Contains(land, key);

                        // A settlement there gets the port's privilege by the rules
                        GameState state = new GameState(board, 4);
                        SettlementAction.UpdatePortPrivileges(state, intersection, 0);
                        Assert.Equal(board.Ports[p].Type.GetPortPrivilege(), state.Players[0].PortPrivileges);
                    }
                }
            }
        }

        [Fact]
        public void EveryClassicBoardHasTheSameGeometryAndTheExportIsJson()
        {
            Export[] exports = Boards().Select(Of).ToArray();
            foreach (Export export in exports.Skip(1))
            {
                Assert.Equal(Json(exports[0].Intersections), Json(export.Intersections));
                Assert.Equal(Json(exports[0].Edges), Json(export.Edges));
                Assert.Equal(Json(exports[0].Ports.Select(p => (p.Water, p.Facing, p.Intersections))), Json(export.Ports.Select(p => (p.Water, p.Facing, p.Intersections))));
            }

            // The seat protocol's shape: lowercase names, cubes as [q, r, s]
            using JsonDocument document = JsonDocument.Parse(JsonSerializer.Serialize(exports[0]));
            JsonElement root = document.RootElement;
            Assert.Equal(["tiles", "intersections", "edges", "ports"], root.EnumerateObject().Select(p => p.Name));
            Assert.Equal(3, root.GetProperty("intersections")[0].GetArrayLength());
            Assert.Equal(3, root.GetProperty("intersections")[0][0].GetArrayLength());
            Assert.Equal(["water", "facing", "type", "intersections"], root.GetProperty("ports")[0].EnumerateObject().Select(p => p.Name));
            Export parsed = JsonSerializer.Deserialize<Export>(root.GetRawText())!;
            Assert.Equal(Json(exports[0]), Json(parsed));
        }

        private static string Json<T>(T value) => JsonSerializer.Serialize(value, new JsonSerializerOptions { IncludeFields = true });

        // The Open Seat Protocol's reference shape of the classic board, shared with the adapters in Bridges
        private static string ReferencePath => Path.Combine(Directory.GetParent(TestUtil.SourceFixtureDirectory())!.Parent!.FullName, "Bridges", "osp", "classic-board.json");

        [Fact]
        public void TheClassicBoardHasTheOpenSeatProtocolsReferenceShape()
        {
            string[] shapes = Boards().Select(board => ToJson(ShapeOf(board))).Distinct().ToArray();
            Assert.Single(shapes);

            // LANZENBRECHER_UPDATE_FIXTURES=1 writes the reference after an intended change
            if (Environment.GetEnvironmentVariable("LANZENBRECHER_UPDATE_FIXTURES") == "1")
            {
                Directory.CreateDirectory(Path.GetDirectoryName(ReferencePath)!);
                File.WriteAllText(ReferencePath, shapes[0]);
            }
            Assert.Equal(File.ReadAllText(ReferencePath).ReplaceLineEndings("\n"), shapes[0]);

            // Its counts and keys as the protocol describes them: axial hexes, three hexes per intersection, two per path
            OSPShape shape = ShapeOf(Boards().First());
            Assert.Equal((19, 18, 54, 72, 9), (shape.Land.Length, shape.Sea.Length, shape.Intersections.Length, shape.Paths.Length, shape.Harbors.Length));
            Assert.All(shape.Intersections, key => Assert.Equal(3, key.Length));
            Assert.All(shape.Paths, path => Assert.Equal(2, path.Length));
            Assert.Equal([0, 0], shape.Land[9]);
            // Every harbor is a path, and every path joins two intersections
            HashSet<string> paths = shape.Paths.Select(p => JsonSerializer.Serialize(p)).ToHashSet();
            HashSet<string> intersections = shape.Intersections.Select(i => JsonSerializer.Serialize(i)).ToHashSet();
            Assert.All(shape.Harbors, harbor => Assert.Contains(JsonSerializer.Serialize(harbor), paths));
            Assert.All(shape.Paths, path => Assert.All(path, end => Assert.Contains(JsonSerializer.Serialize(end), intersections)));
        }
    }
}
