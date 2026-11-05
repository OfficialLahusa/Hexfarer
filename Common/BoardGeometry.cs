using System.Numerics;
using System.Text.Json.Serialization;

namespace Common
{
    /// <summary>
    /// The board's elements keyed by their place, so that another engine laying out the classic board can map its own
    /// indices to ours without hand-written tables (see Bridges): a tile by its cube coordinates relative to the center tile
    /// (Q grows to the east, R to the south-east, S = -Q - R, as Direction's offsets), an intersection by the cubes of its
    /// three tiles (land or water), an edge by its two intersections, a port by the two intersections it serves. Everything
    /// is read from the board's own adjacency, never derived from a formula of its own, so the keys describe exactly the board
    /// the rules play on.
    /// </summary>
    public static class BoardGeometry
    {
        /// <summary>
        /// A tile's place in cube coordinates relative to the center tile
        /// </summary>
        public readonly record struct Cube(int Q, int R, int S) : IComparable<Cube>
        {
            /// <summary>
            /// Steps from the center tile: 0 to 2 for land, 3 for water on the classic board
            /// </summary>
            public int Distance => Math.Max(Math.Abs(Q), Math.Max(Math.Abs(R), Math.Abs(S)));

            public Cube Shift(Direction.Tile direction)
            {
                Vector3 offset = direction.ToCubeOffset();
                return new Cube(Q + (int)offset.X, R + (int)offset.Y, S + (int)offset.Z);
            }

            public int CompareTo(Cube other) => (Q, R, S).CompareTo((other.Q, other.R, other.S));

            public int[] ToArray() => [Q, R, S];

            public override string ToString() => $"({Q}, {R}, {S})";
        }

        // The center tile of the classic map (land within distance 2 of it, water at 3)
        private const int CenterX = 3, CenterY = 3;

        /// <summary>
        /// The cube coordinates of a tile, by its index in the map's storage (y * width + x, as the robber's)
        /// </summary>
        public static Cube CubeOf(Board board, int tileIndex)
        {
            int width = (int)board.Map.Width;
            // Relative to the center, in the map's even-r offset layout
            Vector3 cube = Coordinates.EvenRToCube(tileIndex % width, tileIndex / width) - Coordinates.EvenRToCube(CenterX, CenterY);
            return new Cube((int)cube.X, (int)cube.Y, (int)cube.Z);
        }

        /// <summary>
        /// The map storage index of the tile at the cube coordinates
        /// </summary>
        public static int TileIndexOf(Board board, Cube cube)
        {
            Vector3 absolute = new Vector3(cube.Q, cube.R, cube.S) + Coordinates.EvenRToCube(CenterX, CenterY);
            (int x, int y) = Coordinates.CubeToEvenR(absolute);
            return y * (int)board.Map.Width + x;
        }

        /// <summary>
        /// The cubes of the three tiles that meet at an intersection (land or water), sorted
        /// </summary>
        public static Cube[] IntersectionKey(Board board, int intersection) => IntersectionKeys(board)[intersection];

        /// <summary>
        /// The key of every intersection (see IntersectionKey), by intersection index. The adjacency links intersections only
        /// to land tiles, so a coastal corner's water tiles follow from the corner's direction at a land tile it belongs to:
        /// the corner in a direction lies between the tile and its two neighbors on either side of it, as BuildClassicGeometry
        /// lays the corners out. Throws if two land tiles at one intersection don't agree on its tiles.
        /// </summary>
        public static Cube[][] IntersectionKeys(Board board)
        {
            Cube[]?[] keys = new Cube[board.Intersections.Count][];
            foreach (Tile tile in board.Map.Values)
            {
                if (!tile.IsLandTile()) continue;
                Cube center = CubeOf(board, tile.Y * (int)board.Map.Width + tile.X);
                foreach (Direction.Corner corner in Enum.GetValues<Direction.Corner>())
                {
                    Intersection? intersection = board.Adjacency.GetIntersection(tile, corner);
                    if (intersection == null) continue;
                    // The tile and its neighbors on either side of the corner
                    (Direction.Tile left, Direction.Tile right) = corner.GetAdjacentTiles();
                    Cube[] key = [center, center.Shift(left), center.Shift(right)];
                    Array.Sort(key);
                    // Every land tile at the intersection has to describe the same three tiles
                    Cube[]? known = keys[intersection.Index];
                    if (known == null) keys[intersection.Index] = key;
                    else if (!known.SequenceEqual(key))
                        throw new InvalidOperationException($"Intersection {intersection.Index} is between {string.Join(" ", known)} and {string.Join(" ", key)}.");
                }
            }
            // Every intersection belongs to at least one land tile
            for (int i = 0; i < keys.Length; i++)
                if (keys[i] == null) throw new InvalidOperationException($"Intersection {i} is at no land tile.");
            return keys!;
        }

        /// <summary>
        /// The two intersections at the ends of an edge, the smaller index first
        /// </summary>
        public static (int a, int b) EdgeKey(Board board, int edge)
        {
            (int top, int bottom) = board.Adjacency.GetIntersectionIndices(edge);
            return (Math.Min(top, bottom), Math.Max(top, bottom));
        }

        /// <summary>
        /// The land tile a port faces and the two intersections it serves (the ends of the edge between its water tile and that
        /// land tile, as the port privileges in SettlementAction), the smaller index first
        /// </summary>
        public static (int land, int a, int b) PortPlace(Board board, Port port)
        {
            Tile sea = board.Map.GetTile(port.AnchorTileX, port.AnchorTileY);
            Tile land = board.Adjacency.GetTile(sea, port.AnchorDirection)!;
            Edge edge = board.Adjacency.GetEdge(land, port.AnchorDirection.Mirror())!;
            (int a, int b) = EdgeKey(board, edge.Index);
            return (land.Y * (int)board.Map.Width + land.X, a, b);
        }

        /// <summary>
        /// A tile of the export: its map storage index, cube coordinates and whether it is land
        /// </summary>
        public sealed record TileEntry(
            [property: JsonPropertyName("index")] int Index,
            [property: JsonPropertyName("cube")] int[] Cube,
            [property: JsonPropertyName("land")] bool Land);

        /// <summary>
        /// A port of the export: its water tile, the direction from it to the land, its type and the intersections it serves
        /// </summary>
        public sealed record PortEntry(
            [property: JsonPropertyName("water")] int[] Water,
            [property: JsonPropertyName("facing")] string Facing,
            [property: JsonPropertyName("type")] string Type,
            [property: JsonPropertyName("intersections")] int[] Intersections);

        /// <summary>
        /// The geometry of a board for the seat protocol's new_game: every tile within reach of an intersection (land and
        /// the water ring), every intersection by its tiles' cubes (in the order of our intersection indices), every edge by
        /// its intersections (in the order of our edge indices), and the ports
        /// </summary>
        public sealed record Export(
            [property: JsonPropertyName("tiles")] TileEntry[] Tiles,
            [property: JsonPropertyName("intersections")] int[][][] Intersections,
            [property: JsonPropertyName("edges")] int[][] Edges,
            [property: JsonPropertyName("ports")] PortEntry[] Ports);

        /// <summary>
        /// A hex in the Open Seat Protocol's axial coordinates [q, r] (see Bridges/PROTOCOL.md): the cube without S
        /// </summary>
        public static int[] Axial(Cube cube) => [cube.Q, cube.R];

        /// <summary>
        /// The shape of a board in the Open Seat Protocol's keys, every list sorted: land and sea hexes, intersections (their
        /// three hexes, sorted), paths (their two intersections, sorted) and the paths of the harbors. The same for every
        /// classic board; Bridges/osp/classic-board.json holds it, and every engine's adapter is tested against that file.
        /// </summary>
        public sealed record OspShape(int[][] Land, int[][] Sea, int[][][] Intersections, int[][][][] Paths, int[][][][] Harbors);

        /// <summary>
        /// The board's shape in the Open Seat Protocol's keys (see OspShape)
        /// </summary>
        public static OspShape ShapeOf(Board board)
        {
            Export export = Of(board);
            // The axial hex of a cube [q, r, s]
            static int[] Hex(int[] cube) => [cube[0], cube[1]];
            int[][] land = export.Tiles.Where(t => t.Land).Select(t => Hex(t.Cube)).ToArray();
            int[][] sea = export.Tiles.Where(t => !t.Land).Select(t => Hex(t.Cube)).ToArray();
            // An intersection by its three hexes, already sorted by (q, r) since the cubes are sorted by (q, r, s)
            int[][][] intersections = export.Intersections.Select(key => key.Select(Hex).ToArray()).ToArray();
            int[][][] Path(int a, int b) => new[] { intersections[a], intersections[b] }.Order(Lexical<int[]>(Lexical<int>())).ToArray();
            int[][][][] paths = export.Edges.Select(edge => Path(edge[0], edge[1])).ToArray();
            int[][][][] harbors = export.Ports.Select(port => Path(port.Intersections[0], port.Intersections[1])).ToArray();

            IComparer<int[]> hex = Lexical<int>();
            IComparer<int[][]> intersection = Lexical(hex);
            IComparer<int[][][]> path = Lexical(intersection);
            return new OspShape(land.Order(hex).ToArray(), sea.Order(hex).ToArray(), intersections.Order(intersection).ToArray(),
                paths.Order(path).ToArray(), harbors.Order(path).ToArray());
        }

        /// <summary>
        /// The shape as the reference file holds it: one hex, intersection or path per line, so that a difference shows
        /// where it is
        /// </summary>
        public static string ToJson(OspShape shape)
        {
            static string Lines<T>(string name, T[] items) =>
                $"  \"{name}\": [\n" + string.Join(",\n", items.Select(item => "    " + System.Text.Json.JsonSerializer.Serialize(item))) + "\n  ]";
            return "{\n" + string.Join(",\n", Lines("land", shape.Land), Lines("sea", shape.Sea), Lines("intersections", shape.Intersections),
                Lines("paths", shape.Paths), Lines("harbors", shape.Harbors)) + "\n}\n";
        }

        // Orders arrays element by element, a shorter one first if it is a prefix of the other
        private static IComparer<T[]> Lexical<T>(IComparer<T>? element = null)
        {
            IComparer<T> inner = element ?? Comparer<T>.Default;
            return Comparer<T[]>.Create((a, b) =>
            {
                for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
                {
                    int order = inner.Compare(a[i], b[i]);
                    if (order != 0) return order;
                }
                return a.Length.CompareTo(b.Length);
            });
        }

        /// <summary>
        /// The board's geometry as described by Export
        /// </summary>
        public static Export Of(Board board)
        {
            // Land and the water ring around it; the corners of the 7x7 map beyond that touch no intersection
            List<TileEntry> tiles = [];
            for (int index = 0; index < board.Map.Length; index++)
            {
                Cube cube = CubeOf(board, index);
                if (cube.Distance <= 3) tiles.Add(new TileEntry(index, cube.ToArray(), board.Map.Values[index].IsLandTile()));
            }

            int[][][] intersections = IntersectionKeys(board).Select(key => key.Select(cube => cube.ToArray()).ToArray()).ToArray();
            int[][] edges = Enumerable.Range(0, board.Edges.Count)
                .Select(e => EdgeKey(board, e)).Select(key => new[] { key.a, key.b }).ToArray();
            PortEntry[] ports = board.Ports.Select(port =>
            {
                (_, int a, int b) = PortPlace(board, port);
                Cube water = CubeOf(board, port.AnchorTileY * (int)board.Map.Width + port.AnchorTileX);
                return new PortEntry(water.ToArray(), port.AnchorDirection.ToString(), port.Type.ToString(), [a, b]);
            }).ToArray();
            return new Export(tiles.ToArray(), intersections, edges, ports);
        }
    }
}
