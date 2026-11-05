using Agents;
using Agents.Inference;
using Common;
using Common.Actions;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using Action = Common.Actions.Action;

namespace Arena
{
    /// <summary>
    /// The challenger at a table with three copies of the baseline, with trades between players: every map is played four
    /// times with the challenger in each seat once (seat advantage and map cancel within a group; all four games of a group
    /// use the same seed). A group scores the share of its games the challenger won, 0.25 if it is as strong as the
    /// baselines; the SPRT tests the win rate 25% + Elo0 percentage points (H0) against 25% + Elo1 (H1) on the group scores.
    /// Unlike a mirror duel, the challenger's moves can only help itself: two copies at one table could also gain from trading
    /// with each other. Besides the result, each game line counts the offers of both sides and how many led to a trade.
    /// </summary>
    internal sealed class TableDuel(DuelSettings settings, TextWriter log)
    {
        public const int Seats = 4;

        private readonly object _lock = new();
        private readonly Dictionary<int, bool?[]> _groups = [];
        private readonly List<double> _scores = [];
        private int _wins, _games;
        private long _challengerOffers, _challengerTrades, _baselineOffers, _baselineTrades, _challengerPartner, _challengerCounters, _baselineCounters;

        public GroupStatistics Statistics { get; private set; } = GroupStatistics.Of([], 0.25, 0.28);

        private double S0 => 0.25 + settings.Elo0 / 100;
        private double S1 => 0.25 + settings.Elo1 / 100;

        /// <summary>
        /// Plays the duel and writes one line per game, one per completed group and a last line with the result
        /// </summary>
        public GroupStatistics Run(CancellationToken cancellation = default)
        {
            var challengerNet = new StateValueNet(settings.Challenger.ModelPath, provider: settings.Provider);
            var baselineNet = settings.Baseline.ModelPath == settings.Challenger.ModelPath
                ? challengerNet : new StateValueNet(settings.Baseline.ModelPath, provider: settings.Provider);
            bool iterationBudget = settings.Iterations > 0;
            int parallel = iterationBudget ? Math.Max(1, settings.Parallel) : 1;
            Statistics = GroupStatistics.Of([], S0, S1, settings.Alpha, settings.Beta);

            Write($"[table duel] {settings.Challenger} against 3x {settings.Baseline}, free trade; maps from seed {settings.SeedBase}, "
                + (iterationBudget ? $"{settings.Iterations} iterations per decision, {parallel} games at once" : $"{settings.Seconds.ToString(CultureInfo.InvariantCulture)}s per decision")
                + FormattableString.Invariant($"; SPRT win rate [{S0:P1}, {S1:P1}], at most {settings.MaxGames} games and {settings.MaxDuration.TotalMinutes:F0} min"));

            TextWriter console = Console.Out;
            Console.SetOut(TextWriter.Null);
            using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            stop.CancelAfter(settings.MaxDuration);
            string reason = "all games played";
            try
            {
                var queue = new ConcurrentQueue<int>(Enumerable.Range(0, settings.MaxGames));
                var threads = Enumerable.Range(0, parallel).Select(_ => new Thread(() =>
                {
                    while (!stop.IsCancellationRequested && queue.TryDequeue(out int game))
                    {
                        try
                        {
                            GameResult result = PlayGame(game, challengerNet, baselineNet, stop.Token);
                            if (!stop.IsCancellationRequested || result.State.HasEnded)
                                Record(game, result, stop);
                        }
                        catch (OperationCanceledException) { }
                        catch (Exception e) { Write($"game {game + 1} failed: {e.Message}"); }
                    }
                })).ToList();
                threads.ForEach(t => t.Start());
                threads.ForEach(t => t.Join());
            }
            finally
            {
                Console.SetOut(console);
            }

            if (Statistics.Verdict != 0) reason = "decided";
            else if (cancellation.IsCancellationRequested) reason = "cancelled";
            else if (stop.IsCancellationRequested) reason = "time limit";
            Write($"{reason}: {Summary()}");
            return Statistics;
        }

        /// <summary>
        /// A game's end, whether the challenger won, and the offers and trades: by the challenger (as the offerer, and as the
        /// partner of a baseline's offer) and by the baselines together
        /// </summary>
        private sealed record GameResult(GameState State, int ChallengerSeat, bool ChallengerWon, double Seconds, int ChallengerOffers,
            int ChallengerTrades, int ChallengerPartner, int BaselineOffers, int BaselineTrades, int ChallengerCounters, int BaselineCounters);

        private GameResult PlayGame(int game, StateValueNet challengerNet, StateValueNet baselineNet, CancellationToken cancellation)
        {
            int challengerSeat = game % Seats;
            Utils.Random.SeedCurrentThread(settings.SeedBase + game / Seats);
            GameState state = new GameState(MapGenerator.GenerateRandomClassic(), Seats);
            state.Settings.FreeTrade = true;
            GameState start = state;
            Agent[] agents = Enumerable.Range(0, Seats).Select(seat => seat == challengerSeat
                ? settings.Challenger.Create((sbyte)seat, start, challengerNet, settings.Iterations, settings.Seconds)
                : settings.Baseline.Create((sbyte)seat, start, baselineNet, settings.Iterations, settings.Seconds)).ToArray();

            int challengerOffers = 0, challengerTrades = 0, challengerPartner = 0, baselineOffers = 0, baselineTrades = 0, challengerCounters = 0, baselineCounters = 0;
            var stopwatch = Stopwatch.StartNew();
            while (!state.HasEnded && state.Turn.RoundCounter < 300)
            {
                cancellation.ThrowIfCancellationRequested();
                sbyte acting = 0;
                for (sbyte player = 0; player < Seats; player++)
                    if (state.CanPlayerAct(player)) { acting = player; break; }
                Action action = agents[acting].Act(state, cancellation);
                cancellation.ThrowIfCancellationRequested();
                // Offers and counteroffers are made outside the legal move lists, so every move is checked before it is played
                if (!action.IsValidFor(state)) throw new InvalidOperationException($"seat {acting + 1} made an illegal move: {action}");
                action.Apply(state);
                foreach (Agent agent in agents) agent.HandleAction(action);

                switch (action)
                {
                    case OfferTradeAction:
                        if (acting == challengerSeat) challengerOffers++; else baselineOffers++;
                        break;
                    case CounterTradeAction:
                        if (acting == challengerSeat) challengerCounters++; else baselineCounters++;
                        break;
                    case ConfirmTradeAction confirm:
                        if (confirm.PlayerIndex == challengerSeat) challengerTrades++;
                        else
                        {
                            baselineTrades++;
                            if (confirm.Partner == challengerSeat) challengerPartner++;
                        }
                        break;
                }
            }
            return new GameResult(state, challengerSeat, state.HasEnded && state.Turn.PlayerIndex == challengerSeat, stopwatch.Elapsed.TotalSeconds,
                challengerOffers, challengerTrades, challengerPartner, baselineOffers, baselineTrades, challengerCounters, baselineCounters);
        }

        private void Record(int game, GameResult result, CancellationTokenSource stop)
        {
            lock (_lock)
            {
                _games++;
                if (result.ChallengerWon) _wins++;
                _challengerOffers += result.ChallengerOffers;
                _challengerTrades += result.ChallengerTrades;
                _challengerPartner += result.ChallengerPartner;
                _baselineOffers += result.BaselineOffers;
                _baselineTrades += result.BaselineTrades;
                _challengerCounters += result.ChallengerCounters;
                _baselineCounters += result.BaselineCounters;
                GameState state = result.State;
                string winner = !state.HasEnded ? "none" : result.ChallengerWon ? "challenger" : $"baseline (seat {state.Turn.PlayerIndex + 1})";
                Write($"game {game + 1} ({_games} done): challenger in seat {result.ChallengerSeat + 1}, winner {winner}, VP {string.Join("/", state.Players.Select(p => p.VictoryPoints.Total))}, "
                    + $"{state.Turn.RoundCounter} rounds, {result.Seconds:F0}s; offers C {result.ChallengerOffers} ({result.ChallengerTrades} traded), "
                    + $"B {result.BaselineOffers} ({result.BaselineTrades} traded, {result.ChallengerPartner} with C); counters C {result.ChallengerCounters} B {result.BaselineCounters}");

                bool?[] group = _groups.TryGetValue(game / Seats, out var known) ? known : _groups[game / Seats] = new bool?[Seats];
                group[game % Seats] = result.ChallengerWon;
                if (group.Any(won => won == null)) return;

                _scores.Add(group.Count(won => won == true) / (double)Seats);
                Statistics = GroupStatistics.Of(_scores, S0, S1, settings.Alpha, settings.Beta);
                Write($"group {game / Seats + 1}: {Summary()}");
                if (Statistics.Verdict != 0) stop.Cancel();
            }
        }

        /// <summary>
        /// Games and the challenger's wins, the win rate over the groups with its interval, the SPRT state and the trades
        /// </summary>
        public string Summary()
        {
            GroupStatistics s = Statistics;
            double games = Math.Max(1, _games);
            string trades = FormattableString.Invariant(
                $"offers per game C {_challengerOffers / games:F2} ({_challengerTrades / games:F2} traded), B {_baselineOffers / games:F2} ({_baselineTrades / games:F2} traded, {_challengerPartner / games:F2} with C); counters per game C {_challengerCounters / games:F2} B {_baselineCounters / games:F2}");
            string text = $"{_games} games, challenger won {_wins} | {s.Groups} groups";
            if (s.Groups < 2) return $"{text} | {trades}";
            string verdict = s.Verdict switch { 1 => $"H1 accepted (win rate >= {S1:P1})", -1 => $"H0 accepted (win rate <= {S0:P1})", _ => "open" };
            return FormattableString.Invariant($"{text}: win rate {s.Mean:P1} [{s.Low:P1}, {s.High:P1}] | SPRT [{S0:P1}, {S1:P1}] LLR {s.Llr:+0.00;-0.00} ({s.LowerBound:0.00}, {s.UpperBound:0.00}) {verdict} | {trades}");
        }

        private void Write(string line)
        {
            lock (log)
            {
                log.WriteLine($"{DateTime.Now:HH:mm} {line}");
                log.Flush();
            }
        }
    }

    /// <summary>
    /// Statistics of a table duel over its completed groups, each scored by the share of games the challenger won: the mean
    /// with its 95% interval, and the SPRT between H0 "the mean is S0" and H1 "the mean is S1" in the normal approximation
    /// (as PairStatistics), deciding only from MinGroups on and with a floor under the variance
    /// </summary>
    internal sealed record GroupStatistics(int Groups, double Mean, double Low, double High, double Llr, double LowerBound, double UpperBound)
    {
        public const int MinGroups = 30;
        public const double MinVariance = 0.01;

        public int Verdict => Groups < MinGroups ? 0 : Llr >= UpperBound ? 1 : Llr <= LowerBound ? -1 : 0;

        public static GroupStatistics Of(IReadOnlyList<double> scores, double s0, double s1, double alpha = 0.05, double beta = 0.05)
        {
            int n = scores.Count;
            double lower = Math.Log(beta / (1 - alpha)), upper = Math.Log((1 - beta) / alpha);
            if (n < 2) return new GroupStatistics(n, n == 1 ? scores[0] : 0, 0, 1, 0, lower, upper);

            double mean = scores.Average();
            double variance = scores.Sum(s => (s - mean) * (s - mean)) / (n - 1);
            double half = 1.96 * Math.Sqrt(variance / n);
            double llr = n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * Math.Max(variance, MinVariance));
            return new GroupStatistics(n, mean, mean - half, mean + half, llr, lower, upper);
        }
    }
}
