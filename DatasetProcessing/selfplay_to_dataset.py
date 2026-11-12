"""
Turn self-play runs (SelfPlayCollector output) into the arrays train.py reads.

A run directory holds metadata.json, games.csv (one line per game: seed, result, where its save file is),
samples.csv (one line per recorded position: game, round, acting player, the search's mean result per player at the
root q0..q3, the row of its features and the root visits) and states_c*.f16 (float16 feature rows).

Writes into the output directory, one row per position of the games with a result (games stopped at the action
limit are left out):
- states.npy (n, features) float16
- root_q.npy (n, 4) float32: the search's value of the position
- winners.npy (n,) int8: the game's winner
- win_rates.npy (n, 4) float32: the training target, lambda * one-hot(winner) + (1 - lambda) * root_q
- placements.npy (n, 4, 4), vp_shares.npy (n, 4), rounds_left.npy (n,): the auxiliary labels from the game's end
- actors.npy (n,): the player deciding at the position
- match_seeds.npy (n,) uint64 and groups.npy (n,) int64: the game's seed, unique across runs
- idx_train.npy, idx_val.npy, idx_test.npy: split by game
If every run has the policy files of SelfPlayCollector (policy_counts.i32 and the rest), also the policy targets:
- policy_offsets.npy (n + 1,) int64: the moves of position i are rows offsets[i]:offsets[i + 1] of the arrays below
- policy_tokens.npy (moves, max_tokens) int16 and policy_weights.npy (moves, max_tokens) uint8: the encoded legal moves
  (token -1 unused), in the order of the root's children
- policy_targets.npy (moves,) float32: the share of the root visits per move, or with --policy-target gumbel the Gumbel
  target from the search's values (policy_targets.gumbel_targets, from the run's policy_values.f32 and policy_ranges.f32:
  raw visits, the decider's value and the prior per move, the value range per position; positions without priors keep
  their visit shares)
- policy_layout.json: the token ranges
- policy_target.json: which target it is ({"kind": "visits"} or {"kind": "gumbel", ...} with its settings), so train.py
  applies its temperature only to visit shares, and where the runs recorded the priors the targets' mean KL from them (the
  distillation margin, for visit shares before the temperature) and their mean entropy

Usage: python selfplay_to_dataset.py RUN [RUN ...] --output DIR [--target-lambda 0.5] [--val 0.1] [--test 0.0] [--seed 42]
       [--policy-target visits|gumbel] [--c-scale 0.05] [--c-visit 50]
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

from playout_results import PLAYER_COUNT, placements
from policy_targets import gumbel_targets
from split_dataset import split_indices

POLICY_TARGETS = ("visits", "gumbel")
VALUES_FILE, RANGES_FILE = "policy_values.f32", "policy_ranges.f32"
# Raw visits, the decider's value and the prior per move in VALUES_FILE
FLOATS_PER_MOVE = 3


def read_run(run_dir: str):
    """The positions of a run's games with a result, as a dict of arrays (states as (chunk, row) references)."""
    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as file:
        feature_dim = json.load(file)["model"]["feature_dim"]

    games = {}
    with open(os.path.join(run_dir, "games.csv"), newline="") as file:
        for game in csv.DictReader(file):
            if game["finished"] == "1":
                games[game["game"]] = game

    columns = {name: [] for name in ("seed", "winner", "vp", "rounds_left", "actor", "q", "chunk", "row", "line", "visits")}
    lines = 0
    with open(os.path.join(run_dir, "samples.csv"), newline="") as file:
        for line, sample in enumerate(csv.DictReader(file)):
            lines += 1
            game = games.get(sample["game"])
            if game is None:
                continue
            columns["line"].append(line)
            columns["visits"].append(sample["visits"])
            columns["seed"].append(int(game["seed"]))
            columns["winner"].append(int(game["winner"]))
            columns["vp"].append([int(game[f"vp{p}"]) for p in range(PLAYER_COUNT)])
            columns["rounds_left"].append(int(game["rounds"]) - int(sample["round"]))
            columns["actor"].append(int(sample["actor"]))
            columns["q"].append([float(sample[f"q{p}"]) for p in range(PLAYER_COUNT)])
            columns["chunk"].append(int(sample["chunk"]))
            columns["row"].append(int(sample["row"]))

    arrays = {
        "seed": np.array(columns["seed"], dtype=np.uint64),
        "winner": np.array(columns["winner"], dtype=np.int8),
        "vp": np.array(columns["vp"], dtype=np.float64).reshape(-1, PLAYER_COUNT),
        "rounds_left": np.array(columns["rounds_left"], dtype=np.float32),
        "actor": np.array(columns["actor"], dtype=np.int64),
        "q": np.array(columns["q"], dtype=np.float32).reshape(-1, PLAYER_COUNT),
        "chunk": np.array(columns["chunk"], dtype=np.int64),
        "row": np.array(columns["row"], dtype=np.int64),
        "line": np.array(columns["line"], dtype=np.int64),
        "lines": lines,
        "visits": columns["visits"],
    }
    return feature_dim, arrays, len(games)


POLICY_FILES = ("policy_counts.i32", "policy_tokens.i16", "policy_weights.u8", "policy_layout.json")
VIEWS_FILE = "views.f16"



def read_policy(run_dir: str, arrays: dict, max_tokens: int, policy_target: str = "visits", c_scale: float = 0.05, c_visit: float = 50.0):
    """The encoded legal moves and targets of the run's kept positions (see the module docstring), how many of them fell
    back to their visit shares for lack of priors, and where the run recorded the priors the sums over those positions of
    the targets' KL from the priors (the distillation margin) and of their entropies, with the number of such positions"""
    counts = np.fromfile(os.path.join(run_dir, "policy_counts.i32"), dtype="<i4").astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)])
    tokens = np.fromfile(os.path.join(run_dir, "policy_tokens.i16"), dtype="<i2").reshape(-1, max_tokens)
    weights = np.fromfile(os.path.join(run_dir, "policy_weights.u8"), dtype=np.uint8).reshape(-1, max_tokens)
    if len(tokens) != starts[-1]:
        raise ValueError(f"{run_dir}: the policy files hold {len(tokens)} moves, the counts {starts[-1]}")

    lines = arrays["line"]
    if len(lines) and lines.max() >= len(counts):
        raise ValueError(f"{run_dir}: samples.csv has more lines than policy_counts.i32 entries")
    kept_counts = counts[lines]
    rows = np.concatenate([np.arange(starts[line], starts[line + 1]) for line in lines]) if len(lines) else np.array([], dtype=np.int64)
    paths = [os.path.join(run_dir, name) for name in (VALUES_FILE, RANGES_FILE)]
    recorded_values = all(os.path.exists(path) for path in paths)
    if policy_target == "gumbel" and not recorded_values:
        raise ValueError(f"{run_dir} has no {VALUES_FILE} and {RANGES_FILE} (collected before they were recorded), so no Gumbel targets")
    if recorded_values:
        move_values = np.fromfile(paths[0], dtype="<f4").reshape(-1, FLOATS_PER_MOVE)
        ranges = np.fromfile(paths[1], dtype="<f4").reshape(-1, 2)
        if len(move_values) != starts[-1] or len(ranges) != len(counts):
            raise ValueError(f"{run_dir}: {VALUES_FILE} or {RANGES_FILE} don't match the policy files")
    targets = np.zeros(len(rows), dtype=np.float32)
    fallbacks = 0
    kl_sum, entropy_sum, with_priors = 0.0, 0.0, 0
    offset = 0
    for i, (count, visits) in enumerate(zip(kept_counts, arrays["visits"])):
        pairs = [pair.split(":") for pair in visits.split()]
        moves = np.array([int(move) for move, _ in pairs], dtype=np.int64)
        if moves.size and moves.max() >= count:
            raise ValueError(f"{run_dir}: a visited move index lies beyond the position's {count} moves")
        values = np.array([float(v) for _, v in pairs], dtype=np.float64)
        shares = np.zeros(count)
        shares[moves] = values / values.sum()
        prior = None
        if recorded_values:
            line = lines[i]
            recorded = move_values[starts[line]:starts[line + 1]].astype(np.float64)
            raw, q, prior = recorded[:, 0], recorded[:, 1], recorded[:, 2]
            if np.isnan(prior).any() or raw.sum() <= 0:
                # No priors (a search that reused its root's): the visits stay the target
                prior = None
                fallbacks += policy_target == "gumbel"
            elif policy_target == "gumbel":
                low, high = ranges[line]
                value_range = None if np.isnan(low) or np.isnan(high) else (float(low), float(high))
                root_value = float(arrays["q"][i][arrays["actor"][i]])
                shares = gumbel_targets(prior, raw, q, c_visit=c_visit, c_scale=c_scale, value_range=value_range, root_value=root_value)
        if prior is not None:
            # How far the target moves from the net's priors, and how sharp it is
            kept = shares > 0
            kl_sum += float(np.sum(shares[kept] * np.log(shares[kept] / np.maximum(prior[kept], 1e-30))))
            entropy_sum += float(-np.sum(shares[kept] * np.log(shares[kept])))
            with_priors += 1
        targets[offset:offset + count] = shares
        offset += count
    return kept_counts, tokens[rows], weights[rows], targets, fallbacks, (kl_sum, entropy_sum, with_priors)


def convert(runs, output_dir: str, target_lambda: float = 0.5, val: float = 0.1, test: float = 0.0, seed: int = 42,
            policy_target: str = "visits", c_scale: float = 0.05, c_visit: float = 50.0):
    if policy_target not in POLICY_TARGETS:
        raise ValueError(f"Unknown policy target \"{policy_target}\" ({', '.join(POLICY_TARGETS)})")
    os.makedirs(output_dir, exist_ok=True)
    parts = [(run, *read_run(run)) for run in runs]
    feature_dims = {dim for _, dim, _, _ in parts}
    if len(feature_dims) != 1:
        raise ValueError(f"The runs have different feature layouts: {sorted(feature_dims)}")
    feature_dim = feature_dims.pop()
    n = sum(len(arrays["seed"]) for _, _, arrays, _ in parts)
    if n == 0:
        raise ValueError("The runs hold no positions of games with a result.")

    # States, copied chunk by chunk in the order of the positions
    states = np.lib.format.open_memmap(os.path.join(output_dir, "states.npy"), mode="w+", dtype=np.float16, shape=(n, feature_dim))
    start = 0
    for run, _, arrays, _ in parts:
        for chunk in np.unique(arrays["chunk"]):
            path = os.path.join(run, f"states_c{chunk:04d}.f16")
            rows = np.memmap(path, dtype="<f2", mode="r").reshape(-1, feature_dim)
            positions = np.flatnonzero(arrays["chunk"] == chunk)
            states[start + positions] = rows[arrays["row"][positions]]
        start += len(arrays["seed"])
    states.flush()
    del states

    # The belief-aware inputs (SelfPlayCollector view-features), one row per line of samples.csv, if every run has them
    if all(os.path.exists(os.path.join(run, VIEWS_FILE)) for run, *_ in parts):
        widths = set()
        for run, _, arrays, _ in parts:
            size = os.path.getsize(os.path.join(run, VIEWS_FILE)) // 2
            if arrays["lines"] == 0 or size % arrays["lines"]:
                raise ValueError(f"{run}: {VIEWS_FILE} doesn't hold one row per line of samples.csv")
            widths.add(size // arrays["lines"])
        if len(widths) != 1:
            raise ValueError(f"The runs' view features have different widths: {sorted(widths)}")
        view_dim = widths.pop()
        views = np.lib.format.open_memmap(os.path.join(output_dir, "views.npy"), mode="w+", dtype=np.float16, shape=(n, view_dim))
        start = 0
        for run, _, arrays, _ in parts:
            rows = np.memmap(os.path.join(run, VIEWS_FILE), dtype="<f2", mode="r").reshape(-1, view_dim)
            views[start:start + len(arrays["line"])] = rows[arrays["line"]]
            start += len(arrays["line"])
        views.flush()
        del views

    def joined(name):
        return np.concatenate([arrays[name] for _, _, arrays, _ in parts])

    winners, vp, q = joined("winner"), joined("vp"), joined("q")
    one_hot = np.eye(PLAYER_COUNT, dtype=np.float32)[winners]
    vp_totals = vp.sum(axis=1, keepdims=True)
    match_seeds = joined("seed")
    labels = {
        "root_q": q,
        "winners": winners,
        "win_rates": (target_lambda * one_hot + (1 - target_lambda) * q).astype(np.float32),
        "placements": placements(winners.astype(np.int64), vp).astype(np.float32),
        "vp_shares": np.divide(vp, vp_totals, out=np.full(vp.shape, 1.0 / PLAYER_COUNT), where=vp_totals > 0).astype(np.float32),
        "rounds_left": joined("rounds_left"),
        "actors": joined("actor"),
        "match_seeds": match_seeds,
        "groups": match_seeds.view(np.int64),
    }
    for name, array in labels.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), array)

    if all(os.path.exists(os.path.join(run, name)) for run, *_ in parts for name in POLICY_FILES):
        layouts = []
        for run, *_ in parts:
            with open(os.path.join(run, "policy_layout.json"), encoding="utf-8") as file:
                layouts.append(json.load(file))
        if any(layout != layouts[0] for layout in layouts):
            raise ValueError("The runs encode the moves differently (policy_layout.json)")
        max_tokens = layouts[0]["max_tokens"][0]
        policy = [read_policy(run, arrays, max_tokens, policy_target, c_scale, c_visit) for run, _, arrays, _ in parts]
        counts = np.concatenate([part[0] for part in policy])
        np.save(os.path.join(output_dir, "policy_offsets.npy"), np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
        for i, name in enumerate(("policy_tokens", "policy_weights", "policy_targets"), start=1):
            np.save(os.path.join(output_dir, f"{name}.npy"), np.concatenate([part[i] for part in policy]))
        with open(os.path.join(output_dir, "policy_layout.json"), "w", encoding="utf-8") as file:
            json.dump(layouts[0], file, indent=2)
        # Which target this is: train.py flattens visit shares with its temperature, a Gumbel target is trained as it is
        kind = {"kind": policy_target}
        if policy_target == "gumbel":
            kind.update(c_scale=c_scale, c_visit=c_visit, visit_fallbacks=int(sum(part[4] for part in policy)))
        # The distillation margin where the runs recorded the priors: the targets' mean KL from them (visit shares before
        # train.py's temperature) and their mean entropy
        with_priors = sum(part[5][2] for part in policy)
        if with_priors:
            kind.update(positions_with_priors=with_priors, mean_kl_prior=sum(part[5][0] for part in policy) / with_priors,
                        mean_entropy=sum(part[5][1] for part in policy) / with_priors)
            print(f"policy targets ({policy_target}): KL from the priors {kind['mean_kl_prior']:.4f}, entropy {kind['mean_entropy']:.4f} "
                  f"on average over {with_priors} positions")
        with open(os.path.join(output_dir, "policy_target.json"), "w", encoding="utf-8") as file:
            json.dump(kind, file, indent=2)
    elif policy_target == "gumbel":
        raise ValueError("Gumbel targets need the runs' policy files")

    # Split by game; without a test share everything that isn't validation is training
    groups = labels["groups"]
    if test > 0:
        train_idx, val_idx, test_idx = split_indices(groups, test, val, seed)
    else:
        unique = np.unique(groups)
        val_groups = np.random.default_rng(seed).choice(unique, size=int(round(len(unique) * val)), replace=False)
        in_val = np.isin(groups, val_groups)
        train_idx, val_idx, test_idx = np.flatnonzero(~in_val), np.flatnonzero(in_val), np.array([], dtype=np.int64)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        np.save(os.path.join(output_dir, f"idx_{name}.npy"), idx.astype(np.int64))

    games = sum(count for _, _, _, count in parts)
    print(f"{n} positions of {games} games with a result from {len(runs)} run(s): "
          f"{len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test, target lambda {target_lambda}")
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Turn self-play runs into the arrays train.py reads")
    parser.add_argument("runs", nargs="+", help="run directories (or glob patterns)")
    parser.add_argument("--output", required=True)
    # Weight of the game result against the search's value in the target
    parser.add_argument("--target-lambda", type=float, default=0.5)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    # The policy target: the visit shares, or Gumbel targets from the recorded values (see the module docstring)
    parser.add_argument("--policy-target", choices=POLICY_TARGETS, default="visits")
    parser.add_argument("--c-scale", type=float, default=0.05, help="Gumbel targets: scale of the values (normalized to the tree's range)")
    parser.add_argument("--c-visit", type=float, default=50.0, help="Gumbel targets: visits added to the most visited move's in the scale")
    args = parser.parse_args(argv)

    runs = sorted({path for pattern in args.runs for path in (glob.glob(pattern) or [pattern])})
    convert(runs, args.output, args.target_lambda, args.val, args.test, args.seed, args.policy_target, args.c_scale, args.c_visit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
