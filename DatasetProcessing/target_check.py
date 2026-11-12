"""Which policy target carries more of what a deeper search knows: reads the positions SelfPlayCollector target-check
wrote (each searched shallow with root noise as in self-play, shallow without noise, and deep, all on the same net) and
scores every target variant built from a shallow search against the deep one.

Variants:
  prior, prior_tX            the net's priors, sharpened with a temperature X
  visits_tX, clean_visits_tX the shallow search's visit shares with temperature X: of the noisy search after target
                             pruning (what self-play records, the loops train on visits_t2), or of the search without noise
  gumbel_cX, gumbel_norm_cX  Gumbel targets (policy_targets.gumbel_targets) from the noisy search with c_scale X, on the
                             raw win probabilities or normalized to the tree's value range; clean_ prefixed: from the
                             search without noise
Metrics per position, against the deep search:
  top1      whether the target's most likely move is the deep search's most visited move
  mass      the target's share on that move
  ce        cross-entropy of the deep search's visit shares under the target (lower is better)
  regret    the value of the deep search's choice minus the target's expected deep value (completed Q, lower is better)
  kl_prior  how far the target moves from the prior (KL(target || prior))
  entropy   the target's entropy

Usage: python DatasetProcessing/target_check.py CHECK.jsonl [--baseline visits_t2] [--json OUT]
"""

import argparse
import json
import math
import re
import sys

import numpy as np

from policy_targets import completed_q, gumbel_targets, visit_targets

DEFAULT_VARIANTS = (["prior", "prior_t0.7", "visits_t1", "visits_t1.5", "visits_t2", "clean_visits_t1", "clean_visits_t2"]
                    + [f"{source}gumbel{norm}_c{c}" for source in ("", "clean_") for norm in ("", "_norm")
                       for c in ("0.02", "0.05", "0.1", "0.2", "0.5")])
METRICS = ("top1", "mass", "ce", "regret", "kl_prior", "entropy")
# Floor of a target share inside a logarithm
EPS = 1e-12


def load(path: str) -> list:
    """The positions of a target-check file, one JSON object per line."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _values(search: dict) -> np.ndarray:
    """A search's values per move, NaN where it has none (null in the file)."""
    return np.array([math.nan if v is None else v for v in search["q"]], dtype=np.float64)


def _root_value(record: dict, search: dict) -> float:
    return float(search["root"][record["decider"]])


def variant_target(record: dict, variant: str) -> np.ndarray:
    """The target a variant builds from a recorded position."""
    prior = np.asarray(record["prior"], dtype=np.float64)
    searches = record["searches"]
    if variant == "prior":
        return prior / prior.sum()
    match = re.fullmatch(r"prior_t([0-9.]+)", variant)
    if match:
        sharp = np.maximum(prior, EPS) ** (1.0 / float(match.group(1)))
        return sharp / sharp.sum()
    match = re.fullmatch(r"(clean_)?visits_t([0-9.]+)", variant)
    if match:
        if match.group(1):
            return visit_targets(searches["clean"]["visits"], float(match.group(2)))
        return visit_targets(searches["shallow"]["pruned"], float(match.group(2)))
    match = re.fullmatch(r"(clean_)?gumbel(_norm)?_c([0-9.]+)", variant)
    if match:
        search = searches["clean" if match.group(1) else "shallow"]
        value_range = tuple(search["range"]) if match.group(2) and search.get("range") else None
        return gumbel_targets(prior, search["visits"], _values(search), c_scale=float(match.group(3)), value_range=value_range,
                              root_value=_root_value(record, search))
    raise ValueError(f"Unknown target variant \"{variant}\".")


def position_metrics(record: dict, target) -> dict:
    """How well a target matches the position's deep search (see the module's description)."""
    target = np.asarray(target, dtype=np.float64)
    prior = np.asarray(record["prior"], dtype=np.float64)
    deep = record["searches"]["deep"]
    deep_visits = np.asarray(deep["visits"], dtype=np.float64)
    deep_share = deep_visits / deep_visits.sum()
    best = int(np.argmax(deep_visits))
    deep_q = completed_q(prior, deep_visits, _values(deep), root_value=_root_value(record, deep))
    safe = np.maximum(target, EPS)
    positive = target > 0
    return {
        "top1": float(int(np.argmax(target)) == best),
        "mass": float(target[best]),
        "ce": float(-(deep_share * np.log(safe)).sum()),
        "regret": float(deep_q[best] - target @ deep_q),
        "kl_prior": float((target[positive] * np.log(target[positive] / np.maximum(prior[positive], EPS))).sum()),
        "entropy": float(-(target[positive] * np.log(target[positive])).sum()),
    }


def compare(positions: list, variants, baseline: str = "visits_t2") -> list:
    """Every variant's mean metrics over the positions, and its paired difference to the baseline with a standard error."""
    per_variant = {variant: [position_metrics(p, variant_target(p, variant)) for p in positions] for variant in variants}
    if baseline not in per_variant:
        per_variant[baseline] = [position_metrics(p, variant_target(p, baseline)) for p in positions]
    base = {metric: np.array([m[metric] for m in per_variant[baseline]]) for metric in METRICS}
    table = []
    for variant in variants:
        row = {"variant": variant, "positions": len(positions)}
        for metric in METRICS:
            values = np.array([m[metric] for m in per_variant[variant]])
            diff = values - base[metric]
            row[metric] = float(values.mean())
            row[f"{metric}_diff"] = float(diff.mean())
            row[f"{metric}_se"] = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        table.append(row)
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check", help="the file SelfPlayCollector target-check wrote")
    parser.add_argument("--baseline", default="visits_t2", help="variant the differences are taken to (default: the loops' target)")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--json", help="also write the table as JSON")
    args = parser.parse_args(argv)

    positions = load(args.check)
    table = compare(positions, args.variants, args.baseline)
    print(f"{len(positions)} positions, differences to {args.baseline} (mean ± 1.96 SE)")
    print(f"{'variant':24} {'top1':>7} {'Δtop1':>15} {'mass':>7} {'ce':>7} {'Δce':>15} {'regret':>8} {'Δregret':>17} {'kl_prior':>8} {'entropy':>7}")
    for row in table:
        def diff(metric, scale=1.0, digits=3):
            return f"{row[metric + '_diff'] * scale:+.{digits}f}±{1.96 * row[metric + '_se'] * scale:.{digits}f}"
        print(f"{row['variant']:24} {row['top1']:7.3f} {diff('top1'):>15} {row['mass']:7.3f} {row['ce']:7.3f} {diff('ce'):>15} "
              f"{row['regret'] * 1000:8.2f} {diff('regret', 1000, 2):>17} {row['kl_prior']:8.3f} {row['entropy']:7.3f}")
    print("(regret in thousandths of a win probability)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(table, file, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
