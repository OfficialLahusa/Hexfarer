"""The offline check of policy targets against a deep search (DatasetProcessing/target_check.py): the targets each variant
builds from a recorded position, and how well a target matches the deep search.

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))

from policy_targets import completed_q, gumbel_targets, visit_targets  # noqa: E402
from target_check import compare, load, position_metrics, variant_target  # noqa: E402


def position(prior, shallow_visits, shallow_pruned, shallow_q, deep_visits, deep_q, clean_visits=None, clean_q=None,
             value_range=(0.1, 0.5), decider=2):
    """A recorded position the way SelfPlayCollector target-check writes it (null for the values of unvisited moves)."""
    def values(q):
        return [None if v is None or (isinstance(v, float) and math.isnan(v)) else v for v in q]
    clean_visits = clean_visits if clean_visits is not None else shallow_visits
    clean_q = clean_q if clean_q is not None else shallow_q
    root = [0.25, 0.25, 0.25, 0.25]
    return {
        "game": 1, "action": 5, "round": 3, "decider": decider, "moves": len(prior), "prior": list(prior),
        "searches": {
            "shallow": {"visits": shallow_visits, "pruned": shallow_pruned, "q": values(shallow_q), "root": root, "range": list(value_range)},
            "clean": {"visits": clean_visits, "pruned": clean_visits, "q": values(clean_q), "root": root, "range": list(value_range)},
            "deep": {"visits": deep_visits, "pruned": deep_visits, "q": values(deep_q), "root": root, "range": list(value_range)},
        },
    }


SAMPLE = position(prior=[0.5, 0.3, 0.2], shallow_visits=[30, 12, 4], shallow_pruned=[30, 10, 0], shallow_q=[0.30, 0.34, 0.20],
                  deep_visits=[100, 800, 50], deep_q=[0.31, 0.36, 0.22], clean_visits=[33, 13, 0], clean_q=[0.30, 0.35, None])


class Variants(unittest.TestCase):
    def test_visit_variants_take_the_pruned_visits_of_the_noisy_search(self):
        np.testing.assert_allclose(variant_target(SAMPLE, "visits_t2"), visit_targets([30, 10, 0], 2.0))
        np.testing.assert_allclose(variant_target(SAMPLE, "visits_t1"), visit_targets([30, 10, 0], 1.0))
        np.testing.assert_allclose(variant_target(SAMPLE, "clean_visits_t2"), visit_targets([33, 13, 0], 2.0))

    def test_prior_variants(self):
        np.testing.assert_allclose(variant_target(SAMPLE, "prior"), [0.5, 0.3, 0.2])
        sharp = np.array([0.5, 0.3, 0.2]) ** (1 / 0.7)
        np.testing.assert_allclose(variant_target(SAMPLE, "prior_t0.7"), sharp / sharp.sum())

    def test_gumbel_variants_use_the_values_with_or_without_the_range(self):
        q = np.array([0.30, 0.34, 0.20])
        np.testing.assert_allclose(variant_target(SAMPLE, "gumbel_c0.1"),
                                   gumbel_targets([0.5, 0.3, 0.2], [30, 12, 4], q, c_scale=0.1, root_value=0.25))
        np.testing.assert_allclose(variant_target(SAMPLE, "gumbel_norm_c0.05"),
                                   gumbel_targets([0.5, 0.3, 0.2], [30, 12, 4], q, c_scale=0.05, value_range=(0.1, 0.5), root_value=0.25))
        # The clean search's unvisited move is completed from its own visited moves
        np.testing.assert_allclose(variant_target(SAMPLE, "clean_gumbel_c0.1"),
                                   gumbel_targets([0.5, 0.3, 0.2], [33, 13, 0], [0.30, 0.35, math.nan], c_scale=0.1, root_value=0.25))

    def test_the_root_value_is_the_deciders(self):
        record = json.loads(json.dumps(SAMPLE))
        record["searches"]["clean"]["root"] = [0.1, 0.2, 0.9, 0.3]
        # Only the unvisited move's completion depends on the root value
        expected = gumbel_targets([0.5, 0.3, 0.2], [33, 13, 0], [0.30, 0.35, math.nan], c_scale=0.1, root_value=0.9)
        np.testing.assert_allclose(variant_target(record, "clean_gumbel_c0.1"), expected)

    def test_unknown_variants_raise(self):
        with self.assertRaises(ValueError):
            variant_target(SAMPLE, "visits_x")


class Metrics(unittest.TestCase):
    def test_metrics_against_the_deep_search(self):
        target = np.array([0.7, 0.2, 0.1])
        m = position_metrics(SAMPLE, target)
        deep_share = np.array([100, 800, 50]) / 950
        self.assertEqual(m["top1"], 0.0)
        self.assertAlmostEqual(m["mass"], 0.2)
        self.assertAlmostEqual(m["ce"], -float(np.sum(deep_share * np.log(target))))
        deep_q = np.array([0.31, 0.36, 0.22])
        self.assertAlmostEqual(m["regret"], 0.36 - float(target @ deep_q))
        self.assertAlmostEqual(m["kl_prior"], float(np.sum(target * np.log(target / np.array([0.5, 0.3, 0.2])))))

    def test_the_deep_search_is_completed_where_it_left_moves_unvisited(self):
        record = json.loads(json.dumps(SAMPLE))
        record["searches"]["deep"]["visits"] = [100, 800, 0]
        record["searches"]["deep"]["q"] = [0.31, 0.36, None]
        target = np.array([0.1, 0.1, 0.8])
        deep_q = completed_q([0.5, 0.3, 0.2], [100, 800, 0], [0.31, 0.36, math.nan], root_value=0.25)
        self.assertAlmostEqual(position_metrics(record, target)["regret"], 0.36 - float(target @ deep_q))

    def test_the_deep_choice_is_its_most_visited_move_even_where_another_has_a_higher_value(self):
        # A rarely visited move can carry a noisy high value: the deep search's choice is still its most visited move
        record = json.loads(json.dumps(SAMPLE))
        record["searches"]["deep"]["q"] = [0.31, 0.36, 0.45]
        m = position_metrics(record, np.array([0.0, 1.0, 0.0]))
        self.assertEqual(m["top1"], 1.0)
        self.assertAlmostEqual(m["regret"], 0.0)

    def test_a_matching_target_scores_best(self):
        m = position_metrics(SAMPLE, np.array([0.0, 1.0, 0.0]))
        self.assertEqual(m["top1"], 1.0)
        self.assertAlmostEqual(m["mass"], 1.0)
        self.assertAlmostEqual(m["regret"], 0.0)


class Compare(unittest.TestCase):
    def test_compare_reads_the_file_and_pairs_every_variant_with_the_baseline(self):
        other = position(prior=[0.2, 0.8], shallow_visits=[5, 40], shallow_pruned=[0, 40], shallow_q=[0.2, 0.4],
                         deep_visits=[300, 700], deep_q=[0.38, 0.40])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "check.jsonl")
            with open(path, "w") as file:
                for record in (SAMPLE, other):
                    file.write(json.dumps(record) + "\n")
            positions = load(path)
        self.assertEqual(len(positions), 2)
        table = compare(positions, ["visits_t2", "gumbel_c0.1"], baseline="visits_t2")
        rows = {row["variant"]: row for row in table}
        self.assertEqual(rows["visits_t2"]["top1_diff"], 0.0)
        # Each variant's mean is the mean of its positions' metrics
        expected = np.mean([position_metrics(p, variant_target(p, "gumbel_c0.1"))["top1"] for p in positions])
        self.assertAlmostEqual(rows["gumbel_c0.1"]["top1"], expected)
        self.assertEqual(rows["gumbel_c0.1"]["positions"], 2)


if __name__ == "__main__":
    unittest.main()
