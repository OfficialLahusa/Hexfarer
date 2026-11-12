"""Policy targets from a search's root (DatasetProcessing/policy_targets.py): visit shares with a temperature, and Gumbel
targets from completed Q values (Danihelka et al. 2022).

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import math
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))

from policy_targets import completed_q, gumbel_targets, visit_targets  # noqa: E402


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits - logits.max())
    return e / e.sum()


class VisitTargets(unittest.TestCase):
    def test_shares_with_and_without_a_temperature(self):
        np.testing.assert_allclose(visit_targets([4, 1, 0]), [0.8, 0.2, 0.0])
        # T 2 takes the square root of the visits before normalizing (what the loops train on)
        np.testing.assert_allclose(visit_targets([4, 1, 0], temperature=2.0), [2 / 3, 1 / 3, 0.0])

    def test_no_visits_raise(self):
        with self.assertRaises(ValueError):
            visit_targets([0, 0])


class CompletedQ(unittest.TestCase):
    def test_unvisited_moves_get_the_mixed_value(self):
        prior = [0.5, 0.3, 0.2]
        visits = [3, 1, 0]
        q = [0.6, 0.2, math.nan]
        # v_mix = (v + N / (sum of the visited moves' priors) * sum over visited of prior * q) / (1 + N), N = 4
        v_mix = (0.4 + 4 / 0.8 * (0.5 * 0.6 + 0.3 * 0.2)) / 5
        np.testing.assert_allclose(completed_q(prior, visits, q, root_value=0.4), [0.6, 0.2, v_mix])

    def test_the_mix_weighs_the_visited_values_by_their_priors_not_their_visits(self):
        # The second move has most visits, the first most prior: v_mix leans to the first move's value
        completed = completed_q([0.9, 0.05, 0.05], [1, 9, 0], [1.0, 0.0, math.nan], root_value=0.0)
        self.assertAlmostEqual(completed[2], (0 + 10 / 0.95 * 0.9) / 11)

    def test_values_of_visited_moves_stay_as_they_are(self):
        np.testing.assert_allclose(completed_q([0.2, 0.8], [5, 7], [0.1, 0.9], root_value=0.5), [0.1, 0.9])


class GumbelTargets(unittest.TestCase):
    def test_a_scale_of_zero_gives_the_prior(self):
        prior = np.array([0.6, 0.3, 0.1])
        np.testing.assert_allclose(gumbel_targets(prior, [5, 3, 0], [0.9, 0.1, math.nan], c_scale=0.0, root_value=0.5), prior)

    def test_the_formula(self):
        prior = [1 / 3, 1 / 3, 1 / 3]
        visits = [2, 2, 0]
        q = [1.0, 0.0, math.nan]
        # sigma(q) = (c_visit + max visits) * c_scale * q, here (50 + 2) * 0.1 = 5.2; the unvisited move's v_mix is (0.5 + 2) / 5
        expected = softmax(np.log(prior) + 5.2 * np.array([1.0, 0.0, 0.5]))
        np.testing.assert_allclose(gumbel_targets(prior, visits, q, c_visit=50, c_scale=0.1, root_value=0.5), expected)

    def test_values_are_normalized_to_the_given_range(self):
        prior = [0.5, 0.5]
        visits = [4, 4]
        q = [0.30, 0.20]
        # In the range [0.2, 0.4] the values become 0.5 and 0
        expected = softmax(np.log(prior) + (50 + 4) * 0.1 * np.array([0.5, 0.0]))
        np.testing.assert_allclose(gumbel_targets(prior, visits, q, value_range=(0.2, 0.4)), expected)
        # Without a range they stay raw win probabilities
        expected_raw = softmax(np.log(prior) + (50 + 4) * 0.1 * np.array([0.30, 0.20]))
        np.testing.assert_allclose(gumbel_targets(prior, visits, q), expected_raw)

    def test_a_large_scale_puts_the_mass_on_the_best_value(self):
        target = gumbel_targets([0.7, 0.2, 0.1], [10, 10, 10], [0.2, 0.5, 0.4], c_scale=10.0)
        self.assertEqual(int(np.argmax(target)), 1)
        self.assertGreater(target[1], 0.99)

    def test_a_single_visited_move_gets_the_whole_target(self):
        # The search's shortcut for a winning move visits only it: the prior alone would teach nothing there
        np.testing.assert_allclose(gumbel_targets([0.2, 0.5, 0.3], [0, 7, 0], [math.nan, 1.0, math.nan]), [0.0, 1.0, 0.0])

    def test_targets_sum_to_one_and_follow_the_moves(self):
        prior = np.array([0.1, 0.4, 0.3, 0.2])
        visits = np.array([3, 20, 0, 9])
        q = np.array([0.35, 0.30, math.nan, 0.32])
        target = gumbel_targets(prior, visits, q, root_value=0.3)
        self.assertAlmostEqual(float(target.sum()), 1.0)
        # Reordering the moves reorders the target the same way
        order = np.array([2, 0, 3, 1])
        np.testing.assert_allclose(gumbel_targets(prior[order], visits[order], q[order], root_value=0.3), target[order])

    def test_a_zero_prior_stays_near_zero(self):
        target = gumbel_targets([0.0, 0.5, 0.5], [0, 3, 3], [math.nan, 0.4, 0.6])
        self.assertLess(target[0], 1e-9)


if __name__ == "__main__":
    unittest.main()
