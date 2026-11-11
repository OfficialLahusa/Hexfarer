"""The self-play loop's evaluation command (run without a shell, so the paths filled into it can't add arguments or commands)
and each generation's training data (a window of the newest generations, which prefilled data from elsewhere can start).

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

from selfplay_loop import clearly_worse, command_arguments, training_data  # noqa: E402

TEMPLATE = ('"D:/Some Dir/Arena.exe" mirror --challenger "NEW={new}" --baseline "OLD={old}" --name {name} '
            '--iterations 5000 --sprt 0 40')


class CommandArgumentsTest(unittest.TestCase):
    def test_quoted_parts_stay_one_argument(self):
        args = command_arguments(TEMPLATE, new="C:/nets/new.onnx", old="C:/nets/old.onnx", name="loop_g003")
        self.assertEqual(args[:3], ["D:/Some Dir/Arena.exe", "mirror", "--challenger"])
        self.assertEqual(args[3], "NEW=C:/nets/new.onnx")
        self.assertEqual(args[5], "OLD=C:/nets/old.onnx")
        self.assertEqual(args[6:8], ["--name", "loop_g003"])

    def test_filled_in_paths_cannot_add_arguments_or_commands(self):
        args = command_arguments(TEMPLATE, new="C:/a b/x;calc & y.onnx", old="C:/o\" --evil.onnx", name="n")
        self.assertEqual(args[3], "NEW=C:/a b/x;calc & y.onnx")
        self.assertEqual(args[5], "OLD=C:/o\" --evil.onnx")
        self.assertEqual(len(args), len(command_arguments(TEMPLATE, new="a", old="b", name="n")))


def split(specs):
    """The data specs as (directory, share) pairs"""
    return [(spec.rsplit("@", 1)[0], float(spec.rsplit("@", 1)[1])) for spec in specs]


class TrainingDataTest(unittest.TestCase):
    PREFILL = [f"old/p{i}" for i in range(1, 11)]
    V2 = ["v2@0.1"]

    def test_without_prefill_the_window_holds_the_newest_generations_newest_first(self):
        prepared = [f"loop/g{i}" for i in range(1, 7)]
        data = split(training_data(prepared, [], 4, ["v2@0.4"]))
        self.assertEqual([d for d, _ in data], ["loop/g6", "loop/g5", "loop/g4", "loop/g3", "v2"])
        for _, share in data[:4]:
            self.assertAlmostEqual(share, 0.15)
        self.assertAlmostEqual(sum(share for _, share in data), 1.0)

    def test_prefilled_data_fills_the_window_behind_the_loops_own_generations(self):
        # Generation 1: its own data first (the validation runs on it), then the prefill from the newest to the oldest
        data = split(training_data(["loop/g1"], self.PREFILL, 12, self.V2))
        self.assertEqual([d for d, _ in data], ["loop/g1"] + self.PREFILL[::-1] + ["v2"])
        for _, share in data[:11]:
            self.assertAlmostEqual(share, 0.9 / 11)
        self.assertAlmostEqual(sum(share for _, share in data), 1.0)

    def test_the_oldest_prefill_leaves_first_and_none_is_left_once_the_window_is_the_loops_own(self):
        prepared = [f"loop/g{i}" for i in range(1, 5)]
        data = split(training_data(prepared, self.PREFILL, 12, self.V2))
        dirs = [d for d, _ in data]
        self.assertEqual(dirs[:4], ["loop/g4", "loop/g3", "loop/g2", "loop/g1"])
        self.assertEqual(dirs[4:12], self.PREFILL[:1:-1])
        self.assertNotIn("old/p1", dirs)
        self.assertNotIn("old/p2", dirs)
        for _, share in data[:12]:
            self.assertAlmostEqual(share, 0.9 / 12)

        prepared = [f"loop/g{i}" for i in range(1, 15)]
        dirs = [d for d, _ in split(training_data(prepared, self.PREFILL, 12, self.V2))]
        self.assertEqual(dirs, [f"loop/g{i}" for i in range(14, 2, -1)] + ["v2"])

    def test_the_extra_data_keep_their_shares(self):
        data = training_data(["loop/g1"], self.PREFILL[:2], 4, ["v2@0.1", "loop3/g6@0.05"])
        self.assertEqual(data[-2:], ["v2@0.1", "loop3/g6@0.05"])
        self.assertAlmostEqual(sum(share for _, share in split(data[:3])), 0.85)


class ClearlyWorseTest(unittest.TestCase):
    def test_only_an_interval_entirely_below_zero_counts(self):
        self.assertTrue(clearly_worse("... -147 Elo [-273, -50] ..."))
        self.assertFalse(clearly_worse("... -14 Elo [-56, +28] ..."))
        self.assertFalse(clearly_worse("no output"))


if __name__ == "__main__":
    unittest.main()
