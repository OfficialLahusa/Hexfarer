"""Averaging checkpoints of one net (weight averaging, e.g. of a loop's plateau generations): every floating tensor is the
mean over the checkpoints in its own dtype, every other tensor must be the same in all of them, the rest of the checkpoint
comes from the last one without optimizer state, and checkpoints that don't fit together are refused.

Skipped when torch is not installed.
Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import os
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

try:
    import torch
    from average_checkpoints import average_checkpoints
except ImportError as error:  # pragma: no cover - depends on the environment
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


def checkpoint(scale, steps=7, extra=None):
    """A small checkpoint whose weights are multiples of scale, with a buffer that isn't floating"""
    state = {
        "net.weight": torch.full((2, 3), float(scale)),
        "net.bias": torch.arange(3, dtype=torch.float32) * scale,
        "norm.mean": torch.tensor([1.0, 2.0]),
        "head.half": torch.full((2,), float(scale), dtype=torch.float16),
        "counter": torch.tensor(steps, dtype=torch.long),
    }
    state.update(extra or {})
    return {"state_dict": state, "hyper_parameters": {"hidden": 8, "scale": scale}, "epoch": scale,
            "optimizer_states": [{"step": torch.tensor(1.0)}], "lr_schedulers": []}


@unittest.skipIf(IMPORT_ERROR is not None, f"torch missing: {IMPORT_ERROR}")
class AverageCheckpointsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def save(self, name, data):
        path = os.path.join(self.dir.name, name)
        torch.save(data, path)
        return path

    def average(self, paths):
        out = os.path.join(self.dir.name, "average.ckpt")
        average_checkpoints(paths, out)
        return torch.load(out, map_location="cpu", weights_only=True)

    def test_floating_tensors_are_the_mean_of_all_checkpoints_in_their_own_dtype(self):
        result = self.average([self.save(f"c{i}.ckpt", checkpoint(i)) for i in (1, 2, 6)])["state_dict"]
        torch.testing.assert_close(result["net.weight"], torch.full((2, 3), 3.0))
        torch.testing.assert_close(result["net.bias"], torch.tensor([0.0, 3.0, 6.0]))
        torch.testing.assert_close(result["norm.mean"], torch.tensor([1.0, 2.0]))
        self.assertEqual(result["head.half"].dtype, torch.float16)
        torch.testing.assert_close(result["head.half"], torch.full((2,), 3.0, dtype=torch.float16))
        self.assertEqual(list(result), ["net.weight", "net.bias", "norm.mean", "head.half", "counter"])

    def test_the_rest_comes_from_the_last_checkpoint_without_optimizer_state(self):
        result = self.average([self.save(f"c{i}.ckpt", checkpoint(i)) for i in (1, 2)])
        self.assertEqual(result["hyper_parameters"], {"hidden": 8, "scale": 2})
        self.assertEqual(result["epoch"], 2)
        self.assertNotIn("optimizer_states", result)
        self.assertNotIn("lr_schedulers", result)
        self.assertEqual(result["state_dict"]["counter"].dtype, torch.long)
        self.assertEqual(result["state_dict"]["counter"].item(), 7)

    def test_one_checkpoint_averages_to_itself(self):
        result = self.average([self.save("c.ckpt", checkpoint(5))])["state_dict"]
        torch.testing.assert_close(result["net.weight"], torch.full((2, 3), 5.0))

    def test_checkpoints_that_do_not_fit_together_are_refused(self):
        base = self.save("a.ckpt", checkpoint(1))
        more_weights = self.save("b.ckpt", checkpoint(2, extra={"more": torch.zeros(1)}))
        reshaped = checkpoint(2)
        reshaped["state_dict"]["net.weight"] = torch.zeros(3, 2)
        reshaped = self.save("c.ckpt", reshaped)
        other_counter = self.save("d.ckpt", checkpoint(2, steps=8))
        for path in (more_weights, reshaped, other_counter):
            with self.subTest(path=os.path.basename(path)), self.assertRaises(ValueError):
                average_checkpoints([base, path], os.path.join(self.dir.name, "x.ckpt"))
        with self.assertRaises(ValueError):
            average_checkpoints([], os.path.join(self.dir.name, "x.ckpt"))


if __name__ == "__main__":
    unittest.main()
