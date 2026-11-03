"""Merging the parts of a reanalysis (DatasetProcessing/merge_reanalysis.py).

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))

from merge_reanalysis import merge  # noqa: E402

ROWS = 7
MAX_TOKENS = 3
LAYOUT = {"move_types": [0, 19], "intersections": [19, 54], "max_tokens": [MAX_TOKENS, 0]}


def moves_of(row, count):
    """Moves of a row that tell their row and index apart in every array"""
    tokens = np.array([[row, 10 + m, -1] for m in range(count)], dtype=np.int16)
    weights = np.array([[1, row + m + 2, 0] for m in range(count)], dtype=np.uint8)
    shares = np.array([(m + 1) / (count * (count + 1) / 2) + row / 1000 for m in range(count)], dtype=np.float32)
    return tokens, weights, shares


def root_value(row):
    return np.array([row, row + 0.25, row + 0.5, row + 0.75], dtype=np.float32) / 10


def write_part(directory, searched, rows=ROWS, status="finished", layout=LAYOUT, seed="20261025", commit="56c10df", model="net.onnx"):
    """A part the way DatasetCollector --reanalyze writes it: searched maps a row to its move count"""
    os.makedirs(directory)
    counts = [searched.get(row, 0) for row in range(rows)]
    per_row = [moves_of(row, count) for row, count in enumerate(counts) if count > 0]
    empty = (np.zeros((0, MAX_TOKENS), np.int16), np.zeros((0, MAX_TOKENS), np.uint8), np.zeros(0, np.float32))
    np.save(os.path.join(directory, "policy_offsets.npy"), np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
    for i, name in enumerate(("policy_tokens", "policy_weights", "policy_targets")):
        np.save(os.path.join(directory, f"{name}.npy"), np.concatenate([moves[i] for moves in per_row]) if per_row else empty[i])
    np.save(os.path.join(directory, "root_q.npy"),
            np.stack([root_value(row) if counts[row] else np.full(4, np.nan, np.float32) for row in range(rows)]))
    with open(os.path.join(directory, "policy_layout.json"), "w", encoding="utf-8") as file:
        json.dump(layout, file)
    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump({"format": 1, "kind": "reanalysis", "status": status, "commit": commit, "model": {"file": model},
                   "source": {"collection_seed": seed},
                   "result": {"rows": rows, "searched": sum(c > 0 for c in counts), "moves": sum(counts)} if status == "finished" else None}, file)
    return directory


class MergeReanalysisTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.output = os.path.join(self.base, "merged")

    def tearDown(self):
        shutil.rmtree(self.base)

    def part(self, name, searched, **kwargs):
        return write_part(os.path.join(self.base, name), searched, **kwargs)

    def load(self, name):
        return np.load(os.path.join(self.output, f"{name}.npy"))

    def test_each_row_takes_its_moves_and_root_value_from_the_part_that_searched_it(self):
        # Interleaved rows, different move counts, and rows 3 and 6 searched by neither part
        first = self.part("first", {0: 2, 2: 3, 5: 2})
        second = self.part("second", {1: 4, 4: 2})
        self.assertEqual(merge([first, second], self.output), (ROWS, 5, 13))

        searched = {0: 2, 1: 4, 2: 3, 4: 2, 5: 2}
        expected = [moves_of(row, searched[row]) for row in sorted(searched)]
        offsets = self.load("policy_offsets")
        np.testing.assert_array_equal(offsets, [0, 2, 6, 9, 9, 11, 13, 13])
        np.testing.assert_array_equal(self.load("policy_tokens"), np.concatenate([e[0] for e in expected]))
        np.testing.assert_array_equal(self.load("policy_weights"), np.concatenate([e[1] for e in expected]))
        np.testing.assert_array_equal(self.load("policy_targets"), np.concatenate([e[2] for e in expected]))
        root_q = self.load("root_q")
        for row in range(ROWS):
            if row in searched:
                np.testing.assert_array_equal(root_q[row], root_value(row))
            else:
                self.assertTrue(np.isnan(root_q[row]).all(), row)

    def test_the_merge_keeps_the_dtypes_layout_and_every_parts_metadata(self):
        first = self.part("first", {0: 2}, commit="aaa", model="net.onnx")
        second = self.part("second", {6: 3}, commit="bbb", model="net-fp16.onnx")
        merge([first, second], self.output)

        self.assertEqual(self.load("policy_offsets").dtype, np.int64)
        self.assertEqual(self.load("policy_tokens").dtype, np.int16)
        self.assertEqual(self.load("policy_weights").dtype, np.uint8)
        self.assertEqual(self.load("policy_targets").dtype, np.float32)
        self.assertEqual(self.load("root_q").dtype, np.float32)
        with open(os.path.join(self.output, "policy_layout.json"), encoding="utf-8") as file:
            self.assertEqual(json.load(file), LAYOUT)
        with open(os.path.join(self.output, "metadata.json"), encoding="utf-8") as file:
            metadata = json.load(file)
        self.assertEqual([(part["commit"], part["model"]["file"]) for part in metadata["parts"]], [("aaa", "net.onnx"), ("bbb", "net-fp16.onnx")])
        self.assertEqual(metadata["result"], {"rows": ROWS, "searched": 2, "moves": 5})
        self.assertFalse(os.path.exists(self.output + ".partial"))

    def test_a_row_searched_in_two_parts_is_refused(self):
        first = self.part("first", {0: 2, 3: 2})
        second = self.part("second", {3: 2, 4: 2})
        with self.assertRaisesRegex(ValueError, "Row 3 was searched in more than one part"):
            merge([first, second], self.output)
        self.assertFalse(os.path.exists(self.output))

    def test_parts_that_dont_fit_together_are_refused(self):
        first = self.part("first", {0: 2})
        cases = {
            "different row counts": self.part("rows", {1: 2}, rows=ROWS + 1),
            "encode the moves differently": self.part("layout", {1: 2}, layout={**LAYOUT, "max_tokens": [4, 0]}),
            "different collections": self.part("seed", {1: 2}, seed="777"),
            "no finished reanalysis": self.part("unfinished", {1: 2}, status="running"),
        }
        for message, other in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                merge([first, other], self.output)

    def test_a_part_whose_moves_and_root_values_disagree_is_refused(self):
        first = self.part("first", {0: 2})
        second = self.part("second", {2: 2, 4: 3})
        root_q = np.load(os.path.join(second, "root_q.npy"))
        root_q[4] = np.nan
        np.save(os.path.join(second, "root_q.npy"), root_q)
        with self.assertRaisesRegex(ValueError, "row 4 has 3 moves"):
            merge([first, second], self.output)

    def test_a_part_whose_offsets_dont_match_its_arrays_is_refused(self):
        first = self.part("first", {0: 2})
        second = self.part("second", {2: 2})
        np.save(os.path.join(second, "policy_targets.npy"), np.zeros(3, np.float32))
        with self.assertRaisesRegex(ValueError, "2 moves by its offsets"):
            merge([first, second], self.output)

    def test_an_existing_output_is_left_alone(self):
        os.makedirs(self.output)
        with self.assertRaisesRegex(ValueError, "already exists"):
            merge([self.part("first", {0: 2})], self.output)


if __name__ == "__main__":
    unittest.main()
