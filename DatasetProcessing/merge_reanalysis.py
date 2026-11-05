"""
Merge the parts of a reanalysis (DatasetCollector --reanalyze over different match runs) into one policy directory for
train.py --policy-dir.

Every part has one entry per row of the whole dataset, but moves and a root value only for the rows it searched (rows
outside its match runs have no moves and NaN in root_q.npy). The merge takes each row from the one part that searched
it, so no row may have moves or a root value in two parts. Writes into the output directory:
- policy_offsets.npy (n + 1,) int64, policy_tokens.npy (moves, max_tokens) int16, policy_weights.npy (moves, max_tokens)
  uint8, policy_targets.npy (moves,) float32 and root_q.npy (n, 4) float32, as in each part
- policy_layout.json: the token ranges, the same in every part
- metadata.json: the parts' metadata (commit, net, search, match runs) and the merged totals
The directory is written as OUT.partial and renamed when complete.

Usage: python merge_reanalysis.py PART [PART ...] --output DIR
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

MOVE_ARRAYS = ("policy_tokens", "policy_weights", "policy_targets")


def npy_header(path: str):
    """Shape and dtype of an .npy file, without reading or mapping its data"""
    with open(path, "rb") as file:
        version = np.lib.format.read_magic(file)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = read_header(file)
    return shape, dtype


def read_part(directory: str):
    """A part's metadata, layout, offsets and root values, checked to be a finished reanalysis with consistent arrays"""
    with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("kind") != "reanalysis" or metadata.get("status") != "finished":
        raise ValueError(f"{directory} is no finished reanalysis (kind {metadata.get('kind')}, status {metadata.get('status')})")
    with open(os.path.join(directory, "policy_layout.json"), encoding="utf-8") as file:
        layout = json.load(file)
    offsets = np.load(os.path.join(directory, "policy_offsets.npy"))
    root_q = np.load(os.path.join(directory, "root_q.npy"))
    rows = metadata["result"]["rows"]
    if offsets.shape != (rows + 1,) or root_q.shape[0] != rows:
        raise ValueError(f"{directory} has {offsets.shape[0] - 1} offsets and {root_q.shape[0]} root values for {rows} rows")
    counts = np.diff(offsets)
    if offsets[0] != 0 or np.any(counts < 0):
        raise ValueError(f"{directory} has offsets that don't start at 0 or go down")
    moves = {name: npy_header(os.path.join(directory, f"{name}.npy"))[0][0] for name in MOVE_ARRAYS}
    if any(count != offsets[-1] for count in moves.values()):
        raise ValueError(f"{directory} has {offsets[-1]} moves by its offsets but {moves} in its arrays")
    # A searched row has its moves and its root value, a row that wasn't searched neither
    searched = ~np.isnan(root_q).all(axis=1)
    if np.any(searched != (counts > 0)):
        row = int(np.flatnonzero(searched != (counts > 0))[0])
        raise ValueError(f"{directory}: row {row} has {counts[row]} moves but root value {root_q[row].tolist()}")
    return metadata, layout, offsets, root_q, searched


def merge(parts, output_dir: str):
    """Writes the merged policy directory; returns the rows, searched rows and moves"""
    if os.path.exists(output_dir):
        raise ValueError(f"{output_dir} already exists")
    read = [read_part(part) for part in parts]
    metadata = [part[0] for part in read]
    # The same token encoding, the same rows and the same collection, or the parts' rows wouldn't line up
    if any(layout != read[0][1] for _, layout, *_ in read):
        raise ValueError("The parts encode the moves differently (policy_layout.json)")
    rows = metadata[0]["result"]["rows"]
    if any(meta["result"]["rows"] != rows for meta in metadata):
        raise ValueError(f"The parts have different row counts: {[meta['result']['rows'] for meta in metadata]}")
    seeds = {meta["source"]["collection_seed"] for meta in metadata}
    if len(seeds) != 1:
        raise ValueError(f"The parts come from different collections (seeds {sorted(seeds)})")

    # How many parts searched each row: at most one
    owners = np.stack([searched for *_, searched in read]).sum(axis=0)
    if np.any(owners > 1):
        row = int(np.flatnonzero(owners > 1)[0])
        raise ValueError(f"Row {row} was searched in more than one part")

    # A row's moves in the merge: the sum over the parts, of which only its owner has any
    counts = sum(np.diff(offsets) for _, _, offsets, _, _ in read)
    merged_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    moves = int(merged_offsets[-1])
    root_q = np.full((rows, read[0][3].shape[1]), np.nan, dtype=np.float32)

    # In memory (a few GB for all of v2), not memory-mapped: Windows can't rename a directory with mapped files in it
    outputs = {}
    for name in MOVE_ARRAYS:
        shape, dtype = npy_header(os.path.join(parts[0], f"{name}.npy"))
        if any(npy_header(os.path.join(part, f"{name}.npy"))[1] != dtype for part in parts):
            raise ValueError(f"The parts store {name} with different types")
        outputs[name] = np.empty((moves,) + shape[1:], dtype=dtype)
    for part, (_, _, offsets, part_q, searched) in zip(parts, read):
        # The root values of the rows this part searched
        root_q[searched] = part_q[searched]
        # Move j of the part belongs to row r (offsets[r] <= j < offsets[r + 1]) and goes to merged_offsets[r] + j - offsets[r]
        shift = np.repeat(merged_offsets[:-1] - offsets[:-1], np.diff(offsets))
        destination = np.arange(offsets[-1], dtype=np.int64) + shift
        for name in MOVE_ARRAYS:
            outputs[name][destination] = np.load(os.path.join(part, f"{name}.npy"))

    # Written as .partial and renamed at the end, so an unfinished merge is never taken for a finished one
    partial = output_dir + ".partial"
    if os.path.exists(partial):
        shutil.rmtree(partial)
    os.makedirs(partial)
    for name, array in outputs.items():
        np.save(os.path.join(partial, f"{name}.npy"), array)
    np.save(os.path.join(partial, "policy_offsets.npy"), merged_offsets)
    np.save(os.path.join(partial, "root_q.npy"), root_q)
    with open(os.path.join(partial, "policy_layout.json"), "w", encoding="utf-8") as file:
        json.dump(read[0][1], file, indent=2)
    searched_rows = int(sum(searched.sum() for *_, searched in read))
    with open(os.path.join(partial, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump({"format": 1, "kind": "reanalysis", "status": "finished",
                   "parts": [{"directory": os.path.abspath(part), **meta} for part, meta in zip(parts, metadata)],
                   "result": {"rows": rows, "searched": searched_rows, "moves": moves}}, file, indent=2)
    os.rename(partial, output_dir)
    print(f"{rows} rows, {searched_rows} searched with {moves} moves, from {len(parts)} parts, written to {output_dir}")
    return rows, searched_rows, moves


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge the parts of a reanalysis into one policy directory")
    parser.add_argument("parts", nargs="+", help="the parts' directories")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    merge(args.parts, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
