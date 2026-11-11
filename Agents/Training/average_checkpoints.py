"""
Averages the weights of several checkpoints of one net, e.g. a self-play loop's plateau generations (weight averaging as in
KataGo's snapshot averages and SWA): every floating tensor of the state_dict becomes the mean over the checkpoints (summed
in float64, stored in its own dtype), every other tensor must be the same in all of them. The rest of the checkpoint
(hyperparameters, versions) comes from the last one, without optimizer or scheduler state, so the result loads like any
checkpoint (export_onnx.py --ckpt, train.py --init-from). Plain averaging only makes sense for checkpoints of one training
lineage with the same frozen input normalization (whose mean and scale are then averaged to themselves).

Usage (from Agents/Training):
  python average_checkpoints.py --out AVERAGE.ckpt CKPT [CKPT ...]
"""

import argparse
import os

import torch

import runlog


def average_checkpoints(paths, out_path):
    """Writes the average of the checkpoints at paths to out_path and returns it"""
    if not paths:
        raise ValueError("No checkpoints to average.")
    sums, first, last = None, None, None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        if first is None:
            first = state
            sums = {name: tensor.double() for name, tensor in state.items() if tensor.is_floating_point()}
        else:
            # Only checkpoints of the same net: the same weights in the same shapes, the other tensors unchanged
            if state.keys() != first.keys():
                raise ValueError(f"{path} has other weights than {paths[0]}")
            for name, tensor in state.items():
                if tensor.shape != first[name].shape:
                    raise ValueError(f"{name} has shape {tuple(tensor.shape)} in {path}, {tuple(first[name].shape)} in {paths[0]}")
                if tensor.is_floating_point():
                    sums[name] += tensor.double()
                elif not torch.equal(tensor, first[name]):
                    raise ValueError(f"{name} differs between {paths[0]} and {path}")
        last = checkpoint

    # The means in each tensor's own dtype, in the first checkpoint's order
    averaged = {name: (sums[name] / len(paths)).to(tensor.dtype) if name in sums else tensor for name, tensor in first.items()}
    result = {key: value for key, value in last.items() if key not in ("state_dict", "optimizer_states", "lr_schedulers")}
    result["state_dict"] = averaged
    torch.save(result, out_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="Average the weights of several checkpoints of one net")
    parser.add_argument("--out", required=True, help="the averaged checkpoint")
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()
    # Entry in the run registry (the registry has no kind of its own for this): the inputs, the output and how many were averaged
    with runlog.begin("other", name=os.path.splitext(os.path.basename(args.out))[0],
                      inputs={"checkpoints": args.checkpoints}, outputs={"checkpoint": args.out}) as run:
        average_checkpoints(args.checkpoints, args.out)
        run.result(checkpoints=len(args.checkpoints))
        print(f"Averaged {len(args.checkpoints)} checkpoints into {args.out}")


if __name__ == "__main__":
    main()
