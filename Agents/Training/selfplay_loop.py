"""
Continuous self-play loop in small generations (as in KataGo instead of a few large iterations with hard gates).

Every generation:
1. collect: SelfPlayCollector plays games with the newest net (policy target pruning and root noise as configured there)
2. convert: DatasetProcessing/selfplay_to_dataset.py turns the run into training arrays
3. train: train.py continues from the previous checkpoint for a fixed number of samples at a constant learning rate, on a
   window of the newest generations (equal shares; --prefill starts the window with data from elsewhere, e.g. an earlier
   loop's newest generations, which leave it as the loop's own come in) plus extra data (e.g. the v2 data)
4. export: export_onnx.py writes the net the next generation collects with (with --fp16 the collection uses a float16
   export of it, the evaluations keep the float32 net)

Every --eval-every generations an evaluation command (e.g. an SPRT mirror duel) compares the newest net with the one
evaluated last, and every --anchor-every generations also with a fixed anchor (by default the start net), so small steps
that each look even cannot add up to a loss unnoticed; the last output line of each is logged. The loop stops if one
reports the newest net as clearly worse (the upper end of the Elo interval below 0), and otherwise keeps going until
--generations are done or it is stopped.
The state is kept in loop_state.json in the loop directory, so a stopped loop continues where it left off.

Usage (from Agents/Training, with the Agents/env interpreter):
  python selfplay_loop.py --loop-dir DIR --start-checkpoint CKPT --start-model ONNX --extra-data DIR@0.3 [options]
"""

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time

import runlog

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
# The collector's Release build: an .exe on Windows, a plain executable elsewhere
COLLECTOR = os.path.join(REPO, "SelfPlayCollector", "bin", "Release", "net10.0",
                         "SelfPlayCollector.exe" if os.name == "nt" else "SelfPlayCollector")


def log(loop_dir, text):
    line = f"{time.strftime('%H:%M')} {text}"
    print(line, flush=True)
    with open(os.path.join(loop_dir, "loop.log"), "a", encoding="utf-8") as file:
        file.write(line + "\n")


def run(command, output_path, kind=None):
    """Runs a command with its output in a file; raises with the end of the output if it fails. With a kind the command
    gets its own entry in the run registry (for programs that don't register themselves, like the collector)."""
    if kind:
        with runlog.track(kind, command, output_path) as finish:
            with open(output_path, "w", encoding="utf-8") as output:
                code = subprocess.call(command, stdout=output, stderr=subprocess.STDOUT)
            finish(code)
    else:
        with open(output_path, "w", encoding="utf-8") as output:
            code = subprocess.call(command, stdout=output, stderr=subprocess.STDOUT)
    if code != 0:
        with open(output_path, encoding="utf-8", errors="replace") as file:
            tail = file.read()[-2000:]
        raise RuntimeError(f"{os.path.basename(command[0])} failed with exit code {code}:\n{tail}")


def load_state(loop_dir):
    path = os.path.join(loop_dir, "loop_state.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    return {"generations": []}


def save_state(loop_dir, state):
    path = os.path.join(loop_dir, "loop_state.json")
    with open(path + ".tmp", "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)
    os.replace(path + ".tmp", path)


def command_arguments(template, **values):
    """The evaluation command as program and arguments, run without a shell: split first, then the placeholders filled in,
    so a path can neither add arguments nor shell syntax. Windows keeps backslashes in paths; quotes group as in a shell."""
    parts = shlex.split(template, posix=os.name != "nt")
    if os.name == "nt":
        parts = [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in parts]
    return [part.format(**values) for part in parts]


def training_data(prepared, prefill, window, extra_data):
    """The --data specs of a generation's training: a window of the newest --window prepared directories, in which data
    from elsewhere (prefill, oldest first) comes before the loop's own generations and leaves first, with equal shares of
    what the extra data leave, the newest first (the validation runs on the first source), then the extra data as given"""
    extra_shares = sum(float(spec.rsplit("@", 1)[1]) for spec in extra_data)
    newest = (list(prefill) + list(prepared))[-window:]
    share = (1 - extra_shares) / len(newest)
    return [f"{path}@{share}" for path in reversed(newest)] + list(extra_data)


def clearly_worse(line):
    """Whether an evaluation line ("... +12 Elo [-30, +55] ...") puts the upper end of the interval below 0"""
    match = re.search(r"([+-]?\d+) Elo \[([+-]?\d+), ([+-]?\d+)\]", line)
    return match is not None and int(match.group(3)) < 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Continuous self-play loop in small generations")
    parser.add_argument("--loop-dir", required=True)
    parser.add_argument("--start-checkpoint", required=True, help="checkpoint of the net generation 1 starts from")
    parser.add_argument("--start-model", required=True, help="its ONNX export, which collects generation 1")
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--games", type=int, default=8000, help="games per generation")
    parser.add_argument("--collector-args", default="--concurrent 1024 --nets 3", help="further SelfPlayCollector options")
    parser.add_argument("--fp16", action="store_true",
                        help="collect with float16 exports of the nets (faster where the GPU limits, e.g. CUDA on the server)")
    parser.add_argument("--seed", type=int, default=20261100, help="generation g collects with seed + g")
    parser.add_argument("--window", type=int, default=4, help="newest generations in the training data")
    parser.add_argument("--prefill", action="append", default=[],
                        help="prepared data that fills the window before the loop's own generations (repeat, oldest first), e.g. an earlier loop's newest generations")
    parser.add_argument("--extra-data", action="append", default=[], help="further prepared data DIR@SHARE, e.g. the v2 data")
    parser.add_argument("--samples", type=int, default=2_000_000, help="training samples per generation")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-lambda", type=float, default=0.5)
    parser.add_argument("--policy-temperature", type=float, default=2.0)
    parser.add_argument("--train-args", default="--normalize-inputs --placement-weight 0.5 --vp-share-weight 0.5 --rounds-weight 0.2 "
                                                  "--augment-rotation --augment-board --policy-weight 1.0")
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--anchor-every", type=int, default=9, help="generations between evaluations against the anchor, 0 for none")
    parser.add_argument("--anchor-model", default=None, help="the fixed net the anchor evaluations compare with (default: --start-model)")
    parser.add_argument("--eval-command", default=None,
                        help="command evaluating {new} (ONNX) against {old} (ONNX), {name} names the run, run without a shell; its last output line is logged")
    args = parser.parse_args(argv)

    # In the run registry: the loop, each generation as its child, each step of a generation as the generation's child
    generation_runs = []
    with runlog.begin("selfplay-loop", name=os.path.basename(os.path.normpath(args.loop_dir)), outputs={"loop_dir": args.loop_dir}) as run:
        try:
            code = run_loop(args, generation_runs)
        except BaseException as error:
            # The generation it stopped in (the only one still open) ends the way the loop does: by an error, or a signal
            ending, fields = runlog.ended_by(error)
            for generation_run in generation_runs:
                generation_run.end(ending, **fields)
            raise
        run.result(stopped_by="an evaluation" if code == 2 else "generations done")
    return code


def run_loop(args, generation_runs):
    os.makedirs(args.loop_dir, exist_ok=True)
    state = load_state(args.loop_dir)
    python = sys.executable
    extra_shares = sum(float(spec.rsplit("@", 1)[1]) for spec in args.extra_data)
    if not 0 <= extra_shares < 1:
        raise SystemExit("The shares of --extra-data must add up to less than 1")

    def newest(key, fallback):
        done = [g for g in state["generations"] if key in g]
        return done[-1][key] if done else fallback

    while len([g for g in state["generations"] if "model" in g]) < args.generations:
        if state["generations"] and "model" not in state["generations"][-1]:
            gen = state["generations"][-1]
        else:
            gen = {"generation": len(state["generations"]) + 1}
            state["generations"].append(gen)
            save_state(args.loop_dir, state)
        g = gen["generation"]
        gen_dir = os.path.join(args.loop_dir, f"gen{g:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        model = newest("model", args.start_model)
        checkpoint = newest("checkpoint", args.start_checkpoint)
        generation_run = runlog.begin("selfplay-gen", name=f"{os.path.basename(os.path.normpath(args.loop_dir))}/gen{g:03d}",
                                      inputs={"model": model, "checkpoint": checkpoint}, outputs={"gen_dir": gen_dir})
        generation_runs.append(generation_run)

        def finish_generation():
            generation_run.result(**{key: gen[key] for key in ("collect", "train", "model", "eval", "anchor") if key in gen})
            generation_run.end(0)

        if "run" not in gen:
            # A collection stopped before it finished left its directory behind; the collector won't reuse it
            unfinished = os.path.join(gen_dir, "run")
            if os.path.exists(unfinished):
                os.replace(unfinished, os.path.join(gen_dir, f"run_unfinished_{time.strftime('%Y%m%d_%H%M%S')}"))
            started = time.time()
            collect_model = model
            if args.fp16:
                # Exported from the checkpoint the newest net came from (the start checkpoint for generation 1)
                collect_model = os.path.join(gen_dir, "collect_fp16.onnx")
                run([python, os.path.join(HERE, "export_onnx.py"), "--ckpt", checkpoint, "--out", collect_model, "--fp16"],
                    os.path.join(gen_dir, "export_fp16.log"))
            command = [COLLECTOR, "--games", str(args.games), "--output", gen_dir, "--name", "run", "--seed", str(args.seed + g),
                       "--model", collect_model] + args.collector_args.split()
            run(command, os.path.join(gen_dir, "collect.log"), kind="collect")
            with open(os.path.join(gen_dir, "run", "metadata.json"), encoding="utf-8") as file:
                result = json.load(file)["result"]
            gen["run"] = os.path.join(gen_dir, "run")
            gen["collect"] = {"games": result["games"], "positions": result["positions"], "errors": result["errors"],
                              "minutes": round((time.time() - started) / 60, 1), "model": collect_model}
            save_state(args.loop_dir, state)
            log(args.loop_dir, f"gen {g}: {result['games']} games, {result['positions']} positions, {result['errors']} errors "
                               f"in {gen['collect']['minutes']} min")

        if "prepared" not in gen:
            prepared = os.path.join(gen_dir, "prepared")
            run([python, os.path.join(REPO, "DatasetProcessing", "selfplay_to_dataset.py"), gen["run"], "--output", prepared,
                 "--target-lambda", str(args.target_lambda)], os.path.join(gen_dir, "convert.log"), kind="convert")
            gen["prepared"] = prepared
            save_state(args.loop_dir, state)

        if "checkpoint" not in gen:
            # The newest generation first: the validation runs on its held-out games
            data = training_data([x["prepared"] for x in state["generations"] if "prepared" in x], args.prefill, args.window, args.extra_data)
            window = len(data) - len(args.extra_data)
            checkpoint_dir = os.path.join(gen_dir, "checkpoints")
            command = [python, os.path.join(HERE, "train.py"), "--init-from", checkpoint, "--target-lambda", str(args.target_lambda),
                       "--policy-temperature", str(args.policy_temperature), "--lr", str(args.lr), "--constant-lr",
                       "--max-epochs", "1", "--epoch-samples", str(args.samples), "--checkpoint-dir", checkpoint_dir]
            for spec in data:
                command += ["--data", spec]
            started = time.time()
            run(command + args.train_args.split(), os.path.join(gen_dir, "train.log"))
            checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")), key=os.path.getmtime)
            if not checkpoints:
                raise RuntimeError(f"train.py wrote no checkpoint into {checkpoint_dir}")
            gen["checkpoint"] = checkpoints[-1]
            gen["train"] = {"minutes": round((time.time() - started) / 60, 1), "window": window,
                            "val_loss": re.search(r"val_loss=([0-9]+\.[0-9]+)", os.path.basename(checkpoints[-1])).group(1)}
            save_state(args.loop_dir, state)

        if "model" not in gen:
            onnx = os.path.join(gen_dir, f"state_value_net-loop-g{g:03d}.onnx")
            run([python, os.path.join(HERE, "export_onnx.py"), "--ckpt", gen["checkpoint"], "--out", onnx],
                os.path.join(gen_dir, "export.log"))
            gen["model"] = onnx
            save_state(args.loop_dir, state)
            log(args.loop_dir, f"gen {g}: trained on {gen['train']['window']} generations in {gen['train']['minutes']} min, "
                               f"val_loss {gen['train']['val_loss']}, net {os.path.basename(onnx)}")

        def evaluate(key, reference, name):
            """Runs the evaluation command, keeps its last line under gen[key]; whether the newest net came out clearly worse"""
            command = command_arguments(args.eval_command, new=gen["model"], old=reference, name=name)
            output = os.path.join(gen_dir, f"{key}.log")
            with runlog.track("duel", command, output, name=name) as finish:
                with open(output, "w", encoding="utf-8") as file:
                    finish(subprocess.call(command, stdout=file, stderr=subprocess.STDOUT))
            with open(output, encoding="utf-8", errors="replace") as file:
                lines = [line.strip() for line in file if line.strip()]
            gen[key] = {"against": reference, "result": lines[-1] if lines else "no output"}
            save_state(args.loop_dir, state)
            log(args.loop_dir, f"gen {g} against {os.path.basename(reference)}: {gen[key]['result']}")
            if clearly_worse(gen[key]["result"]):
                log(args.loop_dir, f"gen {g} is clearly worse than {os.path.basename(reference)}, the loop stops")
                return True
            return False

        if args.eval_command and g % args.eval_every == 0 and "eval" not in gen:
            reference = state.get("evaluated", args.start_model)
            worse = evaluate("eval", reference, f"loop_g{g:03d}")
            state["evaluated"] = gen["model"]
            save_state(args.loop_dir, state)
            if worse:
                finish_generation()
                return 2
        if args.eval_command and args.anchor_every > 0 and g % args.anchor_every == 0 and "anchor" not in gen:
            if evaluate("anchor", args.anchor_model or args.start_model, f"loop_g{g:03d}_anchor"):
                finish_generation()
                return 2
        finish_generation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
