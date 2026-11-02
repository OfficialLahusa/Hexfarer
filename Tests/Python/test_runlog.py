"""Run registry (Agents/Training/runlog.py): the wrapper records command, commit, exit code and output; a tool started by
the wrapper adds to its run instead of starting a second one, anything it starts becomes a child; events merge per run;
sync merges without duplicates; .NET build commits, dirty diffs and W&B runs are read.

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

import runlog  # noqa: E402
import selfplay_loop  # noqa: E402


class RunlogTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.directory.name, "runs")
        self.saved = {key: os.environ.get(key) for key in ("HEXFARER_RUNS", runlog.ENV_ID, runlog.ENV_DIRECT, runlog.ENV_PARENT)}
        os.environ["HEXFARER_RUNS"] = self.root
        for key in (runlog.ENV_ID, runlog.ENV_DIRECT, runlog.ENV_PARENT):
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.directory.cleanup()

    def python(self, code):
        return [sys.executable, "-c", f"import sys; sys.path.insert(0, {os.path.join(REPO, 'Agents', 'Training')!r}); {code}"]

    def test_the_wrapper_records_command_commit_exit_and_output(self):
        command = self.python("print('first'); print('the last line')")
        self.assertEqual(runlog.wrap("other", command, name="ok", quiet=True), 0)
        failing = self.python("print('before'); sys.exit(3)")
        self.assertEqual(runlog.wrap("other", failing, name="fails", quiet=True), 3)

        runs = runlog.load()
        self.assertEqual([run["name"] for run in runs], ["ok", "fails"])
        ok, fails = runs
        self.assertEqual(ok["argv"], command)
        self.assertEqual(ok["end"]["exit"], 0)
        self.assertEqual(ok["end"]["last_line"], "the last line")
        self.assertEqual(fails["end"]["exit"], 3)
        self.assertEqual(fails["end"]["last_line"], "before")
        self.assertEqual(ok["repo"]["commit"], subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip())
        with open(os.path.join(self.root, ok["id"], "console.log"), encoding="utf-8") as file:
            self.assertEqual(file.read().split(), ["first", "the", "last", "line"])
        self.assertIsNone(ok["parent"])

    def test_a_tool_started_by_the_wrapper_adds_to_its_run_and_its_children_hang_below(self):
        child = ("import runlog, subprocess; run = runlog.begin('train', name='inner'); run.result(best=0.5); "
                 "subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, ' + repr(sys.path[0]) + '); import runlog; "
                 "r = runlog.begin(\"export\", name=\"grandchild\"); r.end(0)'], check=True); run.end(0)")
        self.assertEqual(runlog.wrap("train", self.python(child), name="outer", quiet=True), 0)

        runs = runlog.load()
        self.assertEqual(len(runs), 2, "the tool must not start a second run for the wrapper's command")
        outer = next(run for run in runs if run["name"] == "outer")
        grandchild = next(run for run in runs if run["name"] == "grandchild")
        self.assertEqual([r.get("best") for r in outer["results"] if "best" in r], [0.5])
        self.assertEqual(grandchild["parent"], outer["id"])
        self.assertEqual(outer["end"]["exit"], 0)

    def test_begin_outside_the_wrapper_starts_a_run_and_restores_the_parent(self):
        os.environ[runlog.ENV_PARENT] = "the-caller"
        with runlog.begin("precision", name="alone") as run:
            self.assertEqual(os.environ[runlog.ENV_PARENT], run.id)
            with runlog.begin("export", name="nested") as nested:
                pass
        self.assertEqual(os.environ[runlog.ENV_PARENT], "the-caller")
        runs = {r["name"]: r for r in runlog.load()}
        self.assertEqual(runs["alone"]["parent"], "the-caller")
        self.assertEqual(runs["nested"]["parent"], runs["alone"]["id"])
        self.assertEqual(runs["nested"]["id"], nested.id)

        with self.assertRaises(ValueError):
            with runlog.begin("train", name="crashes"):
                raise ValueError("boom")
        crashed = next(r for r in runlog.load() if r["name"] == "crashes")
        self.assertEqual(crashed["end"]["exit"], 1)
        self.assertIn("boom", crashed["end"]["error"])

    def test_a_run_ends_once_and_tracked_commands_record_their_last_line(self):
        run = runlog.start("other", name="twice", argv=["x"])
        run.end(0)
        run.end(1)
        output = os.path.join(self.directory.name, "collect.log")
        selfplay_loop.run(self.python("print('100 games'); print('done: 4200 positions')"), output, kind="collect")
        with self.assertRaises(RuntimeError):
            selfplay_loop.run(self.python("sys.exit(5)"), output, kind="convert")

        runs = {r["name"] or r["kind"]: r for r in runlog.load()}
        self.assertEqual(runs["twice"]["end"]["exit"], 0)
        with open(os.path.join(self.root, runlog.REGISTRY_FILE), encoding="utf-8") as file:
            self.assertEqual(sum(1 for line in file if '"end"' in line and runs["twice"]["id"] in line), 1)
        self.assertEqual(runs["collect"]["end"]["last_line"], "done: 4200 positions")
        self.assertEqual(runs["convert"]["end"]["exit"], 5)

    def test_events_merge_per_run_and_sync_adds_no_duplicates(self):
        run = runlog.start("duel", name="d", argv=["arena"])
        run.result(games=10)
        run.result(games=20)
        run.end(0)
        with open(os.path.join(self.root, runlog.REGISTRY_FILE), encoding="utf-8") as file:
            lines = file.readlines()
        (merged,) = runlog.load()
        self.assertEqual([r["games"] for r in merged["results"]], [10, 20])
        self.assertEqual(merged["kind"], "duel")

        other = os.path.join(self.directory.name, "other")
        self.assertEqual(runlog.merge(other, lines), 4)
        self.assertEqual(runlog.merge(other, lines + lines[:2]), 0)
        self.assertEqual(len(runlog.load(other)), 1)

    def test_build_commit_of_a_dotnet_program_and_the_diff_of_a_dirty_repository(self):
        sha = "0123456789abcdef0123456789abcdef01234567"
        program = os.path.join(self.directory.name, "Arena.exe")
        with open(program, "wb") as file:
            file.write(b"\x00apphost without version\x00")
        with open(os.path.join(self.directory.name, "Arena.dll"), "wb") as file:
            file.write(b"\x00\x01" + f"1.0.0+{sha}".encode() + b"\x00")
        self.assertEqual(runlog.binary_commit(program), sha)
        self.assertIsNone(runlog.binary_commit(os.path.join(self.directory.name, "missing.exe")))

        repo = os.path.join(self.directory.name, "repo")
        os.makedirs(repo)
        for args in (["init", "-q"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"]):
            subprocess.run(["git", *args], cwd=repo, check=True)
        self.assertFalse(runlog.repo_state(repo)["dirty"])
        with open(os.path.join(repo, "new.py"), "w", encoding="utf-8") as file:
            file.write("x = 1\n")
        state = runlog.repo_state(repo)
        self.assertTrue(state["dirty"], "an untracked file makes the run irreproducible from the commit")
        patch = os.path.join(self.directory.name, "diff.patch")
        runlog.save_diff(repo, patch)
        with open(patch, encoding="utf-8") as file:
            self.assertIn("new.py", file.read())
        self.assertIsNone(runlog.repo_state(self.directory.name + os.sep + "nowhere"))

    def test_a_selfplay_loop_registers_its_generations_and_their_steps(self):
        loop_dir = os.path.join(self.directory.name, "loop")
        steps = []

        def stand_in(command, output_path, kind=None):
            # Creates what each step leaves behind and registers like the real step: the collector and the conversion
            # through run(kind=...), train.py and export_onnx.py by themselves
            script = os.path.basename(command[1]) if len(command) > 1 else ""
            steps.append(kind or script)
            with open(output_path, "w", encoding="utf-8") as file:
                file.write("step output\n")
            if kind:
                with runlog.track(kind, command, output_path) as finish:
                    finish(0)
            if kind == "collect":
                gen_dir = command[command.index("--output") + 1]
                os.makedirs(os.path.join(gen_dir, "run"), exist_ok=True)
                with open(os.path.join(gen_dir, "run", "metadata.json"), "w", encoding="utf-8") as file:
                    json.dump({"result": {"games": 8, "positions": 300, "errors": 0}}, file)
            elif kind == "convert":
                os.makedirs(command[command.index("--output") + 1], exist_ok=True)
            elif script == "train.py":
                with runlog.begin("train", name="stand-in"):
                    checkpoint_dir = command[command.index("--checkpoint-dir") + 1]
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    open(os.path.join(checkpoint_dir, "epoch=0-step=1-val_loss=0.500000.ckpt"), "w").close()
            elif script == "export_onnx.py":
                with runlog.begin("export", name="stand-in"):
                    open(command[command.index("--out") + 1], "w").close()

        evaluation = os.path.join(self.directory.name, "evaluate.py")
        with open(evaluation, "w", encoding="utf-8") as file:
            file.write("print('decided: +5 Elo [-10, +20]')\n")
        original = selfplay_loop.run
        selfplay_loop.run = stand_in
        try:
            code = selfplay_loop.main(["--loop-dir", loop_dir, "--start-checkpoint", "start.ckpt", "--start-model", "start.onnx",
                                       "--generations", "2", "--eval-every", "1", "--anchor-every", "0",
                                       "--eval-command", f'"{sys.executable}" "{evaluation}" {{new}} {{old}}'])
        finally:
            selfplay_loop.run = original
        self.assertEqual(code, 0)

        runs = runlog.load()
        (loop,) = [run for run in runs if run["kind"] == "selfplay-loop"]
        generations = [run for run in runs if run["kind"] == "selfplay-gen"]
        self.assertEqual([run["name"] for run in generations], ["loop/gen001", "loop/gen002"])
        self.assertEqual(loop["end"]["exit"], 0)
        for generation in generations:
            self.assertEqual(generation["parent"], loop["id"])
            self.assertEqual(generation["end"]["exit"], 0)
            results = {key for result in generation["results"] for key in result}
            self.assertTrue({"collect", "train", "model", "eval"} <= results, results)
            children = sorted(run["kind"] for run in runs if run.get("parent") == generation["id"])
            self.assertEqual(children, ["collect", "convert", "duel", "export", "train"])
        duel = next(run for run in runs if run["kind"] == "duel")
        self.assertEqual(duel["end"]["last_line"], "decided: +5 Elo [-10, +20]")
        self.assertIsNone(os.environ.get(runlog.ENV_PARENT), "the loop must leave the environment as it found it")

    def test_the_command_line_adds_shows_and_lists_runs(self):
        code = runlog.main(["add", "--kind", "collect", "--name", "v2", "--machine", "desktop", "--time", "2026-10-25T02:00:00+02:00",
                            "--command", "DatasetCollector --seed 20261025", "--result", "1M matches"])
        self.assertEqual(code, 0)
        (run,) = runlog.load()
        self.assertEqual((run["kind"], run["name"], run["command"], run["argv"]), ("collect", "v2", "DatasetCollector --seed 20261025",
                                                                               ["DatasetCollector", "--seed", "20261025"]))
        self.assertEqual(run["results"][0]["summary"], "1M matches")
        self.assertTrue(run["end"]["imported"], "a run entered after the fact is finished, not running")
        self.assertEqual(runlog.main(["show", "v2"]), 0)
        self.assertEqual(runlog.main(["show", "nothing-like-this"]), 1)
        self.assertEqual(runlog.main(["list", "--kind", "collect"]), 0)

        # A name that is also part of other runs' ids (duels named after it) still finds its own run
        for name in ("v2p-structured-ft-vs-kt5fwcz5", "v2p-structured-ft-vs-mlp-ft"):
            # As the backfill writes them: the duel's name inside the id
            duel_id = f"20261102-121800-server-duel-{name}"
            runlog.append({"event": "start", "id": duel_id, "time": "2026-11-02T12:18:00+01:00", "kind": "duel", "name": name})
            runlog.append({"event": "end", "id": duel_id, "time": "2026-11-02T12:50:00+01:00", "exit": None, "imported": True})
        trained = runlog.start("train", name="structured-ft", argv=["train.py"])
        trained.end(0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runlog.main(["show", "structured-ft"]), 0)
        self.assertEqual(json.loads(output.getvalue())["id"], trained.id)
        # Ambiguous: lists the candidates
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runlog.main(["show", "v2p-structured"]), 1)
        self.assertIn("2 runs match", output.getvalue())
        self.assertIn("v2p-structured-ft-vs-mlp-ft", output.getvalue())

    def test_wandb_runs_are_imported_once(self):
        wandb = os.path.join(self.directory.name, "wandb")
        files = os.path.join(wandb, "run-20261027_234503-mhn8ihqb", "files")
        os.makedirs(files)
        with open(os.path.join(files, "wandb-metadata.json"), "w", encoding="utf-8") as file:
            json.dump({"startedAt": "2026-10-27T23:45:03.1Z", "host": "desktop", "program": "train.py", "executable": "python",
                       "args": ["--lr", "3e-4"], "git": {"commit": "abc"}}, file)
        with open(os.path.join(files, "wandb-summary.json"), "w", encoding="utf-8") as file:
            json.dump({"val_loss": 0.25, "epoch": 3, "train_loss": 0.2, "_runtime": 9}, file)
        # An offline run keeps its metadata only in its .wandb file
        try:
            from wandb.proto import wandb_internal_pb2 as pb
            from wandb.sdk.internal.datastore import DataStore
        except ImportError:
            pb = None
        if pb is not None:
            offline = os.path.join(wandb, "offline-run-20261102_101323-7fu9314k")
            os.makedirs(offline)
            store = DataStore()
            store.open_for_write(os.path.join(offline, "run-7fu9314k.wandb"))
            record = pb.Record()
            record.environment.host, record.environment.program, record.environment.executable = "server", "train.py", "python"
            record.environment.args.extend(["--monitor", "val_total_loss", "--checkpoint-dir", "/runs/v2policy/structured-ft/"])
            record.environment.git.commit = "def"
            record.environment.started_at.seconds = 1793614403
            store.write(record)
            for key, value in (("val_loss", "0.4"), ("val_loss", "0.3"), ("train_loss", "0.1")):
                record = pb.Record()
                item = record.summary.update.add()
                item.key, item.value_json = key, value
                store.write(record)
            record = pb.Record()
            record.exit.exit_code, record.exit.runtime = 0, 120
            store.write(record)
            store.close()

        # A run that registered itself live keeps its W&B id in a result and must not come in a second time
        live = os.path.join(wandb, "offline-run-20261102_120000-livelive")
        os.makedirs(os.path.join(live, "files"))
        with open(os.path.join(live, "files", "wandb-metadata.json"), "w", encoding="utf-8") as file:
            json.dump({"startedAt": "2026-11-02T11:00:00Z", "host": "desktop", "args": []}, file)
        loop_run = os.path.join(wandb, "offline-run-20261031_062520-kt5fwcz5", "files")
        os.makedirs(loop_run)
        with open(os.path.join(loop_run, "wandb-metadata.json"), "w", encoding="utf-8") as file:
            json.dump({"startedAt": "2026-10-31T05:25:20Z", "host": "server",
                       "args": ["--checkpoint-dir", "/home/x/loop2/gen006/checkpoints"]}, file)
        with runlog.begin("train", name="live") as run:
            run.result(wandb_id="livelive")

        self.assertEqual(runlog.import_wandb([wandb]), 2 if pb is None else 3)
        self.assertEqual(runlog.import_wandb([wandb]), 0)
        self.assertEqual(len(runlog.load()), 3 if pb is None else 4)
        self.assertEqual(next(r for r in runlog.load() if r.get("wandb_id") == "kt5fwcz5")["name"], "loop2/gen006",
                         "a self-play generation's training is named after its generation")
        runs = {run["wandb_id"]: run for run in runlog.load() if "wandb_id" in run}
        run = runs["mhn8ihqb"]
        self.assertEqual(run["time"], runlog.datetime.datetime(2026, 10, 27, 23, 45, 3, tzinfo=runlog.datetime.timezone.utc).astimezone().isoformat(),
                         "imported times are local like live ones")
        self.assertEqual((run["kind"], run["repo"]["commit"], run["machine"]), ("train", "abc", "desktop"))
        self.assertEqual(run["argv"], ["python", "train.py", "--lr", "3e-4"])
        self.assertEqual(run["results"][0], {"id": run["id"], "time": run["time"], "val_loss": 0.25, "epoch": 3})
        self.assertEqual((run["end"]["exit"], run["end"]["seconds"]), (None, 9.0))
        if pb is not None:
            offline = runs["7fu9314k"]
            self.assertEqual((offline["machine"], offline["repo"]["commit"]), ("server", "def"))
            self.assertEqual(offline["argv"], ["python", "train.py", "--monitor", "val_total_loss", "--checkpoint-dir", "/runs/v2policy/structured-ft/"])
            self.assertEqual(offline["name"], "structured-ft")
            self.assertEqual(run["name"], "mhn8ihqb", "without a checkpoint folder the W&B id names the run")
            self.assertEqual(offline["results"][0]["val_loss"], 0.3, "the last summary update counts")
            self.assertNotIn("train_loss", offline["results"][0])
            self.assertEqual((offline["end"]["exit"], offline["end"]["seconds"]), (0, 120.0))


if __name__ == "__main__":
    unittest.main()
