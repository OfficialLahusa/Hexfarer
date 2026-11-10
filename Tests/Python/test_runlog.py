"""Run registry (Agents/Training/runlog.py): the wrapper records command, commit, exit code and output; a tool started by
the wrapper adds to its run instead of starting a second one, anything it starts becomes a child; events merge per run;
a run stopped by a signal records its end (the wrapper passes the signal on to its command and waits for it); sync merges
without duplicates, copies every run folder however many there are and reports those that did not arrive; .NET build
commits, dirty diffs and W&B runs are read.

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import contextlib
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

import runlog  # noqa: E402
import selfplay_loop  # noqa: E402


def posix_shell():
    """bash for a stand-in of another machine: on Windows Git's (System32's bash.exe would be WSL's), else the system's"""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    folder = os.path.dirname(git) if git else None
    # git.exe lies in Git\cmd or Git\mingw64\bin; bash.exe in Git\bin
    for _ in range(3):
        if folder is None:
            break
        candidate = os.path.join(folder, "bin", "bash.exe")
        if os.path.isfile(candidate):
            return candidate
        folder = os.path.dirname(folder)
    return None


class RunlogTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.directory.name, "runs")
        self.saved = {key: os.environ.get(key) for key in ("LANZENBRECHER_RUNS", runlog.ENV_ID, runlog.ENV_DIRECT, runlog.ENV_PARENT)}
        os.environ["LANZENBRECHER_RUNS"] = self.root
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
        # A program by a relative path with forward slashes, as written in shell scripts
        relative = os.path.relpath(sys.executable).replace(os.sep, "/")
        self.assertEqual(runlog.wrap("other", [relative, "-c", "print('relative')"], name="relative", quiet=True), 0)

        runs = runlog.load()
        self.assertEqual([run["name"] for run in runs], ["ok", "fails", "relative"])
        ok, fails, relative_run = runs
        self.assertEqual(relative_run["end"]["last_line"], "relative")
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

    def run_python(self, code):
        """Runs code in a fresh Python with runlog importable; returns the finished process"""
        return subprocess.run(self.python(code), capture_output=True, text=True, timeout=60)

    def test_a_run_stopped_by_a_signal_records_its_end(self):
        # SIGTERM inside a run (as from kill): the run ends with 128 + 15 and names the signal, the process exits with it
        stopped = self.run_python("import runlog, signal, time\n"
                                  "with runlog.begin('train', name='terminated'):\n"
                                  "    signal.raise_signal(signal.SIGTERM)\n"
                                  "    time.sleep(30)\n")
        self.assertEqual(stopped.returncode, 143, stopped.stderr)
        self.assertNotIn("Traceback", stopped.stderr)
        # Ctrl-C: 128 + 2
        self.run_python("import runlog\n"
                        "with runlog.begin('train', name='interrupted'):\n"
                        "    raise KeyboardInterrupt\n")
        # A second signal while the first one unwinds must not cut the cleanup (and the run's end) short
        twice = self.run_python("import runlog, signal\n"
                                "with runlog.begin('train', name='twice'):\n"
                                "    try:\n"
                                "        signal.raise_signal(signal.SIGTERM)\n"
                                "    finally:\n"
                                "        signal.raise_signal(signal.SIGTERM)\n"
                                "        print('the cleanup ran to its end')\n")
        self.assertIn("the cleanup ran to its end", twice.stdout, twice.stderr)
        # A tracked command and the run around it both end by the signal
        self.run_python("import runlog, signal\n"
                        "with runlog.begin('selfplay-loop', name='around'):\n"
                        "    with runlog.track('collect', ['collector'], None) as finish:\n"
                        "        signal.raise_signal(signal.SIGTERM)\n")

        runs = {r["name"] or r["kind"]: r for r in runlog.load()}
        for name in ("terminated", "twice", "around", "collect"):
            self.assertEqual((runs[name]["end"]["exit"], runs[name]["end"].get("signal")), (143, "SIGTERM"), name)
        self.assertEqual((runs["interrupted"]["end"]["exit"], runs["interrupted"]["end"].get("signal")), (130, "SIGINT"))
        self.assertNotIn("error", runs["terminated"]["end"])

    def test_a_selfplay_loop_stopped_by_a_signal_ends_its_open_runs(self):
        loop_dir = os.path.join(self.directory.name, "loop").replace(os.sep, "/")
        stopped = self.run_python("import runlog, selfplay_loop, signal\n"
                                  "def stand_in(command, output_path, kind=None):\n"
                                  "    with runlog.track(kind or 'other', command, output_path):\n"
                                  "        signal.raise_signal(signal.SIGTERM)\n"
                                  "selfplay_loop.run = stand_in\n"
                                  f"selfplay_loop.main(['--loop-dir', {loop_dir!r}, '--start-checkpoint', 'start.ckpt', "
                                  "'--start-model', 'start.onnx', '--generations', '2'])\n")
        self.assertEqual(stopped.returncode, 143, stopped.stderr)
        runs = {r["kind"]: r for r in runlog.load()}
        for kind in ("selfplay-loop", "selfplay-gen", "collect"):
            self.assertEqual((runs[kind]["end"]["exit"], runs[kind]["end"].get("signal")), (143, "SIGTERM"), kind)

    def test_begin_keeps_a_programs_own_signal_handlers(self):
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("signal handlers can only be set in the main thread")
        before = signal.getsignal(signal.SIGTERM)
        try:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            with runlog.begin("train", name="default"):
                self.assertNotEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL, "a run catches SIGTERM to record its end")
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL, "and gives it back when it ends")

            def own(signum, frame):
                pass
            signal.signal(signal.SIGTERM, own)
            with runlog.begin("train", name="own"):
                self.assertIs(signal.getsignal(signal.SIGTERM), own, "a handler the program set stays")
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            with runlog.begin("train", name="ignored"):
                self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN, "an ignored signal stays ignored")
        finally:
            signal.signal(signal.SIGTERM, before)

    def wrap_with_a_signal(self, signum, child_seconds, name):
        """Runs a child that prints a line every 0.1 s for child_seconds under the wrapper and raises signum in this process
        after 1 s; a guard handler catches the signal if the wrapper does not. Returns (exit code, seconds, guard hits)."""
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("signal handlers can only be set in the main thread")
        hits = []

        def guard(s, f):
            hits.append(s)
        before = signal.signal(signum, guard)
        timer = threading.Timer(1.0, signal.raise_signal, [signum])
        try:
            timer.start()
            started = time.time()
            code = runlog.wrap("other", self.python(f"import time\nfor i in range({int(child_seconds * 10)}):\n"
                                                    "    print(i, flush=True); time.sleep(0.1)\nprint('done')"),
                               name=name, quiet=True)
            seconds = time.time() - started
            self.assertEqual(len(hits), 0, "the signal reached this process's handler instead of the wrapper's")
            self.assertIs(signal.getsignal(signum), guard, "the wrapper restores the handler it found")
        finally:
            timer.cancel()
            signal.signal(signum, before)
        return code, seconds, hits

    def test_the_wrapper_passes_sigterm_on_and_records_the_end(self):
        code, seconds, _ = self.wrap_with_a_signal(signal.SIGTERM, 30, "terminated")
        self.assertNotEqual(code, 0)
        self.assertLess(seconds, 15, "the command must be stopped, not run to its end")
        (run,) = runlog.load()
        self.assertEqual(run["end"]["signal"], "SIGTERM")
        self.assertEqual(run["end"]["exit"], code)
        with open(os.path.join(self.root, run["id"], "console.log"), encoding="utf-8") as file:
            self.assertIn("5", file.read().split(), "what the command printed before the signal is kept")

    def test_the_wrapper_waits_for_a_command_that_got_the_terminals_ctrl_c(self):
        # From the terminal Ctrl-C reaches the command as well; the wrapper doesn't pass it on a second time and keeps
        # recording until the command ends (Arena prints its result so far after a Ctrl-C)
        original = runlog._terminal_sent_it
        runlog._terminal_sent_it = lambda: True
        try:
            code, seconds, _ = self.wrap_with_a_signal(signal.SIGINT, 2, "interrupted")
        finally:
            runlog._terminal_sent_it = original
        self.assertEqual(code, 0, "this command does not react to Ctrl-C, so it finishes")
        (run,) = runlog.load()
        self.assertEqual((run["end"]["signal"], run["end"]["last_line"]), ("SIGINT", "done"))

    def test_the_wrapper_leaves_an_ignored_signal_ignored(self):
        # A background job ignores Ctrl-C; so does its wrapper, which doesn't record or pass on what it ignores
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("signal handlers can only be set in the main thread")
        before = signal.signal(signal.SIGINT, signal.SIG_IGN)
        timer = threading.Timer(1.0, signal.raise_signal, [signal.SIGINT])
        try:
            timer.start()
            code = runlog.wrap("other", self.python("import time\nfor i in range(20):\n    print(i, flush=True); time.sleep(0.1)"),
                               name="background", quiet=True)
            self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
        finally:
            timer.cancel()
            signal.signal(signal.SIGINT, before)
        self.assertEqual(code, 0)
        (run,) = runlog.load()
        self.assertNotIn("signal", run["end"])

    @unittest.skipIf(os.name == "nt", "Windows' console sends Ctrl-C to every process itself; it can't be passed on")
    def test_the_wrapper_passes_on_a_ctrl_c_that_was_sent_to_it_alone(self):
        original = runlog._terminal_sent_it
        runlog._terminal_sent_it = lambda: False
        try:
            code, seconds, _ = self.wrap_with_a_signal(signal.SIGINT, 30, "interrupted")
        finally:
            runlog._terminal_sent_it = original
        self.assertNotEqual(code, 0)
        self.assertLess(seconds, 15)
        (run,) = runlog.load()
        self.assertEqual(run["end"]["signal"], "SIGINT")

    def test_a_registry_that_fails_during_a_run_does_not_stop_the_work(self):
        os.environ[runlog.ENV_PARENT] = "the-caller"
        run = runlog.begin("train", name="long")
        # The registry can't be written any more (a directory took the file's place)
        registry = os.path.join(self.root, runlog.REGISTRY_FILE)
        os.replace(registry, registry + ".saved")
        os.makedirs(registry)
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            run.result(best_checkpoint="epoch=7.ckpt")
            run.end(0)
        self.assertIn("could not record the result", errors.getvalue())
        self.assertIn("could not record the end", errors.getvalue())
        self.assertEqual(os.environ[runlog.ENV_PARENT], "the-caller")
        # A run that can't record its start doesn't start
        with self.assertRaises(OSError):
            runlog.start("train", name="unrecorded", argv=["x"])

    def test_every_kind_of_run_can_be_registered(self):
        for kind in ("train", "export", "precision", "selfplay-loop", "selfplay-gen", "collect", "convert", "reanalysis", "duel", "bench", "other"):
            runlog.start(kind, name=kind, argv=["x"]).end(0)
        self.assertEqual([run["kind"] for run in runlog.load()][-3:], ["duel", "bench", "other"])
        with self.assertRaises(ValueError):
            runlog.start("unknown", argv=["x"])
        self.assertEqual(runlog.main(["run", "--kind", "reanalysis", "--name", "via-cli", "--", sys.executable, "-c", "print('ok')"]), 0)

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

    def fake_remote(self, count, folders):
        """Another machine's registry with count runs, the first `folders` of them with a run folder (later ones were
        entered by hand and have none); and an ssh stand-in that runs the command in a POSIX shell after cutting it to
        8191 characters, as Git for Windows' ssh does. Returns (the --ssh string, the remote dir, the ids with folders)."""
        shell = posix_shell()
        if shell is None:
            self.skipTest("no POSIX shell (bash) to stand in for the other machine")
        remote = os.path.join(self.directory.name, "remote")
        os.makedirs(remote)
        ids = [f"20261110-{i:06d}-otherhost-duel-{i:04x}" for i in range(count)]
        with open(os.path.join(remote, runlog.REGISTRY_FILE), "w", encoding="utf-8") as file:
            for run_id in ids:
                file.write(json.dumps({"event": "start", "id": run_id, "time": "2026-11-10T12:00:00+01:00", "kind": "duel"}) + "\n")
        for run_id in ids[:folders]:
            os.makedirs(os.path.join(remote, run_id))
            with open(os.path.join(remote, run_id, "console.log"), "w", encoding="utf-8") as file:
                file.write(f"output of {run_id}\n")
        stand_in = os.path.join(self.directory.name, "fake_ssh.py")
        with open(stand_in, "w", encoding="utf-8") as file:
            file.write("import os, subprocess, sys\n"
                       "command = sys.argv[1][:8191]\n"
                       "data = None\n"
                       "if os.environ.get('FAKE_SSH_DROP_LAST_NAME') and 'tar c' in command:\n"
                       "    data = b''.join(sys.stdin.buffer.read().splitlines(keepends=True)[:-1])\n"
                       f"sys.exit(subprocess.run([{shell!r}, '-c', command], input=data).returncode)\n")
        return f"{shlex.join([sys.executable, stand_in])}", remote.replace(os.sep, "/"), ids[:folders]

    def test_sync_copies_every_run_folder_however_many_there_are(self):
        # 300 runs with folders: their names alone are ~12,000 characters, more than the ssh command line carries
        ssh, remote, with_folders = self.fake_remote(320, 300)
        os.makedirs(self.root)
        os.makedirs(os.path.join(self.root, with_folders[5]))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runlog.main(["sync", "--ssh", ssh, "--remote-dir", remote]), 0)
        self.assertIn("320 new events, 299 run folders", output.getvalue())
        for run_id in with_folders:
            path = os.path.join(self.root, run_id, "console.log")
            # A folder already here is left as it is
            self.assertEqual(os.path.isfile(path), run_id != with_folders[5], run_id)
        with open(os.path.join(self.root, with_folders[-1], "console.log"), encoding="utf-8") as file:
            self.assertEqual(file.read(), f"output of {with_folders[-1]}\n")
        self.assertEqual(len(runlog.load()), 320)
        # Nothing new the second time
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runlog.main(["sync", "--ssh", ssh, "--remote-dir", remote]), 0)
        self.assertIn("0 new events, 0 run folders", output.getvalue())

    def test_sync_reports_run_folders_that_did_not_arrive(self):
        ssh, remote, with_folders = self.fake_remote(4, 3)
        os.environ["FAKE_SSH_DROP_LAST_NAME"] = "1"
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(runlog.main(["sync", "--ssh", ssh, "--remote-dir", remote]), 1)
        finally:
            os.environ.pop("FAKE_SSH_DROP_LAST_NAME")
        self.assertIn("4 new events, 2 run folders", output.getvalue())
        self.assertIn(f"1 run folder did not arrive: {with_folders[-1]}", output.getvalue())

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
