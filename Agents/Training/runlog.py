"""
Central run registry: every training, export, evaluation, self-play generation, duel or collection gets an entry with its
full command line, the commit it ran from and its result, in a folder outside the repository.

Layout of the registry (HEXFARER_RUNS, by default ~/Hexfarer-data/runs, or ~/hexfarer-arena/runs on a machine that only
has the arena folder):
- runs.jsonl: one JSON event per line, appended only. A run has a "start" event (kind, name, machine, command, working
  directory, commit and whether the code had uncommitted changes, inputs, outputs, parent run, note), any number of
  "result" events and an "end" event (exit code, duration). load() merges them per run id.
- <id>/: the run's console output (when the wrapper ran it) and diff.patch if the code was dirty.

Two ways in:
- the wrapper for any command, Python or C#:  python runlog.py run --kind duel --name NAME -- COMMAND ...
  It also reads the commit a .NET binary was built from (the SDK puts it into the informational version).
- begin() in the Python tools (train.py, export_onnx.py, evaluate_precision.py, selfplay_loop.py). Started directly by
  the wrapper, a tool adds its results to the wrapper's run instead of starting a second one; anything it starts itself
  becomes a child run (HEXFARER_RUN_PARENT).

More commands: list, show ID, sync (pulls another machine's registry over ssh), import-wandb (backfill from W&B runs),
add (enters a run by hand, e.g. reconstructed from logs).
"""

import argparse
import datetime
import json
import os
import re
import secrets
import shlex
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REGISTRY_FILE = "runs.jsonl"
KINDS = ("train", "export", "precision", "selfplay-loop", "selfplay-gen", "collect", "convert", "duel", "bench", "other")

# Set for the processes a run starts: the run they belong to, and whether they were started directly by the wrapper
ENV_ID, ENV_DIRECT, ENV_PARENT = "HEXFARER_RUN_ID", "HEXFARER_RUN_DIRECT", "HEXFARER_RUN_PARENT"


def registry_root():
    if os.environ.get("HEXFARER_RUNS"):
        return os.environ["HEXFARER_RUNS"]
    home = os.path.expanduser("~")
    if not os.path.isdir(os.path.join(home, "Hexfarer-data")) and os.path.isdir(os.path.join(home, "hexfarer-arena")):
        return os.path.join(home, "hexfarer-arena", "runs")
    return os.path.join(home, "Hexfarer-data", "runs")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def append(event, root=None):
    """Appends one event as a line of JSON (one write, so concurrent runs don't interleave within a line)"""
    root = root or registry_root()
    os.makedirs(root, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    with open(os.path.join(root, REGISTRY_FILE), "a", encoding="utf-8") as file:
        file.write(line)


def git(args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None


def repo_state(path=REPO):
    """Commit, branch and whether the working tree has uncommitted changes (untracked files count), or None outside git"""
    commit = git(["rev-parse", "HEAD"], path)
    if not commit:
        return None
    status = git(["status", "--porcelain"], path) or ""
    return {"commit": commit.strip(), "branch": (git(["rev-parse", "--abbrev-ref", "HEAD"], path) or "").strip(),
            "dirty": bool(status.strip()), "path": os.path.abspath(path)}


def save_diff(path, target):
    """The uncommitted changes of the repository at path (status and diff against HEAD) into target"""
    with open(target, "w", encoding="utf-8") as file:
        file.write(git(["status", "--porcelain"], path) or "")
        file.write("\n")
        file.write(git(["diff", "HEAD"], path) or "")


def binary_commit(executable):
    """The commit a .NET program was built from: the SDK appends it to the informational version ("1.0.0+<sha>"), stored
    in the program's .dll (next to an apphost) or in the file itself. None if there is none."""
    candidates = [executable]
    stem, extension = os.path.splitext(executable)
    if extension.lower() != ".dll":
        candidates.insert(0, stem + ".dll")
    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, "rb") as file:
                match = re.search(rb"\d+\.\d+\.\d+\+([0-9a-f]{40})", file.read())
            if match:
                return match.group(1).decode("ascii")
    return None


class Run:
    """A run in the registry. As a context manager it ends with exit code 0, or 1 on an exception (which goes on)."""

    def __init__(self, run_id, root, attached=False):
        self.id, self.root, self.attached = run_id, root, attached
        self.started = time.time()
        self.ended = False
        self._env = None

    @property
    def directory(self):
        return os.path.join(self.root, self.id)

    def result(self, **fields):
        append({"event": "result", "id": self.id, "time": now(), **fields}, self.root)

    def end(self, exit_code=0, **fields):
        """Writes the end event once; later calls do nothing"""
        if self.ended:
            return
        self.ended = True
        if not self.attached:
            append({"event": "end", "id": self.id, "time": now(), "exit": exit_code,
                    "seconds": round(time.time() - self.started, 1), **fields}, self.root)
        self._restore_env()

    def _enter_env(self):
        # Processes started from here on belong to this run
        self._env = {key: os.environ.get(key) for key in (ENV_PARENT, ENV_DIRECT)}
        os.environ[ENV_PARENT] = self.id
        os.environ[ENV_DIRECT] = "0"

    def _restore_env(self):
        if self._env is None:
            return
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._env = None

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.end(0 if kind is None else 1, **({} if kind is None else {"error": f"{kind.__name__}: {value}"[:500]}))
        return False


def last_line(text):
    """The last non-empty line of a program's output, without terminal color codes"""
    lines = [line.strip() for line in re.split(r"[\r\n]+", re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)) if line.strip()]
    return lines[-1][:500] if lines else ""


def new_id(kind):
    return f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{socket.gethostname().lower()}-{kind}-{secrets.token_hex(2)}"


def start(kind, name=None, argv=None, inputs=None, outputs=None, note=None, parent=None, root=None, binary=None, **fields):
    """Writes the start event of a new run and returns it; saves the diff if the code has uncommitted changes"""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}, one of {', '.join(KINDS)}")
    root = root or registry_root()
    run = Run(new_id(kind), root)
    argv = list(argv if argv is not None else [sys.executable, *sys.argv])
    repo = repo_state()
    event = {"event": "start", "id": run.id, "time": now(), "kind": kind, "name": name, "machine": socket.gethostname(),
             "argv": argv, "command": shlex.join(argv), "cwd": os.getcwd(), "repo": repo,
             "parent": parent if parent is not None else os.environ.get(ENV_PARENT) or None,
             "inputs": inputs or {}, "outputs": outputs or {}, "note": note, **fields}
    if binary:
        event["binary"] = {"path": os.path.abspath(binary), "commit": binary_commit(binary)}
    os.makedirs(run.directory, exist_ok=True)
    if repo and repo["dirty"]:
        save_diff(repo["path"], os.path.join(run.directory, "diff.patch"))
    append(event, root)
    return run


def begin(kind, name=None, inputs=None, outputs=None, note=None, **fields):
    """For a Python tool's own entry: a new run (a child of the run that started this process, if any), or, when the wrapper
    started this process directly, the wrapper's run (results go there, it ends there). Processes started inside it
    belong to it."""
    if os.environ.get(ENV_DIRECT) == "1" and os.environ.get(ENV_ID):
        run = Run(os.environ[ENV_ID], registry_root(), attached=True)
        run.result(kind_detail=kind, name=name, inputs=inputs or {}, outputs=outputs or {}, **fields)
    else:
        run = start(kind, name=name, inputs=inputs, outputs=outputs, note=note, **fields)
    run._enter_env()
    return run


def wrap(kind, command, name=None, note=None, inputs=None, outputs=None, root=None, quiet=False):
    """Runs a command as a run: its output goes to the console and into <id>/console.log, the last non-empty line is kept
    as the result. Returns the exit code."""
    executable = command[0]
    binary = executable if os.path.isfile(executable) and not os.path.basename(executable).lower().startswith("python") else None
    run = start(kind, name=name, argv=command, inputs=inputs, outputs=outputs, note=note, root=root, binary=binary)
    env = dict(os.environ, **{ENV_ID: run.id, ENV_DIRECT: "1", "HEXFARER_RUNS": run.root})
    env.pop(ENV_PARENT, None)
    tail = b""
    with open(os.path.join(run.directory, "console.log"), "wb") as log:
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        except OSError as error:
            run.end(127, error=str(error))
            raise
        with process:
            for chunk in iter(lambda: process.stdout.read1(65536), b""):
                log.write(chunk)
                if not quiet:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                tail = (tail + chunk)[-4096:]
        code = process.wait()
    run.end(code, last_line=last_line(tail.decode("utf-8", errors="replace")))
    return code


def track(kind, command, output_path=None, name=None):
    """Start and end events around a command the caller runs itself (e.g. with its output in a file): use as
    `with track(...) as finish: code = ...; finish(code)`; the last line of output_path becomes the result"""
    executable = command[0]
    binary = executable if os.path.isfile(executable) and not os.path.basename(executable).lower().startswith("python") else None
    run = start(kind, name=name, argv=command, binary=binary)

    class Finisher:
        def __init__(self):
            self.done = False

        def __call__(self, code):
            last = ""
            if output_path and os.path.exists(output_path):
                with open(output_path, encoding="utf-8", errors="replace") as file:
                    last = last_line(file.read())
            run.end(code, last_line=last, log=output_path)
            self.done = True

    class Context:
        def __enter__(self):
            self.finish = Finisher()
            return self.finish

        def __exit__(self, kind_, value, traceback):
            if not self.finish.done:
                self.finish(1 if kind_ else 0)
            return False

    return Context()


def load(root=None):
    """The runs of a registry, oldest first: each the start event's fields with all results merged in, then the end's"""
    root = root or registry_root()
    path = os.path.join(root, REGISTRY_FILE)
    runs = {}
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            event = json.loads(line)
            run = runs.setdefault(event["id"], {"id": event["id"], "results": []})
            kind = event.pop("event")
            if kind == "start":
                run.update(event)
            elif kind == "result":
                run["results"].append(event)
            elif kind == "end":
                run["end"] = event
    def started(run):
        try:
            return datetime.datetime.fromisoformat(run["time"].replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError, AttributeError):
            return 0.0
    return sorted(runs.values(), key=started)


def merge(root, lines):
    """Appends the event lines not already in the registry (same text); returns how many were new"""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, REGISTRY_FILE)
    known = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as file:
            known = {line.rstrip("\n") for line in file}
    new = [line.rstrip("\n") for line in lines if line.strip() and line.rstrip("\n") not in known]
    unique = list(dict.fromkeys(new))
    if unique:
        with open(path, "a", encoding="utf-8") as file:
            file.write("".join(line + "\n" for line in unique))
    return len(unique)


def summary(run):
    """One line per run for list"""
    end = run.get("end") or {}
    repo = run.get("repo") or {}
    commit = (run.get("binary") or {}).get("commit") or repo.get("commit") or ""
    state = "imported" if end.get("imported") and end.get("exit") is None else f"exit {end['exit']}" if "exit" in end else "running?"
    duration = f"{end['seconds'] / 60:.0f} min" if end.get("seconds") is not None else ""
    result = end.get("last_line") or ""
    for item in run.get("results", []):
        fields = {k: v for k, v in item.items() if k not in ("id", "time", "inputs", "outputs", "kind_detail", "name")}
        if fields:
            result = json.dumps(fields, ensure_ascii=False)[:160]
    return (f"{run['id']:<46} {run.get('kind', ''):<13} {str(run.get('name') or '')[:28]:<28} {commit[:8]}{'*' if repo.get('dirty') else ' '} "
            f"{state:<8} {duration:>7}  {result[:120]}")


def _wandb_run_file(path):
    """Metadata, final summary and exit of an offline W&B run, read from its .wandb file (offline runs keep them only
    there): ({startedAt, host, program, executable, args, git}, {summary key: value}, {exit_code, runtime}), or None"""
    try:
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal.datastore import DataStore
    except ImportError:
        return None
    store = DataStore()
    store.open_for_scan(path)
    metadata, final, exit_info = None, {}, {}
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        kind = record.WhichOneof("record_type")
        if kind == "environment" and metadata is None:
            env = record.environment
            metadata = {"startedAt": datetime.datetime.fromtimestamp(env.started_at.seconds, datetime.timezone.utc).isoformat(),
                        "host": env.host, "program": env.program, "executable": env.executable, "args": list(env.args),
                        "git": {"commit": env.git.commit}}
        elif kind == "summary":
            for item in record.summary.update:
                final[item.key or "/".join(item.nested_key)] = json.loads(item.value_json)
        elif kind == "exit":
            exit_info = {"exit_code": record.exit.exit_code, "runtime": record.exit.runtime}
    return (metadata, final, exit_info) if metadata else None


def import_wandb(directories, root=None):
    """Backfill: a train run for every W&B run folder not yet in the registry (args, commit and host from its metadata,
    final metrics from its summary, exit code and runtime if known); returns how many were added"""
    root = root or registry_root()
    # Live runs keep their W&B id in a result, imported ones at the top
    known = {run.get("wandb_id") for run in load(root)} | {result.get("wandb_id") for run in load(root) for result in run["results"]}
    added = 0
    for directory in directories:
        for entry in sorted(os.listdir(directory)):
            folder = os.path.join(directory, entry)
            wandb_id = entry.rsplit("-", 1)[-1]
            if not os.path.isdir(folder) or wandb_id in known:
                continue
            metadata_path = os.path.join(folder, "files", "wandb-metadata.json")
            summary_path = os.path.join(folder, "files", "wandb-summary.json")
            run_file = os.path.join(folder, f"run-{wandb_id}.wandb")
            exit_info = {}
            if os.path.isfile(metadata_path):
                with open(metadata_path, encoding="utf-8") as file:
                    metadata = json.load(file)
                final = {}
                if os.path.isfile(summary_path):
                    with open(summary_path, encoding="utf-8") as file:
                        final = json.load(file)
            elif os.path.isfile(run_file) and (parsed := _wandb_run_file(run_file)):
                metadata, final, exit_info = parsed
            else:
                continue
            metrics = {k: v for k, v in final.items() if k.startswith(("val_", "epoch")) and isinstance(v, (int, float))}
            argv = [metadata.get("executable") or "python", metadata.get("program") or "train.py", *metadata.get("args", [])]
            # W&B keeps UTC; the registry keeps local time like its live entries
            local = datetime.datetime.fromisoformat(metadata["startedAt"].replace("Z", "+00:00")).astimezone()
            started = local.isoformat(timespec="seconds")
            run_id = f"{local.strftime('%Y%m%d-%H%M%S')}-{(metadata.get('host') or 'unknown').lower()}-train-{wandb_id}"
            args = metadata.get("args", [])
            # Named like live runs after the checkpoint folder, if the command gave one
            name = wandb_id
            if "--checkpoint-dir" in args[:-1]:
                parts = os.path.normpath(args[args.index("--checkpoint-dir") + 1]).replace("\\", "/").split("/")
                # A self-play generation trains into <loop>/<generation>/checkpoints
                name = "/".join(parts[-3:-1]) if parts[-1] == "checkpoints" and len(parts) >= 3 else parts[-1]
            append({"event": "start", "id": run_id, "time": started, "kind": "train", "name": name, "machine": metadata.get("host"),
                    "argv": argv, "command": shlex.join(argv), "cwd": None, "parent": None, "inputs": {}, "outputs": {},
                    "repo": {"commit": (metadata.get("git") or {}).get("commit"), "dirty": None},
                    "wandb_id": wandb_id, "note": "imported from W&B (no dirty flag)"}, root)
            if metrics:
                append({"event": "result", "id": run_id, "time": started, **metrics}, root)
            runtime = exit_info.get("runtime", final.get("_runtime"))
            append({"event": "end", "id": run_id, "time": started, "exit": exit_info.get("exit_code"),
                    "seconds": round(float(runtime), 1) if isinstance(runtime, (int, float)) else None, "imported": True}, root)
            known.add(wandb_id)
            added += 1
    return added


def main(argv=None):
    parser = argparse.ArgumentParser(description="Central run registry")
    sub = parser.add_subparsers(dest="action", required=True)

    run_parser = sub.add_parser("run", help="run a command as a registered run: runlog.py run --kind K [--name N] -- COMMAND ...")
    run_parser.add_argument("--kind", required=True, choices=KINDS)
    run_parser.add_argument("--name")
    run_parser.add_argument("--note")
    run_parser.add_argument("--input", action="append", default=[], help="KEY=VALUE, repeatable")
    run_parser.add_argument("--output", action="append", default=[], help="KEY=VALUE, repeatable")
    run_parser.add_argument("rest", nargs=argparse.REMAINDER)

    list_parser = sub.add_parser("list", help="list runs, newest last")
    list_parser.add_argument("--kind")
    list_parser.add_argument("--name", help="substring of the name")
    list_parser.add_argument("--last", type=int, default=30)

    show_parser = sub.add_parser("show", help="all fields of a run")
    show_parser.add_argument("id", help="the run id, a unique part of it, or the run's name")

    sync_parser = sub.add_parser("sync", help="pull another machine's registry over ssh into this one")
    sync_parser.add_argument("--ssh", required=True, help='e.g. "ssh -i KEY -o IdentitiesOnly=yes user@host"')
    sync_parser.add_argument("--remote-dir", required=True, help="the registry folder on the other machine (relative to its home)")

    wandb_parser = sub.add_parser("import-wandb", help="backfill train runs from W&B run folders")
    wandb_parser.add_argument("directories", nargs="+")

    add_parser = sub.add_parser("add", help="enter a run by hand (e.g. reconstructed from logs)")
    add_parser.add_argument("--kind", required=True, choices=KINDS)
    add_parser.add_argument("--name", required=True)
    add_parser.add_argument("--command", required=True, help="the command line as it ran")
    add_parser.add_argument("--time", required=True, help="start, ISO 8601")
    add_parser.add_argument("--machine", required=True)
    add_parser.add_argument("--commit")
    add_parser.add_argument("--result", help="the outcome in one line")
    add_parser.add_argument("--note", default="reconstructed")
    args = parser.parse_args(argv)

    if args.action == "run":
        rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
        if not rest:
            parser.error("run needs a command after --")
        pairs = lambda items: dict(item.split("=", 1) for item in items)  # noqa: E731
        return wrap(args.kind, rest, name=args.name, note=args.note, inputs=pairs(args.input), outputs=pairs(args.output))
    if args.action == "list":
        runs = [run for run in load() if (not args.kind or run.get("kind") == args.kind) and (not args.name or args.name in str(run.get("name")))]
        for run in runs[-args.last:]:
            print(summary(run))
        return 0
    if args.action == "show":
        runs = load()
        matches = [run for run in runs if args.id in run["id"]] or [run for run in runs if run.get("name") == args.id]
        if len(matches) != 1:
            print(f"{len(matches)} runs match {args.id!r}")
            return 1
        print(json.dumps(matches[0], indent=2, ensure_ascii=False))
        return 0
    if args.action == "sync":
        ssh = shlex.split(args.ssh)
        remote = subprocess.run(ssh + [f"cat {shlex.quote(args.remote_dir)}/{REGISTRY_FILE} 2>/dev/null"], capture_output=True, check=True)
        lines = remote.stdout.decode("utf-8").splitlines()
        added = merge(registry_root(), lines)
        # The run folders (console logs, diffs) of the runs not here yet
        ids = sorted({json.loads(line)["id"] for line in lines if line.strip()} - set(os.listdir(registry_root())))
        if ids:
            # Only runs started live have a folder; tar skips the others
            pack = subprocess.Popen(ssh + [f"cd {shlex.quote(args.remote_dir)} && tar cf - " + " ".join(shlex.quote(i) for i in ids) + " 2>/dev/null"],
                                    stdout=subprocess.PIPE)
            subprocess.run(["tar", "xf", "-"], stdin=pack.stdout, cwd=registry_root(), check=False)
            pack.wait()
        copied = sum(1 for run_id in ids if os.path.isdir(os.path.join(registry_root(), run_id)))
        print(f"{added} new events, {copied} run folders")
        return 0
    if args.action == "import-wandb":
        print(f"{import_wandb(args.directories)} runs imported")
        return 0
    if args.action == "add":
        run_id = f"{args.time[:19].replace('-', '').replace(':', '').replace('T', '-')}-{args.machine.lower()}-{args.kind}-{secrets.token_hex(2)}"
        append({"event": "start", "id": run_id, "time": args.time, "kind": args.kind, "name": args.name, "machine": args.machine,
                "argv": shlex.split(args.command), "command": args.command, "cwd": None, "parent": None, "inputs": {}, "outputs": {},
                "repo": {"commit": args.commit, "dirty": None}, "note": args.note})
        if args.result:
            append({"event": "result", "id": run_id, "time": args.time, "summary": args.result})
        # Entered after the fact: finished, exit code and duration unknown
        append({"event": "end", "id": run_id, "time": args.time, "exit": None, "seconds": None, "imported": True})
        print(run_id)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
