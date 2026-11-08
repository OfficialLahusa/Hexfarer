"""
Builds and runs the StacSettlers adapter (Java, in src/) against StacSettlers' jar with its dependencies
(STACSettlers-1.0-bin.jar), which STACSETTLERS_JAR names (StacSettlers is a dependency, never copied in). The classes go
to build/ next to this file. simulate() runs StacSettlers' headless games (its Simulation, through OSPSimulation, which
knows the agent kind "osp") in a working directory of their own.
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.join(HERE, "src")
CLASSES = os.path.join(HERE, "build")
# StacSettlers' MCTS agents read this file from the working directory
TYPE_PDF = "mcts.game.catan.typepdf.HumanActionTypePdf.dat"


def jar():
    return os.environ.get("STACSETTLERS_JAR", "")


def available():
    """Whether a JDK and StacSettlers' jar are there"""
    return bool(jar()) and os.path.isfile(jar()) and shutil.which("javac") is not None and shutil.which("java") is not None


def classpath():
    return os.pathsep.join([CLASSES, jar()])


def build():
    """Compiles every source file into build/"""
    files = [os.path.join(root, name) for root, _, names in os.walk(SOURCES) for name in names if name.endswith(".java")]
    os.makedirs(CLASSES, exist_ok=True)
    subprocess.run(["javac", "-encoding", "UTF-8", "-nowarn", "-cp", jar(), "-d", CLASSES, *files], check=True)


def run(main_class, *args, stdin=None, timeout=600):
    """Runs a class's main and returns its standard output"""
    result = subprocess.run(["java", "-cp", classpath(), main_class, *args], input=stdin, capture_output=True, text=True,
                            encoding="utf-8", timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{main_class} {' '.join(args)} failed ({result.returncode}):\n{result.stderr}")
    return result.stdout


def config(name, games, agents, chat=True, fully_observable=False, force_end_turns=False, log=False):
    """A Simulation config: the suite's settings, then the agents as (count, name, kind[, flags]), e.g.
    (1, "Hexfarer", "osp") and (3, "Stac", "stac", "TRY_N_BEST_BUILD_PLANS:0|FAVOUR_DEV_CARDS:-5")"""
    lines = [f"Games={games}", f"Log={'true' if log else 'false'}", f"ChatNegotiation={'true' if chat else 'false'}"]
    if fully_observable:
        lines.append("FullyObservable=true")
    if not force_end_turns:
        lines.append("ForceEndTurns=false")
    lines += ["~", name, "Trades=true"]
    lines += ["Agent=" + ",".join(str(part) for part in agent) for agent in agents]
    return "\n".join(lines) + "\n"


def simulate(directory, config_text, agent_command=None, properties=None, timeout=1800, java_options=()):
    """Runs StacSettlers' Simulation with the config in the working directory (created, with the MCTS agents' file),
    our robots' agent given by agent_command (osp.agent); returns its standard output and error together"""
    os.makedirs(directory, exist_ok=True)
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(jar()))), TYPE_PDF)
    if os.path.isfile(source):
        shutil.copy(source, os.path.join(directory, TYPE_PDF))
    with open(os.path.join(directory, "config.txt"), "w", encoding="utf-8") as file:
        file.write(config_text)
    arguments = ["java", *java_options]
    if agent_command:
        arguments.append(f"-Dosp.agent={agent_command}")
    arguments += [f"-D{name}={value}" for name, value in (properties or {}).items()]
    arguments += ["-cp", classpath(), "osp.stacsettlers.OSPSimulation", "config.txt"]
    result = subprocess.run(arguments, cwd=directory, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return result.stdout + result.stderr


if __name__ == "__main__":
    build()
    if len(sys.argv) > 1:
        print(run(*sys.argv[1:]), end="")
