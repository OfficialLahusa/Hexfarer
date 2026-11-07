"""
Builds and runs the JSettlers adapter (Java, in src/) against JSettlers' full jar, which JSETTLERS_JAR names (JSettlers
is a dependency, never copied in). The classes go to build/ next to this file.
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = os.path.join(HERE, "src")
CLASSES = os.path.join(HERE, "build")


def jar():
    return os.environ.get("JSETTLERS_JAR", "")


def available():
    """Whether a JDK and JSettlers' jar are there"""
    return bool(jar()) and os.path.isfile(jar()) and shutil.which("javac") is not None and shutil.which("java") is not None


def classpath():
    return os.pathsep.join([CLASSES, jar()])


def build():
    """Compiles every source file into build/"""
    files = [os.path.join(root, name) for root, _, names in os.walk(SOURCES) for name in names if name.endswith(".java")]
    os.makedirs(CLASSES, exist_ok=True)
    subprocess.run(["javac", "-encoding", "UTF-8", "-cp", jar(), "-d", CLASSES, *files], check=True)


def run(main_class, *args, stdin=None, timeout=600):
    """Runs a class's main and returns its standard output"""
    result = subprocess.run(["java", "-cp", classpath(), main_class, *args], input=stdin, capture_output=True, text=True,
                            encoding="utf-8", timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{main_class} {' '.join(args)} failed ({result.returncode}):\n{result.stderr}")
    return result.stdout


if __name__ == "__main__":
    build()
    if len(sys.argv) > 1:
        print(run(*sys.argv[1:]), end="")
