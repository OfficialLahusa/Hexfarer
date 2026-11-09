#!/bin/sh
# The run registry (Agents/Training/runlog.py) from the repository root, e.g.
#   ./chronicle.sh run --kind duel --name NAME -- COMMAND ...      ./chronicle.sh list --kind train      ./chronicle.sh show NAME
# Uses the Agents/env interpreter if there is one, otherwise python3 (list, show, run and sync need only the standard library).
here=$(cd "$(dirname "$0")" && pwd)
for python in "$here/Agents/env/bin/python" "$here/Agents/env/python.exe" python3; do
    if [ -x "$python" ] || command -v "$python" >/dev/null 2>&1; then
        exec "$python" "$here/Agents/Training/runlog.py" "$@"
    fi
done
echo "chronicle: no Python found" >&2
exit 1
