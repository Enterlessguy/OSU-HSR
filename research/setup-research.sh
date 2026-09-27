#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project="$root/research/human-sim"
python="${PYTHON:-python3}"

"$python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else "Python 3.12 or newer is required")'
venv_python="$project/.venv/bin/python"
if [[ ! -x "$venv_python" ]]; then
    "$python" -m venv "$project/.venv"
fi

"$venv_python" -m pip install --upgrade 'pip>=26.2' setuptools
"$venv_python" -m pip install -r "$project/requirements-release.txt"
"$venv_python" -m pip install --no-deps --no-build-isolation -e "$project"
printf '%s\n' 'Intelligence Database HSR Python environment is ready.'
