#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
session="${XDG_SESSION_TYPE:-}"
if [[ "$session" == "wayland" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    printf '%s\n' 'HSR dispatch requires an X11 session; Wayland is rejected because global focus and target-window checks are unavailable.' >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}" ]]; then
    printf '%s\n' 'No X11 DISPLAY is available. Start an X11 desktop session to run HSR.' >&2
    exit 1
fi
command -v xdotool >/dev/null 2>&1 || { printf '%s\n' 'Install xdotool for X11 window and focus checks.' >&2; exit 1; }
command -v dotnet >/dev/null 2>&1 || { printf '%s\n' 'Install the .NET 8 SDK/runtime and make dotnet available on PATH.' >&2; exit 1; }

python="$root/research/human-sim/.venv/bin/python"
human_sim="$root/research/human-sim/.venv/bin/human-sim"
if [[ ! -x "$human_sim" ]]; then
    "$root/research/setup-research.sh"
fi
"$root/research/build-research.sh" Debug

client="$root/osu.Desktop/bin/Debug/net8.0/osu!"
if [[ ! -x "$client" ]]; then
    printf 'Research client apphost was not built: %s\n' "$client" >&2
    exit 1
fi

printf 'Profile skill [50]: '
read -r skill
skill="${skill:-50}"
printf 'Profile effort [68]: '
read -r effort
effort="${effort:-68}"

export PYTHONPATH="$root/research/human-sim/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$human_sim" auto-run "$client" --mode profile --skill "$skill" --effort "$effort"
