#!/usr/bin/env bash
set -euo pipefail

client="${1:?Usage: test-linux-desktop.sh /absolute/path/to/osu!.dll}"
[[ "$client" = /* ]] || { printf '%s\n' 'Use an absolute client path.' >&2; exit 2; }
command -v xvfb-run >/dev/null
command -v weston >/dev/null
command -v openbox >/dev/null
state="$(mktemp -d -t hsr-desktop-check-XXXXXXXX)"
compositor_pid=''
cleanup() {
    local status=$?
    if [[ -n "$compositor_pid" ]]; then kill "$compositor_pid" 2>/dev/null || true; wait "$compositor_pid" 2>/dev/null || true; fi
    if [[ "$status" != 0 ]]; then
        for log in "$state/x11.log" "$state/wayland.log" "$state/weston.log"; do [[ ! -f "$log" ]] || tail -100 "$log" >&2; done
    fi
    printf 'Desktop check logs: %s\n' "$state"
}
trap cleanup EXIT
export XDG_DATA_HOME="$state/data" XDG_CACHE_HOME="$state/cache" XDG_STATE_HOME="$state/state"
export XDG_RUNTIME_DIR="$state/runtime" LIBGL_ALWAYS_SOFTWARE=1 OSU_SDL3=1
mkdir -m 700 "$XDG_RUNTIME_DIR"

export HSR_SMOKE_CLIENT="$client" HSR_SMOKE_STATE="$state"
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 SDL_VIDEODRIVER=x11 xvfb-run -a -s '-screen 0 1280x720x24' bash -c '
    openbox > "$HSR_SMOKE_STATE/openbox.log" 2>&1 & wm=$!
    trap "kill $wm 2>/dev/null || true" EXIT
    timeout 90s dotnet "$HSR_SMOKE_CLIENT" --verify-research-host > "$HSR_SMOKE_STATE/x11.log" 2>&1
    grep -q "native_research_host=loaded;focus=true;session=x11" "$HSR_SMOKE_STATE/x11.log"
'

weston --backend=headless-backend.so --renderer=pixman --socket=hsr-smoke-wayland --idle-time=0 > "$state/weston.log" 2>&1 &
compositor_pid=$!
for _ in $(seq 1 100); do
    [[ -S "$XDG_RUNTIME_DIR/hsr-smoke-wayland" ]] && break
    kill -0 "$compositor_pid" 2>/dev/null || { cat "$state/weston.log" >&2; exit 1; }
    sleep 0.1
done
[[ -S "$XDG_RUNTIME_DIR/hsr-smoke-wayland" ]] || { cat "$state/weston.log" >&2; exit 1; }
env -u DISPLAY XDG_SESSION_TYPE=wayland WAYLAND_DISPLAY=hsr-smoke-wayland SDL_VIDEODRIVER=wayland \
    timeout 90s dotnet "$client" --verify-research-host > "$state/wayland.log" 2>&1
grep -q 'native_research_host=loaded;focus=true;session=wayland' "$state/wayland.log"
printf '%s\n' 'Native X11 and Wayland research client checks passed.'
