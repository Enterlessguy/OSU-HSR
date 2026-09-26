# Research isolation boundary

This fork is for offline, visibly synthetic runs against its own local build.

- `Human Simulator Research` is unranked and incompatible with built-in
  automation mods.
- Selecting it prevents both score-token retrieval and score submission inside
  the player, before any network submission request can be created.
- Gameplay requires a one-time local named-pipe token from `HumanSim.Runner`.
- The runner launches the client itself and checks the executable, process,
  beatmap, mods, window, DPI, focus, and playfield transform.
- Focus loss, movement/resizing, DPI change, process exit, malformed trace, or
  pipe/hash mismatch aborts and releases held keys.
- Traces carry a permanent `synthetic: true` marker. The runner rejects traces
  without it.

The runner uses ordinary Windows `SendInput`, or unprivileged X11 XTest on
Linux. Linux dispatch requires an X11 session and rejects Wayland/XWayland; it
does not open `/dev/input` or request elevated access. The Linux window guard
uses `xdotool` to check the launched process, visible window, focus and geometry.
There is no driver, injection, process-memory access, or support for attaching
to a production osu! client. It is not an anti-cheat bypass and must not be
changed into one. Windows is the verified dispatch platform; live Linux
gameplay remains unverified.
