# Research isolation boundary

This fork is for offline, visibly synthetic runs against its own local build.

- `Human Simulator Research` is unranked and incompatible with built-in
  automation mods.
- Selecting it prevents both score-token retrieval and score submission inside
  the player, before any network submission request can be created.
- Gameplay requires a one-time local named-pipe token from `HumanSim.Runner`.
- The runner launches the client itself and checks the executable, process,
  beatmap and mods. Windows guards window position/DPI; Linux guards native
  client focus and playfield size/scale.
- Focus loss, guarded geometry changes, process exit, malformed trace, or
  pipe/hash mismatch aborts and releases held keys.
- Traces carry a permanent `synthetic: true` marker. The runner rejects traces
  without it.

The runner uses ordinary Windows `SendInput`. On Linux it writes a private
synthetic timeline, transfers its descriptor over the authenticated local pipe,
and verifies the client's consumed-frame counters. The client checks the file
SHA-256, token, map, clock rate, bounds and learned exposure before attaching
its private input handler. The payload is limited to 30 minutes / 1.8 million
frames, 64 MiB compressed and 256 MiB decompressed, and is removed after the run.
The Linux handler observes native client focus and playfield scale/size,
releases actions on abort, and rejects backward clocks. This route works with
X11 and Wayland without global input, `/dev/input` or elevated access.
There is no driver, injection, process-memory access, or support for attaching
to a production osu! client. It is not an anti-cheat bypass and must not be
changed into one. Timeline accounting and OS input latency are different
measurements; the Windows latency calibration is not a Linux result.
