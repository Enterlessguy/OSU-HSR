# Offline osu! human-movement simulator

The implementation is split at an explicit process boundary:

- `HumanSim.MapExporter` decodes and transforms `.osu` files through the pinned
  lazer projects and streams a schema-v1 gzip NDJSON MapPlan.
- `human-sim` collects public replays, extracts matched training rows, fits
  context-conditioned quantile models, creates deterministic traces, validates
  them, and runs defensive comparisons.
- `HumanSim.ReplayExtractor` decodes `.osr` files through lazer and hashes player
  identifiers before emitting normalized NDJSON.
- `HumanSim.Runner` is the guarded Windows macro.
- `OsuModHumanSimulatorResearch` supplies the visible banner and authenticated
  gameplay clock/playfield handshake while leaving normal input handling active.

Build the three research utilities with the SDK selected by the repository's
`global.json`. The normal release client does not need to be installed.

```powershell
.\research\build-research.ps1
.\.dotnet\dotnet.exe build .\research\HumanSim.MapExporter\HumanSim.MapExporter.csproj
.\.dotnet\dotnet.exe build .\research\HumanSim.ReplayExtractor\HumanSim.ReplayExtractor.csproj
.\.dotnet\dotnet.exe build .\research\HumanSim.Runner\HumanSim.Runner.csproj
```

The local client is produced at
`osu.Desktop\bin\Debug\net8.0\osu!.exe`. A minimal fixture flow is:

```powershell
.\.dotnet\dotnet.exe .\research\HumanSim.MapExporter\bin\Debug\net8.0\HumanSim.MapExporter.dll --map .\research\fixtures\basic.osu --output map.ndjson.gz --mods HD,HR,DT
.\research\human-sim\.venv\Scripts\human-sim.exe plan map.ndjson.gz trace.ndjson.gz --percentile 95 --seed 42
.\research\human-sim\.venv\Scripts\human-sim.exe run ".\osu.Desktop\bin\Debug\net8.0\osu!.exe" trace.ndjson.gz
```

After the client opens, import/select the matching map and enable the `HSR`
mod. The runner will not arm until the executable, process, map, and mods match.

For normal research use, automatic mode removes the manual MapPlan/trace step:

```powershell
.\research\human-sim\.venv\Scripts\human-sim.exe auto-run ".\osu.Desktop\bin\Debug\net8.0\osu!.exe" --mode perfect
```

Select any supported osu!standard map or difficulty and enable `HSR`. The
runner resolves the exact content-addressed beatmap reported by the client,
verifies its SHA-256, decodes it through `HumanSim.MapExporter`, generates the
trace, and only then acknowledges gameplay. While you are still in song select,
the research client additionally reports every highlighted map and the enabled
mods over a dedicated selection pipe, so the runner parses and pre-plans that
difficulty in real time before you enter gameplay; the loading-screen
handshake then completes immediately instead of waiting for parsing. Generated
traces are cached by beatmap hash, mods, and profile configuration, so
re-selecting the same difficulty is instant and only invalidated by changed
content or configuration. The runner stays alive between maps: after each
completed or aborted run it releases held keys and waits for the next map's
handshake, so you can keep playing maps in the same session. Runner output is
mirrored to `research/human-sim/output/auto-run-*.log` for diagnostics.
`--mode profile` uses the selected corpus-percentile baseline; `--mode perfect`
is the zero-error infrastructure calibration mode. Automatic runs apply an
8 ms OS input-delivery lead by default without changing canonical beatmap or
trace timestamps; override it with `--input-lead-ms` when calibrating a
different host.

Supported gameplay mods are `HD`, `HR`, `DT` (including custom speeds up to
2.00x via the in-game speed slider), `HT`, and `FL`; each combination is cached
separately by its exact clock rate. `HSR` remains incompatible with autoplay,
relax, autopilot, and spun-out.

Skill/effort customisation: `auto-run` accepts `--skill 0-100` and
`--effort 0-100` (defaults: percentile anchor, full effort). Phase 1 implements
the aim structure (skill-scaled aim error, per-player aim habit, effort-gated
aim lapses) and reports per-plan `strength`/`aim` diagnostics; see
`research/human-sim/output/phase1-aim-benchmark.md`.

## Real-time timing

During execution the runner raises the process to `High` priority, requests
1 ms Windows timer resolution (`timeBeginPeriod`), busy-spins short waits, and
tracks the client's gameplay clock with a drift-correcting linear fit over the
recent heartbeat stream (the research client heartbeats every 50 ms). These
keep dispatch lateness at single-digit microseconds. Cursor positions are
dispatched through absolute `SendInput` mouse moves (one event teleports any
distance) at a cadence matching the trace sample rate (1000 Hz by default,
clamped to the trace rate), with key events batched in the same call. For the
lowest end-to-end latency the dev client should run with `FrameSync = Unlimited` in
`%APPDATA%\osu-development\framework.ini` so input is sampled at the uncapped
render rate; the remaining OS delivery delay is compensated by
`--input-lead-ms`.

For headless latency benchmarks the runner accepts test-only flags
(`--timing-only`, `--disable-timer-resolution`, `--disable-clock-fit`); the
temporary harness, synthetic trace, and captured results live under
`timing-tests/`.

See [human-sim/README.md](human-sim/README.md) for the data workflow and
[SECURITY_BOUNDARY.md](SECURITY_BOUNDARY.md) for the enforced isolation rules.
