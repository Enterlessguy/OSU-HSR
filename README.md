# HSR research fork (release preparation)

HSR (Human Simulator Research) is an offline research fork of osu!lazer for
generating and replaying deterministic, visibly synthetic osu!standard input
traces. This checkpoint contains the second-generation mathematical planner,
guarded Windows input runner, replay research pipeline, and cross-map
validation tooling.

> [!IMPORTANT]
> HSR is not an osu! cheat and must not be used with the production client or
> online score submission. The research build disables login and score
> submission, generated traces are permanently marked `synthetic: true`, and
> the runner only accepts its own authenticated local build.

This repository is an independent research fork and is not affiliated with,
endorsed by, or supported by ppy Pty Ltd. The upstream project is
[ppy/osu](https://github.com/ppy/osu), pinned here to tag
`2026.726.0-lazer`.

## Checkpoint status

- Mathematical planner version in this checkout: `timing-sync-v2.17-coherent-test`.
- Skill and effort profiles from 0 to 100, plus a machine-perfect diagnostic
  baseline.
- Shared timing/aim pressure, persistent bias and drift, context-conditioned
  mistaps, ghost-press handling, fatigue, idle movement, slider tracking, and
  spinner motion.
- Continuous minimum-jerk transitions with correlated residual movement,
  correction submovements, curvature state, speed ceilings, and exact key-event
  frames.
- Automatic 500 Hz planning, promoted to 1000 Hz for ultra-dense maps.
- Hash-bound map/mod/clock-rate validation and deterministic trace caching.
- 105-map library audit passing across 49,685 objects and more than 9 million
  generated frames.
- The historical checkpoint passed its Python planner suite; rerun current
  tests on the exact release commit before publishing.

The mathematical system is the deterministic baseline and safety envelope for
experimental learned movement. A gated direct-residual checkpoint has delivered
nonzero changes on 43 opened validation maps, and its adapter and runner wiring
are now present in this checkout. Its draft OSI V2 circle component improved
over math, but the full realism benchmark and sealed confirmation are
unfinished. The active desktop shortcut still targets a separate build with an
earlier guarded model. See
[`research/human-sim/models/experimental/README.md`](research/human-sim/models/experimental/README.md)
for the candidate and [`RELEASE_READINESS.md`](RELEASE_READINESS.md) for the
publication blockers and release checks.

## Architecture

```text
osu! beatmap
    |
    v
HumanSim.MapExporter ----> canonical MapPlan (gzip NDJSON)
    |
    v
Python mathematical planner ----> synthetic trace + hash-bound manifest
    |
    v
HumanSim.Runner <---- authenticated named-pipe handshake ----> HSR client mod
    |
    v
ordinary Windows SendInput, guarded by process/window/focus/DPI/map checks
```

### Components

- `osu.Game.Rulesets.Osu/Mods/OsuModHumanSimulatorResearch.cs` implements the
  visible research mod and client handshake.
- `research/HumanSim.MapExporter` exports lazer-decoded beatmaps with effective
  modded timing, geometry, slider paths, and hit windows.
- `research/human-sim` contains planning, trace validation, corpus extraction,
  model fitting, evaluation, and library-audit commands.
- `research/HumanSim.Runner` validates and dispatches traces through ordinary
  Windows input while monitoring focus, clock drift, transforms, and process
  identity.
- `research/HumanSim.ReplayExtractor` decodes local `.osr` research captures
  and hashes player identity with a private salt.

See `HANDOFF.md` for detailed implementation history and
`research/SECURITY_BOUNDARY.md` for the non-negotiable isolation boundary.

## Requirements

- Windows 10 or newer.
- PowerShell 7 recommended.
- Python 3.12 or newer.
- The repository-local .NET 8 SDK under `.dotnet` or a compatible system SDK.
- A local osu! beatmap library for automatic map selection and live research
  runs.

Do not commit local beatmaps, replay exports, credentials, raw player data,
generated traces, or trained binary models.

## Build

The build script isolates .NET and NuGet state inside the repository and enables
the compile-time research build flag:

```powershell
.\research\build-research.ps1 -Configuration Release
```

Build the research tools individually when required:

```powershell
$dotnet = ".\.dotnet\dotnet.exe"
& $dotnet build .\research\HumanSim.MapExporter\HumanSim.MapExporter.csproj --configfile .\NuGet.Config
& $dotnet build .\research\HumanSim.ReplayExtractor\HumanSim.ReplayExtractor.csproj --configfile .\NuGet.Config
& $dotnet build .\research\HumanSim.Runner\HumanSim.Runner.csproj --configfile .\NuGet.Config
```

## Python setup

```powershell
Set-Location .\research\human-sim
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

## Run

The interactive launcher builds all components, updates the Python environment,
prompts for skill and effort, then starts the guarded automatic-planning runner:

```powershell
.\research\run-dev-build.ps1
```

Alternatively, after building:

```powershell
.\research\human-sim\.venv\Scripts\human-sim.exe auto-run `
  ".\osu.Desktop\bin\Debug\net8.0\osu!.exe" `
  --mode profile --skill 50 --effort 68
```

Machine-perfect infrastructure calibration uses `--mode perfect`. It is not a
human profile.

## Test and audit

```powershell
Set-Location .\research\human-sim
.\.venv\Scripts\python.exe -m pytest tests -q

# Deterministic sample of the installed osu!standard library.
.\.venv\Scripts\human-sim.exe audit-library .\output\library-audit.json --limit 100
```

The phase-two benchmark harness is
`research/human-sim/output/analysis/phase2_analyze.py`. Its estimates are useful
for regression testing but do not replace authoritative in-client judgements or
held-out real-player validation.

## Data and credentials

Corpus collection reads `OSU_CLIENT_ID` and `OSU_CLIENT_SECRET` only from the
environment. Replay extraction requires `HUMAN_SIM_PLAYER_SALT`; raw usernames
are not written to derived datasets. Raw `.osr` files, Parquet corpora, generated
traces, logs, credentials, and fitted models are ignored by Git.

The small `human.parquet` and `synthetic.parquet` files included in local
development output are fixtures, not evidence of human realism. A real,
consented and player/map-grouped corpus remains the next calibration stage.

## Security

Read `SECURITY.md` and `research/SECURITY_BOUNDARY.md` before changing the
runner, authentication handshake, online API behavior, score submission, or
trace markers. Changes that weaken those controls are outside project scope.

## Documentation

- `HANDOFF.md` - implementation history, key files, commands, and open work.
- `research/human-sim/README.md` - planner and corpus-tool documentation.
- `research/SECURITY_BOUNDARY.md` - enforced research isolation model.
- `research/human-sim/output/phase1-aim-benchmark.md` - phase-one evidence.
- `research/human-sim/output/phase2-timing-analysis.md` - phase-two analysis.

## Licence and upstream attribution

The fork remains under the upstream MIT licence in `LICENCE`. The osu! name,
branding, and resources are subject to ppy's separate trademark and resource
terms. Do not present this fork as an official osu! build.
