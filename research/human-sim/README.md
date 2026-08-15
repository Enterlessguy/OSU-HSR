# Offline human-simulator research tools

This directory contains the model, trace planner, public replay collector, and
validation tools. It is intentionally separate from real-time input dispatch.
Generated files are permanently marked as synthetic.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

## Typical flow

1. Export a canonical map plan with `HumanSim.MapExporter`.
2. Generate a deterministic trace:

```powershell
human-sim plan map-plan.ndjson.gz run.trace.ndjson.gz --percentile 99.5 --seed 42
```

3. Validate it with `human-sim validate run.trace.ndjson.gz`.
4. Execute it only through `HumanSim.Runner`, which requires the research mod
   handshake from the pinned local lazer build.

The Python entrypoint also exposes `human-sim run <client> <trace>` as a thin
launcher for that separate compiled runner; Python never emits OS input itself.
Use `human-sim auto-run <client> --mode perfect` to launch without a prepared
trace. Selecting a supported map or difficulty with HSR then automatically
exports that exact lazer beatmap and generates its matching trace. The client
starts pre-parsing as soon as a map is highlighted in song select (reported over
the `HUMAN_SIM_SELECTION_PIPE` channel), so by the time you click play the
loading-screen handshake is served from the pre-planned or cached result
instead of parsing on the spot. Use `--mode profile` when moving from
infrastructure calibration to a percentile profile. The trace remains at 500 Hz
by default for key-transition precision; ultra-dense maps are automatically
promoted to 1000 Hz. The cursor cadence is capped at 1000 Hz and cannot exceed
the trace rate, and every key transition carries an exact trace position. The
runner stays alive between maps (a per-map abort such
as focus loss or a window change only ends that map, not the session), and
writes a timestamped `auto-run-*.log` beside the output traces. The runner
executes at high process priority with 1 ms timer resolution and a
drift-correcting gameplay clock model; the client heartbeats every 50 ms so the
model stays tight. The cursor is moved at a cadence that matches the effective
trace rate via absolute `SendInput` moves,
teleporting any distance in a single event with keys batched alongside. The planner reserves a short settle window before every hit
so the cursor is already on the target when the OS/game input state catches up;
this prevents fast-jump head misses. A planner version is folded into the
configuration hash so trace caches invalidate automatically when generation
changes. Supported mods are `HD`, `HR`, `DT`/custom speeds up to 2.00x, `HT`,
and `FL`, cached separately per clock rate.

## Skill, effort, aim, and timing

`plan` and `auto-run` accept `--skill 0-100` and `--effort 0-100`. Skill sets
the aim and timing ceiling, while effort controls consistency, fatigue, lapse
frequency, and stress response. The overall evaluation is the strength function
`str(t) = 1 - weakness(t)`, a weighted sum of the six criteria (aim, timing,
randomness, correlation, curvature, fatigue). Aim, timing, and fatigue are
wired into the checkpoint-two metric; randomness, correlation, and curvature
remain explicit future criteria. Each plan reports `strength`, `aim`, and
`timing` diagnostics; profile values are part of the configuration hash so
caches separate per skill and effort.

The desktop launcher (`research\run-dev-build.ps1`, shortcut "OSU Human
Simulator") prompts for both values and recommends an effort sweet spot for
the entered skill:

```
e_effort = clamp(1 - 0.08 / (1 - skill/100)^2, 0.40, 1.00)
```

The recommendation is a mathematical calibration aid rather than a claim about
the global osu! player population. Real-player corpus calibration remains open.

Before live testing, `human-sim audit-library report.json --limit 100` runs a
deterministic cross-section of the installed osu!standard library through the
official lazer exporter and machine-perfect planner. It verifies one key-down
per object, exact head timing/position, slider-tail holds, and trace continuity.

## Corpus pipeline

`collect` resumes official API replay downloads from a score-ID list. Decode
each replay with `HumanSim.ReplayExtractor` (which uses lazer's decoder), then
run `extract` with its matching MapPlan. Concatenate the Parquet shards, run
`fit`, and pass the resulting bundle to `plan --model-bundle`. `evaluate`
trains a logistic baseline and a gradient-boosted defensive comparison and
writes held-out ROC-AUC, PR-AUC, calibration error, and per-feature KS values.

The collector reads OAuth credentials only from `OSU_CLIENT_ID` and
`OSU_CLIENT_SECRET`. The replay extractor requires a private
`HUMAN_SIM_PLAYER_SALT`; raw usernames are never written to derived data.

Percentile labels describe the collected corpus, not the global osu!
population. Raw replays, credentials, trained binary models, and generated
outputs are ignored by Git.
