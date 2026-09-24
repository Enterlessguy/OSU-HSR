# Offline human-simulator research tools

This directory contains the model, trace planner, public replay collector, and
validation tools. It is intentionally separate from real-time input dispatch.
Generated files are permanently marked as synthetic.

The [movement roadmap](MOVEMENT_ROADMAP.md) covers the v2.16 short-interval and
free-roam changes, human-only replay collection across four skill groups,
the CPU movement-training pilot, evaluation, and repeatable retraining.
No existing local HSR runs are eligible for that training corpus.

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
teleporting any distance in a single event with keys batched alongside. Ordinary
circles use a rolling local waypoint horizon and persistent position, velocity,
acceleration, wander, and coloured-noise state. Quintic Hermite segments share
interior waypoint derivatives, so shallow flows carry through object boundaries
while reversals slow naturally. Explicit pre-hit parking is reserved for idle
or break time, stacked repeats, and slider-specific tracking. A planner version
is folded into the
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

## Statistical regression benchmark

Use the reusable offline benchmark for routine planner checks:

```powershell
human-sim benchmark --scope compact --output output/analysis/statistical-benchmark.json
```

`compact`, `default`, and `full` select progressively larger checked-in map
sets; `--map` can be repeated for an explicit set and `--seeds` accepts a
comma-separated deterministic seed list. The JSON report and text summary
include exact same-seed reproducibility, one-to-one object/press coverage,
cross-seed landing delta/correlation, lag-2..8 periodicity, normalized landing
radius (mean/p95/rim share), angular entropy, successful landings, timing
distribution, cursor velocity/acceleration/jerk, structured context-label
shares, spinner-only rotation/radial roughness, long-gap idle hover/distance,
and skill/effort monotonicity. Spinner and idle gates use mode-specific time
windows, so ordinary circle samples cannot hide a defective spinner or
free-roam path. Each matched landing also reports the
approach-aligned longitudinal/lateral means and spreads, covariance
anisotropy, undershoot share, skewness/kurtosis, sector entropy,
`corr(|u|,|v|)`, tail/rim rates, and the explicit axis-wedge score. These
metrics use the actual one-to-one matched press error, so a screen-space
aggregate cannot hide a local-frame arrow. The same fields are emitted as
`planned_approach_aligned` from sampled landing offsets, isolating the aim
distribution from timing/late-arrival contamination while retaining the
press-time view for replay comparison.

The broad default gates are artifact detectors rather than snapshot targets:
absolute cross-seed correlation <= 0.35 when every seed pair has at least 32
common object samples, lag correlation <= 0.35, rim share <= 0.20, and normalized angular
entropy >= 0.60 when at least 32 samples are available. The benchmark's primary
`radial` values describe planned landing
offsets for correctly matched presses; `press_radial` is retained separately
to expose cursor/timing contamination. Normal stochastic variation should not
be handled by tightening these limits.

Spinner gates require at least 98% visual counter-clockwise angular steps and
radial second-difference p95 <= 0.35 px at the configured cursor cadence.
Long-gap dwell windows require target-hover share below 45 px <= 0.30 and a
median target distance >= 60 px. Reports include the evaluated window counts;
zero evidence is visible rather than being mistaken for a measured pass.

The approach-aligned artifact gates are intentionally broad: at least 64
samples per map, undershoot share between 0.42 and 0.78, covariance
anisotropy <= 2.00, absolute-axis correlation <= 0.55, sector entropy >= 0.70,
and wedge share <= 0.42. The reference skill-50/effort-90 target is tighter
(roughly 0.55-0.68 undershoot, 1.05-1.35 anisotropy, correlation < 0.30,
and wedge < 0.30), but those values are calibration guidance rather than
brittle snapshots. Press-time gates use only maps with at least 85% success
and timing p95 <= 60 ms; excluded maps remain listed in JSON diagnostics.
The planned sampler has a separate cloud/wedge gate (undershoot 0.40-0.75,
anisotropy <= 1.75, absolute-axis correlation <= 0.40, sector entropy >=
0.75, wedge <= 0.32), so a difficult map cannot hide a directional sampler
artifact while its raw press-time metrics remain visible.

Continuity diagnostics are stratified by ordinary-circle corner angle and ignore
explicit idle breaks, slider tracking, and stacked repeats (which are allowed
to dwell or reverse). Default gates require shallow
0-45-degree flow to have severe-stop share <= 0.15 and median carry >= 0.50;
45-120-degree turns must not exceed shallow-flow carry by more than 0.05,
reversals may approach zero without a
 restart spike (default carry ratio <= 2.5 in well-sampled reversal windows),
 tangent-reset excess p95 must stay <= 90 degrees, and the base Hermite path must meet position (1e-4 px), shared
velocity (1e-5 px/s), and shared acceleration (1e-3 px/s²) tolerances. Cursor
kinematics are differentiated only after resampling to the configured cadence;
the five-sample triangular resampling filter is recorded in JSON, and p95
speed/acceleration/lateral-acceleration/jerk bounds are checked. Unfiltered
uniform-cadence velocity, velocity-vector jump, acceleration, and jerk
distributions are retained too. The maximum velocity-vector jump is gated;
raw acceleration/jerk maxima remain diagnostic because a real boundary change
divided by a sub-millisecond event interval can inflate the derivative without
creating a visible displacement. The
uniform-cadence motion report additionally records launch-delay share, transit
share, active-motion share, p50/p95/max speed, speed normalized by available
gap and speed ceiling, velocity carry, severe-stop share, and an unnecessary
flick count/share. A flick is gated only when the map gap has spare time and
the measured 10%-to-90% travel is both compressed and unusually fast; genuine
dense late-arrival misses are not relabelled as flicks. Separately, every
non-spinner transition of 2-600 ms is checked regardless of object kind. The
report measures travelled/direct path ratio, endpoint-corridor deviation,
projection outside the endpoint segment, and maximum velocity reversal. This
catches slider-tail overshoot/return yanks that the older circle-only metric
could not see. Slider-handoff coverage is gated, so eligible handoffs cannot
pass on empty evidence. Longer gaps use the intentional idle-wander model and
are excluded from the target-to-target excursion gate.

Skill and effort monotonicity runs cover every selected map and the first three
benchmark seeds by default. Paired profile changes are aggregated across seeds;
only consistent inversions fail the gate. Use `--monotonicity-seeds` to choose a
different seed subset.

Trace headers and manifests carry `planner_version`, `git_commit`, and
`build_identity`. The runner rejects missing or stale identity fields before
using a cached trace. `research/run-dev-build.ps1` prints the checkout
identity, supports an opt-in `-Update` (fetch plus `--ff-only` when clean;
otherwise it safely skips updating and launches the local work), reinstalls
the editable package, builds the client and runner, verifies Python/runner
identity, and launches only after those checks.
Use `-VerifyTrace <trace.gz>` when validating a specific existing trace; old
v2.11/v2.12 artifacts are retained as raw evidence but are not accepted as
current cache entries.

The benchmark is classified as
`planner-only/not-runtime-validated` when no runner telemetry is supplied.
That is intentional: a clean offline planner result is not evidence that a
Windows dispatch run was clean.

## Shared pattern context

Planner, replay extraction, modeling, and analysis use one structured context
implementation. Each object exposes its kind, interval, distance, approach
velocity, direction-change angle, rhythm ratio/continuation, local density and
strain history, robust geometry, clock rate, active mods, and available
difficulty metadata. The small compatibility labels remain (`stream`, `burst`,
`jump`, `transition`, `slider`, `spinner`), but dense long jumps and sharp
angle changes are not labelled `stream` solely because their intervals are
short. Missing metadata and first-object history use finite neutral defaults.

## Runtime quality gate

The runner writes its existing human-readable diagnostics plus a structured
`Runtime telemetry: {...}` record to the same log. Assess it with:

```powershell
human-sim runtime-quality output/auto-run-YYYYMMDD-HHMMSS.log
```

Default thresholds are dispatch p95 1 ms, dispatch p99 5 ms, dispatch max
100 ms, key-down p95 2.5 ms, `SendInput` p95 2.5 ms, `SendInput` max 100 ms,
deadline coalescing <= 2% of delivered frames, and heartbeat gaps <= 2 s.
Override individual values with the corresponding `--max-*`,
`--max-coalesced-fraction`, or `--max-heartbeat-gap-ms` options. A run is
`runtime-validated/clean`, `runtime-degraded`, or `runtime-invalid`; focus,
window/DPI, protocol, and heartbeat failures invalidate the run. Missing
telemetry is reported as `planner-only/not-runtime-validated`, with the raw
diagnostics preserved in the JSON result. `benchmark --runtime-telemetry`
attaches the same assessment to its machine-readable report and returns a
failure code for degraded/invalid runtime evidence.

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
