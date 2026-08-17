# Statistical planner regression benchmark

The roadmap benchmark is implemented in
`src/human_sim/benchmark.py` and exposed as `human-sim benchmark`. It is an
offline planner benchmark; it does not create OS input and does not certify a
runtime dispatch run unless a runner log is explicitly passed with
`--runtime-telemetry`.

## Scopes and reproducibility

- `compact`: the small fixture plus Kingborn NM, suitable for routine local
  checks.
- `default`: Kingborn NM, Heavenly HRDT, and the generated Centipede map.
- `full`: every checked-in `*.map.ndjson.gz` under `timing-tests/benchmark`.

The default profile is skill 50 / effort 80 with seeds 42, 43, 44, 45, and 46.
Use `--seeds`, `--skill`, `--effort`, and repeated `--map` arguments to make a
smaller or targeted run. Same-seed checks compare the complete trace frame
signature, including timestamps, coordinates, and key state.

## Measurements

Each run uses rising key edges and a global one-to-one object/press assignment;
no press can be reused. The report retains matched and unmatched object rows.
It records:

- planned landing radius and separate press-time radius, with mean, p95, and
  rim share (`normalized radius > 0.8`);
- normalized angular entropy, cross-seed per-object absolute correlation and
  landing deltas, and lag-2..8 periodicity;
- structured context-label counts and shares for calibration review;
- successful landing count/rate and signed/absolute timing distribution;
- velocity, acceleration, and jerk summaries after resampling to the configured
  cursor cadence;
- approach-aligned longitudinal/lateral moments, covariance eigenvalue
  anisotropy, undershoot, skewness/excess kurtosis, sector entropy,
  `corr(|u|,|v|)`, tail/rim, and axis-wedge rates from actual matched press
  errors; the same fields from `planned_approach_aligned` sampled landing
  offsets;
- trace motion allocation: launch delay, transit and active-motion shares,
  speed p50/p95/max, speed normalized by the available object gap and profile
  ceiling, boundary carry/severe-stop rates, and spare-gap flick candidates;
- skill and effort monotonicity comparisons with broad tolerances.

The primary radial/angle fields and cross-seed landing deltas use the planner's
per-object landing offset only after that object has a correctly matched press.
This keeps the heatmap measurement about aim rather than accidentally measuring
a cursor halfway through a transition caused by a timing miss. The separate
`press_radial` field retains that contaminated view for diagnosis.

## Gates

The default artifact gates are deliberately broad:

| check | default gate |
|---|---:|
| same-seed trace | exact |
| cross-seed absolute correlation (>= 32 common samples per seed pair) | <= 0.35 |
| lag-2..8 absolute correlation | <= 0.35 |
| rim share | <= 0.20 |
| angular entropy | >= 0.60 when >= 32 angular samples |
| successful landings | >= 1 across the report |

Approach-aligned gates require at least 64 samples per map and allow
undershoot share 0.42-0.78, covariance anisotropy <= 2.00, absolute-axis
correlation <= 0.55, sector entropy >= 0.70, and wedge share <= 0.42. The
skill-50/effort-90 calibration target is tighter but remains a diagnostic
range, not a snapshot. Press-time gates apply only to maps with success >=
0.85 and timing p95 <= 60 ms; exclusions are retained in report diagnostics.
The planned sampler gate is separate (undershoot 0.40-0.75, anisotropy <=
1.75, absolute-axis correlation <= 0.40, sector entropy >= 0.75, wedge <=
0.32), ensuring timing contamination cannot hide a cone/wedge. Trace flick
share is <= 0.20 among transitions with enough samples. All kinematic
derivatives use the uniform cadence; event frames are preserved separately
for dispatch analysis.

Continuity corner gates exclude explicit breaks, slider tracking, and stacked
repeats; stacked repeats retain raw trace diagnostics because a short dwell or
near-reversal is an allowed geometry-specific behavior.

Skill and effort comparisons aggregate paired results across every selected map
and the first three benchmark seeds. Small stochastic reversals are tolerated
per pair, but a consistent multi-seed inversion fails the gate. These limits are
regression alarms, not claims about a population-level human distribution.

## Runtime evidence

The runner's final human-readable latency line is parsed for compatibility,
and current builds additionally emit a structured `Runtime telemetry` JSON
line. `human_sim.runtime_quality.assess_runtime_quality` preserves the raw
source and parsed values. Missing telemetry never becomes `clean`: offline
reports are explicitly `planner-only/not-runtime-validated`. Major latency or
coalescing thresholds produce `runtime-degraded`; focus/window/DPI, protocol,
completion, or heartbeat failures produce `runtime-invalid`.

Trace identity

Every newly generated trace header and manifest contains `planner_version`,
`git_commit`, and `build_identity`. The runner requires these fields and
rejects cache entries whose planner or checkout commit differs. Existing
v2.11/v2.12 traces remain useful raw evidence, but their missing identity is
reported by validation and they cannot be accepted as current runtime cache
entries.
