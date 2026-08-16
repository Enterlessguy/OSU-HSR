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
- successful landing count/rate and signed/absolute timing distribution;
- velocity, acceleration, and jerk summaries from the trace;
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
| cross-seed absolute correlation (>= 32 common samples per seed pair) | <= 0.60 |
| lag-2..8 absolute correlation | <= 0.65 |
| rim share | <= 0.20 |
| angular entropy | >= 0.60 when >= 32 angular samples |
| successful landings | >= 1 across the report |

Skill and effort comparisons allow small stochastic reversals and fail only
when more than one-third of comparisons are clearly reversed. These limits
are regression alarms, not claims about a population-level human distribution.

## Runtime evidence

The runner's final human-readable latency line is parsed for compatibility,
and current builds additionally emit a structured `Runtime telemetry` JSON
line. `human_sim.runtime_quality.assess_runtime_quality` preserves the raw
source and parsed values. Missing telemetry never becomes `clean`: offline
reports are explicitly `planner-only/not-runtime-validated`. Major latency or
coalescing thresholds produce `runtime-degraded`; focus/window/DPI, protocol,
completion, or heartbeat failures produce `runtime-invalid`.
