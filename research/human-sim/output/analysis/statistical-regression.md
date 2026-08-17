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

## v2.13 launcher and short benchmark validation (2026-08-17)

Main was fast-forwarded to `ec40d456d312626b6e1862f4a4898d328dfc0282`
(`timing-sync-v2.13`). The desktop shortcut was corrected to invoke the main
checkout's `research/run-dev-build.ps1 -Update`; it had still referenced the
temporary Codex worktree. Launcher validation also found and fixed the runner
artifact path: `HumanSim.Runner` targets `net8.0-windows`, not `net8.0`.

The launcher's complete pre-launch toolchain was then exercised directly:

- the osu! research client and MapExporter, ReplayExtractor, and Runner builds
  completed with zero warnings and zero errors using local .NET SDK 8.0.100;
- the editable Python package and test dependencies installed successfully in
  the project virtual environment;
- source, Python runtime, and Runner all reported `timing-sync-v2.13`;
- the client executable existed at the shortcut's configured icon/launch path.

Focused planner, continuity, and benchmark tests passed `23/23`. A short
technical benchmark used the exact Haiboku plan at skill 50, effort 90, 1000 Hz,
seeds 41-43, and monotonicity seed 42. The machine-readable and text reports are
written locally as `output/analysis/v213-haiboku-short-benchmark.json` and
`.txt` (generated reports remain ignored by Git).

The report passed every gate. Across 1,830 object opportunities it produced
1,825 successful matched landings (99.73%, five misses). Planned local-frame
errors had anisotropy 1.050, wedge share 0.233, absolute-axis correlation 0.073,
sector entropy 0.987, and undershoot share 0.619. Matched press-time errors had
anisotropy 1.102, wedge share 0.244, absolute-axis correlation 0.094, sector
entropy 0.985, and undershoot share 0.631. These values are consistent with a
mildly undershoot-biased elliptical cloud rather than the former approach-axis
arrow.

Motion used a mean 60.0% transit share with 93.9% active motion, p95 speed about
1,601 px/s, and zero unnecessary flicks among 149 measured transitions. Exact
base-path position, velocity, and acceleration continuity gates passed. The
maximum uniform-cadence kinematic summaries were 5,422 px/s speed, 161,558
px/s^2 acceleration p95, 68,926 px/s^2 lateral acceleration p95, and 76.36
million px/s^3 jerk p95, all within the deliberately broad regression limits.

This run is correctly classified `planner-only/not-runtime-validated`: it
validates generated behavior and the launch build/identity chain, but a user's
interactive play session is still required to collect Windows dispatch,
focus, heartbeat, and SendInput runtime evidence.

## v2.14 slider-handoff jerk postmortem (2026-08-17)

The exported Haiboku replay disproved the v2.13 benchmark result. At 53,331 ms,
58 ms before circle 203, the cursor moved 76.3 px in one 18 ms replay frame
immediately after slider 202. Its velocity reversed by 169.9 degrees, with a
535,947 px/s² finite-difference acceleration. Equivalent overshoot/return
events recurred after other slider tails. The old benchmark reported zero
flicks because `_trace_motion_summary` evaluated only circle-to-circle
transitions, while broad filtered-p95 kinematic gates hid rare impulses. The
user's criticism was therefore correct.

The root cause was a derivative handoff error. `_sync_motion_state_from_points`
computed the new slider-tail velocity from its final two samples and then used
`a_start = (v_tail - v_previous_mode) / delta_t`. The previous velocity belonged
to another motion primitive and `delta_t` was usually about 1 ms. This amplified
a normal cross-mode velocity difference into start accelerations above
1,000,000 px/s². The next quintic Hermite segment inherited that acceleration.
Its endpoint-only derivative scaling could not reduce the start derivative, so
the curve travelled away from the target before the final speed clamp folded it
back, producing the visible yank.

v2.14 estimates both derivatives from one physical tail. For samples in the
final 20 ms, with local times `t_i <= 0` and final position `p_0`, it solves the
two-axis least-squares model `p_i - p_0 = v_tail*t_i + 0.5*a_tail*t_i²`.
Two-sample tails fall back to finite-difference velocity and zero acceleration.
Non-finite values are rejected, and velocity/acceleration are bounded by the
same skill-conditioned slider and waypoint envelopes used elsewhere. A
dedicated slider-to-circle test failed on v2.13 at 500,690 px/s² and passes on
v2.14 below 100,000 px/s² with the base path inside its speed envelope.

The benchmark now retains unfiltered uniform-cadence derivative diagnostics and
gates the maximum velocity-vector jump. More importantly, every eligible
2-600 ms non-spinner transition is measured, including slider-to-circle,
slider-to-slider, and circle-to-slider. It records travelled/direct path ratio,
endpoint-corridor deviation, projection beyond the endpoint segment, maximum
direction reversal, yank count/share, and explicit slider-handoff coverage.
The 600 ms ceiling separates target movement from intentional idle doodles.

| Metric | v2.13 | v2.14 |
| --- | ---: | ---: |
| evaluated ordinary transitions | 602 | 602 |
| slider-handoff coverage | 100% | 100% |
| detected yanks | 81 (13.455%) | 0 (0.000%) |
| path-length ratio p95 | 4.406 | 1.209 |
| corridor-deviation ratio p95 | 1.773 | 0.243 |
| maximum projection excursion ratio | 8.752 | 0.486 |

The revised one-map technical benchmark passes all gates. It evaluates all 602
ordinary transitions and every eligible slider handoff, reports zero
unnecessary flicks and zero yanks, a maximum raw velocity-vector jump of
6,092 px/s, and a filtered speed maximum of 5,421 px/s. This remains
planner-only evidence; the old replay cannot be repaired and a new interactive
replay is required to verify runtime dispatch and subjective cursor feel.
