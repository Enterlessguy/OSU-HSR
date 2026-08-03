# HSR checkpoint 2 - version 1.5 (jerk/continuity tuning)

Date: 2026-08-03

Published to a new private repository (HSR-Checkpoint-2). Planner version
`jerk-tuned-v1.5`.

## What changed since checkpoint 1

- **Continuous motion** replaces per-transition random noise: a slow absolute
  wander and a persistent curve state evolve smoothly across the whole run.
  Result (tokiko's Hard, skill 20 / effort 70, display scale): path length /
  displacement p50 1.60 -> 1.06, visible direction reversals 21.5 -> 10.4 / s,
  angular speed p95 2127 -> 850 rad/s, median cursor speed 610 -> 287 px/s.
- **Flicks made impossible**: fixed the speed-ceiling units bug (distance /
  speed compared in seconds vs milliseconds - the original flick root cause),
  added the min-jerk peak factor and a hard per-step trace guard
  (ceiling x 1.35). Display-scale max speed at skill 20 is ~2.4k px/s.
- **Dense-section calm**: transition speed multiplier narrowed (0.82-1.08);
  chord detection no longer treats out-of-order presses as chords; slider
  lateral noise stays on one side of the curve at reversals; spinner RPM is
  skill-dependent (120 at skill 0 vs 300-420 before).
- **Accuracy nerf preserved** (from v4, unchanged in v1.5): aim sigma 16 px and
  timing sigma 34 ms at skill 0, effort inflates both, entry-side-biased
  landings with speed-coupled spread. Estimated acc (500 Hz, seed 42):
  Crystalia [Luminosity] sk10/ef40 16.7%, sk50/ef68 48.2%, sk99 98.7%;
  Kingborn NM sk10 21.9%, sk50 52.5%, sk99 98.2%; tokiko's Hard sk20/ef70
  61.3%; Kawa's HEAVENLY+ HR+DT sk50/ef68 19.2%.

## Validation

- 13/13 planner tests pass.
- All benchmark traces validate (status "valid", speed guard respected).
- User-reported symptoms reproduced and measured on the 2026-08-03 14:25
  tokiko's Hard replay before the fix; the replay's 9k px/s "max" was a
  pre-map startup artifact, not an in-game move.

## Open items

- Live re-test: tokiko's Hard at skill 20/effort ~70, Kawa's HEAVENLY+ at
  skill 50/effort 68.
- Phase 1 remaining criteria: randomness, correlated pattern, curvature
  habits, fatigue variance.
- Phase 2 corpus calibration; optional in-process input mode decision.
  See `HANDOFF.md`.
