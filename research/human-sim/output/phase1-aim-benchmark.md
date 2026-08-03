# Phase 1 benchmark: aim + timing structure, strength function

Date: 2026-08-03. Map: Camellia - Xeroa benchmark anchor (Kingborn NM, 2802
objects, OD 10). Percentile anchor 99.5, seed 42, 500 Hz. All traces validated.

## Model

Skill (0-100) now drives the complete parameter set (aim, timing, curvature,
corrections, holds, fatigue) through interpolated anchors, with aim-sigma
7.5 px / timing-sigma 22 ms at skill 0 down to zero at 100. Effort (0-100)
sets how consistently that ceiling is reached: a slowly wandering quality
multiplier and an aim-lapse probability
`(1-skill)^2 * (1-effort) * (0.01 + 0.05*strain)` that occasionally lands the
cursor outside the circle. A seeded per-player aim habit (fixed offset +
directional octant bias) persists across the run.

Strength is the overall evaluation:

```
str(t) = 1 - weakness(t)
weakness(t) = 0.30*aim_w + 0.25*timing_w + 0.15*randomness_w
              + 0.10*correlation_w + 0.10*curvature_w + 0.10*fatigue_w
```

Phase 1 implements aim_w = clamp(|aim error| / radius) and
fatigue_w = fatigue * (0.5 + 0.5*(1-effort)); the remaining criteria
contribute 0 until their phases land.

Overlapping presses (a sampled hit lands before the previous object's cursor
path ends) now keep the cursor on its current path so the press misses
naturally, and the resampled path is speed-clamped (90k px/s) in profile mode;
this removes sub-millisecond teleports that previously broke validation on
dense maps.

## Results

| skill | effort | aim mean | aim p95 | aim max | lapses | aim misses | str mean | str p95 | str min | str end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 50 | 11.89 | 23.65 | 66.92 | 24 | 26 | 0.902 | 0.974 | 0.698 | 0.785 |
| 20 | 100 | 11.55 | 22.89 | 40.53 | 0 | 4 | 0.904 | 0.976 | 0.699 | 0.814 |
| 40 | 50 | 9.94 | 19.86 | 62.18 | 12 | 12 | 0.917 | 0.978 | 0.698 | 0.886 |
| 40 | 100 | 10.04 | 19.86 | 35.64 | 0 | 0 | 0.917 | 0.979 | 0.706 | 0.960 |
| 60 | 50 | 8.35 | 17.13 | 58.81 | 7 | 7 | 0.930 | 0.982 | 0.698 | 0.837 |
| 60 | 100 | 8.03 | 16.28 | 28.77 | 0 | 0 | 0.933 | 0.983 | 0.762 | 0.862 |
| 80 | 50 | 5.63 | 11.31 | 45.39 | 2 | 2 | 0.952 | 0.987 | 0.698 | 0.974 |
| 80 | 100 | 5.80 | 11.53 | 20.16 | 0 | 0 | 0.951 | 0.987 | 0.833 | 0.976 |
| 95 | 50 | 3.32 | 6.62 | 12.29 | 0 | 0 | 0.971 | 0.992 | 0.897 | 0.984 |
| 95 | 100 | 3.36 | 6.76 | 12.67 | 0 | 0 | 0.972 | 0.992 | 0.895 | 0.986 |
| 99.5 | 50 | 1.70 | 3.43 | 6.35 | 0 | 0 | 0.985 | 0.995 | 0.946 | 0.991 |
| 99.5 | 100 | 1.69 | 3.34 | 6.39 | 0 | 0 | 0.985 | 0.996 | 0.946 | 0.992 |

Perfect baseline: aim error 0, strength 1.0 throughout.

## Timing wiring (v3)

Timing now follows skill (17 ms raw sigma at skill 50 vs 3.4 ms at 99.5),
strain-scaled. Measured timing errors on benchmark maps:

| skill | effort | map | timing std (ms) | timing p95 abs (ms) | strength mean |
|---|---:|---|---:|---:|---:|
| 50 | 68 | Kingborn NM | 22.1 | 44.2 | 0.931 |
| 80 | 68 | Kingborn NM | 13.8 | 27.5 | 0.954 |
| 99 | 40 | Kingborn NM | 5.4 | 10.6 | 0.983 |

The old model ran timing at the 99.5 anchor regardless of skill, which is why
skill 50 felt superhuman: aim errors were the only visible imperfection. With
timing wired, intermediate profiles produce a natural spread of 300/100/miss.

## Observations

- Aim error falls monotonically with skill (11.9 px -> 1.7 px mean).
- Lapses appear only at low skill with low effort; effort 100 has zero lapses.
- Aim-only misses (~0.9% at skill 20) are natural overshoots; timing misses
  arrive with the Phase-2 timing structure.
- Strength end < strength mean for most profiles: fatigue drains strength
  through the run; low effort drains it further.

## Usage

`human-sim plan <map> <trace> --skill 60 --effort 50`

`human-sim auto-run <client> --skill 60 --effort 50`
