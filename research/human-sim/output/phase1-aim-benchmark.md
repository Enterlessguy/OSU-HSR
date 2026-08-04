# Phase 1 benchmark: aim + timing structure, strength function

Date: 2026-08-03. Map: Camellia - Xeroa benchmark anchor (Kingborn NM, 2802
objects, OD 10). Percentile anchor 99.5, seed 42, 500 Hz. All traces validated.

## v1.6 (distance-tuned) results

Planner version `distance-tuned-v1.6` (2026-08-04). Short moves are now a
little easier to land and long moves a little harder, without changing the
human speed envelope:

Note: aim landing uses the v4 entry-side-biased Gaussian model (most likely on
the side of the circle the cursor enters, spread growing with speed/strain;
rare lapse for overshoots). The beta "where-to-hit" landing experiments
(v1.7-v2.0) were tried and reverted on 2026-08-04 because they made movement
erratic and misses compounding.

- Aim spread gets a smooth distance factor: ~0.94x at very short range (<=80
  px) ramping to ~1.06x on long jumps (>=320 px), with a neutral crossover
  around 160 px.
- The per-move speed ceiling gets a matching small bias (+3% on short moves,
  -7% at long range) so the ratio between Fitts' required pace and the
  available ceiling shifts the same way, and late-arrival misses are slightly
  more common on long jumps.

Measured on Crystalia [Luminosity] (OD10, seed 42, 500 Hz) vs the v1.5
baseline, segmented by jump distance (aim-miss % by band):

| skill/effort | band | v1.5 miss % | v1.6 miss % |
|---|---:|---:|---:|
| 10/40 | <=40 px | 43.9 | 33.8 |
| 10/40 | 40-80 | 47.6 | 39.8 |
| 10/40 | 120-180 | 75.0 | 72.7 |
| 10/40 | >260 | 89.4 | 90.3 |
| 20/70 | <=40 px | 23.0 | 18.0 |
| 20/70 | 40-80 | 23.3 | 20.4 |
| 20/70 | 120-180 | 59.1 | 65.9 |
| 20/70 | >260 | 85.8 | 90.3 |
| 50/68 | <=40 px | 4.3 | 4.3 |
| 50/68 | 40-80 | 1.9 | 2.9 |
| 50/68 | 120-180 | 30.7 | 22.7 |
| 50/68 | >260 | 52.2 | 60.2 |

Aggregate accuracy moved ~+0.5-2 pp at low skill (short moves dominate those
maps); the per-distance split is the intended effect. Skill 99 aim remains
miss-free; high-skill accuracy is unchanged in shape.

## v1.5 (jerk-tuned) results

Planner version `jerk-tuned-v1.5` (2026-08-03). Motion is now continuous:
a slow absolute wander plus a persistent curve state replace the old
per-transition random noise, and the low-skill speed ceiling (with the
min-jerk peak factor and hard per-step guard) makes flicks impossible. The
skill/effort accuracy profile is unchanged in shape (measured on the same
maps at 500 Hz, seed 42):

| map | skill | effort | est. acc % | aim miss % | max speed px/s |
|---|---:|---:|---:|---:|---:|
| Crystalia [Luminosity] | 10 | 40 | 16.7 | 67.4 | 3558 |
| Crystalia [Luminosity] | 20 | 70 | 28.8 | 51.8 | 3782 |
| Crystalia [Luminosity] | 50 | 68 | 48.2 | 24.5 | 4359 |
| Crystalia [Luminosity] | 99 | 40 | 98.7 | 0.0 | 10320 |
| Kingborn NM | 10 | 40 | 21.9 | 66.0 | 3527 |
| Kingborn NM | 20 | 70 | 30.5 | 54.7 | 3785 |
| Kingborn NM | 50 | 68 | 52.5 | 26.6 | 5685 |
| Kingborn NM | 99 | 40 | 98.2 | 0.1 | 16230 |
| tokiko's Hard | 20 | 70 | 61.3 | 28.3 | 3802 |
| Kawa's HEAVENLY+ (HR+DT) | 50 | 68 | 19.2 | 69.5 | 5677 |
| Kawa's HEAVENLY+ (HR+DT) | 99 | 40 | 87.9 | 3.2 | 16625 |

Motion quality (tokiko's Hard, skill 20/effort 70; 60 Hz display scale):

| metric | v4 (before) | v1.5 (after) |
|---|---:|---:|
| path/displacement ratio p50 | 1.603 | 1.057 |
| visible direction reversals / s | 21.5 | 10.4 |
| angular speed p95 (rad/s) | 2127 | 850 |
| median cursor speed px/s | 610 | 287 |
| display-scale max speed px/s | 2438 | 2461 |

The old v3 aim sweep remains below for reference.

## Skill categories vs real players (estimates)

osu! does not publish a skill-percentile table, so the mapping below blends
real-world anchors with the simulator's own measured ladder:

- Real anchors: ~26.8M registered accounts (osu! site footer); the oii+
  dataset of 134,953 players with 250+ hours fits expected PP ~ 226 x
  hours^0.488 (so ~3.3k pp at 250 h, ~6.6k pp at 1000 h); community PP
  consensus tiers (500 pp = starting out, 2k = decent, 4k = good,
  6-8k = great/top-tier, 10k+ = elite); rank ~100k was around the top 4% of
  ranked players (2021 forum data).
- Simulator ladder (this file): skill 10 ~ 61% acc on a 270 BPM Hard (OD6),
  ~17% on a 7.5* OD10; skill 50 ~ 48-52% on OD10; skill 80 ~ 75%; skill
  99 ~ 98-99%.

| mode | skill % | rough real-world equivalent |
|---|---:|---|
| beginner | 0-14 | <~500 pp; first weeks-months; fails/struggles on Hard |
| beginner+ | 15-29 | ~500-1000 pp; passes Hard, low acc on Insane |
| intermediate | 30-44 | ~1k-2k pp; comfortable Hard/Insane |
| intermediate+ | 45-59 | ~2k-4k pp; plays 6-7* with moderate acc |
| expert | 60-74 | ~4k-6k pp; solid on 7* |
| expert+ | 75-87 | ~6k-8k pp; high acc on 7-8* |
| competitive | 88-95 | ~8k-10k+ pp; top ~1-2%; near-FC on 7-8* |
| superhuman | 96-99.9 | 10k+ pp; top ~0.1%; 99%+ consistency |
| max | 100 | machine-perfect baseline (calibration, not a human) |

These are estimates: rank/pp percentiles only cover ranked players, and
skill in osu! is multi-dimensional (aim/speed/reading). The launcher presets
use the midpoints (10, 20, 35, 50, 65, 78, 90, 97, 100) and manual % entry
remains available.

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
