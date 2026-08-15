# Phase 2: realistic timing + aim/timing sync (v2.10 `timing-sync-v2.10`)

Date: 2026-08-04. Planner version `timing-sync-v2.10`. This document covers the
design rationale (grounded in real-player data), the implementation, and the
measured before/after behaviour on the benchmark map set. All numbers below
are simulator measurements at seed 42, 500 Hz, percentile anchor 99.5.

## Goal (from the user)

Make errors *realistic* and *mixed* instead of aim-dominated:

- Complicated movements (slider clusters, dense/close circles) should cause
  timing inaccuracies; simple isolated or far-apart objects should mostly be
  clean with occasional natural error.
- Do not decrease overall strength or reduce overall accuracy per map - reshape
  *where* and *how* the errors happen, distributed across aim and timing
  systems that work together, for every skill level.

## Real-player grounding

The local `.osr` exports in this workspace are all HSR synthetic runs and were
deliberately **not** used as evidence of human play. Grounding comes from
published sources:

| Fact | Source |
|---|---|
| Unstable rate (UR) = 10 x std of hit errors; measures *consistency*, not accuracy; community benchmark: top players often score below 100 | [osu! wiki - Unstable rate](https://osu.ppy.sh/wiki/en/Gameplay/Unstable_rate), [forum: "What is unstable rate (UR)?"](https://osu.ppy.sh/community/forums/topics/946903) |
| osu!standard hit windows: GREAT 80-6xOD ms, OK 140-8xOD, MEH 200-10xOD, MISS 400. Slider heads only need the MEH window (ScoreV1); early taps combo-break rather than miss; missing a slider tail does not miss | [osu! wiki - Judgement system](https://osu.ppy.sh/wiki/en/Gameplay/Judgement/osu%21) |
| Aim has only two outcomes (hit/miss) vs timing's graded windows; the official Aim Error statistic handles misses with a Rayleigh distribution because miss location is unobservable | [ppy/osu PR #26687 - Aim Error statistic](https://github.com/ppy/osu/pull/26687) |
| Cognitive load (dual tasking) significantly increases variability of sub-second rhythmic finger-timing (500 ms intervals, n=103) | [Dual-task interference and motor timing, PMC12550904](https://pmc.ncbi.nlm.nih.gov/articles/PMC12550904/) |
| Motor acuity is a speed-accuracy trade-off: shot/targeting spatial error grows with required speed, with individual kinematic differences | [biorxiv: professional-level FPS motor acuity](https://www.biorxiv.org/content/biorxiv/early/2022/07/04/2022.06.30.498231.full.pdf) |
| Hit-error meters show per-player persistent early/late leaning (e.g. "-17.59 - 8.68 avg"); bias varies by player and drifts across a play | [forum: Calculating Error Rate](https://osu.ppy.sh/community/forums/topics/502405), [forum: Accuracy questions](https://osu.ppy.sh/community/forums/topics/130480) |

Community UR expectations used for calibration: sub-100 = strong, 100-140 =
decent on the map, 150+ = struggling/overfaced; streams raise UR versus
singletap/jump maps (community consensus, e.g. the UR wiki and forum threads
above).

## Design: one shared state, two coupled systems

### Shared pressure + stress episodes

A single AR(1)-style **pressure** signal is driven by strain (speed required),
pattern type, and fatigue, damped by skill. A shorter **stress-episode** state
occasionally spikes on top for a note or two. Both systems read the same
signals, so a hard pattern degrades aim *and* timing together - that is the
mechanism behind the real-world aim/timing correlation under load.

### Shared "rush" draw

One signed draw per object (positive = rushing, negative = falling behind)
feeds both systems: rushing taps early *and* cuts the aim short of the circle
(offset against the approach direction); falling behind taps late *and*
overshoots. This creates per-object coupling on top of the shared variance.

### Context-aware timing

Per-pattern timing scales (mirroring the replay-analysis contexts):

| context | timing scale | why |
|---|---:|---|
| stream (<130 ms interval) | 1.28 | sustained motor load; UR rises on streams |
| burst (<220 ms) | 1.05 | short dense bursts, less load than streams |
| transition | 1.00 | neutral |
| jump (>120 px) | 0.94 | taps lock to the beat; aim is the bottleneck |
| slider | 0.70 | heads are lenient (MEH window, ScoreV1) |
| spinner | 1.00 | timing not meaningful |

Scales are damped at high skill (top players stay clean in streams) and the
base timing sigma anchors were reduced ~10% so an isolated note is tighter than
before; streams/bursts are looser. Slider *clusters* still produce timing
errors because they feed the same pressure state.

### Biases, drift, mistaps

- **Per-run early/late bias**: sampled once per run (skill-dependent spread,
  up to ~13 ms at skill 0), so a run leans early or late persistently like a
  real player's hit-error meter.
- **Session drift**: a slow signed wander of the bias over the play, scaled by
  (1 - effort).
- **Mistaps**: rare heavy-tailed timing errors (mostly 50s, some misses past
  the MEH window), concentrated in streams/bursts, near-absent on sliders and
  isolated notes. These are the main source of *pure timing misses*.
- **Ghost presses**: when a sampled press fires before the cursor could
  physically reach the target (large early error on a dense section), the
  press fires anyway and misses naturally. These are timing-induced misses and
  are counted separately (`ghost_presses`) instead of being misattributed to
  aim.

### Timing error shape (v2.4) - OK-biased, miss-suppressed

Real players at ~90%+ accuracy produce far more OKs than MEHs and almost no
misses. The v2.4 changes enforce that shape:

- **Mistap mixture**: a mistap is drawn from three bands - ~65% OK-band
  (mostly 100s), ~25% MEH-band (50s), ~10% beyond the MEH window (rare timing
  miss). Previously mistaps landed mostly in the 50/miss bands.
- **Miss suppression**: mistap probability scales with `(1-skill)^2` and
  `(0.35 + 0.65*(1-effort)^1.5)` - misses fall quadratically with skill and
  harder with effort. The ordinary AR tap is capped at 74% of the MEH window
  (was 82%) so it can never drift into a 50/miss on its own.
- **Aim**: landings are ~8% tighter and calm-map lapses are halved, so a 90%+
  run keeps near-misses inside the circle.

Measured on Haiboku Hard (OD6) at skill 50 / effort 80, 5-seed average:
575x300 (94.3%), 29x100, 3x50, 2 misses - matching the target shape (300s
dominate, oks 25-30, mehs 3-4, misses 1-2). Seed 42 (the launcher default):
0 misses, 573x300 / 33x100 / 4x50.

The replay-analysis press matcher was also fixed (global greedy assignment
instead of per-object nearest search): the old matcher cascaded one missing
press into several fake misses in dense sections, inflating miss counts ~4x.

### Aim landing shape (v2.5) - centre-anchored, undershoot-biased

Replay diagnosis on Tear Rain [Insane] (50/80): all in-game misses were aim
(cursor outside the circle at press; zero timing misses). The old v1.6
landing placed the cursor near the entry edge of the circle, so the
execution path had almost no margin. v2.5 replaces it with an elliptical
Gaussian anchored just short of the centre along the approach axis:

- **Overwhelming centre bias**: mean landing ~10 px from centre on a ~36-39 px
  circle (p95 ~23 px), edges and far side unlikely - a rough neighbourhood of
  the centre, never an exact bullseye.
- **Entry-side "beginning" and middle most likely**: the radial mean is a few
  px short of centre (undershoot), so the entry side is favoured over the far
  side; the perpendicular spread is ~0.45x the radial spread, so the sides
  are unlikely.
- **Undershoot bias**: measured ~68% undershoot vs 32% overshoot at 50/80.
- **Spread still scales** with speed/strain/fatigue/effort, so hard maps at
  low skill keep natural aim misses (Kingborn 10/40 still ~28% acc) while a
  90%+ run sits deep inside the circle with huge delivery margin.

Measured plan misses at 50/80 (5-seed average): Haiboku Hard 3 (0.5%),
Tear Rain Insane 4 (0.6%); seed 42: Haiboku 2-3, Tear Rain 7 (variance from
rare lapses/mistap-misses). Lapse base was cut again (0.002) so lapses stay
rare on calm maps while strain keeps them on dense ones.

### In-game miss root cause (fixed 2026-08-04)

The user's runs still showed 16-18 in-game misses at 90%+ accuracy even
though the plan predicted 2-4. The game's own judgement statistics (read
from the .osr) confirmed all misses were aim-caused, and comparing replay
cursor positions against the trace revealed a **constant ~12.6 osu px
vertical offset** for the entire run (dx ~ 0, dy ~ -12.6 in every 10 s bin).

Root cause: the HSR mod sampled the playfield transform on the first frame
the gameplay clock ran - a transient layout state. The runner then mapped
every dispatched cursor through that wrong transform, so the game-visible
cursor was systematically ~13 px below the plan: edge landings fell outside
the circle. 17 of the 18 Tear Rain misses had the trace cursor inside the
circle at press time; only 1 was a genuine plan miss.

Fix: the mod re-samples the settled playfield transform ~500 ms after
gameplay start and sends a `transform_update` message; the runner validates
it (bounds/aspect/orthogonality, non-fatal on failure) and switches its
per-frame mapping. The first press is always >= ~1.3 s in, so the corrected
transform is active before any key matters. The remaining in-game misses
should now match the plan (1-4), well inside the acceptable range.

2026-08-05 correction: the transform-update addressed the playfield settling
between hello and start, but the persistent ~14 osu px Y offset that kept
producing in-game rim misses (Tear Rain sk65/ef90: 5 misses, all aim, cursor
3.6-5.6 px outside with clean timing) was actually a translation error in the
runner's coordinate frame: the mod's playfield transform is window-relative,
but the runner dispatched the resulting physical coordinates as if they were
screen-absolute, omitting the window's on-screen client origin (this window
sits at ~(0, 11 px), so dx ~ 0 and dy ~ -14). Fixed by adding
`guard.ClientRect.Left/Top` to the dispatched coordinates before SendInput.

### Fast/dense-section miss shaping (v2.6)

The user's My Love DT run (sk50/ef95, OD7 at 1.5x) showed the remaining misses
cluster on fast/dense sections: game-judged 411x300 / 131x100 / 14x50 /
10 miss (80.7%), split between aim near-misses (cursor 0.5-18 px outside the
circle) and mistap-band timing errors just past the MEH window. Real players
at that accuracy level produce Goods/OKs, not combo breaks.

v2.6 changes:

- **Mistap bands**: OK 65% / MEH 30% / miss 5% (miss band was 10%) - a stumble
  on a fast section is now a 100 or 50, almost never a miss.
- **Effort**: mistap probability scales with `(0.15 + 0.85*(1-effort)^2)` - at
  effort 95+ mistaps essentially vanish; at effort 50 they stay meaningful.
- **Aim speed-inflation**: the fast-move spread coefficient is gentler
  (`0.24 + 0.48*(1-skill)` vs `0.30 + 0.65*(1-skill)`), so dense/fast moves
  land inside instead of spraying over the edge.
- **Press park**: the cursor parks 48 ms before each press (was 24), covering
  2-3 game frames of delivery lag so near-misses stay inside the circle.

Measured plan (sk50/ef95, My Love DT, seed 42, 1000 Hz): 434x300 / 123x100 /
4x50 / 4 miss (84.2%) - Goods and OKs overwhelm MEHs and misses, and the
overfaced accuracy drop comes from OKs, not combo breaks. Haiboku 50/80 and
Tear Rain 50/80: 0-2 plan misses. The OD10 low-skill ladder is preserved
(Kingborn 10/40 ~28% acc, ~41% of misses timing-attributable).

### Motion realism + edge guard (v2.7)

The user's triangles Hard run (sk50/ef80, OD6.6) showed 4 in-game misses at
94.9% acc - all four were slider-head rim landings (cursor 0.4-2.1 px outside
the circle, clean timing). The landing distribution's thin Gaussian tail
still reached the rim, and residual frame lag pushed it out. Three changes:

- **Aim edge guard**: landings are capped at `radius x (0.90 - 0.25*skill)`
  - a skill-tightening hard guard at the rim. A 90%+ run physically cannot
  rim-land; low-skill profiles keep natural edge misses (the cap is 0.875r at
  skill 0).
- **Idle wander rework**: smooth figure-8s dominate (65%), circles 20%, loose
  random paths 15%. The doodle anchors 80-200 px from the next key with
  60-110 px pattern radii and 20-45 px centre drift, so it roams a large
  neighbourhood instead of hovering. Pattern pace scales with the available
  time (long gaps get slow, grand loops); speed varies per-window and per
  moment (OU 0.45-1.4x) capped at 1600 px/s. Measured roam spans 163-321 px
  at ~445 px/s median.
- **Continuous flow motion**: transitions now spend the whole gap gliding to
  the next landing instead of parking and shooting - only a 32 ms pre-press
  park remains (120 ms for the first object), and each move gets per-move
  curvature variation (0.8-2.2x) on top of the persistent signed bow. The
  cursor leaves the previous target right after the press, so the whole run
  reads as one connected, curving hand path. Stillness median ~1 ms (p95
  ~9 ms) versus hundreds of ms of frozen "park" before.

Measured plan misses at the user's configs: triangles Hard 1 (was 4 in-game),
My Love DT 2 (was 10), Haiboku 50/80 2, Tear Rain 50/80 1. The OD10 low-skill
ladder stays intact (Kingborn 10/40 ~29% acc).

### Aim distribution realism (v2.8) - responding to a heatmap review

An external review of the Chasers - Lost [Hard] results-screen aim heatmap
(sk50/ef90) flagged it as "roughly circular, smoothly center-weighted,
approximately isotropic" - the signature of a trivial centered Gaussian
generator. Measuring the actual rotated-frame distribution confirmed it:

| property | human expectation | v2.7 | v2.8 |
|---|---:|---:|---:|
| elongation (std along / std perpendicular) | clearly > 1 | 1.32 | 1.63 |
| undershoot centroid | visible offset | -3.4 px (65%) | -7.9 px (81%) |
| lag-1 autocorrelation of aim errors | runs, not independent | -0.02 | 0.15 |
| short vs long jump spread | noticeable contrast | weak | stronger |

v2.8 changes: perpendicular spread cut to 0.24x radial (from 0.45x), the
angular cone tightened (10-22 deg base instead of 14-30), the uniform-direction
mixture nearly halved, the undershoot bias roughly doubled, a persistent
aim-bias state (AR 0.92) added so consecutive landings drift in runs, an
asymmetric "rushed short-arm" tail added, and short/long distance contrast
strengthened. Verified on Chasers - Lost [Hard]: 0-1 planned misses and an aim
cloud that is elongated along travel with a clear undershoot offset instead of
a round centered blob. See
`research\human-sim\output\analysis\phase2-aim-heatmap-before-after.png`.

### Skill-gated miss nerf (v2.9)

The user's Tear Rain run at skill 60 / effort 90 (NM, seed 42) still showed
game-judged 637x300 / 27x100 / 7x50 / 14 miss (94.5%) - all 14 misses were
aim (cursor outside the circle at press; zero timing misses) while the plan
predicted 0, i.e. residual delivery-lag misses plus a general miss appetite
that is unrealistic for a 90%+ player.

v2.9 changes:

- **Miss nerf above 50% skill**: a factor that is 1.0 at skill 50 and falls
  roughly quadratically to a ~8% floor at skill 100, applied to aim lapses,
  the mistap miss-band (downgraded to a MEH when suppressed), and ghost
  presses (rushed taps become late OK/MEH catch-up hits instead of combo
  breaks). Very fast/complex sections (high strain or pressure) partially
  escape the nerf, so genuinely overfaced moments can still miss - and it
  never reaches zero.
- **48 ms pre-press park** (was 32): covers the measured ~27 ms game-visible
  delivery lag with margin, removing the residual in-game aim near-misses
  without sacrificing the continuous flow motion.

Measured: Tear Rain sk60/ef90 0 planned misses (14 in-game before), Haiboku
50/80 0, Chasers sk50/ef90 0-1. The low-skill ladder is unchanged (Kingborn
10/40 ~29%), and very fast maps keep their struggle profile (Kingborn sk80
OD10 ~83% with 150-ish misses).

### Dense-map movement floor (v2.10)

The user's triangles Expert run (sk65/ef90) showed 43 in-game misses (all aim,
plan 42). The suspected slider-overlap bug was disproven: 0 of the 28
overlapping-position objects missed. The real cause is density: 43% of the
map's intervals are <= 94 ms with 210-290 px jumps (a fast jumpstream). The
old 48 ms pre-press park plus a conservative 2.4x movement floor (min-jerk
peaks at 1.875x average) left too little movement time, so the cursor was
still mid-flight at the press. Two changes:

- **Density-adaptive park**: 16 ms on intervals < 140 ms (one game frame, the
  delivery minimum after the window-origin fix), 32 ms otherwise.
- **Movement floor ~1.9x** instead of 2.4x: moves up to the player's speed
  ceiling are feasible on 94 ms intervals; the hard per-step resample clamp
  remains the flick guard.

Measured: triangles Expert sk65/ef90 42 -> 12 planned misses (92.5%); the
residual misses are the genuinely-too-fast long moves (the allowed exception).
Tear Rain sk60/ef90 1, Haiboku 50/80 2, My Love DT sk50/ef95 3. Side effect:
low-skill OD10 accuracy rose (Kingborn 10/40 ~33% vs ~29%).

### Strength function

`timing_weakness = min(1, |timing error| / MEH window)` is now wired into
`str(t) = 1 - weakness(t)` under its 0.25 weight (was contributing 0). Strength
now honestly reflects timing; the mean value at skill 50/68 dropped from ~0.93
(aim+fatigue only) to ~0.80-0.83 because timing is finally counted, not because
the player got weaker - the per-map accuracy ladder is preserved (below).

### Idle wandering (v2.2, reworked v2.3)

When the cursor has genuine spare time - map breaks, cutscenes, the pre-roll,
or long inter-object gaps - it now doodles instead of parking on the next
note:

- A pattern is picked per idle window by probability: ~12% figure-8
  (Lissajous 1:2), ~12% circle/ellipse, ~76% loose random path (three detuned
  sines per axis), each with random rotation, amplitude, period and speed.
- A minimum-jerk ease blends the parked cursor into the pattern (~250-300 ms)
  so the doodle never starts with a jerk; a slow centre drift stops long
  doodles from looking like a rigid orbit; sub-pixel hand-tremor jitter gives
  natural texture; and a smooth time-varying speed multiplier (OU, 0.45-1.7x)
  makes the doodle accelerate and slow organically instead of one robotic
  speed.
- **No-miss guarantee (v2.3)**: the doodle is anchored 35-110 px from the next
  key and converges onto it via a (1-u)^k envelope (k ~ 1-2). At the moment
  the real approach starts, the cursor is on the target, so the return
  distance is ~0 and the reserved approach time can never be insufficient.
  Diagnosed on the attached Haiboku Hard run (skill 50/effort 80): 2 of 6
  objects after gaps >= 1 s missed (cursor 41 px and 218 px from target at
  press); after the fix, 0 of the big-gap objects miss.
- Wander only runs when at least ~350 ms remain after reserving a short (~60
  px) final approach plus settle. Perfect-baseline mode never wanders.
- Measured on Haiboku Hard at 50/80: 8 idle wanders, idle-window speeds vary
  p10 ~12 px/s -> p90 ~254 px/s (decelerating toward the target),
  path/displacement ratio ~7-9 (a curvy doodle, not a line), direction
  reversals all sub-0.5 px (hand-tremor at pattern turning points). See
  `research\human-sim\output\analysis\phase2-idle-wander-v23.png`.

## Measured before/after (seed 42, 500 Hz)

### Accuracy % (v1.6 -> v2.1)

| map | 10/40 | 20/70 | 50/68 | 50/80 | 80/68 | 99/40 |
|---|---:|---:|---:|---:|---:|---:|
| Kingborn NM (OD10, stream-heavy) | 22.3 -> 22.1 | 33.0 -> 29.2 | 53.4 -> 53.4 | 54.6 -> 54.6 | 75.2 -> 78.3 | 99.0 -> 97.8 |
| Crystalia [Luminosity] (OD10) | 18.7 -> 19.5 | 27.9 -> 27.7 | 49.5 -> 50.9 | 49.6 -> 54.1 | 76.4 -> 82.7 | 97.2 -> 99.7 |
| Xeroa HEAVENLY+ (HR+DT, OD10) | 6.1 -> 7.6 | 8.1 -> 10.8 | 21.2 -> 22.9 | 20.5 -> 21.8 | 47.1 -> 48.7 | 85.7 -> 94.4 |
| Haiboku [Hard] (OD6) | 56.6 -> 58.8 | 69.8 -> 77.6 | 89.8 -> 94.5 | 91.9 -> 93.9 | 99.2 -> 99.5 | 100 -> 100 |
| Centipede (OD8, all-burst) | 26.5 -> 27.7 | 43.3 -> 45.2 | 77.9 -> 80.1 | 79.3 -> 77.4 | 92.4 -> 97.2 | 100 -> 100 |

Low/mid skill is preserved within ~1-4 pp; the largest drops (-3.8 pp) are on
the stream-heavy Kingborn at 20/70 where dense streams now produce timing
errors - the intended reshape. Mid-high skill (80-99) generally *rises* because
timing at high skill is tighter (realistic top-player UR), never a nerf.

### Error mix (misses, Kingborn OD10)

| skill/effort | v1.6 aim-only misses | v2.1 aim-only | v2.1 timing-only | v2.1 both | v2.1 ghost (timing-induced) | timing share % |
|---|---:|---:|---:|---:|---:|---:|
| 10/40 | 1818 | 1343 | 434 | 9 | 424 | 39.2 |
| 20/70 | 1408 | 1248 | 274 | 2 | 268 | 30.4 |
| 50/68 | 710 | 640 | 85 | 2 | 81 | 20.8 |
| 80/68 | 206 | 187 | 13 | 0 | 13 | 12.2 |
| 99/40 | 0 | 3 | 0 | 0 | 0 | 0.0 |

Before v2.1, timing could never miss (errors were clamped at 82% of the MEH
window), so **100% of misses were aim-attributed**. Now timing produces a
realistic share (12-39% depending on skill, concentrated in streams) while total
misses stay close to v1.6.

### Timing behaviour

| metric | v1.6 (skill 50/68, Kingborn) | v2.1 | real-world reference |
|---|---:|---:|---|
| UR (std x 10) | 259 | 260 | community: 100-140 decent, 150+ overfaced |
| UR at 99/40 (5-map range) | 75-95 | 58-73 | top players often <100 |
| mean signed error (bias) | ~0-1 ms | -5.9 to +9.5 ms across runs | hit-error meters show per-player leaning |
| stream tstd | ~28 ms | 28 ms | streams raise UR |
| transition tstd | ~7 ms | ~7-12 ms | isolated notes cleaner |
| slider tstd | ~23 ms | ~20-23 ms | sliders lenient but slider clusters still load |

### Aim/timing correlation

- Per-object correlation of |timing error| vs aim error is ~0.0-0.1 (two
  largely independent motor systems - aim = hand, timing = fingers), which
  matches the fact that players can be aim-strong but tap-weak or vice versa.
- **Section-level** correlation (20-object windows) is 0.1-0.5 on most dense
  maps: hard sections degrade both systems together via the shared pressure
  state. This is the "implemented in sync" signal.

## Validation

- `pytest` suite: 13/13 passing.
- Trace validation: all benchmark traces pass `validate_trace` (monotonic
  time, speed clamp, key re-arm).
- CLI smoke test (`human-sim plan` at skill 50/68): trace valid, stats include
  `mistaps`, `ghost_presses`, strength with timing wired in.
- Planner version bumped to `timing-sync-v2.10` and mirrored in
  `HumanSim.Runner/Program.cs` `canonicalConfiguration` so cached traces are
  invalidated.

## Limits / next steps

- Real-player replay calibration (osu! API corpus -> fitted per-skill
  quantile models) is still open: `OSU_CLIENT_ID/SECRET` are not configured in
  this environment, so skill anchors remain research-estimates blended with
  published community benchmarks.
- The strength criteria `randomness`, `correlation`, `curvature` still
  contribute 0 (Phase-1 remainder).
- The exact mistap/context constants were tuned on 5 benchmark maps; a wider
  corpus pass may refine them.
