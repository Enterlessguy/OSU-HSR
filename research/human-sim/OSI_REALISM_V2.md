# OSI V2: human-reference movement realism benchmark (draft to freeze)

This is a scoring protocol for the research client, not a probability that a
trace is human. Do not score the final cohort until the feature code, scales,
eligibility, model identities, maps and exclusions are frozen. ORI V1 remains a
coverage/abstention specification; earlier SHES results remain separate.

## Unit of comparison and coverage

Use matched transition windows defined from beatmap events before generated
traces are inspected. Pair pure math and hybrid on the exact same frozen map,
route inputs, event/key schedule, skill/effort profile and random seed. Include
every predeclared eligible window, including hybrid fallback. Group repeated
windows by player, replay/session and map family; no player or family may span
fit, calibration, validation or final confirmation. Report movement modes and
transition intervals separately. Unsupported cells have score N/A, never a
renormalized aggregate or a free pass.

Report source sampling cadence, timestamp gaps, playfield transform, mods and
clock rate. Process human and generated traces through the same fixed effective
bandwidth. Preserve native samples for spike and gap diagnostics. Do not treat
interpolated 2 ms replay positions as new human observations. A jerk estimate
from sparse replays is estimator-specific, not a true physiological jerk limit.
For the continuity gate, also sample each generated trace at its matched human
window's original timestamps before comparing displacement and reversal tails;
report its unmodified 2 ms tails separately. Never score raw 2 ms and 16 ms
sample-to-sample jumps as if their observation cadence were identical.

## Scored human features

Each applicable feature is a *distribution* over matched windows and players,
not a target mean to optimize. Measure both tails and within-session structure.

| Criterion | Weight | Features and failure modes |
|---|---:|---|
| Speed and phase organization | 20 | movement onset, time-to-peak, peak/mean speed, early/late asymmetry, dwell fractions, speed quantiles and pauses; penalize constant-speed and identical templates |
| Spatial path and contact | 20 | route-relative lateral error, curvature, path/direct distance, undershoot/overshoot, legal-hit-region contact and miss distribution; distinguish center error from legal success |
| Continuity and flicker | 20 | native jump tails, reversal bursts, velocity/acceleration changes at contacts and mode joins, stop/restart rate, physically implausible snaps; report raw tail rate and severity as gates |
| Temporal structure and diversity | 15 | within-player autocorrelation, motif recurrence, between-player variation, session-consistent style, over-smoothing and excess independent noise |
| Mode behavior | 15 | circle/stream transitions, slider lag and reversal, spinner angular speed, break occupancy/pause/return onset; apply only to supported modes |
| Coupled motor behavior | 10 | relation of speed, curvature, error, timing, distance, angle and density; use a frozen low-dimensional joint distance so matching marginals cannot hide wrong correlations |

Balance by player and map family before aggregating. Calibrate feature distances
against independent human-versus-human comparisons and deliberately degraded
controls: deadline snaps, objectwise resets, straight constant-speed paths,
over-smoothed repetition, excess jitter and forced break wandering. A feature
without clear human/control separation is uncalibrated and gets N/A. The
calibration must include sensitivity to reasonable bandwidth choices. Freeze
weights, robust scales, corruption severities and random seeds on calibration.

Condition circle features on route distance, effective target width, approach
angle and local object density before calling a speed or correction pattern
unusual. Mouse-pointing experiments have found that distance, target size and
orientation affect movement kinematics and that late secondary accelerations
can reflect corrective submovements ([kinematic constraint study](https://www.sciencedirect.com/science/article/pii/S0167945706000704),
[rapid cursor-positioning experiments](https://pubmed.ncbi.nlm.nih.gov/8244410/)).
Treat these as reasons to test conditional descriptors, not as universal osu!
timing laws. Candidate V2 descriptors include late path correction fraction
and secondary filtered speed peaks; retain them only if calibration distinguishes
human-human variation from straight-path, snap and jitter controls. Set
minimum supported windows per mode and the handling of unsupported maps on
calibration, before looking at a final confirmation score.

## Success and integrity gates

Publish both arm scores, raw feature distances, per-mode cells, worst supported
cell, population coverage, exclusions, fallback reasons, model and code hashes,
and a paired player/map-family clustered bootstrap interval for hybrid-minus-
math. A win requires the entire predeclared eligible population, positive
lower 95% confidence bound, no critical-cell regression beyond a frozen margin,
and native-sample flicker tail equivalence. Repeat seeds are paired simulations,
not independent human subjects. Hold an untouched confirmation set until model
selection and all scoring rules are frozen.

The hybrid must show model invocation, accepted nonzero blend, measurable
delivered coordinate change on human-reference-covered windows, unchanged
event/key schedule, and valid runtime dispatch. A configured hybrid label or
training metric is insufficient. Report generated traces separately from
observed game cursor capture. Use a blinded player-view study with fixed
rendering and multiple raters as an independent realism check; do not choose
clips after seeing which arm scores better.

## 2026-09-24 source and development status

The new o!rdr version-155 source has a metadata-only split proposal with 400
train, 100 calibration, 100 validation, 100 sealed confirmation and 300 ordered
reserves. Player-name and map-set selection is provisional until public score
IDs and old-exposure checks resolve stable identities. Of the first 100
calibration and 100 validation pairs, all 200 file/hash checks passed; 31 and
43 respectively matched public score player IDs and exact map checks. This
corroborates identity and excludes known automation mods but does not certify
manual input. The strict circle extractor yielded 2,825 calibration and 4,075
validation transition windows. The old scoped SHES implementation covers
circle transitions only; V2 must add spatial contact, continuity, coupled
movement and supported slider/spinner/break cells before its status can be
frozen or promoted. The confirmation partition's replay bytes remain unopened.

The first common-bandwidth human mode inventory covers all stable-ID admitted
development maps. Calibration has 6,289 supported slider windows across 31
players/families, 40 spinner windows across 19, and 83 break windows across 22.
Validation has 10,393 slider windows across 43, 61 spinner windows across 24,
and 124 break windows across 31. Very short sliders are excluded by the
pre-existing minimum support rule (2,317 calibration and 4,014 validation).
These are coverage counts only. They do not constitute calibrated mode scores,
and the generated arms still need matching per-mode evaluation.

A draft circle-component probe now uses 31 independent calibration players,
100 player-disjoint human/human splits, four deliberately corrupted controls,
and a calibration-derived minimum of 12 supported windows per map. Nineteen
of 23 candidate circle features separated at least one corruption from the
human/human reference; four remain uncalibrated. On 43 opened validation maps,
only 38 meet that circle minimum. Both current hybrid models have positive
paired intervals for this **circle-only component**, but this is not an OSI V2
aggregate or a model-promotion result. Temporal diversity, mode distances,
matched-cadence flicker gates, independent perceptual review and the sealed
confirmation cohort remain outstanding. The older exploratory circle distance
ranks the models differently, so retain raw feature and map-level reports and
do not select a winner from one partial metric.
