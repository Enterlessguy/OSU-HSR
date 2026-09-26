# Osu Realism Index V1

Specification 1.0, 2026-09-08. Status: design; calibration and independent
validation pending. Machine ID: `osu-realism-index-v1` (ORI-V1).
Model family: Path execution AI-V1 / peAI. This specification supersedes using
completion, accuracy, or phase RMSE as the headline realism benchmark.

## What the benchmark measures

ORI measures how closely synthetic cursor behaviour resembles an independently
sampled human population of the requested skill and movement context. It is a
distributional benchmark, not an exact detector of humanity. A score of 100 means
agreement within calibrated human sampling variation on measured criteria, not
proof that a trace is human. No numerical score is publishable before calibration.
All minimum sample counts and weights below are design choices, not established
physiological constants; pilot calibration must validate them before freezing V1.

Two separately labelled outputs prevent execution quality and route quality from
being confused:

1. **ORI-V1 execution:** human likeness of timing along a supplied route. Primary
   development endpoint for peAI. Condition on route geometry and task context.
2. **ORI-V1 whole movement:** cursor behaviour including route shape, spatial
   exploration and tracking. Required for any claim about overall gameplay realism.
   Timing-only AI cannot repair an implausible mathematical route.

Hit accuracy, combo, misses and completion are contextual diagnostics, never
positive rewards or multipliers. Keep failed human plays and imperfect motion.
Do not clip unrealistic outputs out of the evaluation population.

## Frozen comparison contract

Compare math-only, hybrid at validation-selected nonzero blend, and the last
released hybrid on identical maps, frozen routes, key schedules, initial states,
skill/effort settings and paired seeds. Blend zero must be an exact identity.
Never substitute straight chords for the actual planner and label that a runtime
comparison. A raw learned proposal is a separate diagnostic, not delivered output.

Math alone owns destinations, path shape, object order, free-roam destinations,
constraints and key scheduling. peAI controls monotone phase/speed only. Execute
per admissible segment with stateful joins; do not warp an entire mixed map across
intermediate key events. Preserve slider time/progress locks and spinner direction.
Unsupported or invalid inference falls back explicitly. Report model/config/route
hashes, requested and effective mode, blend, supported modes, fraction of duration
and segments actually learned, projection magnitude and every fallback reason.
Zero learned exposure cannot establish an AI improvement.

Evaluate BOTH offline planned traces and observed runtime cursor traces, clearly
separated. Log dispatch deadlines, actual timestamps, dropped/coalesced samples
and stalls where available. Runtime flicker may be a delivery problem, not a
training problem. Replay-only measurements cannot certify sub-frame behaviour.

## Human reference and skill bounds

Four reporting strata: beginner, intermediate, expert, top competitive. Require
dated, independent evidence (rank/pp history or documented assessment) and a
frozen label rule; never infer skill from this benchmark's smoothness, synthetic
skill slider, or success on the evaluated map. Rank is a proxy, not a universal
motor-skill scale. Publish label uncertainty and sensitivity to tier boundaries.
Unknown labels remain unknown and cannot count as four-tier coverage.

Match or stratify within tier by task duration, distance, angle, curvature,
object spacing, circle size, approach time, relative difficulty, mods, device if
known, client timing convention, frame-rate/bandwidth and session position.
Missing device metadata gets an explicit unknown stratum. Do not compare a slow
beginner pattern with a top player's dense stream and call the difference realism.
Require overlapping human support; unsupported conditions return N/A, not zero
and not extrapolated confidence. Publish unmatched fractions.

Movement modes: isolated circles/jumps; streams/dense transitions; slider tracking
and reversals; spinner motion; breaks/free roam; and joins between modes/objects.
Time bins for transitions: <50, 50–100, 100–200, 200–500, >500 ms. These are
reporting bins, not human limits. Preserve continuous duration as a covariate.
Break analysis includes no-motion periods: humans need not wander every break.
Treat skill selection and execution selection (math/hybrid/blend) as different axes.

Disjoint training, model-validation, human-calibration and final-test partitions.
Group by player and source replay/session; near duplicates stay together. Main
generalisation test also holds out beatmap families/sets. If strict player AND map
partitioning collapses the corpus, report the deficit instead of splitting windows
at random. Keep a separate same-map/unseen-player result if useful. Source archive
IDs and hashes prevent duplicate train/test material across downloads.

Provisional scored tier×mode cell minimum: 30 held-out human players, 20 map sets,
300 valid windows, and at least 10 players/10 sets in each duration subcell used
for claims. Human calibration needs independent comparable coverage. These are
floors, not a power guarantee. Use a pilot power/precision study to increase them
as needed. Smaller datasets produce exploratory diagnostics only. Balance by
player, then map/session; a prolific player must not dominate thousands of windows.

## Measurement pipeline

Store original .osr and exact matching .osu, content hashes, source/version/license,
provenance, assistance-mod flags, pseudonymous player ID, dated skill evidence and
sampling diagnostics. All local mod-generated runs are forbidden human references.
Do not equate an archive license or matching username with proven human execution.
Document the confidence and exclusion rule. Quarantine uncertain automation,
malformed timing, unmatched maps and duplicate scores; publish counts/reasons.

Decode variable frame deltas and special records correctly; normalize coordinates
to osu playfield units and time to seconds, including speed-mod transformations
exactly once. Validate parser against known official-extractor fixtures. Define
segmentation once using map events, with windows spanning joins. Selection must
not depend on whether the synthetic output looks good.

Maintain a native-sampling artifact channel and a bandwidth-matched kinematic
channel. Never differentiate raw irregular samples repeatedly. Freeze resampling,
low-pass and derivative estimators on calibration only; apply identical processing
to humans and synthetics. Mask long gaps and filter edges. Do not interpolate sparse
human frames into fictional high-frequency ground truth. Report effective bandwidth,
valid duration and estimator sensitivity. Short windows lacking sufficient samples
are unscorable for jerk/spectrum; report coverage, not a fabricated value.

Use duration- and path-length-normalised descriptors where appropriate, alongside
absolute units. Handle stationary/near-zero-length windows separately; no division
by zero. Never smooth away jumps in the artifact channel. Arc-length projection
must be ordered and mode-aware: nearest-point projection alone is ambiguous at
self-intersections. Preserve pauses and distinguish backtracking from decode noise.

## Criteria and fixed initial weights

| Criterion | Execution weight | Whole movement weight | Measurements |
|---|---:|---:|---|
| Speed organisation | 25 | 20 | Phase increments, peak timing, speed quantiles, acceleration/deceleration asymmetry, dwell fractions; full distributions, not one mean |
| Smoothness and flicker | 30 | 25 | Bandwidth-controlled dimensionless jerk and spectral smoothness where valid; raw displacement spikes, reversals, stop/restart bursts, tail event frequency and magnitude |
| Continuity at joins | 20 | 15 | Velocity/acceleration changes around object and mode boundaries, deadline snap frequency, approach/departure continuity |
| Variation and temporal structure | 15 | 15 | Within-player vs between-player variation, autocorrelation, repeated motifs, speed-profile diversity, session-consistent style; penalise both collapse and excess randomness |
| Mode-specific execution | 10 | 10 | Tracking phase lag, reversal dynamics, spinner angular-speed variation, break pause/onset/dwell timing and entry/exit timing |
| Spatial behaviour | — | 15 | Path curvature/length distribution, tracking deviations, break occupancy/excursion, distance to future target over time, destination diversity |

Compute each feature's distance to its matched human distribution, never reward
“less jerk is always better”. A perfectly smooth repeated template can be unlike
human movement. Do not set universal handedness, spinner direction or wander-radius
targets. Mode-inapplicable features have a predeclared applicability mask; missing
required data is not the same thing as inapplicability.

Use robust Wasserstein-1 distances for scalar descriptors (scale using independent
human calibration IQR, with a frozen resolution floor), plus multivariate MMD for
joint relationships after calibration-only scaling. Kernel and bandwidth selection
are frozen before test. Keep dimensionality modest and disclose redundant features.
MMD is a distribution comparison method, not a probability that a trace is human.
Reference: [Gretton et al., 2012](https://www.jmlr.org/papers/v13/gretton12a.html).
Spectral smoothness is motivated by
[Balasubramanian et al.](https://pubmed.ncbi.nlm.nih.gov/22180502/);
validate the chosen implementation and bandwidth for replay cursor data rather
than importing clinical thresholds.

## Calibration and scoring

For each supported feature distance D, build repeated size-matched, player/map
grouped human-vs-human comparisons in calibration. Let h95 be their 95th-percentile
distance. Build a preregistered graded corruption ladder from held-out calibration
humans: inserted snaps, deadline compression, objectwise velocity resets, exaggerated
pauses, over-smoothed templates, repeated break motifs and excess noise. Preserve
the original samples and distinguish geometry corruptions from execution ones.

Let b be the median distance of the preregistered severe corruption relevant to
that feature. Require b > h95 with stable separation, otherwise this feature has no
validated scoring scale and the required cell remains uncalibrated. Map:

    feature_score = 100 * clamp(1 - max(0, D - h95) / (b - h95), 0, 1)

This is a declared engineering scale, not a natural unit of realism. Freeze the
corruptions, seeds and b in the versioned calibration bundle. Equal-average scalar
feature scores within each criterion; give the joint MMD score half the criterion
weight and scalar scores the other half. Freeze feature lists and applicability
before test. Weighted arithmetic mean across criteria gives the cell score.

Always publish tier×mode cells, transition bins, worst supported cell and confidence
intervals alongside any aggregate. Aggregate equally across the four tiers and
predeclared modes only when ALL required cells are calibrated and sufficiently
covered. No renormalisation over missing tiers/modes. Otherwise overall = N/A and
publish a clearly named scoped circle-only or other partial result.

Flicker is also a non-compensatory gate: human-calibrated tail event rate and
severity must each meet preregistered equivalence margins in every supported
critical cell, with uncertainty. “No statistically significant difference” is
not equivalence. Define margins from calibration variability and a perceptual
pilot before test; do not choose them after seeing candidate results. A failed or
indeterminate gate blocks a realism-improvement claim regardless of mean score.

Use at least 2,000 paired, two-way player/map clustered bootstrap replicates,
preserving paired generated seeds and the declared weighting. Include calibration
uncertainty; report 95% intervals for scores and hybrid-minus-math differences.
Repeated seeds are not independent human subjects. Correct the preregistered
family of cell-level claim tests (Holm), and mark extra analyses exploratory.

## Independent perceptual validation

Randomised blinded clips: human reference, pure math and hybrid in identical
rendering, playback rate and contexts, with score/name/mod overlays hidden.
Include calibration clips and attention checks; counterbalance order. Ask human
likeness, abruptness, repetitive behaviour and confidence, not which play wins.
Recruit raters with varied osu experience; define rater/clip sample sizes from
pilot power analysis. Analyse with rater and map/player clustering. Keep this
external validation distinct from the automatic score. If raters and ORI disagree,
investigate and version the benchmark; do not silently retune V1 on its final test.

## Acceptance and reporting

An improvement claim needs: valid coverage/calibration; positive paired ORI
execution improvement with uncertainty; no critical cell realism regression beyond
predeclared margins; flicker gates passed; nonzero learned exposure; stable route,
keys and runtime integrity. Whole-movement claims additionally require whole-movement
coverage and perceptual corroboration. Integrity gates ensure the experiment is
valid; they do not add realism points. Archive all failures and abstentions.

Every report includes spec/calibration/code/model/data hashes, split manifest,
coverage and exclusions, identity/mode/fallback evidence, criterion and cell scores,
intervals, raw tail metrics, paired baseline deltas, runtime/offline distinction,
and success statistics in a separate context section. Never call endpoint phase
completion hit accuracy. Existing 191-row, four-player, unlabelled training pilot
cannot receive a certified ORI-V1 score.

## Implementation and further training order

1. Audit current launcher/trace identity and sudden flickers. Add explicit
   math-only/hybrid selection and effective-mode display. Preserve existing changes
   and the research isolation boundary. Add tests of joins, zero blend identity,
   intermediate key locks and unsupported-mode fallback before new deployment.
2. Implement versioned feature extraction, coverage/abstention report and calibration
   protocol. Validate on human-vs-human and corruption ladders before scoring models.
   Keep the final test sealed; use validation realism metrics for model selection.
3. Expand the existing o!rdr corpus incrementally using its existing index/cache;
   investigate additional lawful .osr + exact .osu sources. Record working links,
   licenses, provenance limitations, tier yield, storage and compatibility. Seek
   ordinary/failed beginner and intermediate sessions as well as expert/top plays.
   Do not use local synthetic runs, fabricate tier labels or call metadata a replay.
4. First expansion target: 50 independently evidenced players per tier and several
   maps per player, then grow until grouped split coverage/power requirements hold.
   This acquisition target alone does not satisfy the benchmark cell minima. Avoid
   downloading a whole archive when selective extraction works; retain disk headroom.
5. Train stateful, geometry-conditioned execution only. Support explicit skill and
   motion-mode conditioning only where evidence exists. Use masked unsupported
   cells; never pretend a circle pilot learned sliders/spinners/free roam. Start
   with continuous joins and short-interval execution; add modes behind their own
   validation. Multi-example/distributional learning is preferable to collapsing
   all human profiles to one average. No score-maximising objective or route outputs.
6. Track phase/velocity fit as diagnostics; choose candidates on held-out human
   likeness and flicker tails, with conservative math feasibility projection. Compare
   blends on validation; all-fallback means no demonstrated model contribution.
   Preserve Path execution AI-V1 identity with immutable candidate IDs/hashes;
   never overwrite a model under an existing benchmark result.
7. Launch a visible CMD progress window as requested, attached to the actual run log.
   Show current acquisition/decode/admission/train/validation stage, current files,
   accepted/rejected counts and reasons, skill coverage, elapsed time, actual fit
   progress/losses and output candidate. Show epochs only for an epoch-based trainer.
   Never fabricate percentages. Omit credentials and player salt. Log terminal
   success/failure and keep the window open for inspection.
8. Continue acquisition/training while useful authorised work is possible. Report
   exact blockers if a source needs credentials or required skill evidence is absent.
   Completion of a small training job is not completion of the four-tier objective.
