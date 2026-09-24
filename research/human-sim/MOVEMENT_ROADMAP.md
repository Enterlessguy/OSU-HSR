# Movement improvement and human-only training plan

Updated 2026-09-07. Baseline: v2.15 / 88f6e86. Working candidate: v2.16.

## Decision and current status

Use a hybrid: persistent mathematical motion and reliable trace execution,
with learned, context-conditioned human movement parameters. Keep the current
planner as the control group and fallback. Train offline, generate traces before
playback, and keep neural inference out of the Windows dispatch loop.

The user reports fast flickering in very short intervals and robotic free roam
that stays too close to the next target. These reports outweigh passing artifact
checks: the checks were incomplete measures of the desired behavior.

Implemented in this working candidate:

- Unreachable ordinary-circle transitions under 50 ms continue inherited
  position/velocity/acceleration with bounded jerk instead of forcing a new
  target-facing endpoint in that interval. Misses are allowed. The 50 ms and
  jerk thresholds are engineering defaults, not measured physiological facts.
- Free-roam centres are sampled in playfield coordinates, with distance from
  the next target used as an exclusion check. Final staging varies from
  110–180 px rather than always 42 px. The return reserves more travel time.
  Fixed circles/figure-eights are less frequent; smooth irregular paths dominate.
- A regression explicitly tests an impossible 5 ms reversal. The long-gap
  regression checks distance before approach and the landing at the actual
  press, replacing the old requirement to park near the target early.
- Python and runner identities are v2.16, invalidating old cached traces.
- An executable CPU movement-training pilot, explicit corpus manifest, and
  provenance/coverage checks are available. No human-data model has been trained.

All existing local runs are excluded from training at the user's instruction.
The empty corpus is intentional. No API credential pair was configured during
this work. Test fixture training exercises software only and is not a trained
human-movement artifact.

## 1. Finish the short-interval correction

The first fix covers unreachable ordinary-circle movements under 50 ms. It does
not solve every short slider handoff, feasible fast movement, or resampling
artifact. Evaluate those separately before claiming the report is resolved.

Next implementation steps:

1. Partition transitions by effective interval: <10, 10–25, 25–50, 50–100,
   100–200, and >200 ms; also record distance, turn angle, and object kinds.
   Compare actual dispatch timestamps with planned timestamps for the same run.
2. Extend the current waypoint horizon into a trajectory feasibility check
   covering position, velocity, acceleration, and jerk across multiple objects.
   Prefer a plausible miss or cut corner when the requested path is infeasible.
3. Preserve realized derivatives across circle/slider/idle mode boundaries.
   Diagnose cases where the final speed clamp changes the delivered path away
   from the planner's inherited state. A speed cap alone permits instant reversal.
4. Suppress high-frequency noise in dense windows using correlated velocity
   perturbations. Adding independent position noise at sub-millisecond key
   events can amplify derivatives even when position errors are small.
5. Add uniform-cadence and event-frame reports, including velocity-vector jumps,
   reversal frequency, transit-time fraction, clamp frequency, and missed targets.

Acceptance: fewer pathological vector jumps and rapid reversals in paired
before/after traces, no regression in ordinary shallow flow, and visually smoother
dense patterns. Report accuracy changes rather than compensating with faster
movement. Existing impossible-event misses are not automatically failures.

## 2. Finish free-roam behavior

The v2.16 change is an incremental baseline improvement. It still uses harmonic
paths and staging; it is not a learned model of a player's break-time intention.

Next replace the fixed pattern menu with persistent, irregular intentions:
coast, broad wander, pause, small doodle, and return. Sample intention durations
and spatial choices once per event, preserve motion between them, and allow
legitimate stillness. Avoid forcing every human to wander or spin one direction.

Use map timing to decide when a return becomes necessary, but choose the roam
path from player state and playfield space. Calculate the return deadline from
the actual position and velocity, rather than from a fixed distance. A late
return should remain plausible even if it produces an imperfect landing.

Benchmark break lengths 0.6–1.5, 1.5–3, 3–8, and >8 seconds; centre and edge
targets; approach direction; and several player profiles. Report occupied area,
target-distance distribution, dwell duration, periodicity, speed, boundary
clipping, return onset, and landing outcome. Compare human distributions instead
of declaring that all target-hovering or all pauses are unnatural.

## 3. Where to obtain human material

Use replay trajectories with corresponding exact beatmaps, not gameplay video
as the primary training source. Video overlays, camera movement, compression,
and limited frame rate make cursor reconstruction and timing less reliable.
Videos remain useful for blind visual review and qualitative examples.

Source routes:

- Official osu! score replay downloads. The API documents public OAuth replay
  download endpoints; the existing `human-sim collect` uses the score-ID route.
  Collect candidate IDs from human player profiles/score histories and available
  leaderboards, then verify replay availability. Public score metadata does not
  guarantee downloadable replay data. See [official API documentation](https://osu.ppy.sh/docs/).
- Player-contributed replay exports, particularly for beginners/intermediates
  and failed attempts that leaderboard collections underrepresent. Ask players
  to export their own replays using their client's replay export function and
  provide the matching beatmap. See [osu! replay documentation](https://osu.ppy.sh/wiki/en/Gameplay/Replay).
- Competitive-player public score pages as an expert source, supplemented by
  ordinary sessions so the corpus is not only exceptional best performances.

Do not use the existing local `.parquet` or replay exports. Do not ingest unknown
bulk datasets merely because they are labelled human. Record source URL, replay
hash, beatmap hash, collection date, mods, player hash, and provenance review.
Keep raw files private unless their redistribution terms are established.

Pilot collection target (an experiment budget, not a claim of statistical sufficiency):

| Group | Players | Replays/player | Collection intent |
|---|---:|---:|---|
| Beginner | 20 | 5–10 | Easy maps, hesitation, misses, imperfect corrections |
| Intermediate | 20 | 5–10 | Mixed accuracy, streams, jumps, ordinary sessions |
| Expert | 20 | 5–10 | Dense patterns, technical maps, controlled transitions |
| Competitive | 20 | 5–10 | High difficulty plus routine human sessions |

Start with 400–800 replays across about 80 players. Expand based on missing
contexts and learning curves, not replay count alone. Population rank is a proxy,
not the simulator's skill percentile. Store dated rank/performance evidence,
input device when volunteered, map difficulty, accuracy, and reviewer rationale.
Treat skill group as provisional and evaluate robustness to that label.

Use the same export salt for the entire corpus so one player cannot acquire
different hashes and leak across splits. Assign players AND maps to one of
train/validation/test before extracting windows. A 60/20/20 allocation is a
starting point; bridges between groups require excluding a replay or assigning
the connected player/map group together. Freeze the assignment ledger.

## 4. Runnable collection and training workflow

Commands below run from the repository root. Paths containing `SCORE_ID` and
`BEATMAP_ID` are placeholders for real, reviewed data; they are not bundled data.

1. Configure `OSU_CLIENT_ID` and `OSU_CLIENT_SECRET` locally using an osu! OAuth
   application. Do not paste secrets into chat or commit them. Put reviewed
   score IDs, one per line, in `training/score-ids.txt`.

```powershell
.\research\human-sim\.venv\Scripts\human-sim.exe collect research/human-sim/training/score-ids.txt research/human-sim/training/human/raw
```

2. Obtain the matching `.osu` map from its official beatmap page or the player's
   export. Use the existing tools to decode NM replays and export canonical maps.
   Configure `HUMAN_SIM_PLAYER_SALT` locally before replay extraction.

```powershell
.\.dotnet\dotnet.exe research/HumanSim.ReplayExtractor/bin/Debug/net8.0/HumanSim.ReplayExtractor.dll --replay research/human-sim/training/human/raw/SCORE_ID.osr --map research/human-sim/training/human/maps/BEATMAP_ID.osu --output research/human-sim/training/human/decoded/SCORE_ID.ndjson
.\.dotnet\dotnet.exe research/HumanSim.MapExporter/bin/Debug/net8.0/HumanSim.MapExporter.dll --map research/human-sim/training/human/maps/BEATMAP_ID.osu --output research/human-sim/training/human/maps/BEATMAP_ID.ndjson.gz
```

3. Add reviewed records to `training/corpus.json`; copy the structure from
   `training/replay-record.example.json`. Paths resolve relative to the manifest.
   The human flag is a provenance assertion to review, not proof by itself.

```powershell
.\research\human-sim\.venv\Scripts\python.exe -m human_sim.movement_training audit research/human-sim/training/corpus.json research/human-sim/output/training/corpus-audit.json
.\research\human-sim\.venv\Scripts\python.exe -m human_sim.movement_training train research/human-sim/training/corpus.json research/human-sim/output/training/run-001 --seed 42
```

Training refuses missing coverage: each tier needs at least 100 training windows,
25 validation windows, and 25 test windows, with at least two players in each
cell. These are software minimums; use the larger collection target above for a
meaningful experiment. Audit reports missing coverage without pretending success.

The pilot resamples NM circle-to-circle windows of 20–600 ms to 32 phase points,
normalizes them by map direction/distance, and fits four smooth residual basis
coefficients per axis. It rejects sparse/non-finite frames, automation mods,
duplicate content, mismatched maps, and player/map leakage. Misses are retained.
An Extra Trees regressor learns all eight coefficients together, with training
weights balancing skill groups and players. It uses two CPU workers and existing
dependencies; no GPU or neural framework is required.

This first model is a deterministic shape-regression baseline. It does NOT yet
learn endpoint error, timing, player memory, free-roam intentions, or a complete
multimodal movement distribution. It does NOT load through `--model-bundle`
(that option is the older object-outcome model). It is deliberately not wired
into gameplay before evaluation. Source frame cadence limits recoverable motion;
resampling to 32 points does not create human high-frequency evidence.

## 5. Where and how to train the later sequence model

Run the pilot locally on CPU. Record elapsed time and peak memory with the real
corpus before renting hardware. The pilot writes an immutable candidate directory
containing `movement.joblib` and `report.json`; use a new directory for each run.

After the baseline establishes usable data, train a small probabilistic temporal
model on recent motion + upcoming geometry + skill/style context. Predict
correlated residual coefficients or short action chunks, keeping a style latent
stable across a session. Fit the human likelihood and trajectory derivatives;
do not optimize only game score. Add slider and free-roam heads only with
mode-specific demonstrations and normalization tests.

An initial neural experiment can use PyTorch on one suitable NVIDIA GPU; select
batch size and hardware from measured memory use. [Colab](https://research.google.com/colaboratory/faq.html)
and [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) offer hosted notebook
runtimes, but accelerator availability and quotas vary. Save checkpoints and
manifests outside ephemeral sessions. Do not assume free notebooks provide an
unattended continuous-training service. No paid compute has been provisioned.

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) is a useful later
reference for multimodal sequence generation, not the mandatory starting model.
Only consider its additional complexity if the smaller model's learning curves
and visual evaluation show a specific limitation.

## 6. Benchmarks and promotion

Use three separate evaluations; one good score cannot substitute for the others.

1. **Implementation/runtime:** unit tests, existing statistical benchmark,
   per-mode and interval metrics above, exact trace identity, and per-run latency
   telemetry. A mixed aborted/completed log must be separated before judging a
   successful live run; the known log aggregation issue is still open.
2. **Human resemblance:** held-out shape error versus the minimum-jerk baseline,
   distributions of speed/acceleration/jerk/curvature/corrections/dwell, temporal
   correlation, and player consistency. Stratify by tier, pattern, interval,
   distance, and device when known. The pilot's reported shape RMSE is only one
   diagnostic; average trajectories can score well while looking artificial.
3. **Blind review:** randomized human/v2.15/v2.16/hybrid clips on the same patterns,
   equal playback rate and rendering, with reviewers scoring smoothness, natural
   variation, and return behavior. Record uncertainty and disagreement.

```powershell
.\research\human-sim\.venv\Scripts\python.exe -m pytest research/human-sim/tests -q
.\research\human-sim\.venv\Scripts\human-sim.exe benchmark --scope compact --output research/human-sim/output/analysis/v216-statistical-benchmark.json
```

The existing `evaluate` utility randomly splits object rows and is NOT the
promotion gate for a movement model. The new pilot honors explicit player/map
splits. Future comparisons should bootstrap by player/map, not treat correlated
frames as independent evidence. Fix acceptance thresholds from human validation
distributions, then lock them before the final test. Include skill-specific
accuracy and miss rates; perfect aim is not the target for beginners.

Promotion requires no unexplained correctness regression, improved dense-motion
and free-roam evidence, and improvement across tiers rather than only the largest
tier. Run paired seeds/maps and ablations: math only, calibrated math, learned
shape hybrid, then sequence hybrid. Preserve every candidate and rollback target.

## 7. Continuous training without self-reinforcing artifacts

Use reviewed batch retraining, not learning from the simulator's own output.

1. Append a new human-data batch to a versioned source inventory. Never scan the
   local HSR output directory. Hash and deduplicate before admission.
2. Apply the frozen split ledger, review skill evidence, and audit coverage/drift.
   Quarantine new maps/players whose assignments would bridge existing splits.
3. Retrain a fresh candidate on old + new training data, with balanced sampling.
   For this modest tree baseline, full retraining is simpler than incremental
   updates and avoids forgetting. Neural models may fine-tune with retained old
   demonstrations after a full-retrain comparison.
4. Use validation metrics for candidate selection. The CLI currently reports
   test diagnostics too; do not repeatedly tune against those outputs. For
   production research, keep a separate sealed acceptance corpus evaluated only
   at release milestones and refresh it when it becomes a tuning target.
5. Save model hash, corpus hash, source hashes, seed, code commit, dependency lock,
   hardware, and benchmark reports. The pilot records hashes, seed, code commit,
   Python/platform and core package versions; retain a complete environment lock
   and accelerator details for a neural/cloud experiment.
6. Promote explicitly after offline and visual gates, then test guarded playback.
   Monitor mode/tier failures and collect missing human contexts. A scheduled
   training job is useful only once collection credentials, corpus, compute, and
   acceptance gates are established. None has been scheduled here.

## Next concrete inputs

An osu! API credential pair configured locally plus a reviewed score-ID cohort,
or externally collected human replay/map pairs with provenance. Beginner and
intermediate contributions are essential; a top leaderboard scrape alone does
not fulfill this project's training objective. No amount of compute removes
this data requirement.

## Verification from this implementation

- 43 Python tests pass, including human-corpus rejection, extraction, leakage,
  model output/version protection, dense reversal, and long-gap return tests.
- The v2.16 runner builds with zero warnings/errors.
- The compact benchmark ran two maps and five generation seeds. Reproducibility
  and motion gates pass. Overall result is FAIL because effort monotonicity fails
  on the small `fixture-2x` map. Kingborn's effort comparisons pass.
- Running v2.15 source from commit 88f6e86 on that fixture also fails effort
  monotonicity, at a different effort step. This is evidence of an existing
  sensitivity, not justification to suppress the failure or call v2.16 fully
  validated. Follow up with larger per-tier samples and independent random
  streams for style versus aim/timing before changing the gate.
- Evidence: `output/analysis/v216-statistical-benchmark.json`,
  `output/analysis/v216-fixture-comparison.json`, and
  `output/training/corpus-audit.json`. These are generated local artifacts.
- Final visual/live validation of the candidate is pending. The trained pilot
  is not connected to the runtime. Real human-data training has not started.
