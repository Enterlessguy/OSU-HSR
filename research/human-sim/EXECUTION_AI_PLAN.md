# Execution-only AI over the mathematical planner

User-approved direction, 2026-09-07. This document supersedes any broader AI
path-generation proposals in MOVEMENT_ROADMAP.md. Implement in a separate task
using gpt-5.6-luna with max reasoning.

## Non-negotiable responsibility split

The mathematical system chooses the route, object order, waypoints, slider/spinner
geometry, free-roam destinations, allowed timing windows, and feasibility limits.
The AI learns HOW to execute the supplied route, never WHERE to go. Do not train
a target selector, navigator, route generator, screen-to-cursor agent, or score
maximizing reinforcement-learning policy.

Start with a strict path-preserving hybrid:

    mathematical route r(s), 0 <= s <= 1
    learned monotone phase law s(t)
    delivered position x(t) = r(s(t))

The AI may control speed progression, acceleration/deceleration, pauses where
allowed, and session-consistent execution style. Mathematical joins preserve
continuous state and enforce feasibility. It cannot create waypoints, change
route geometry, reorder targets, invent a free-roam destination, or retime key
presses. Keep key scheduling in the current mathematical timing system.

Use explicit mode constraints: slider tracking must respect the slider's time
and progress requirements; spinner progress must remain in its planned direction;
unsupported modes use pure math. Do not assume ordinary-circle time warping is
valid for sliders. AI inference occurs during offline trace planning, never in
the real-time Windows dispatcher.

Lateral motor-error residuals are OUT OF SCOPE for the first strict comparison.
They could be a separate bounded execution experiment later, but must not become
an implicit alternative path generator. The current movement_training.py learns
2D shape coefficients and therefore is NOT the desired execution-only model.
Keep it clearly labelled legacy experimental or replace it with migration notes;
do not silently wire those coefficients into the hybrid.

## Existing work and constraints

- Committed baseline is main at 88f6e86, planner v2.15, seven commits ahead of
  origin/main. There are uncommitted v2.16 changes to carry into this task.
- v2.16 continues inherited motion with bounded jerk for unreachable circle
  transitions under 50 ms and broadens free-roam staging/irregular patterns.
- Both planner.py and HumanSim.Runner/Program.cs now declare v2.16. Preserve
  version/cache agreement whenever generation behavior changes.
- Existing tests: 43 passed. Runner build passed. Compact statistical benchmark
  passes motion gates but fails effort monotonicity on fixture-2x. v2.15 also
  fails that fixture at a different effort step. Do not hide/relax this failure.
- Every existing local replay/run is mod-generated. The user explicitly forbids
  using ANY of them as human training data, even files called human.parquet.
- The new corpus manifest is deliberately empty; there are no configured API
  credentials and no verified human training data. Never claim training occurred
  on real humans without real evidence. Artificial fixtures are software tests.
- Preserve the research-only HSR handshake, synthetic trace labels, and submission
  isolation. Read research/SECURITY_BOUNDARY.md. No production-client integration.
- HANDOFF.md is stale; prefer this plan, MOVEMENT_ROADMAP.md, code and reports.

## Deliverable 1: separate geometry from execution

Introduce explicit route and execution contracts. A frozen route stores stable
identity, geometry, arc-length lookup, mode, duration/time window, entry/exit
state constraints and skill/effort inputs. Execution returns monotone phase/speed
controls and diagnostics, not arbitrary XY waypoints. Keep objects immutable.

Expose math-only and hybrid modes. Both must consume the EXACT SAME frozen route
and key schedule for a paired comparison; do not regenerate routes with different
random streams after enabling the AI. Separate geometry, timing, and execution
RNG streams. At zero hybrid blend, reproduce math-only output exactly. A missing,
invalid, incompatible, or out-of-distribution model falls back to math, with a
recorded reason. Use content hashes for route/model/normalization/config identity.

For each route segment, sample positive speeds or positive phase increments,
normalize subject to permitted duration, then reconstruct a smooth monotone phase
function. Avoid numerical backtracking, forced endpoint snaps and per-object
velocity resets. Solve joins across neighboring segments, not independently.
Projection to a feasible execution envelope belongs to math. Log how often and
how much constraints modify the learned proposal; excessive projection is a model
failure, not hidden evidence of success. Preserve plausible misses when the route
or deadline cannot be followed physically.

## Deliverable 2: human execution dataset

Build on explicit-manifest provenance, hashes, duplicate checks and grouped
splits in movement_training.py, but change the training labels to execution.
The corpus must include beginner, intermediate, expert and top competitive human
players. Balance players/tier rather than letting prolific expert downloads
dominate. Record dated skill evidence without equating global rank with the
simulator's numeric percentile. Keep one player hash salt across the corpus.

Start with verified NM circle transitions, then extend modes with separate tests.
Collect .osr and exact map pairs through official replay downloads and human
contributions. Do not rely solely on leaderboards: beginner/intermediate ordinary
and failed plays need deliberate coverage. The prior roadmap contains commands
and source links; verify API/auth requirements before actual downloading.

Distinguish observed human geometry from execution labels. For the strict pilot,
derive normalized cumulative arc length and a smoothed speed/phase profile along
each observed human movement. Supply geometry descriptors (length, local
curvature, turn angle), timing budget, prior execution state and tier as context;
do NOT predict the observed route coordinates. At inference these descriptors
come from the frozen mathematical route. Document this train/inference geometry
shift and evaluate it explicitly. For alignment against a mathematical reference,
use order-constrained local projection, report cross-track error, and reject or
separately classify ambiguous loops/backtracks instead of converting them into
arbitrary jumps in phase. Human correction loops cannot be represented fully by
a strict monotone path model; report this limitation and lost coverage.

Keep source cadence/gap masks. Resampling does not create high-frequency human
data. Reject automation/synthetic data regardless of manifest human flags. Keep
imperfect human executions; exclude unsupported transformations and corrupted
windows with reason counts. Split by BOTH player and map before windows, freeze
the split ledger, and prohibit duplicate replay/frame streams across splits.

## Deliverable 3: train, evaluate and register candidates

Implement an executable CPU-first execution baseline using available dependencies:
predict a compact log-speed/phase representation with joint temporal structure,
then constrained reconstruction. Compare with a constant/minimum-jerk execution
baseline. Preserve reproducibility; add a probabilistic sequence model only if
the corpus and validation show a need. A large neural network is not required
merely to call this AI.

Provide audit, extract, train, evaluate and model-inspect commands, documented
examples, and immutable candidate directories. Store dataset/model hashes,
normalization/feature versions, seed, code identity, dependency versions, coverage,
training history and per-tier metrics. Keep test results sealed during routine
candidate selection; validation is for tuning. Do not repeat the previous pilot's
habit of reporting the test set during every training iteration.

Actually train and benchmark if verified human data can be obtained. If not,
finish implementation, validate on explicitly artificial fixtures, write a precise
blocked-training report naming the missing credentials/data/coverage, and give
the exact next command once those inputs exist. Do not train on local HSR data,
fabricate a human model, purchase compute, or leave the implementation half done
just because data collection requires user input.

## Deliverable 4: paired hybrid versus pure-math benchmark

Freeze routes, key schedules, maps, skills, efforts and seeds once; execute each
with math-only and hybrid. Record frozen-route hashes to prove the geometry is
identical. Report execution differences without conflating them with route choice.

Include:

- Position-on-route error, monotone phase, route/target order, shared boundary
  position/velocity/acceleration, event-time interpolation and reproducibility.
- Speed, acceleration, jerk, velocity-vector jumps, reversal/flick frequency,
  excessive early arrival, pause duration, and constraint/fallback frequency.
- Interval buckets <10, 10–25, 25–50, 50–100, 100–200 and >200 ms; stratify by
  distance, corner angle, mode, map and tier. Include impossible tight reversals
  and slider/idle handoffs even if these initially fall back to pure math.
- Accuracy/misses and timing alongside smoothness. Smoother movement is not a
  win if it merely destroys tracking; high accuracy is not proof of realism.
- Human execution-distribution distances on held-out players/maps, persistence
  of player style, and uncertainty estimated by player/map rather than frames.
- Paired trace/plot exports and randomized math/hybrid/human visual comparisons
  with equal rendering/playback. Keep path-planning quality outside this score.

Free roam is an especially important separation: the mathematical planner still
chooses where it wanders. An execution-only AI may improve its pace but cannot
fix bad destinations or a target-hugging route. Report geometry and execution
limitations separately rather than allowing the model to invent a different path.

Do not automatically promote the first trained model. Promotion requires valid
contracts, better execution evidence across tiers, acceptable task performance,
low constraint intervention, and visual review. Preserve pure math fallback and
candidate rollback. Live runtime validation is separate from offline validation.

## Deliverable 5: repeatable retraining and final handoff

Implement reviewed batch retraining: append verified human batches, deduplicate,
apply frozen split assignments, audit coverage, train a new candidate on retained
old plus new training data, validate against the prior candidate/math baseline,
and produce a promotion recommendation. Never learn from generated hybrid output.
Start with full retraining for the compact baseline; no always-on service or cloud
spend is needed. Document local CPU usage and optional later GPU notebook workflow.

Add meaningful tests for geometry immutability, zero-blend identity, monotonic
phase, continuity, impossible targets, model rejection/fallback, data leakage,
human-only admission, skill coverage, reproducible training and paired benchmarks.
Run relevant tests and build checks. Provide an honest final inventory of what
works, what was trained on real data, measured results and remaining blockers.

Work autonomously within this scope. Do not revert other work. Changes belong in
the new task's checkout; do not edit the original task's working directory.
