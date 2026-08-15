# OSU Human Simulator - Project Handoff

Created 2026-08-03. Read this first in any new session, then continue from the
"Open items" section. Workspace: `C:\Users\mani1\Documents\OSU map  human simulator`.

## What this is

An offline osu!lazer-based human-movement simulator (research build). It parses
beatmaps through osu!'s own decoder, generates human-like input traces, and
executes them with an external Windows macro. It is strictly research-only:
unranked, no login, no score submission, traces permanently marked synthetic.
Pinned to upstream lazer tag `2026.726.0-lazer`, branch
`codex/offline-human-simulator`.

## Architecture (process boundary)

- `research\HumanSim.MapExporter` (C#): `.osu` -> schema-v1 MapPlan (gzip
  NDJSON), through lazer's own `FlatWorkingBeatmap`/`GetPlayableBeatmap`.
- `research\human-sim` (Python 3.12, editable venv): planner generates traces
  (500/1000 Hz), plus corpus/modeling/validation/evaluate tooling.
- `research\HumanSim.Runner` (C#): guarded macro; launches the client, resolves
  the selected beatmap from lazer storage by SHA-256, plans, acks, dispatches
  SendInput (absolute mouse + keys) at up to 1000 Hz.
- `osu.Game.Rulesets.Osu\Mods\OsuModHumanSimulatorResearch.cs` (HSR mod):
  named-pipe handshake with the runner; gates gameplay start; 50 ms heartbeats;
  blocks score submission; incompatible with autoplay/relax/autopilot/spunout.
- `research\HumanSim.ReplayExtractor` (C#): decodes `.osr` replays to NDJSON.
- Desktop shortcut "OSU Human Simulator" -> `research\run-dev-build.ps1`
  (prompts skill/effort with recommended effort, rebuilds everything, runs
  auto-run; window stays open).

## Current capabilities (all working)

- Perfect performer (criterion 1): dispatch p95 ~10-600 µs, 1000 Hz cursor
  cadence (clamped to trace rate), `timeBeginPeriod(1)`, High process priority,
  drift-correcting fitted clock model, spin-tail waits, 50 ms heartbeats.
  Client runs with `FrameSync = Unlimited` (framework.ini).
- Auto parsing: song-select pre-plan over a second named pipe; ultra-dense maps
  automatically plan at 1000 Hz (pre-cache density detection); per-rate/mod/
  skill cache keys; multi-map persistent runner (waits between maps, keeps the
  pipe open after `complete` until the client disconnects).
- Mods: HD, HR, DT (1.01-2.00x via speed slider), HT, FL. Clock-rate included
  in cache keys (fixed the 2x handshake rejection).
- Skill/effort (criterion 2, Phase 1): `--skill 0-100`, `--effort 0-100`.
  Skill drives ALL parameters via interpolated anchors (v4 nerf: aim sigma
  16->0 px and timing 34->0 ms across 0..100); effort drives consistency
  wandering, aim-lapse probability
  `(1-skill)^2*(1-effort)*(0.02+0.06*strain)`, and inflates aim/timing sigma.
  Overlapping presses keep the cursor on-path and miss naturally.
- Timing/aim sync (`timing-sync-v2.11`, Phase 2): one shared pressure
  state plus short stress episodes drive both aim and timing, so dense
  patterns degrade both systems together. Context-aware timing scales
  (streams/bursts looser, sliders/jumps tighter), per-run early/late bias +
  session drift, heavy-tailed mistaps (50s and timing misses, concentrated in
  streams), and ghost presses (press fired before the cursor could arrive) are
  tracked separately so timing-induced misses are not misattributed to aim.
  Timing is wired into the strength function (0.25 weight). The miss mix at
  low/mid skill moved from 100% aim to ~12-39% timing-attributable while the
  accuracy ladder is preserved. See
  `research\human-sim\output\phase2-timing-analysis.md`. Idle wandering
  (v2.2, reworked v2.3): when a long gap (>= ~350 ms) exists - map breaks,
  cutscenes, pre-roll - the cursor doodles (12% figure-8, 12% circle, ~76%
  loose random 3-sine paths) with smooth time-varying speed, a slow centre
  drift, and hand-tremor jitter. The doodle anchors near the next key and
  converges onto it via a (1-u)^k envelope, so the return distance at the
  approach start is ~0 - mathematically eliminating wander-induced
  late-arrival misses (verified: 0 misses after gap >= 900 ms on Haiboku
  50/80 vs 2 before). Perfect baseline never wanders.
  Timing error shape (v2.4): mistaps are now an OK-biased 3-band mixture
  (~65% OK, ~25% MEH, ~10% miss), the miss probability falls quadratically
  with skill and with (1-effort)^1.5, the AR tap is capped tighter (74% of
  MEH), and aim landings are ~8% tighter, so a 90%+ run produces a realistic
  judgement ladder: on Haiboku Hard at 50/80 the 5-seed average is
  575x300 / 29x100 / 3x50 / 2 miss (seed 42: 0 misses).
  Aim landing (v2.5): landings are now an elliptical Gaussian anchored just
  short of the circle centre along the approach axis - overwhelming bias for
  the centre neighbourhood, entry-side "beginning" and middle most likely,
  far side and edges unlikely - with a real-player undershoot bias (measured
  ~68% undershoot vs 32% overshoot at 50/80). Plan misses on Haiboku 50/80:
  2-3; on Tear Rain Insane 50/80: 4 average (0.4-0.6%), all well inside the
  old entry-edge placement's margin. Lapse base cut again (0.002) so 90%+
  runs keep near-zero misses; hard-map misses stay driven by strain/speed.
- Motion (v1.5 `jerk-tuned-v1.5`): continuous slow wander + persistent curve
  state replace per-transition random noise, so paths stay nearly straight
  with subtle curves instead of per-frame squiggles. Per-skill distance-scaled
  speed ceiling with per-move randomness; hard per-step trace guard makes
  flicks impossible (display-scale max ~2.4k px/s at skill 20). Spinner RPM is
  skill-dependent (120 at skill 0 -> 430 at 99.5). Deterministic octant/habit
  bias removed: landings are entry-side-biased with speed-coupled spread.
- Strength function (overall evaluation): `str(t) = 1 - weakness(t)` with
  weights aim .30, timing .25, randomness .15, correlation .10, curvature .10,
  fatigue .10. Implemented so far: aim + fatigue. Reported per plan as
  `strength`/`aim`/`timing` stats.
- Recommended effort in the launcher:
  `e = clamp(1 - 0.08/(1-skill/100)^2, 0.40, 1.00)` (keeps lapse rate near
  typical human ~0.2%/dense object).

## Key files

- `research\human-sim\src\human_sim\planner.py` - trace generation, skill/effort
  model, strength function, continuous-motion states, shared pressure/timing
  state, overlap/speed guards. `PLANNER_VERSION` at top (currently
  `timing-sync-v2.11`); bump it when generation changes and mirror in the
  runner's `canonicalConfiguration`.
- `research\human-sim\src\human_sim\cli.py` - CLI; `--skill/--effort`;
  configuration hash (must match runner's canonical mirror byte-for-byte).
- `research\HumanSim.Runner\Program.cs` - runner, cache keys, config hash,
  dense-map detection, pre-plan dedupe, teardown grace.
- `research\HumanSim.MapExporter\Program.cs` - exporter; supported mods.
- `research\human-sim\output\phase1-aim-benchmark.md` - skill/effort benchmark.
- `research\human-sim\output\phase2-timing-analysis.md` - Phase-2 timing/aim
  sync design, real-player grounding, before/after evidence.
- `timing-tests\benchmark\` - plans, traces, replays, acceptance logs.
- `research\SECURITY_BOUNDARY.md` - isolation rules; do not change without
  explicit user approval (in-process input mode would alter it).

## Commands

```powershell
# build everything + launch (or use the desktop shortcut)
.\research\run-dev-build.ps1

# manual builds
.\research\build-research.ps1
.\.dotnet\dotnet.exe build .\research\HumanSim.Runner\HumanSim.Runner.csproj --configfile .\NuGet.Config --no-restore --no-incremental

# run
.\research\human-sim\.venv\Scripts\human-sim.exe auto-run ".\osu.Desktop\bin\Debug\net8.0\osu!.exe" --mode profile --skill 50 --effort 68
# perfect: --mode perfect; plan-only: human-sim plan <map.ndjson.gz> <out> --skill X --effort Y

# tests
Set-Location research\human-sim; .\.venv\Scripts\python.exe -m pytest tests -q
```

Runner logs: `research\human-sim\output\auto-run-*.log`. Client logs:
`%APPDATA%\osu-development\logs\*.runtime.log`. Beatmap store:
`%APPDATA%\osu-development\files\<h[0]>\<h[0:2]>\<hash>`. Replay exports:
`%APPDATA%\osu-development\exports\*.osr`.

## Gotchas

- Runner/client binaries are locked while a game session runs; rebuilds fail
  with MSB3021 until the user closes the session.
- Planner changes are live via the editable venv; runner changes need a rebuild.
- Cache validation uses the configuration hash (percentile, seed, sample rate,
  perfect flag, planner version, skill, effort) - keep Python and C# mirrors
  identical when adding fields.
- The first object gets a long settle (up to 120 ms) to absorb map-start input
  lag (EXCEED first-circle miss fix).

## Open items / next steps

1. **Pending user feedback**: live re-test of v1.5 - tokiko's Hard at
   skill 20/effort ~70 and Kawa's HEAVENLY+ at skill 50/effort 68; tune the
   jerk amplitude / speed ceiling if the motion still feels too stiff or too
   loose.
2. Phase 1 remaining criteria: randomness, correlated pattern (movement
   memory keyed by direction+distance), curvature/jitter habits, fatigue
   strength curve with variance - wire each into `STRENGTH_WEIGHTS` (timing is
   now wired; randomness/correlation/curvature still contribute 0).
3. Phase 2 corpus calibration: real-player replay fitting (collect/extract/fit
   via the osu! API) is still open - `OSU_CLIENT_ID/SECRET` are not configured
   in this environment, so skill anchors blend published community benchmarks
   with simulator measurements. A wider map corpus may refine the mistap and
   context constants tuned on the current 5-map benchmark set.
4. Optional: in-process replay-style input mode (needs user decision - changes
   the security boundary; the only way to hit offscreen objects and remove
   frame-rate sampling limits).
5. Taitoki [Warned] is structurally impossible (offscreen/stacked objects);
   ~75% is the OS-path floor. A playable generated variant exists
   ([Generated Playable]) and a "skip offscreen" option was offered.

## Recent history (most valuable fixes)

- v2.10 (2026-08-05): dense-map miss reduction after triangles Expert sk65/ef90
  showed 43 in-game misses (all aim, plan 42). Not a slider-overlap bug (0 of
  28 overlapping objects missed) - the map is ultra-dense (43% intervals
  <= 94 ms) with 210-290 px jumps, and the old 48 ms park + 2.4x movement
  floor left too little time to complete moves. Now: density-adaptive park
  (16 ms on intervals < 140 ms, else 32 ms) and a realistic ~1.9x movement
  floor (min-jerk peak is 1.875x) with the hard per-step resample clamp
  retained as the flick guard. triangles Expert: 42 -> 12 planned misses
  (92.5%); Tear Rain sk60/ef90 1; Haiboku 50/80 2; My Love DT 3. Low-skill
  OD10 crept up (Kingborn 10/40 ~33%) as a side effect.
- v2.9-fix (2026-08-05): true residual-miss root cause found. The remaining
  in-game aim misses (Tear Rain sk65/ef90: 5 misses, all aim, cursor 3.6-5.6 px
  outside) traced to a constant -14 osu px Y offset between the dispatched and
  game-visible cursor - a translation error, not the transform. The runner was
  sending window-relative playfield coordinates without adding the window's
  on-screen client origin (the window sits at ~(0, 11 px), hence dx ~ 0 and
  dy ~ -14). Fixed by adding `guard.ClientRect.Left/Top` to the dispatched
  physical coordinates before SendInput. Applies on the next launch (the
  current session holds the runner exe).
- v2.9 (2026-08-04): skill-gated miss nerf above 50% skill (quadratic collapse
  to an 8% floor, context-exempt on fast/complex sections) + 48 ms pre-press
  park for delivery lag. Tear Rain sk60/ef90: 14 in-game misses -> 0 planned.
- v2.8 (2026-08-04): aim-distribution realism after heatmap review - stronger
  along-travel elongation (1.63), visible undershoot centroid (-7.9 px, 81%),
  persistent bias state (lag-1 autocorr 0.15), tighter angular cone, rarer
  uniform-direction landings, asymmetric short-arm tail, distance contrast.
  Chasers - Lost [Hard]: 0-1 planned misses, heatmap now reads as human.
- v2.7 (2026-08-04): aim edge guard (no rim landings at high acc), figure-8-
  dominant large-range idle wander (65/20/15), and continuous flow motion
  across gaps (no park-and-shoot; 32 ms pre-press park; curved paths).
  triangles Hard: 4 in-game rim misses -> 1 planned. Motion: stillness median
  ~1 ms, wander spans up to ~320 px at human speeds.
- v2.6 (2026-08-04): fast-section miss shaping - mistaps become OK/MEH-biased
  (5% miss band), effort kills mistaps quadratically at ef 95+, aim spreads
  less on fast moves, 48 ms press park covers game-frame lag. User's My Love
  DT run: 10 in-game misses -> 4 planned at 84% acc; 300/100/50 shape matches
  real high-accuracy play (Goods and OKs overwhelm MEHs and misses).
- v2.5-fix (2026-08-04): in-game miss root cause found and fixed. The
  playfield transform sampled on the first running-clock frame is transient,
  which shifted every dispatched cursor by a constant ~12.6 osu px in Y for
  the whole run (game-judged misses: Tear Rain 18, Haiboku 16 - all aim).
  The HSR mod now re-samples the settled transform ~500 ms after gameplay
  start and sends a `transform_update`; the runner validates and adopts it
  per-frame (before the first press, which is >= 1.3 s in). Verified against
  the attached replays: 17 of 18 Tear Rain misses had the trace cursor
  inside the circle at press (fixable by the correction); 1 was genuine.
  Fast/dense-section miss shaping (v2.6): mistap miss band cut 10% -> 5%
  (OK 65% / MEH 30% / miss 5%), effort suppresses mistaps quadratically
  (ef 95+ ~none), aim speed-inflation gentler on fast moves, and the press
  parks 48 ms early (was 24) so game-frame lag cannot push near-misses out.
  Measured on the user's My Love DT run (sk50/ef95): game 10 misses ->
  plan 4 (434x300 / 123x100 / 4x50); Haiboku 50/80 and Tear Rain 50/80:
  0-2 plan misses.
  v2.7 (motion + misses): aim landings get a skill-tightening edge guard
  (max = radius x (0.90 - 0.25*skill)) so 90%+ runs never rim-land; the idle
  wander is now figure-8 dominant (65% / 20% circle / 15% random) roaming an
  80-200 px neighbourhood of the next key with time-scaled speeds; and
  transitions flow - the cursor glides continuously across each gap (only a
  32 ms pre-press park) with per-move curvature variation, eliminating the
  park-and-shoot "bullet" look. Measured: triangles Hard 4 in-game misses ->
  1 planned; My Love DT 10 -> 2; Haiboku 50/80 2; Tear Rain 1. Stillness
  median ~1 ms; wander spans 163-321 px at ~445 px/s.
  Aim distribution realism (v2.8): after an external reviewer flagged the aim
  heatmap as too round/isotropic (like a centered Gaussian), the landing cloud
  is now clearly anisotropic and human: elongation std_along/std_perp ~1.6
  (was 1.3), a visible undershoot centroid (mean -7.9 px, 81% short-of-centre
  hits vs 65% before), a persistent aim-bias state giving lag-1 autocorrelation
  ~0.15 (was ~0) so consecutive landings drift in runs, tighter angular cone,
  weaker uniform-direction mixture, an asymmetric short-arm tail, and stronger
  short/long distance contrast. Verified on Chasers - Lost [Hard] sk50/ef90.
  Skill-gated miss nerf (v2.9): above 50% skill, miss probability collapses
  roughly quadratically to an 8% floor at skill 100 - applied to aim lapses,
  the mistap miss-band (downgrades to MEH), and ghost presses (rushed taps
  become late OK/MEH catch-up hits). Very fast/complex sections (high
  strain/pressure) partially escape the nerf so overfaced maps keep misses.
  Pre-press park raised 32 -> 48 ms to cover the measured ~27 ms game-visible
  delivery lag. Verified: Tear Rain sk60/ef90 14 in-game misses -> 0 planned;
  Haiboku 50/80 0; Kingborn sk80 OD10 stays ~83% with its fast-section misses.
- v2.5 (2026-08-04): centre-anchored undershoot-biased aim landings (was
  entry-edge-leaning). Landings cluster around the circle centre with the
  entry side/middle most likely and edges unlikely; undershoot outnumbers
  overshoot ~2:1. Plan misses at 50/80 drop to ~0.3-0.6% on 90%+ maps, which
  also gives the SendInput execution path far more margin inside the circle.
- v2.4 (2026-08-04): OK-biased timing inaccuracies - mistap mixture ~65% OK /
  ~25% MEH / ~10% miss, quadratic skill + effort^1.5 miss suppression, tighter
  AR cap and aim landings. 90%+ runs now show the real-player judgement shape
  (300s dominate, oks 25-30, mehs 3-4, misses 1-2 on Haiboku 50/80). Also
  fixed the replay-analysis press matcher (global greedy instead of per-object
  nearest), which was inflating miss counts ~4x on dense sections.
- v2.3 (2026-08-04): idle-wander rework after replay diagnosis - deterministic
  doodles cut to 24%, ~76% random 3-sine paths, smooth time-varying speed
  (measured p10 12 -> p90 254 px/s), doodle anchored 35-110 px from the next
  key and converging onto it via a (1-u)^k envelope. Removes wander-induced
  late-arrival misses by construction (verified on the attached Haiboku run:
  2 big-gap misses before, 0 after).
- v2.2 (2026-08-04): idle wandering for long gaps - figure-8/circle/random
  smooth doodles with hand-tremor jitter, entering via min-jerk and ending in
  time for the real approach (perfect baseline unaffected).
- v2.1 (2026-08-04): Phase-2 timing/aim sync - shared pressure + stress
  episodes, context-aware timing, early/late bias + drift, mistaps, ghost-press
  attribution, timing wired into strength. Miss mix at low/mid skill went from
  100% aim to ~23-44% timing-attributable with the accuracy ladder preserved;
  UR at skill 99 now sits at 58-73 (top players often <100).
- v1.5: continuous wander/curve states (squiggles gone: path/disp p50 1.06,
  display reversals halved); speed-ceiling min-duration unit fix + min-jerk
  peak factor + hard trace guard (flicks impossible); chord negative-delta
  fix; slider tangent-flip fix; skill-dependent spinner RPM.
- v4: beginner nerf (aim sigma 7.5->16, timing 22->34 at skill 0; effort
  inflates both), entry-side-biased landings, distance-scaled speed ceilings,
  speed-coupled aim spread.
- SendInput mouse path reverted (SetCursorPos regressed game-visible cursor).
- Runner keeps pipe open after `complete` (end-of-map abort fix).
- Clock rate added to cache/pre-plan keys (2x DT handshake rejection).
- FL support; ultra-dense maps auto-plan at 1000 Hz before cache lookup.
- First-object long settle; overlap presses miss naturally; profile speed
  clamp; timing wired to skill (v3).
