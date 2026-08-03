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
  model, strength function, continuous-motion states, overlap/speed guards.
  `PLANNER_VERSION` at top (currently `jerk-tuned-v1.5`); bump it when
  generation changes and mirror in the runner's `canonicalConfiguration`.
- `research\human-sim\src\human_sim\cli.py` - CLI; `--skill/--effort`;
  configuration hash (must match runner's canonical mirror byte-for-byte).
- `research\HumanSim.Runner\Program.cs` - runner, cache keys, config hash,
  dense-map detection, pre-plan dedupe, teardown grace.
- `research\HumanSim.MapExporter\Program.cs` - exporter; supported mods.
- `research\human-sim\output\phase1-aim-benchmark.md` - skill/effort benchmark.
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
   strength curve with variance - wire each into `STRENGTH_WEIGHTS`.
3. Phase 2: corpus calibration (collect/extract/fit) so skill levels map to
   realistic score distributions per map difficulty; score-mix validation loop.
4. Optional: in-process replay-style input mode (needs user decision - changes
   the security boundary; the only way to hit offscreen objects and remove
   frame-rate sampling limits).
5. Taitoki [Warned] is structurally impossible (offscreen/stacked objects);
   ~75% is the OS-path floor. A playable generated variant exists
   ([Generated Playable]) and a "skip offscreen" option was offered.

## Recent history (most valuable fixes)

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
