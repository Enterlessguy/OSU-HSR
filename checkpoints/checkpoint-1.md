# HSR checkpoint 1

Date: 2026-08-03

Initial project import of the offline osu!lazer-based human-movement
simulator (research build). Private repository, never made public.

## Included

- osu!lazer source pinned to tag `2026.726.0-lazer`, with the research-only
  modifications (HSR mod, login/score-submission gating, research song-select
  notifier).
- `research\HumanSim.MapExporter` - `.osu` -> schema-v1 MapPlan exporter.
- `research\human-sim` - Python planner, corpus/modeling/validation tooling.
- `research\HumanSim.Runner` - guarded SendInput macro runner.
- `research\HumanSim.ReplayExtractor` - `.osr` replay decoder.
- `research\HumanSim.IsolationVerifier` - isolation checks.
- `timing-tests` - timing harness source and benchmark evidence (maps,
  traces, replay exports).
- `research\human-sim\output\phase1-aim-benchmark.md` - Phase 1 skill/effort
  benchmark report.
- `HANDOFF.md`, `checkpoints\` - continuation state.

## Excluded (see .gitignore)

- Local toolchains and caches: `.dotnet\`, `.dotnet-home\`,
  `.dotnet-cli-home\`, `.nuget\`.
- Python venv, generated output traces/logs, models, `__pycache__`.
- Compiled artifacts: `timing-tests\baseline-bin\`, `**\bin\`, `**\obj\`.
- No API keys, tokens, or credentials are committed. The osu! API client
  secret is read from the `OSU_CLIENT_SECRET` environment variable only.

## State at checkpoint

- Phase 1 skill/effort model (planner `settle-chord-v3`) implemented and
  benchmarked on Camellia - Xeroa Kingborn NM; aim + fatigue strength
  criteria wired.
- Perfect-performer dispatch pipeline validated at 1000 Hz (dispatch p95
  ~10 us in recent runs).
- Open items: HEAVENLY+ re-test at skill 50/effort 68; remaining Phase 1
  strength criteria (randomness, correlation, curvature, fatigue variance);
  Phase 2 corpus calibration; optional in-process input mode decision.
  See `HANDOFF.md` for details.
