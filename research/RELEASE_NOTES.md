# Intelligence Database HSR — local V2 research preview

This source release includes osu!lazer with Human Simulator Research
preinstalled, a gated learned movement model, the mathematical control arm,
and local evaluation tools. MIT source; independent fork of ppy/osu.

## Changes

- Pinned v4 residual model, fitted on 198 contexts from 100 TRAIN maps.
- Context-dependent learned curve shape, lateral residual gain 1.5, continuous
  position/velocity/acceleration joins, and adaptive feasible strength.
- Exact continuous extrema are reused across the strength search. Numerical
  equivalence tests cover the optimization; delivered cases reproduce the
  previous tuned result.
- Compiler planning selects this checkout's Python source. Missing, corrupt,
  zero-blend and entirely math-only coherent output fail explicitly.
- Research client uses its own instance pipe and storage; production updating,
  legacy IPC, login and score submission are disabled.
- Visible **Intelligence Database | SYNTHETIC RESEARCH RUN** banner.
- HTTPS acquisition, bounded downloads, numeric score IDs, credential-safe
  redirects, and finite/timestamp/model-size checks.

## Local benchmark

The [aggregate snapshot](human-sim/benchmarks/LOCAL_V2_20260926.json) records
43 opened validation maps, with every fallback included. The circle component
supports 38 maps and 2,814 common-bandwidth windows: math 62.768, hybrid 63.887,
paired gain +1.119, bootstrap 95% interval [+0.517, +1.959]. There were 5,124
accepted learned windows, 146 fallbacks, and 1,006,801 changed samples.
No math-legal circle was made illegal among 27,387 circle contacts.

Slider, spinner and break movement is unchanged. Temporal score is 4.656
versus 4.376, with an interval crossing zero. Its low absolute score is a
remaining limitation. The flicker audit is descriptive, without a calibrated
equivalence claim. Five circle cells are unsupported. This is development
component evidence, not a full OSI V2 aggregate, independent confirmation,
player-indistinguishability claim, or observed OS cursor-capture result.
Confirmation data remains unopened; no independent reviewers were available.

## Both distribution directions

1. **Official-client extension:** custom-ruleset feasibility documented, not
   implemented or shipped. Official osu!lazer does not load an arbitrary Mod
   DLL into its standard ruleset. Host-level score isolation must be verified
   before an extension is released.
2. **Separate build:** the implemented offline fork with HSR preinstalled.
   Build from this source with `HumanSimResearchBuild=true`.

See [distribution options](DISTRIBUTION_OPTIONS.md) and the
[release audit](RELEASE_AUDIT.md). This preview distributes source and the
small JSON model; it does not redistribute the acquired training/validation
beatmaps, replay archive, player movement, a Python environment, or a .NET
runtime. Existing upstream test beatmaps and four replay fixtures remain in
the source tree with upstream attribution; they are not HSR training data. Install a patched .NET 8 SDK
and Python 3.12, run `research/setup-research.ps1`, then
`research/run-dev-build.ps1` on Windows. The launcher builds the research client
and tools. Import your own licensed beatmaps into the research client.

The project's branding is Intelligence Database (inteldatabase.org).
No website deployment is part of this release.
