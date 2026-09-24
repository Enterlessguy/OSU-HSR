# HSR open-source release readiness

Status: **research preview; not yet ready to publish as the desktop hybrid build.**

The intended public repository is the configured `origin` (`Enterlessguy/HSR-Checkpoint-2-final`). Verify the final repository and branch before pushing. The website mentioned by the project owner has not been verified as a deployment target.

## Verified locally

- The root license is MIT (`LICENCE`), inherited from the upstream osu! fork. Keep upstream attribution and the independent research-fork notice.
- The HSR client is restricted to an unranked research mod with login and score submission disabled. Preserve the local handshake and synthetic trace labels in any release.
- Static boundary review found that `research/build-research.ps1` sets `HumanSimResearchBuild=true`, `osu.Game.csproj` defines `HUMAN_SIM_RESEARCH`, `ResearchBuild.AllowsLogin` is false under that flag, and both score-token retrieval and score submission check `ResearchBuild.AllowsScoreSubmission`. The exact shipped binaries still need a build-flag and runtime smoke check.
- `python research/human-sim/scripts/check_open_source_release.py --strict --history` scanned 6,308 local Git blobs and 5,760 tracked or unignored current files on 2026-09-25; it reported no recognized secrets, local user paths in HSR source, raw replay extensions, or oversized unignored files. This pattern scan is not a security proof. Run it again on the exact release commit and inspect any generated files before staging.
- `research/human-sim/models/experimental/seed101-math-residual-g100.json` is a research checkpoint. Its card records provenance, hashes, and limitations. Raw replays and decoded player movement are excluded.
- The active desktop shortcut was previously launch-smoked in its actual worktree; that shortcut still uses the older guarded model. The root checkout now contains the gated lateral-residual adapter, matching runner identity and a profile launcher path to the pinned model. The root research client and runner build without warnings using local restored assets. A root CLI smoke produced seven accepted learned segments, zero fallbacks, and a valid trace. The active desktop shortcut has not been switched to this checkout or smoke-tested with this model.
- The gated lateral-residual adapter was evaluated on all 43 opened validation maps. It delivered learned movement on every map (5,124 accepted windows, 146 fallbacks, 1,006,801 changed samples). Its draft circle component improved over the untuned hybrid by 0.158 points on 38 supported maps, paired bootstrap 95% interval [+0.044, +0.285]. The full OSI V2 score remains unavailable.
- The root planner, continuity, movement-training, and coherent runtime contract tests passed (33 tests) with the repository source selected as `PYTHONPATH`. The coherent tests cover pinned model resolution, hash rejection, zero blend, break-window fallback, and refusal to write an all-math coherent trace. The exact release commit and desktop worktree still need their own build and smoke test.
- The root launcher parses and all three research-tool projects build from existing restored assets with zero warnings; clean checkouts retain normal NuGet restore. The active desktop shortcut still needs a live smoke run against the integrated checkout.

## Publication blockers

1. Reconcile the active desktop shortcut's target worktree with the repository root. The root now contains the experimental runtime modules and compiler wiring, but the desktop shortcut still targets its separate older-model worktree. Preserve existing desktop edits while integrating.
2. Commit and build the exact source intended for publication, then launch-smoke it from the desktop shortcut. Confirm that the compiled runner dispatches the selected model in a live research run. Count invocation, accepted learned windows, fallbacks, delivered coordinate changes, and key/timestamp identity. The root command-line smoke does not prove live desktop dispatch.
3. Finish OSI V2 gates on TRAIN/calibration, freeze scorer, model, case list, exclusions, and critical margins, then run the sealed player/map-family-disjoint confirmation set once. Require a positive paired confidence interval, nonregressing critical cells, native flicker equivalence, and blinded player-view corroboration. The current positive score is for an opened-validation **circle component only**.
4. Review public artifact provenance and reuse rights. The o!rdr source is listed as CC0 by its publisher, but raw replay data and player identifiers are excluded. Review the upstream osu! fork notices and the final diff before pushing.
5. The portable path refactor changed `train_ordr_math_residual.py` after the current model was fitted. The existing model hash remains pinned, but that trainer's current file hash no longer matches the historical fit source. Retrain under a new model identity before claiming exact source-to-model reproducibility from the publishable code.

## Release check

Use `git status --short` and review the staged diff, especially `research/human-sim/output`, `output`, replay files, logs, model artifacts, and any developer credentials. Run the release scanner with `--strict --history` on the final commit. Build the research client and runner, launch the actual shortcut, and record hashes of the shipped binaries, model, adapter, and OSI report. Push only the reviewed commit to the verified GitHub repository.
