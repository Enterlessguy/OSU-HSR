# Release security and open-source review — 2026-09-26

## Scope and evidence

Reviewed HSR changes, model provenance, compiler wiring, acquisition, research
boundary, public file inventory, local Git history and dependency advisories.
This is a local release review, not an independent penetration test or proof
that no vulnerabilities exist.

- Python: 69 tests passed, including model/hash rejection, all-math refusal,
  invalid timestamps, break isolation, polynomial scaling equivalence, path
  traversal rejection and credential-safe redirects.
- Bandit: no medium/high findings after fixes. Remaining 21 low findings are
  subprocess imports/calls and PATH-resolved development tools. Calls pass
  argument lists without a shell; the runner pins its selected Python source.
  A compromised local checkout or executable PATH remains outside this guard.
- Python dependency feed: no known vulnerabilities after replacing old pip.
  The 31 runtime/test packages are pinned in `human-sim/requirements-release.txt`.
- NuGet identified System.Net.Http 4.3.0 and RegularExpressions 4.3.0; direct
  references now select patched 4.3.4 and 4.3.1 respectively.
- **Retained advisory:** AutoMapper 13.0.1 remains flagged by
  [GHSA-rvv3-g6hj-g44x](https://github.com/LuckyPennySoftware/AutoMapper/security/advisories/GHSA-rvv3-g6hj-g44x).
  Upstream retained this MIT version because later releases change licensing.
  All 33 configured Realm maps now have explicit recursion limits at most 32,
  retaining stricter depth 1/2 maps. `--verify-research-mapping-depth` checks the
  actual configured maps without entering gameplay. This follows the maintainer's
  recommended depth mitigation; it does not make the package advisory disappear
  or establish coverage of arbitrary upstream object graphs.
- Research boundary diagnostic reports `research=True;login=False;submission=False`.
  The client and tools build in Debug and Release. A compiled runner
  planning smoke validates the exporter → pinned model → trace-loader path.
  A research client reached menu/song selection during this review; final live
  gameplay dispatch and OS cursor capture were not recorded. Desktop automation
  was stopped when the user pressed Escape.

## Data and licensing

The root `LICENCE` retains upstream MIT terms and ppy attribution. HSR source
and the bundled derived JSON weights are offered under those MIT terms.
Keep third-party package notices when building or redistributing binaries.
The o!rdr v155 archive is listed as CC0 by its publisher; this is publisher
metadata, not independent proof of every replay's rights or manual origin.
Existing upstream test maps and four replay fixtures are retained with their
upstream attribution; the acquired HSR research corpus is excluded.
Only reviewed TRAIN inputs entered fitting. Public score identity corroboration
does not certify human input. No acquired research replay, map, player identifier, OAuth
credential, private salt or decoded movement is included in the release.

The model card and aggregate benchmark contain source/artifact hashes.
Reproducing the historical numerical fit requires reacquiring the same reviewed
inputs and admission metadata; those private datasets are not bundled. Public
source supports fitting reviewed inputs and verifying the released model hash.
No claim of one-command recreation of the private historical cohort is made.

## GitHub actions and publication

Upstream deployment, Sentry and diff-calculation workflows are retained as
`.yml.upstream` reference files, which GitHub does not execute. The active HSR
workflow has read-only repository permission and builds with the research flag;
it does not publish, contact upstream deployment services or use credentials.
GitHub CI itself has not run until the branch is uploaded.

The public-file/history scanner found no recognized secrets or private
artifacts in 6,337 local Git blobs and 5,782 current tracked/unignored files
before final staging. Recheck the exact final commit and inspect the
staged diff before uploading. Source ZIPs are made from a clean Git commit,
never by archiving the working directory. Generated reports, stores, toolchains,
logs, replay bytes and environments are ignored. Review asset checksums and
repository identity before creating a GitHub release.
