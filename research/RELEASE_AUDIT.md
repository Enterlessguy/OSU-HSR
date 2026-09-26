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
The initial source audit preceded remote CI. The Linux review branch now
runs Windows, Ubuntu and Arch checks; see the dated evidence below.

The public-file/history scanner found no recognized secrets or private
artifacts in 6,337 local Git blobs and 5,782 current tracked/unignored files
before final staging. Recheck the exact final commit and inspect the
staged diff before uploading. Source ZIPs are made from a clean Git commit,
never by archiving the working directory. Generated reports, stores, toolchains,
logs, replay bytes and environments are ignored. Review asset checksums and
repository identity before creating a GitHub release.

## Linux package review — 2026-09-27

The public `codex/hsr-linux-readiness` branch replaces the initial global X11
prototype with authenticated private client input on X11 and Wayland.
Transport is hashed and bounded, creates Unix files with mode 0600, checks
token/map/rate and preserves explicit learned/fallback counters. Every recorded
frame is important to the gameplay frame-stability scheduler. Focus loss,
backward clocks and settled playfield-scale changes abort playback and release
held actions. The research login/submission gates remain enforced.

[CI run 36279524900](https://github.com/Enterlessguy/OSU-HSR/actions/runs/36279524900)
passed all three jobs at `ba77630`: Windows builds/contracts, Ubuntu's 74
Python tests and builds, unprivileged Arch makepkg, generated .SRCINFO comparison,
namcap (zero errors), installation, installed command diagnostics, and native
client startup on X11 and nested Wayland with keyboard focus. Handler checks
cover 153 scheduled frames, exact key edges, duplicate accounting, three clock
rates, backward-clock release and scale-change release; transport checks cover
14 identity/frame/hash cases. Native startup checks exercise rendering and
storage but do not play through a complete HSR map or measure physical latency.

The final package includes vendor BASS-family notices, the intact BASS FX
archive, FFmpeg LGPL text and checksum-pinned upstream source. The HSR source
remains MIT; the complete binary stack includes proprietary BASS components.
NuGet metadata alone is not a native-library licence determination. Namcap's
upstream ELF hardening/private-library warnings and old FFmpeg 4.3.3 are retained
limitations, not silently dismissed security findings. See the native dependency
document. No vulnerability-free or fully open-source binary-stack claim is made.

The source/history pattern scan inspected 6,461 Git blobs and 5,797 current
tracked/unignored files with zero recognized findings before final packaging
commit. Raw data and local logs remain excluded. Final public-pin CI, package
inventory/hash review and a physical Linux desktop gameplay check are distinct
release evidence levels. No AUR credentials or AUR publication are included.

### Public-pin package and compiler evidence

[Run 36280322335](https://github.com/Enterlessguy/OSU-HSR/actions/runs/36280322335)
passed the actual public source pin `d08bad9`, including install, focused native
X11/Wayland startup and uninstall. Its downloaded 163,285,274-byte package has
SHA-256 `a34feb161cb92ac00a6a6f983c1ff0f2cab6f759b45dc24ca1ebeb178e3440fd`.
The 1,362-entry inventory contains no raw maps/replays/traces, private supervision,
environments or caches. The installed model digest matches the reviewed checkpoint.
BASS FX and FFmpeg archives inside the package match their recipe checksums.
Relative runtime symlinks preserve tool-local paths without duplicating binaries.

[Run 36280532846](https://github.com/Enterlessguy/OSU-HSR/actions/runs/36280532846)
adds the installed compiler integration gate: the compiled runner invokes the
installed exporter and Python planner, loads the reviewed model and validates
41,317 frames, 26 learned segments, 39,078 changed samples and zero fallbacks.
The fixture is an 80-circle synthetic map generated in temporary storage; it
contains no human replay or validation cohort. No input is dispatched by this
planning diagnostic. This is model-wiring evidence, not a realism score.

The current recipe pins `632f100ae008b29eb00d8498ff3042e1cfe58a25`, which corrects
platform help and preserves the Windows log default. Its final public-pin
[CI run 36280792929](https://github.com/Enterlessguy/OSU-HSR/actions/runs/36280792929)
passed all three jobs, including installed learned planning, focused native
X11/Wayland startup and removal. Its artifact supersedes the earlier package. The source
scan at metadata commit `932ef67` reports 6,483 Git blobs, 5,798 files and zero
recognized findings. The AUR RPC reported the package name unregistered on
2026-09-27; availability must be checked again at upload time.

Final package from run `36280792929`:
`intelligence-database-hsr-0.1.2-1-x86_64.pkg.tar.zst`, SHA-256
`a14ea8ae6882dbd3d6e0b7cf1b33c2a6a05d286e1e516c14b57ac6d357220b72`.
The downloaded hash matches CI's checksum file. Inventory and installed model
digest checks passed again. This is the candidate artifact; no GitHub binary
release or AUR upload has been made.
