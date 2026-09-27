# Intelligence Database HSR: distribution directions

## 1. Extension in the official game — feasibility track

**Status: investigated, not a shipped extension.** The official osu!lazer
client supports community ruleset assemblies. Its loader discovers a public
`Ruleset` subclass; it does not discover arbitrary `Mod` classes and merge
them into the built-in osu!standard mod list. An HSR DLL containing only this
fork's mod therefore cannot be dropped into the official client to add HSR.

Sources inspected on 2026-09-26:

- [Official ruleset loader](https://github.com/ppy/osu/blob/master/osu.Game/Rulesets/RulesetStore.cs)
- [Official custom game mode documentation](https://osu.ppy.sh/wiki/en/Game_mode#custom-game-modes)

The feasible extension direction is a separate **HSR Research custom ruleset**
with its own ruleset identity and mod catalog, using normal extension APIs.
It would render visibly synthetic traces inside its own gameplay input
pipeline, retain the Intelligence Database watermark, and keep results local.
It would not modify the official osu!standard ruleset or attach a macro to
production gameplay. Compatibility must be pinned and tested against each
supported official client version.

Before shipping that direction, verify that the official host actually blocks
score-token requests, uploads, multiplayer and leaderboard effects for the
custom ruleset. `Mod.Ranked == false` alone is not equivalent to disabling all
score submission. This fork's `IResearchOnlyMod`, gameplay-start hooks and
compile-time login/submission guards are changes to core code and are not
available merely by referencing an unmodified official client. The extension
must supply equivalent isolation through supported interfaces, or remain an
offline replay viewer. No DLL for this track is included in this release.

osu!stable is not covered by this ruleset-extension direction.

## 2. Separate build with HSR preinstalled — implemented track

The shipped source is an osu!lazer fork with HSR in its osu!standard mod list,
the continuous learned residual planner, and a guarded Windows runner. Linux
uses an authenticated private client timeline on X11 and Wayland. Build
with `HumanSimResearchBuild=true`. The compiler flag disables login and score
submission; the runner requires its authenticated local client handshake.

Research desktop builds use `osu-development` storage and a separate
`osu-human-sim-research` instance pipe. Production updater/legacy IPC and
Discord presence are disabled in research builds. The generated traces are
permanently marked synthetic. The visible gameplay banner includes
**Intelligence Database** and **SYNTHETIC RESEARCH RUN**.

This track is suitable for the local V2 benchmark and experimental demos.
There is no player-indistinguishability or anti-cheat-evasion claim. The
extension feasibility document and bundled build are both release materials;
only the bundled build is currently implemented.

### Arch Linux package preparation

`packaging/arch/` contains an x86_64 PKGBUILD and launcher files. The package
targets X11 and Wayland desktops through private client input, stages Linux
runtime files under `/usr/lib`, uses Arch's
Python packages without network installs during `package()`, and keeps user
data under XDG directories. It is not published to the AUR. See
[installation and package review](../packaging/arch/README.md) for verification
and upload gates.

The build targets .NET 8. Microsoft support for .NET 8 ends 2026-11-10; the
package must move to a supported runtime before then.
