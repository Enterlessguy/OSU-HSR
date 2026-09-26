# Arch Linux installation and AUR preparation

HSR's Linux runner transfers a bounded, hashed synthetic timeline to its own
research client. The client consumes each recorded cursor/key frame against
the gameplay clock. This works independently of X11/Wayland global input APIs;
it requires no root, input group, uinput driver or compositor automation access.
Windows retains the existing guarded SendInput route.

## Install locally

On Arch Linux x86_64, install the build dependencies and build as a normal user:

```sh
sudo pacman -S --needed base-devel git dotnet-sdk-8.0 python-pytest
git clone --branch codex/hsr-linux-readiness https://github.com/Enterlessguy/OSU-HSR.git
cd OSU-HSR/packaging/arch
makepkg --syncdeps --cleanbuild
namcap PKGBUILD ./*.pkg.tar.zst
sudo pacman -U ./*.pkg.tar.zst
intelligence-database-hsr --diagnostics
```

These commands currently select the public Linux review branch. Main still
contains the earlier source release. Do not run makepkg as root.

## Run

```sh
intelligence-database-hsr --skill 65 --effort 75
intelligence-database-hsr --client-only
```

The first command launches the research client through its authenticated
runner. Import your own maps, select an osu!standard difficulty and enable HSR.
The second command opens only the isolated client for map import/settings;
HSR gameplay still requires the runner. The desktop entry opens a terminal to
keep progress and abort messages visible. Both X11 and Wayland use the same
private timeline handler. The official client extension is not implemented.

User paths: maps in `$XDG_DATA_HOME/osu-development/files`, generated traces
in `$XDG_CACHE_HOME/intelligence-database-hsr/auto`, logs in
`$XDG_STATE_HOME/intelligence-database-hsr/logs`. Unset variables default to
`~/.local/share`, `~/.cache`, and `~/.local/state`. Relative XDG values are ignored.
Installed files beneath `/usr/lib/intelligence-database-hsr` are never writable
application state. Removing the package preserves your user maps and logs.

## Package contents and licensing

The recipe publishes framework-dependent `linux-x64` output, installs the
planner source and pinned v4 JSON model, and generates a NuGet license inventory.
It uses Arch Python packages; `package()` stages files without network calls
or system Python installation. Windows/macOS native outputs, test data,
training corpora, local environments and generated traces are excluded.

Project source is MIT. Native BASS audio libraries have separate vendor terms,
including free non-commercial use and restrictions on resale/sublicensing.
The complete application must not be advertised as having exclusively MIT
dependencies. Commercial/advertising-supported distribution requires checking
the applicable vendor licence. See [BASS terms](https://www.un4seen.com/bass.html)
and the generated inventory. The package includes BASS's vendor licence text.
This is an end-user research application;
it does not sublicense BASS for use in other applications.

.NET 8 support ends 2026-11-10. A supported runtime migration must precede that
date; this package currently follows the fork's net8.0 project targets.

## Verification and AUR upload checklist

The workflow tests Windows and Linux source plus an unprivileged Arch package
build/install. Diagnostics exercise the actual timeline handler at 0.75x,
1x and 1.5x, exact key edges, stalled updates, duplicate accounting and
backward-clock aborts. Transport checks cover hashes, token/map/rate, malformed
frames and mandatory learned exposure. These do not measure Windows OS input
latency or substitute for playing through representative maps on your desktop.

Before uploading to AUR:

1. Publish the reviewed source commit and pin that exact commit in PKGBUILD.
2. Run `makepkg --printsrcinfo > .SRCINFO` and review both files together.
3. Complete `makepkg --cleanbuild`, namcap, installation, diagnostics and
   uninstall checks in a clean Arch environment. Record logs and SHA-256.
4. Exercise maps with circles, sliders, spinners, breaks, DT/HT, focus loss,
   pause/quit, resize/scale changes, missing models and interrupted IPC on X11
   and Wayland. Retain the full fallback/delivered-frame counters.
5. Confirm the application remains a free/non-commercial end-user distribution
   under its native-library terms. No raw player/map data belongs in the package.
6. Create an AUR account and SSH key, check the name is available, clone
   `ssh://aur@aur.archlinux.org/intelligence-database-hsr.git`, copy PKGBUILD and
   .SRCINFO, inspect the staged diff, commit, and push after approval.

AUR holds build recipes, not prebuilt binaries. The recipe fetches the pinned
source; its launchers and license helper are part of that source. Only PKGBUILD
and .SRCINFO need to be uploaded. AUR publication is a separate authorized step.
