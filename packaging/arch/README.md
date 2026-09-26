# Arch Linux package preparation

This directory contains an x86_64 `makepkg` recipe for the isolated HSR
research build. It uses Arch's Python packages and an installed .NET 8 runtime;
`package()` only stages files and does not install Python packages or access the
network. The X11/XTest input backend is required for dispatch. Wayland and
XWayland sessions are rejected. The framework's Linux run requirements call
for system-wide FFmpeg [upstream requirement](https://github.com/ppy/osu-framework).
The Microsoft [.NET support policy](https://dotnet.microsoft.com/en-us/platform/support/policy)
lists .NET 8 end of support as 2026-11-10.

The application build also includes native BASS runtime binaries through the
upstream framework dependency. NuGet license metadata is staged under the
package license directory, but it does not establish redistribution rights for
those binaries. The vendor's [BASS licensing terms](https://www.un4seen.com/bass.html)
restrict redistribution/resale and separately describe non-commercial use.
Resolve this licensing question before publishing a binary package to AUR.

## Current status

The recipe is prepared for review but has not been built with `makepkg`, checked
with `namcap`, tested on an Arch desktop, or published to the AUR. This host is
Windows-only and has no Arch, WSL, container or `makepkg` runtime. The source
revision in `PKGBUILD` is pinned to the reviewed compatibility-source commit;
it will be fetchable by `makepkg` only after that source commit is explicitly
authorized and made public. No AUR publication is part of this preparation.

Linux X11 dispatch is also experimental. The Windows dispatch path remains the
verified runtime; no live Arch gameplay or timing calibration is claimed.
The BASS redistribution review is an additional publication gate.

## Build and inspect on Arch

After the pinned source revision is publicly available, build the package as an
unprivileged user:

```sh
makepkg --cleanbuild
makepkg --printsrcinfo > .SRCINFO.generated
diff -u .SRCINFO .SRCINFO.generated
namcap PKGBUILD ./*.pkg.tar.zst
```

Inspect the package contents and install it locally in a disposable Arch user
environment before release:

```sh
bsdtar -tf ./*.pkg.tar.zst
sudo pacman -U ./*.pkg.tar.zst
```

Then verify the research-boundary diagnostic, map/exporter and replay-extractor
wrappers, Python CLI, X11 backend preflight, Wayland rejection, and a synthetic
offline run. Confirm all user data is written beneath XDG data/cache/state
directories, and uninstall the package to verify owned files are removed.

## Package layout

- `/usr/lib/intelligence-database-hsr` contains the research client, .NET tools,
  Python source and pinned JSON model.
- `/usr/bin/intelligence-database-hsr` starts the labeled profile launcher.
- `/usr/bin/human-sim`, `/usr/bin/hsr-map-exporter` and
  `/usr/bin/hsr-replay-extractor` expose the research tools.
- The launcher opens in a terminal so runner logs and safety aborts remain
  visible. The desktop entry does not hide the synthetic-research context.

## Maintenance

The package currently targets .NET 8 because that is the fork's project target.
Microsoft support for .NET 8 ends 2026-11-10. Port the source, CI and package to
a supported .NET runtime before that date. Review runtime dependencies and
rebuild the package on each supported Arch update; do not claim the package is
verified solely because its `PKGBUILD` parses.
