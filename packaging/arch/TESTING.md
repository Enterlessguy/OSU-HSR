# Intelligence Database HSR — Arch testing preview

This preview targets Arch Linux x86_64 desktops with X11 or Wayland. It contains
the separate osu!lazer research build with HSR preinstalled. Login and score
submission are disabled; synthetic gameplay carries the Intelligence Database
watermark. Download from [the Arch testing release](https://github.com/Enterlessguy/OSU-HSR/releases/tag/v0.1.2-arch-preview).

## Install the prebuilt package

Download `intelligence-database-hsr-0.1.2-1-x86_64.pkg.tar.zst` and `SHA256SUMS`
into the same directory. In a terminal in that directory:

```sh
sha256sum --ignore-missing --check SHA256SUMS
sudo pacman -Syu
sudo pacman -U ./intelligence-database-hsr-0.1.2-1-x86_64.pkg.tar.zst
intelligence-database-hsr --diagnostics
intelligence-database-hsr --client-only
```

Pacman installs the declared runtime dependencies from your configured
repositories. Import your own maps in this isolated client, then close it and run:

```sh
intelligence-database-hsr --skill 65 --effort 75 --execution-mode coherent
```

Select an osu!standard difficulty, enable HSR in the mod selection, and start.
Keep the terminal open for compiler, learned-movement and abort messages. Close
the client before changing the launch profile. Skill and effort accept 0–100.
The desktop menu entry uses skill 50 and effort 68 with the same learned model.

Run as your normal desktop user; no global input permissions or privileged input
devices are required. If focus or playfield scale changes abort a run, restore
focus/settings and restart the run.

## Build the same package from source

Download `intelligence-database-hsr-0.1.2-arch-recipe.tar.gz` and verify it with
the same checksum file. Build as a normal user:

```sh
sudo pacman -S --needed base-devel git dotnet-sdk-8.0 python-pytest
tar -xzf intelligence-database-hsr-0.1.2-arch-recipe.tar.gz
cd intelligence-database-hsr-arch-recipe
makepkg --syncdeps --cleanbuild
sudo pacman -U ./intelligence-database-hsr-0.1.2-1-x86_64.pkg.tar.zst
intelligence-database-hsr --diagnostics
```

The recipe fetches the exact public source commit used in the verified CI
package. Locally built archives need not be byte identical to the published
package. Distribution is through GitHub; no AUR registration is needed.

## The two product distribution directions

1. **Extension inside the official client:** not implemented. No extension DLL
   is included. The custom-ruleset design and submission-isolation requirements
   are in [DISTRIBUTION_OPTIONS.md](../../research/DISTRIBUTION_OPTIONS.md).
2. **Separate build with HSR preinstalled:** implemented here. Both installation
   methods above install this build; neither modifies your official client.

## Friend testing checklist

CI verified build, installation/removal, native X11 and nested Wayland startup,
input/transport contracts and the installed compiler/model chain. Full HSR
gameplay on a physical Linux desktop remains the preview's testing gate.

Please test on each desktop session available to you:

- Circles, sliders, spinners and breaks on several local maps.
- Normal speed, Double Time and Half Time; low, medium and high skill profiles.
- Smooth cursor motion and preserved key edges through dense patterns.
- Pause/resume, focus loss, resize/scale changes and quit while a key is held.
- A complete map: check learned movement, fallback counts and completion in the
  terminal; report any abort or missing completion.
- Login/submission remain disabled and the watermark is visible.

Record desktop session, GPU/driver, map characteristics, profile, enabled mods,
what happened, and relevant diagnostic lines. Logs normally live in
`~/.local/state/intelligence-database-hsr/logs`; generated traces live in
`~/.cache/intelligence-database-hsr/auto`. Inspect logs before sharing and remove
personal paths/map identifiers. Do not upload human replays or private datasets.

Remove the application with `sudo pacman -R intelligence-database-hsr`.
Your imported maps and user logs are preserved.

## Audit and known limits

The source/history signature scan found no recognized secret findings; package
inventory excludes private research data. This does not guarantee vulnerability-free
dependencies. See [the release audit](../../research/RELEASE_AUDIT.md) and
[native dependency notices](NATIVE_DEPENDENCIES.md): old bundled FFmpeg, native
hardening warnings, BASS licence conditions and the documented AutoMapper
mitigation remain. Source/model are MIT; the full native dependency stack is not
exclusively open source. This is a free, non-commercial testing preview.

.NET 8 support ends on 2026-11-10. This preview has not established a full realism
win or calibrated Linux physical input latency.
