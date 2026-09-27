# Native runtime dependencies

The HSR source and model are MIT licensed. Native binaries retain their own
terms. The generated NuGet inventory describes package metadata, which does
not override the licences of native libraries inside a package.

## BASS family

The framework supplies BASS, BASSmix and BASS FX through
`ppy.osu.Framework.NativeLibs 2025.806.0-nativelibs`. Vendor notices are included
alongside this document. BASS is proprietary and free for non-commercial use;
commercial distribution requires the applicable vendor licence. BASSmix is
free to use with BASS. BASS FX has its own free-distribution conditions.
The package retains the complete, unmodified vendor BASS FX ZIP under
`/usr/share/doc/intelligence-database-hsr/vendor/`.

Notice sources reviewed on 2026-09-27:

- [BASS terms and downloads](https://www.un4seen.com/bass.html)
- [BASSmix documentation](https://www.un4seen.com/doc/bassmix/bassmix.html)
- [BASS FX Linux vendor archive](https://www.un4seen.com/files/z/0/bass_fx24-linux.zip)

These notice snapshots do not upgrade the framework's bundled binaries.
They do not grant permission to resell or sublicense BASS.

## FFmpeg

The bundled Linux libraries identify themselves as FFmpeg 4.3.3 and LGPL 2.1
or later. They are dynamically loaded; the wrapper's MIT licence does not
relicense them. The package includes the LGPL text and upstream 4.3.3 source
archive under `/usr/share/doc/intelligence-database-hsr/native-source/`.
The archive SHA-256 is
`9f0a68fbd74feb4e50dc220bddd59d84626774a53687fb737806ae00e5c6e9e6`.

Source: [FFmpeg 4.3.3](https://ffmpeg.org/releases/ffmpeg-4.3.3.tar.xz).
The installed binaries come from the pinned framework NuGet package, not a
new HSR compilation of this source. Their embedded configure flags are:

```text
--disable-static --enable-shared --disable-all --disable-autodetect
--enable-lto --enable-avcodec --enable-avformat --enable-swscale
--enable-demuxer='avi,flv,asf' --enable-parser=mpeg4video
--enable-decoder='flv,msmpeg4v1,msmpeg4v2,msmpeg4v3,mpeg4,vp6,vp6f,wmv2'
--enable-demuxer='mov,matroska' --enable-parser='h264,hevc,vp8,vp9'
--enable-decoder='h264,hevc,vp8,vp9' --enable-protocol=pipe
--target-os=linux
```

Rebuilding a replacement also requires a compatible toolchain and output
paths (`--prefix` and `--shlibdir`). Binary-identical reproduction has not
been established. Keep replacement libraries compatible with the 58/56/5
ABI used by this fork; the current Arch FFmpeg package uses a different ABI.

## Audit limits

The package preserves the framework's supplied native binaries without
stripping them. Namcap reports missing full RELRO in several upstream ELFs
and cannot resolve every private-directory shared-library dependency.
Managed assemblies and dynamically loaded dependencies also produce unused
dependency warnings. Build, installation and startup checks are evidence of
compatibility; they do not establish that these dependencies are free of
vulnerabilities. FFmpeg 4.3.3 is an old dependency and needs a separate
decoder/runtime migration. Use reviewed local research media.

The complete published application therefore includes proprietary components
and is not an exclusively open-source dependency stack.
