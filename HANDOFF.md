# HSR maintainer entry points

HSR is an offline osu!lazer research fork, maintained under Intelligence Database.
Start with the [technical guide](docs/TECHNICAL_GUIDE.md) for architecture,
mathematical planning, learned residual composition, runtime protocol, benchmark,
build instructions, distribution directions and troubleshooting.

Current release source uses the v4 residual checkpoint. Preserve its frozen
numerical source and model identities when changing documentation or packaging.
Use a new model/runtime identity for substantive movement changes.

- [Release notes](research/RELEASE_NOTES.md)
- [Release readiness](RELEASE_READINESS.md)
- [Security and provenance audit](research/RELEASE_AUDIT.md)
- [Model card](research/human-sim/models/experimental/README.md)
- [Distribution directions](research/DISTRIBUTION_OPTIONS.md)
- [Local benchmark snapshot](research/human-sim/benchmarks/LOCAL_V2_20260926.json)

No acquired research dataset or local machine state is part of this repository.
Do not commit replay corpora, credentials, logs, generated traces or environments.
