# Security policy

## Scope

HSR is an offline research fork. Supported security work includes protecting
the local research boundary, preventing accidental online use, safeguarding
research credentials and player data, and validating generated artifacts.

The following are outside project scope:

- attaching to or modifying a production osu! client;
- bypassing anti-cheat or score-submission controls;
- process injection, kernel drivers, memory access, or hidden automation;
- weakening the permanent synthetic marker or research-build restrictions;
- publishing raw replay identities, OAuth credentials, or private datasets.

The enforced design is documented in `research/SECURITY_BOUNDARY.md`.

## Reporting

Report suspected vulnerabilities privately to the repository owner. Include the
affected commit, component, reproduction conditions, and expected impact. Do
not include real credentials, raw player identities, or private replay data in
an issue or log attachment.

Issues inherited unchanged from upstream osu! should be reported through the
upstream project's security process rather than this fork.

## Secrets

HSR reads sensitive values only from environment variables:

- `OSU_CLIENT_ID`
- `OSU_CLIENT_SECRET`
- `HUMAN_SIM_PLAYER_SALT`

Never place their values in source files, command examples, manifests, logs,
fixtures, or generated datasets. Local `.env` files, private keys, replay
exports, corpora, logs, and unreviewed model artifacts must remain untracked.
The audited small JSON research checkpoint is intentionally published with its
model card and aggregate development evidence.

## Research data

Player names are hashed with a private salt before derived features are written.
Raw replay files and salts must be stored separately from published or shared
feature datasets. A hash is pseudonymous, not anonymous, if the salt or source
replays are disclosed.

## Release checklist

Before publishing a checkpoint:

1. Scan the tracked tree and commit history for credentials and private keys.
2. Audit Python and .NET dependencies for known vulnerabilities.
3. Build the research client and all research tools.
4. Run the planner test suite and trace validation checks.
5. Confirm login and score submission remain disabled in the research build.
6. Confirm generated traces still require `synthetic: true`.
7. Confirm ignored local outputs are not staged.
