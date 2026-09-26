# Intelligence Database HSR residual model card

The release default is `seed101-math-residual-g100-v4.json`.
Older JSON files are historical checkpoints. This is an offline research model.

| Field | Value |
|---|---|
| File SHA-256 | `3ba4d157124fa078b862062a38578561265bf03aaa2180a566440874b22bc7a2` |
| Canonical SHA-256 | `7b87af291bca415c1f053ca515659c091ce9f2c1b280283fc129211a79ce0d87` |
| Fit | 198 contexts, 100 independent TRAIN map/player components, seed 101 |
| Architecture | 22 context features, 45 curve coefficients, ridge fit in integrated curve space |
| Target | task baseline + human minus math at matched timestamps |
| Composition | math path + feasible learned residual; lateral gain 1.5 |
| Runtime | `math-residual-lateral-gated-runtime-v1` |
| Runtime source SHA-256 | `5f69c10fd2be615b93cff847a4343a85ca8ab753d6fd2ddd32230b4818bf5d87` |
| Frozen fit specification | `d4ba7daae084093a2154c9b7aa6e18a5a2eb2936ab5cce3ddfee52be4c86e5d2` |

## Movement and limitations

Learned shape depends on route geometry, timing, incoming motion and dwell.
The residual preserves contact positions and zero outer position/velocity/
acceleration deltas, producing continuous joins. The controller admits only
positive strength satisfying continuous residual and sampled relative gates.
Those gates bound added motion; they do not guarantee an already imperfect
math baseline becomes globally legal. Every rejected window is reported.
Maps with no delivered learned movement fail the coherent planning route.

The learned adapter covers eligible four-circle windows. It leaves slider,
spinner and break movement to the mathematical planner. It is not a large
neural network, generalized player model or proof of human realism. Low
session-diversity scores remain a documented limitation.

## Data, release and reproduction

The [o!rdr replay archive](https://www.kaggle.com/datasets/skihikingkevin/ordr-replay-dump/data),
version 155, is listed as CC0 by its publisher. Public score identity checks
exclude mismatched/known automated records but cannot certify manual input.
133 admitted TRAIN inputs were staged; bounded fitting selected 198 contexts
from 100 independent components. Raw replays, player IDs, beatmaps and decoded
movement are excluded. Code and derived JSON model are released under the root
MIT license, retaining upstream attribution.

`coherent_training.py` contains the public numerical fitting/admission core.
After independently acquiring and reviewing the expected development inputs,
`train_ordr_mixed_coherent.py --prepare-only` stages TRAIN and
`train_ordr_math_residual.py --output <new-directory>` performs the fit.
Source hashes are embedded in the frozen fit spec. Exact historical fitting
requires the same private input/admission records, which are not distributed.
The released JSON is directly hash-verifiable and needs no training downloads
for normal research use.

## Opened validation evidence

All 43 maps delivered learned movement: 5,124 accepted windows, 146 fallbacks,
1,006,801 changed samples. The local V2 circle component supports 38 maps:
math 62.768, hybrid 63.887; paired gain +1.119, bootstrap 95% interval
[+0.517, +1.959]. No math-legal circle became illegal among 27,387 contacts.
Other mode coordinates were unchanged. Temporal improvement is inconclusive.
These are opened development results; the scorer was inspected during model
selection. No full V2 aggregate, independent confirmation, perceptual review,
anti-cheat-evasion or player-indistinguishability claim is made.

See the [aggregate benchmark snapshot](../../benchmarks/LOCAL_V2_20260926.json)
and [release notes](../../../RELEASE_NOTES.md).
