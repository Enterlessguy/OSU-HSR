# Experimental direct residual model

`seed101-math-residual-g100.json` is a research checkpoint. It is not the
desktop default and is not evidence of human indistinguishability.

| Field | Value |
|---|---|
| Model file SHA-256 | `e1cd5bcd7ca40d8f060277c9a0ee176fbf625bcc0401c66716044a61891a7074` |
| Canonical model SHA-256 | `302e4fe996664366f498ac74301cd294ee39b4736dfb812e4851281183c4fce1` |
| Fit | 198 contexts from 100 independent TRAIN map/player components |
| Target | Task curve plus observed human minus generated math at matched times |
| Runtime | Experimental math plus learned residual adapter, not the desktop loader |

The public source was the [o!rdr replay dump](https://www.kaggle.com/datasets/skihikingkevin/ordr-replay-dump/data),
version 155, listed as CC0 by its publisher. The model does not contain raw
replays or beatmaps. Public score metadata corroborates source identity but
does not prove that each trace was manually played. The replay archive and
decoded player movement are intentionally excluded from this repository.

On 43 opened validation maps, the experimental runtime delivered learned
movement on all maps: 5,182 accepted segments, 88 fallbacks, and 1,016,461
changed coordinate samples. A **draft circle-only** OSI V2 component scored
63.729 for the hybrid versus 62.768 for pure math on 38 supported maps;
paired gain +0.961, bootstrap 95% interval [+0.422, +1.725]. Five map circle
cells were unsupported. Slider, spinner and break coordinate samples were
unchanged, and a contact audit found no math-legal circle made illegal by the
hybrid. These are development results because the benchmark and model were
inspected on opened validation data.

Full OSI V2 is unfinished: temporal diversity, calibrated mode and native
flicker gates, bandwidth sensitivity, critical-cell margins, sealed
confirmation, and blinded player-view corroboration remain. The desktop
runtime still uses the earlier guarded model. Do not configure this file as
the default or advertise a superior full realism score until those gates pass
and the actual desktop compiler dispatch is verified.
