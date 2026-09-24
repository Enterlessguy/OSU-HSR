"""Run the draft OSI V2 circle descriptors on one matched real/pair window."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from human_sim.io import load_map_plan
from human_sim.scoped_realism import clip_transition_samples
from osi_v2_features import circle_window_features


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"


def main() -> None:
    with gzip.open(OUT / "development-batches" / "validation" / "human-windows.json.gz", "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    row = rows[0]
    map_md5 = row["map"]
    plan = load_map_plan(OUT / "development-plans" / (map_md5 + ".map.ndjson.gz"))
    index = row["source_window_index"]
    previous, current = plan.objects[index - 1], plan.objects[index]
    start, end = previous.start_time_ms, current.start_time_ms
    case = np.load(OUT / "development-cases" / "validation" / (map_md5 + "-seed42.npz"))
    math_t, math_xy = clip_transition_samples(case["times_ms"], case["math_positions"], start, end)
    kwargs = {"event_start_ms": start, "event_end_ms": end, "origin": previous.position, "target": current.position, "target_radius_px": current.radius}
    human = circle_window_features(row["raw_times_ms"], row["raw_positions"], **kwargs)
    math = circle_window_features(math_t, math_xy, **kwargs)
    print(json.dumps({"map": map_md5, "index": index, "human_support": human["support"], "math_support": math["support"], "human_contact": human["spatial_contact"], "math_contact": math["spatial_contact"]}, indent=2))


if __name__ == "__main__":
    main()
