"""Explain opened validation maps absent from the exploratory circle report."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from human_sim.execution_training import _extract_phase_label_arrays, transition_route
from human_sim.io import load_map_plan
from human_sim.scoped_realism import clip_transition_samples, movement_descriptors, profile_descriptors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/external-ordr-v155"
CASE = OUT / "development-cases/validation/gapguard-v1"


def main() -> None:
    with gzip.open(OUT / "development-batches/validation/human-windows.json.gz", "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    scored = {row["map_md5"] for row in json.loads((CASE / "circle-pilot-distance.json").read_text(encoding="utf-8"))["cases"]}
    for summary_path in sorted(CASE.glob("*-seed42.json")):
        map_md5 = summary_path.name.split("-")[0]
        if map_md5 in scored:
            continue
        plan = load_map_plan(OUT / "development-plans" / f"{map_md5}.map.ndjson.gz")
        with np.load(CASE / f"{map_md5}-seed42.npz") as arrays:
            times = arrays["times_ms"]
            arms = {name: arrays[f"{name}_positions"] for name in ("math", "hybrid")}
        counts = Counter()
        for row in (value for value in rows if value["map"] == map_md5):
            counts["attempted"] += 1
            try:
                movement_descriptors(row["raw_times_ms"], row["raw_positions"])
                profile_descriptors(row["increments"])
            except ValueError as error:
                counts[f"human:{error}"] += 1
                continue
            route = transition_route(plan, row["source_window_index"], skill_group="competitive", record_id=f"probe:{map_md5}:{row['source_window_index']}")
            start, end = route.start_time_ms, route.start_time_ms + route.duration_ms
            for name, positions in arms.items():
                try:
                    arm_times, arm_xy = clip_transition_samples(times, positions, start, end)
                    label = _extract_phase_label_arrays(route, arm_times, arm_xy)
                    profile_descriptors(np.exp(label["log_phase_increments"]))
                    movement_descriptors(arm_times, arm_xy)
                except ValueError as error:
                    counts[f"{name}:{error}"] += 1
                else:
                    counts[f"{name}:valid"] += 1
        print(map_md5, json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
