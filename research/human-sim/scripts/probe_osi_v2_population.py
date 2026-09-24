"""Count circle-feature support on opened validation cases only."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from human_sim.execution_training import transition_route
from human_sim.io import load_map_plan
from osi_v2_features import circle_window_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"


def main() -> None:
    with gzip.open(OUT / "development-batches/validation/human-windows.json.gz", "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    by_map = {}
    for row in rows:
        by_map.setdefault(row["map"], []).append(row)
    counts = Counter()
    case_root = OUT / "development-cases/validation/gapguard-v1"
    for case_path in sorted(case_root.glob("*-seed42.npz")):
        map_md5 = case_path.name.split("-")[0]
        plan = load_map_plan(OUT / "development-plans" / f"{map_md5}.map.ndjson.gz")
        with np.load(case_path) as case:
            times = case["times_ms"]
            positions = {arm: case[f"{arm}_positions"] for arm in ("math", "hybrid")}
        for row in by_map[map_md5]:
            index = int(row["source_window_index"])
            route = transition_route(plan, index, skill_group="competitive", record_id=f"osi-v2-probe:{map_md5}:{index}")
            target = plan.objects[index]
            origin = plan.objects[index - 1]
            args = dict(event_start_ms=route.start_time_ms, event_end_ms=route.start_time_ms + route.duration_ms,
                        origin=origin.position, target=target.position, target_radius_px=target.radius)
            counts["attempted"] += 1
            good = True
            for arm in ("human", "math", "hybrid"):
                try:
                    values = circle_window_features(
                        row["raw_times_ms"] if arm == "human" else times,
                        row["raw_positions"] if arm == "human" else positions[arm], **args)
                    if any(not np.isfinite(float(value)) for group in ("speed_phase", "spatial_contact", "continuity", "coupled_motor") for value in values[group].values()):
                        raise ValueError("nonfinite feature")
                except ValueError as error:
                    counts[f"{arm}:{error}"] += 1
                    good = False
            if good:
                counts["common_supported"] += 1
        print(f"{map_md5[:10]} {counts['common_supported']}/{counts['attempted']}", flush=True)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
