"""Find supported human slider, spinner and break examples in calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from human_sim.io import load_map_plan
from osi_v2_features import break_window_features, slider_window_features, spinner_window_features


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"


def main() -> None:
    manifest = json.loads((OUT / "development-batches" / "calibration" / "window-manifest.json").read_text(encoding="utf-8"))
    found = {}
    failures = {"slider": 0, "spinner": 0, "break_free_roam": 0}
    for row in manifest["records"]:
        plan = load_map_plan(OUT / "development-plans" / (row["map"] + ".map.ndjson.gz"))
        with (OUT / "development-decoded" / (row["replay_sha256"] + ".ndjson")).open("r", encoding="utf-8") as stream:
            next(stream)
            frames = [json.loads(line) for line in stream if line.strip()]
        times = np.asarray([frame["time_ms"] for frame in frames], dtype=float)
        positions = np.asarray([(frame["x"], frame["y"]) for frame in frames], dtype=float)
        for obj in plan.objects:
            if obj.kind in ("slider", "spinner") and obj.kind not in found:
                try:
                    value = slider_window_features(times, positions, obj) if obj.kind == "slider" else spinner_window_features(times, positions, obj)
                    found[obj.kind] = {"map": row["map"], "object_index": obj.index, "features": value["features"], "support": value["support"]}
                except ValueError:
                    failures[obj.kind] += 1
        if "break_free_roam" not in found:
            for previous, current in zip(plan.objects, plan.objects[1:]):
                if current.start_time_ms - previous.end_time_ms < 1000:
                    continue
                try:
                    value = break_window_features(times, positions, break_start_ms=previous.end_time_ms, break_end_ms=current.start_time_ms, next_target=current.position, next_radius_px=current.radius)
                    found["break_free_roam"] = {"map": row["map"], "previous_index": previous.index, "next_index": current.index, "features": value["features"], "support": value["support"]}
                    break
                except ValueError:
                    failures["break_free_roam"] += 1
        if len(found) == 3:
            break
    report = {"schema_version": "osi-v2-mode-feature-probe-v1", "calibration_only": True, "confirmation_access": False, "found": found, "failed_windows_before_first_support": failures}
    path = OUT / "development-batches" / "calibration" / "osi-v2-mode-feature-probe.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"found_modes": list(found), "failures": failures, "examples": {key: value["features"] for key, value in found.items()}}, indent=2))


if __name__ == "__main__":
    main()
