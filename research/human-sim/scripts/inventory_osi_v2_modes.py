"""Inventory supported OSI V2 slider, spinner and break cells on development."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from human_sim.io import load_map_plan
from osi_v2_features import break_window_features, slider_window_features, spinner_window_features


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calibration", "validation"), required=True)
    args = parser.parse_args()
    batch = OUT / "development-batches" / args.split
    manifest_path = batch / "window-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    mode_players: dict[str, set[str]] = defaultdict(set)
    mode_families: dict[str, set[str]] = defaultdict(set)
    output_path = batch / "osi-v2-mode-windows.jsonl.gz"
    with gzip.open(output_path, "wt", encoding="utf-8") as output:
        for number, row in enumerate(manifest["records"], 1):
            plan = load_map_plan(OUT / "development-plans" / (row["map"] + ".map.ndjson.gz"))
            decoded = OUT / "development-decoded" / (row["replay_sha256"] + ".ndjson")
            with decoded.open("r", encoding="utf-8") as stream:
                next(stream)
                frames = [json.loads(line) for line in stream if line.strip()]
            times = np.asarray([frame["time_ms"] for frame in frames], dtype=float)
            positions = np.asarray([(frame["x"], frame["y"]) for frame in frames], dtype=float)

            def keep(mode: str, object_index: int, value: dict) -> None:
                counts[mode + "_supported"] += 1
                mode_players[mode].add(row["player"])
                mode_families[mode].add(row["map_family"])
                output.write(json.dumps({"split": args.split, "map": row["map"], "map_family": row["map_family"], "player": row["player"], "replay_sha256": row["replay_sha256"], "mode": mode, "object_index": object_index, "features": value["features"], "support": value["support"]}, separators=(",", ":")) + "\n")

            for obj in plan.objects:
                if obj.kind not in ("slider", "spinner"):
                    continue
                counts[obj.kind + "_attempted"] += 1
                try:
                    value = slider_window_features(times, positions, obj) if obj.kind == "slider" else spinner_window_features(times, positions, obj)
                    keep(obj.kind, obj.index, value)
                except ValueError as error:
                    reasons[obj.kind + ":" + str(error)] += 1
            for previous, current in zip(plan.objects, plan.objects[1:]):
                if current.start_time_ms - previous.end_time_ms < 1000:
                    continue
                counts["break_free_roam_attempted"] += 1
                try:
                    value = break_window_features(times, positions, break_start_ms=previous.end_time_ms, break_end_ms=current.start_time_ms, next_target=current.position, next_radius_px=current.radius)
                    keep("break_free_roam", current.index, value)
                except ValueError as error:
                    reasons["break_free_roam:" + str(error)] += 1
            print(f"[{number}/{len(manifest['records'])}] {row['map'][:10]}", flush=True)
    report = {
        "schema_version": "osi-v2-development-mode-inventory-v1",
        "split": args.split, "source_manifest_sha256": sha(manifest_path),
        "confirmation_access": False, "human_only": True,
        "counts": counts, "rejection_reasons": reasons,
        "supported_players": {mode: len(values) for mode, values in mode_players.items()},
        "supported_map_families": {mode: len(values) for mode, values in mode_families.items()},
        "features_file_sha256": sha(output_path),
    }
    report_path = batch / "osi-v2-mode-inventory.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
