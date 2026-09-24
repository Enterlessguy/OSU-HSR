"""Unscored continuity audit at each human window's original timestamps.

Generated 2 ms traces are interpolated onto the matched human replay times.
This avoids treating synthetic frames as independent human observations. No
threshold or realism score is selected on the validation cohort here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/external-ordr-v155"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def metrics(times: np.ndarray, positions: np.ndarray, row: dict) -> dict[str, float]:
    dt = np.diff(times)
    step = np.diff(positions, axis=0)
    valid = (dt > 0.0) & (dt <= 40.0)
    if np.count_nonzero(valid) < 3:
        raise ValueError("fewer than three positive <=40ms native intervals")
    velocity = step[valid] / (dt[valid, None] / 1000.0)
    speed = np.linalg.norm(velocity, axis=1)
    distance = max(float(row["distance_px"]), 1e-6)
    duration_s = float(row["duration_ms"]) / 1000.0
    dot = np.sum(velocity[:-1] * velocity[1:], axis=1)
    norm = np.linalg.norm(velocity[:-1], axis=1) * np.linalg.norm(velocity[1:], axis=1)
    reversal = (norm > 1e-9) & (dot < -0.25 * norm)
    return {
        "normalized_speed_p99": float(np.quantile(speed * duration_s / distance, .99)),
        "normalized_max_step": float(np.max(np.linalg.norm(step[valid], axis=1)) / distance),
        "direction_reversal_fraction": float(np.mean(reversal)) if len(reversal) else 0.0,
        "valid_intervals": int(np.count_nonzero(valid)),
    }


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-tag", required=True)
    args = parser.parse_args()
    case_root = OUT / "development-cases/validation" / args.case_tag
    human = rows(OUT / "development-batches/validation/human-windows.json.gz")
    by_map: dict[str, list[dict]] = defaultdict(list)
    for row in human:
        by_map[row["map"]].append(row)
    rejection = Counter()
    map_results = []
    for summary_path in sorted(case_root.glob("*-seed42.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trace_path = case_root / f"{summary['map_md5']}-seed42.npz"
        if summary["trace_sha256"] != sha(trace_path) or not summary["timestamp_and_key_identity"]:
            raise ValueError(f"Paired artifact identity mismatch: {trace_path}")
        with np.load(trace_path) as case:
            times = case["times_ms"]
            traces = {name: case[f"{name}_positions"] for name in ("math", "hybrid")}
        features: dict[str, list[dict]] = {name: [] for name in ("human", "math", "hybrid")}
        for row in by_map[summary["map_md5"]]:
            raw_times = np.asarray(row["raw_times_ms"], dtype=float)
            raw_positions = np.asarray(row["raw_positions"], dtype=float)
            if len(raw_times) < 4 or raw_times[0] < times[0] or raw_times[-1] > times[-1]:
                rejection["outside_generated_coverage"] += 1
                continue
            try:
                candidate = {"human": metrics(raw_times, raw_positions, row)}
                for arm, positions in traces.items():
                    matched = np.column_stack([np.interp(raw_times, times, positions[:, axis]) for axis in range(2)])
                    candidate[arm] = metrics(raw_times, matched, row)
            except ValueError as error:
                rejection[str(error)] += 1
                continue
            for arm in features:
                features[arm].append(candidate[arm])
        if not features["human"]:
            map_results.append({"map_md5": summary["map_md5"], "windows": 0, "status": "no_matched_native_support"})
            continue
        fields = ("normalized_speed_p99", "normalized_max_step", "direction_reversal_fraction")
        result = {"map_md5": summary["map_md5"], "windows": len(features["human"]), "status": "supported"}
        for arm in features:
            result[arm] = {name: {"median": median([row[name] for row in features[arm]]), "p95": float(np.quantile([row[name] for row in features[arm]], .95))} for name in fields}
        map_results.append(result)
    supported = [row for row in map_results if row["status"] == "supported"]
    report = {
        "schema_version": "coherent-matched-human-cadence-flicker-audit-v1",
        "status": "descriptive_unscored_no_validation_tuned_thresholds",
        "case_tag": args.case_tag, "confirmation_access": False,
        "map_count": len(map_results), "supported_maps": len(supported),
        "matched_windows": sum(row["windows"] for row in supported),
        "rejections": rejection,
        "cohort_mean_map_median": {
            arm: {name: float(np.mean([row[arm][name]["median"] for row in supported])) for name in ("normalized_speed_p99", "normalized_max_step", "direction_reversal_fraction")}
            for arm in ("human", "math", "hybrid")
        },
        "maps": map_results,
    }
    path = case_root / "matched-cadence-flicker-audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "maps"}, indent=2))


if __name__ == "__main__":
    main()
