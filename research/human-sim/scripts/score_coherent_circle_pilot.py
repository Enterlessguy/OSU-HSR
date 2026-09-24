"""Exploratory paired distribution distances on already-open development maps.

This is a diagnostic for model selection, not OSI V2 or a superiority claim.
Calibration supplies descriptor scales only; no negative-control anchor or
confidence interval is computed here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from human_sim.execution_training import _extract_phase_label_arrays, transition_route
from human_sim.io import load_map_plan
from human_sim.scoped_realism import (
    DESCRIPTORS, MOVEMENT_DESCRIPTORS, _validation_scale,
    clip_transition_samples, movement_descriptors, profile_descriptors,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"
SEED = 42


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def distance(human: list[dict], arm: list[dict], names: list[str], scales: dict[str, float]) -> tuple[float, dict[str, float]]:
    parts = {name: float(wasserstein_distance([row[name] for row in human], [row[name] for row in arm]) / scales[name]) for name in names}
    return float(np.mean(list(parts.values()))), parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-tag")
    args = parser.parse_args()
    calibration_path = OUT / "development-batches" / "calibration" / "human-windows.json.gz"
    validation_path = OUT / "development-batches" / "validation" / "human-windows.json.gz"
    cal = rows(calibration_path)
    val = rows(validation_path)
    cal_phase = [profile_descriptors(row["increments"]) for row in cal]
    cal_movement = []
    cal_rejects: Counter[str] = Counter()
    for row in cal:
        try:
            cal_movement.append(movement_descriptors(row["raw_times_ms"], row["raw_positions"])[0])
        except ValueError as error:
            cal_rejects[str(error)] += 1
    phase_names = [name for name in DESCRIPTORS if np.ptp(np.percentile([row[name] for row in cal_phase], [5, 95])) > 1e-4]
    movement_names = [name for name in MOVEMENT_DESCRIPTORS if np.ptp(np.percentile([row[name] for row in cal_movement], [5, 95])) > (1e-4 if name in {"stop_fraction", "high_frequency_power_share"} else 1e-3)]
    phase_scales = {name: _validation_scale([row[name] for row in cal_phase], floor=1e-4) for name in phase_names}
    movement_scales = {name: _validation_scale([row[name] for row in cal_movement], floor=(1e-4 if name in {"stop_fraction", "high_frequency_power_share"} else 1e-3)) for name in movement_names}

    case_dir = OUT / "development-cases" / "validation"
    if args.case_tag:
        case_dir = case_dir / args.case_tag
    results = []
    for summary_path in sorted(case_dir.glob(f"*-seed{SEED}.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        map_md5 = summary["map_md5"]
        trace_path = case_dir / (map_md5 + f"-seed{SEED}.npz")
        if summary["trace_sha256"] != sha(trace_path):
            raise ValueError("Paired trace artifact changed")
        case = np.load(trace_path)
        times = case["times_ms"]
        plan = load_map_plan(OUT / "development-plans" / (map_md5 + ".map.ndjson.gz"))
        map_rows = [row for row in val if row["map"] == map_md5]
        valid = []
        failures: Counter[str] = Counter()
        for row in map_rows:
            try:
                human_movement = movement_descriptors(row["raw_times_ms"], row["raw_positions"])[0]
            except ValueError as error:
                failures["human:" + str(error)] += 1
                continue
            route = transition_route(plan, row["source_window_index"], skill_group="competitive", record_id=f"pilot:{map_md5}:{row['source_window_index']}")
            start, end = route.start_time_ms, route.start_time_ms + route.duration_ms
            record = {"human": {"phase": profile_descriptors(row["increments"]), "movement": human_movement}}
            for arm in ("math", "hybrid"):
                try:
                    candidate_times, candidate_positions = clip_transition_samples(times, case[f"{arm}_positions"], start, end)
                    label = _extract_phase_label_arrays(route, candidate_times, candidate_positions)
                    movement = movement_descriptors(candidate_times, candidate_positions)[0]
                    record[arm] = {"phase": profile_descriptors(np.exp(label["log_phase_increments"])), "movement": movement}
                except ValueError as error:
                    failures[arm + ":" + str(error)] += 1
                    record[arm] = None
            valid.append(record)
        common = [record for record in valid if record["math"] is not None and record["hybrid"] is not None]
        if not common:
            continue
        group_result = {"map_md5": map_md5, "map_family": summary["map_family"], "eligible_human_windows": len(valid), "common_valid_windows": len(common), "failure_counts": dict(failures), "learned_segment_count": summary["learned_segment_count"], "fallback_segment_count": summary["fallback_segment_count"], "changed_sample_count": summary["changed_sample_count"]}
        for arm in ("math", "hybrid"):
            phase, phase_parts = distance([r["human"]["phase"] for r in common], [r[arm]["phase"] for r in common], phase_names, phase_scales)
            movement, movement_parts = distance([r["human"]["movement"] for r in common], [r[arm]["movement"] for r in common], movement_names, movement_scales)
            group_result[arm] = {"phase_distance": phase, "movement_distance": movement, "composite_distance": 0.4 * phase + 0.6 * movement, "phase_feature_distances": phase_parts, "movement_feature_distances": movement_parts}
        group_result["hybrid_minus_math_distance"] = group_result["hybrid"]["composite_distance"] - group_result["math"]["composite_distance"]
        results.append(group_result)
        print(f"{map_md5[:10]} common={len(common)}/{len(valid)} math={group_result['math']['composite_distance']:.4f} hybrid={group_result['hybrid']['composite_distance']:.4f} delta={group_result['hybrid_minus_math_distance']:+.4f}", flush=True)

    report = {
        "schema_version": "ordr-circle-pilot-distance-v1",
        "status": "exploratory_scaled_distance_not_osi_v2",
        "source_calibration_sha256": sha(calibration_path),
        "source_validation_sha256": sha(validation_path),
        "calibration_rows": len(cal), "calibration_movement_rows": len(cal_movement),
        "calibration_movement_rejections": dict(cal_rejects),
        "phase_names": phase_names, "movement_names": movement_names,
        "phase_scales": phase_scales, "movement_scales": movement_scales,
        "confirmation_access": False,
        "cases": results,
        "mean_hybrid_minus_math_distance": float(np.mean([row["hybrid_minus_math_distance"] for row in results])) if results else None,
    }
    output_path = case_dir / "circle-pilot-distance.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"mean_delta={report['mean_hybrid_minus_math_distance']} report={output_path}")


if __name__ == "__main__":
    main()
