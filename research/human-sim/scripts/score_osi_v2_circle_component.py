"""Development-only calibrated OSI V2 circle component.

This is a component diagnostic, not an overall OSI V2 score: temporal,
slider/spinner/break, matched-cadence gate and perceptual checks remain open.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from human_sim.execution_training import transition_route
from human_sim.io import load_map_plan
from human_sim import scoped_realism
from osi_v2_features import circle_window_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/external-ordr-v155"
WEIGHTS = {"speed_phase": 20.0, "spatial_contact": 20.0,
           "continuity": 20.0, "coupled_motor": 10.0}
SEED = 20260924


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-tag", required=True)
    args = parser.parse_args()
    case_root = OUT / "development-cases/validation" / args.case_tag
    cal_path = OUT / "development-batches/calibration/osi-v2-circle-calibration-probe.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if cal["schema_version"] != "osi-v2-circle-calibration-probe-v1" or cal["confirmation_access"]:
        raise ValueError("Calibration artifact is not the training-independent draft")
    if cal["bandwidth_source_sha256"] != sha(Path(scoped_realism.__file__)):
        raise ValueError("Calibration and scoring bandwidth implementations differ")
    minimum_map_windows = int(cal["minimum_supported_windows_per_map"])
    if minimum_map_windows < 1:
        raise ValueError("Calibration yielded no usable map support threshold")
    selected = {key: value for key, value in cal["features"].items()
                if value["separates_any_control_at_1p5x_null_p95"]}
    if any(not any(key.startswith(group + ".") for key in selected) for group in WEIGHTS):
        raise ValueError("Uncalibrated required circle criterion")
    human_path = OUT / "development-batches/validation/human-windows.json.gz"
    with gzip.open(human_path, "rt", encoding="utf-8") as stream:
        human = json.load(stream)
    by_map = defaultdict(list)
    for row in human:
        by_map[row["map"]].append(row)
    manifest = json.loads((OUT / "development-batches/validation/window-manifest.json").read_text(encoding="utf-8"))
    expected_maps = {row["map"] for row in manifest["records"]}
    failures = Counter()
    maps = []
    model_hashes = set()
    adapter_hashes = set()
    trace_hashes = []
    learned = fallback = changed = 0
    for summary_path in sorted(case_root.glob("*-seed42.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        map_md5 = summary["map_md5"]
        trace_path = case_root / f"{map_md5}-seed42.npz"
        if summary["trace_sha256"] != sha(trace_path) or not summary["timestamp_and_key_identity"] or summary["time_basis"] != "absolute_beatmap_ms":
            raise ValueError(f"Paired trace integrity failed: {map_md5}")
        model_hashes.add(summary["model_file_sha256"])
        adapter_hashes.add(summary["adapter_source_sha256"])
        trace_hashes.append(summary["trace_sha256"])
        learned += summary["learned_segment_count"]
        fallback += summary["fallback_segment_count"]
        changed += summary["changed_sample_count"]
        plan = load_map_plan(OUT / "development-plans" / f"{map_md5}.map.ndjson.gz")
        with np.load(trace_path) as arrays:
            times = arrays["times_ms"]
            positions = {arm: arrays[f"{arm}_positions"] for arm in ("math", "hybrid")}
        values = {arm: [] for arm in ("human", "math", "hybrid")}
        for row in by_map[map_md5]:
            index = int(row["source_window_index"])
            route = transition_route(plan, index, skill_group="competitive", record_id=f"osi-v2-circle:{map_md5}:{index}")
            origin, target = plan.objects[index - 1:index + 1]
            kwargs = dict(event_start_ms=route.start_time_ms,
                          event_end_ms=route.start_time_ms + route.duration_ms,
                          origin=origin.position, target=target.position,
                          target_radius_px=target.radius)
            try:
                candidate = {"human": circle_window_features(row["raw_times_ms"], row["raw_positions"], **kwargs)}
                for arm in ("math", "hybrid"):
                    candidate[arm] = circle_window_features(times, positions[arm], **kwargs)
                if any(not np.isfinite(float(candidate[arm][group][name]))
                       for arm in candidate for key in selected
                       for group, name in (key.split(".", 1),)):
                    raise ValueError("nonfinite selected feature")
            except ValueError as error:
                failures[str(error)] += 1
                continue
            for arm in values:
                values[arm].append(candidate[arm])
        result = {
            "map_md5": map_md5, "map_family": summary["map_family"],
            "attempted_windows": len(by_map[map_md5]), "common_supported_windows": len(values["human"]),
            "learned_segments": summary["learned_segment_count"],
            "fallback_segments": summary["fallback_segment_count"],
        }
        if len(values["human"]) < minimum_map_windows:
            result["status"] = "circle_support_below_calibration_minimum"
            maps.append(result)
            continue
        result["status"] = "supported"
        for arm in ("math", "hybrid"):
            criterion = {}
            feature_distances = {}
            for group in WEIGHTS:
                distances = []
                for key, item in selected.items():
                    if not key.startswith(group + "."):
                        continue
                    _, name = key.split(".", 1)
                    raw = float(wasserstein_distance(
                        [window[group][name] for window in values["human"]],
                        [window[group][name] for window in values[arm]]))
                    normalized = raw / item["scale"]
                    null = item["human_human_median"]
                    control = float(np.median(list(item["corruption_distances"].values())))
                    anchor = max(control, 1.5 * item["human_human_p95"], null + .1)
                    calibrated = max(0.0, (normalized - null) / (anchor - null))
                    distances.append(calibrated)
                    feature_distances[key] = {"raw": raw, "normalized": normalized, "calibrated_excess": calibrated}
                criterion[group] = float(np.mean(distances))
            aggregate = sum(WEIGHTS[group] * criterion[group] for group in WEIGHTS) / sum(WEIGHTS.values())
            result[arm] = {"criterion_calibrated_excess": criterion,
                           "calibrated_excess_distance": aggregate,
                           "circle_component_score": float(100.0 * np.exp(-aggregate)),
                           "feature_distances": feature_distances}
        result["hybrid_minus_math_score"] = result["hybrid"]["circle_component_score"] - result["math"]["circle_component_score"]
        maps.append(result)
        print(f"{map_md5[:10]} windows={len(values['human'])}/{len(by_map[map_md5])} score math={result['math']['circle_component_score']:.3f} hybrid={result['hybrid']['circle_component_score']:.3f}", flush=True)
    scored = [row for row in maps if row["status"] == "supported"]
    if not scored:
        raise ValueError("No scoreable circle maps")
    if len({row["map_family"] for row in maps}) != len(maps):
        raise ValueError("Map families are not independent for bootstrap")
    delta = np.asarray([row["hybrid_minus_math_score"] for row in scored])
    rng = np.random.default_rng(SEED)
    boot = np.mean(delta[rng.integers(0, len(delta), (20000, len(delta)))], axis=1)
    result = {
        "schema_version": "osi-v2-circle-component-development-v1",
        "status": "draft_circle_component_only_not_overall_osi_v2",
        "confirmation_access": False,
        "case_tag": args.case_tag,
        "expected_maps": len(expected_maps), "generated_maps": len(maps),
        "all_generated_maps_present": {row["map_md5"] for row in maps} == expected_maps,
        "scored_circle_maps": len(scored),
        "minimum_supported_windows_per_map": minimum_map_windows,
        "unsupported_circle_maps": [row["map_md5"] for row in maps if row["status"] != "supported"],
        "attempted_windows": sum(row["attempted_windows"] for row in maps),
        "common_supported_windows": sum(row["common_supported_windows"] for row in maps),
        "feature_failures": failures,
        "model_file_sha256": sorted(model_hashes), "adapter_source_sha256": sorted(adapter_hashes),
        "paired_trace_hash_collection_sha256": hashlib.sha256("\n".join(sorted(trace_hashes)).encode()).hexdigest(),
        "learned_segments_all_maps": learned, "fallback_segments_all_maps": fallback,
        "changed_samples_all_maps": changed,
        "calibration_sha256": sha(cal_path), "feature_source_sha256": sha(Path(__file__).with_name("osi_v2_features.py")),
        "bandwidth_source_sha256": sha(Path(scoped_realism.__file__)),
        "scorer_source_sha256": sha(Path(__file__)),
        "math_mean_supported_map_score": float(np.mean([row["math"]["circle_component_score"] for row in scored])),
        "hybrid_mean_supported_map_score": float(np.mean([row["hybrid"]["circle_component_score"] for row in scored])),
        "hybrid_minus_math_mean_supported_map_score": float(np.mean(delta)),
        "hybrid_minus_math_bootstrap_95": [float(x) for x in np.quantile(boot, [.025, .975])],
        "full_osi_v2_score_available": False,
        "maps": maps,
    }
    path = case_root / "osi-v2-circle-component.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "maps"}, indent=2))


if __name__ == "__main__":
    main()
