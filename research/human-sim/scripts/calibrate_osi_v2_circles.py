"""Calibrate draft OSI V2 circle features on calibration humans only.

Produces independent-player human/human distances and deliberately corrupted
controls. It does not inspect validation scores or open confirmation payloads.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
GROUPS = ("speed_phase", "spatial_contact", "continuity", "coupled_motor")
CONTROLS = ("straight", "smoothstep", "deadline_snap", "alternating_jitter")
SEED = 20260924


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corrupt(name: str, times: np.ndarray, positions: np.ndarray, start: float, end: float,
            origin: np.ndarray, target: np.ndarray, radius: float) -> np.ndarray:
    u = np.clip((times - start) / (end - start), 0.0, 1.0)
    if name == "straight":
        return origin + u[:, None] * (target - origin)
    if name == "smoothstep":
        progress = 3.0 * u**2 - 2.0 * u**3
        return origin + progress[:, None] * (target - origin)
    if name == "deadline_snap":
        progress = np.clip((u - .75) / .25, 0.0, 1.0)
        return origin + progress[:, None] * (target - origin)
    if name == "alternating_jitter":
        route = target - origin
        lateral = np.asarray([-route[1], route[0]]) / max(np.linalg.norm(route), 1e-9)
        return positions + ((-1.0) ** np.arange(len(times)) * .18 * radius)[:, None] * lateral
    raise ValueError(name)


def flatten(feature: dict) -> dict[str, float]:
    return {f"{group}.{name}": float(value) for group in GROUPS for name, value in feature[group].items()}


def balanced_weights(players: list[str]) -> np.ndarray:
    count = Counter(players)
    return np.asarray([1.0 / count[player] for player in players], dtype=float)


def distance(left: list[dict], right: list[dict], name: str, scale: float) -> float:
    return float(wasserstein_distance(
        [row["features"][name] for row in left],
        [row["features"][name] for row in right],
        u_weights=balanced_weights([row["player"] for row in left]),
        v_weights=balanced_weights([row["player"] for row in right]),
    ) / scale)


def main() -> None:
    source_path = OUT / "development-batches/calibration/human-windows.json.gz"
    with gzip.open(source_path, "rt", encoding="utf-8") as stream:
        source = json.load(stream)
    plans = {}
    human = []
    controls: dict[str, list[dict]] = {name: [] for name in CONTROLS}
    rejected = Counter()
    supported_by_map = Counter()
    for row in source:
        map_md5 = row["map"]
        if map_md5 not in plans:
            plans[map_md5] = load_map_plan(OUT / "development-plans" / f"{map_md5}.map.ndjson.gz")
        plan = plans[map_md5]
        index = int(row["source_window_index"])
        route = transition_route(plan, index, skill_group="competitive", record_id=f"osi-v2-cal:{map_md5}:{index}")
        start, end = route.start_time_ms, route.start_time_ms + route.duration_ms
        origin = plan.objects[index - 1]
        target = plan.objects[index]
        kwargs = dict(event_start_ms=start, event_end_ms=end, origin=origin.position,
                      target=target.position, target_radius_px=target.radius)
        times = np.asarray(row["raw_times_ms"], dtype=float)
        positions = np.asarray(row["raw_positions"], dtype=float)
        try:
            observed = flatten(circle_window_features(times, positions, **kwargs))
            converted = {
                name: flatten(circle_window_features(times, corrupt(
                    name, times, positions, start, end,
                    np.asarray([origin.position.x, origin.position.y]),
                    np.asarray([target.position.x, target.position.y]),
                    float(target.radius)), **kwargs))
                for name in CONTROLS
            }
            if any(not np.isfinite(value) for candidate in (observed, *converted.values()) for value in candidate.values()):
                raise ValueError("nonfinite feature")
        except ValueError as error:
            rejected[str(error)] += 1
            continue
        record = {"player": row["player"], "map": map_md5, "features": observed}
        human.append(record)
        supported_by_map[map_md5] += 1
        for name in CONTROLS:
            controls[name].append({"player": row["player"], "map": map_md5, "features": converted[name]})
    if len({row["player"] for row in human}) < 20:
        raise ValueError("Too few independent calibration players")
    names = sorted(human[0]["features"])
    scales = {name: max(float(np.quantile([row["features"][name] for row in human], .95) -
                              np.quantile([row["features"][name] for row in human], .05)), 1e-3)
              for name in names}
    players = np.asarray(sorted({row["player"] for row in human}))
    rng = np.random.default_rng(SEED)
    null = defaultdict(list)
    for _ in range(100):
        selected = set(rng.permutation(players)[:len(players) // 2])
        left = [row for row in human if row["player"] in selected]
        right = [row for row in human if row["player"] not in selected]
        for name in names:
            null[name].append(distance(left, right, name, scales[name]))
    calibration = {}
    for name in names:
        distances = {control: distance(human, controls[control], name, scales[name]) for control in CONTROLS}
        calibration[name] = {
            "scale": scales[name],
            "human_human_median": float(np.median(null[name])),
            "human_human_p95": float(np.quantile(null[name], .95)),
            "corruption_distances": distances,
            "separates_any_control_at_1p5x_null_p95": any(value > 1.5 * np.quantile(null[name], .95) for value in distances.values()),
        }
    report = {
        "schema_version": "osi-v2-circle-calibration-probe-v1",
        "status": "draft_calibration_not_frozen_score",
        "source_human_sha256": sha(source_path),
        "feature_source_sha256": sha(Path(__file__).with_name("osi_v2_features.py")),
        "bandwidth_source_sha256": sha(Path(scoped_realism.__file__)),
        "calibrator_source_sha256": sha(Path(__file__)),
        "seed": SEED, "confirmation_access": False, "validation_movement_access": False,
        "attempted_windows": len(source), "common_supported_windows": len(human),
        "players": len(players), "maps": len(plans), "rejections": rejected,
        "supported_windows_by_map": {map_md5: supported_by_map[map_md5] for map_md5 in sorted(plans)},
        "minimum_supported_windows_per_map": int(np.floor(np.quantile([supported_by_map[map_md5] for map_md5 in plans], .1))),
        "minimum_support_rule": "floor of the calibration-map 10th percentile of common supported windows; below this circle cell is N/A",
        "features": calibration,
    }
    path = OUT / "development-batches/calibration/osi-v2-circle-calibration-probe.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "features"}, indent=2))
    print("separating_features", sum(value["separates_any_control_at_1p5x_null_p95"] for value in calibration.values()), "/", len(calibration))


if __name__ == "__main__":
    main()
