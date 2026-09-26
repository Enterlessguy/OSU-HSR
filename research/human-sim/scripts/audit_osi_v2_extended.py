"""Development mode and temporal checks; never opens confirmation.

Mode distances are calibrated on independent human halves and real corrupted
cursor samples. Session descriptors test recurrence, variation and serial
dependence of normalized movement shapes. Missing cells stay N/A. This report
does not silently manufacture a full OSI aggregate from partial coverage.
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
from human_sim.io import load_map_plan
from human_sim import scoped_realism
from osi_v2_features import slider_window_features, spinner_window_features, break_window_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/external-ordr-v155"
SEED = 20260926
MODES = ("slider", "spinner", "break_free_roam")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_human(split):
    with gzip.open(OUT / f"development-batches/{split}/human-windows.json.gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


def shape(times, positions, start, end):
    t, xy = scoped_realism.clip_transition_samples(times, positions, start, end)
    _, xy, _ = scoped_realism.common_bandwidth_positions(t, xy)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.sum(speed) <= 1e-9:
        return np.zeros(16)
    return np.interp(np.linspace(0, 1, 16), np.linspace(0, 1, len(speed)), speed / np.mean(speed))


def session_features(shapes, indices):
    if len(shapes) < 12:
        raise ValueError("fewer_than_12_temporal_windows")
    x = np.asarray(shapes)
    adjacent = np.diff(indices) == 1
    difference = np.linalg.norm(np.diff(x, axis=0), axis=1) / np.sqrt(x.shape[1])
    peak = np.argmax(x, axis=1) / (x.shape[1] - 1)
    pairs = np.flatnonzero(adjacent)
    if len(pairs) < 4:
        raise ValueError("fewer_than_4_adjacent_temporal_pairs")
    a, b = peak[pairs], peak[pairs + 1]
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else 0.0
    return {"shape_variation": float(np.mean(np.std(x, axis=0))),
            "adjacent_shape_distance": float(np.median(difference[adjacent])),
            "peak_phase_serial_correlation": corr,
            "peak_phase_iqr": float(np.quantile(peak, .75) - np.quantile(peak, .25)),
            "near_repeated_motif_fraction": float(np.mean(difference[adjacent] < .05))}


def events(plan):
    for obj in plan.objects:
        if obj.kind in ("slider", "spinner"):
            yield obj.kind, obj.index, obj.start_time_ms, obj.end_time_ms, obj
    for prev, obj in zip(plan.objects, plan.objects[1:]):
        if obj.start_time_ms - prev.end_time_ms >= 1000:
            yield "break_free_roam", obj.index, prev.end_time_ms, obj.start_time_ms, obj


def mode_features(mode, times, positions, start, end, obj):
    if mode == "slider":
        return slider_window_features(times, positions, obj)["features"]
    if mode == "spinner":
        return spinner_window_features(times, positions, obj)["features"]
    return break_window_features(times, positions, break_start_ms=start, break_end_ms=end,
                                 next_target=obj.position, next_radius_px=obj.radius)["features"]


def corrupt(name, t, xy, rng):
    if name == "stall":
        return np.tile(xy[0], (len(xy), 1))
    if name == "straight":
        u = (t - t[0]) / max(t[-1] - t[0], 1e-9)
        return xy[0] + u[:, None] * (xy[-1] - xy[0])
    if name == "snap":
        result = np.tile(xy[0], (len(xy), 1))
        result[int(.85 * len(xy)):] = xy[-1]
        return result
    return xy + rng.normal(0, 20, xy.shape)


def balanced_distance(left, right, name, scale):
    def weights(rows):
        counts = Counter(row["map"] for row in rows)
        return [1 / counts[row["map"]] for row in rows]
    return float(wasserstein_distance([r["features"][name] for r in left],
                                     [r["features"][name] for r in right],
                                     weights(left), weights(right)) / scale)


def calibrate_feature_set(human, controls, rng):
    maps = np.asarray(sorted({r["map"] for r in human}))
    if len(maps) < 12:
        return {"status": "N/A_too_few_independent_maps", "maps": len(maps)}
    result = {}
    for name in sorted(human[0]["features"]):
        values = np.asarray([r["features"][name] for r in human])
        scale = max(float(np.quantile(values, .95) - np.quantile(values, .05)), 1e-3)
        null = []
        for _ in range(100):
            chosen = set(rng.permutation(maps)[:len(maps) // 2])
            left = [r for r in human if r["map"] in chosen]
            right = [r for r in human if r["map"] not in chosen]
            null.append(balanced_distance(left, right, name, scale))
        bad = {k: balanced_distance(human, v, name, scale) for k, v in controls.items() if v}
        p95 = float(np.quantile(null, .95))
        result[name] = {"scale": scale, "null_median": float(np.median(null)), "null_p95": p95,
                        "control_distances": bad,
                        "selected": any(d > 1.5 * p95 for d in bad.values())}
    return {"status": "calibrated_development_only", "maps": len(maps), "features": result}


def calibration():
    dest = OUT / "development-batches/calibration/osi-v2-extended-calibration.json"
    identities = {"script": sha(__file__), "features": sha(Path(__file__).with_name("osi_v2_features.py")),
                  "bandwidth": sha(scoped_realism.__file__)}
    if dest.exists():
        saved = json.loads(dest.read_text())
        if saved["source_hashes"] != identities:
            raise ValueError("Existing extended calibration uses different source; use a new artifact identity")
        return saved
    manifest = json.loads((OUT / "development-batches/calibration/window-manifest.json").read_text())
    if len({r["player"] for r in manifest["records"]}) != len(manifest["records"]) or len({r["map_family"] for r in manifest["records"]}) != len(manifest["records"]):
        raise ValueError("Calibration is not player/family disjoint")
    humans = {mode: [] for mode in MODES}
    controls = {mode: defaultdict(list) for mode in MODES}
    temporal = []
    temporal_controls = defaultdict(list)
    rejections = Counter()
    rng = np.random.default_rng(SEED)
    windows = defaultdict(list)
    for row in read_human("calibration"):
        windows[row["map"]].append(row)
    for row in manifest["records"]:
        map_id = row["map"]
        plan = load_map_plan(OUT / f"development-plans/{map_id}.map.ndjson.gz")
        with (OUT / f"development-decoded/{row['replay_sha256']}.ndjson").open(encoding="utf-8") as stream:
            next(stream)
            frames = [json.loads(line) for line in stream if line.strip()]
        times = np.asarray([f["time_ms"] for f in frames])
        xy = np.asarray([[f["x"], f["y"]] for f in frames])
        for mode, index, start, end, obj in events(plan):
            try:
                t, points = scoped_realism.clip_transition_samples(times, xy, start, end)
                observed = mode_features(mode, t, points, start, end, obj)
                bad = {k: mode_features(mode, t, corrupt(k, t, points, rng), start, end, obj)
                       for k in ("stall", "straight", "snap", "jitter")}
                if not all(np.isfinite(v) for d in (observed, *bad.values()) for v in d.values()):
                    raise ValueError("nonfinite_mode_feature")
            except ValueError as error:
                rejections[f"{mode}:{error}"] += 1
                continue
            humans[mode].append({"map": map_id, "features": observed})
            for k, v in bad.items():
                controls[mode][k].append({"map": map_id, "features": v})
        shapes, indices = [], []
        for window in sorted(windows[map_id], key=lambda r: r["source_window_index"]):
            t, points = np.asarray(window["raw_times_ms"]), np.asarray(window["raw_positions"])
            try:
                shapes.append(shape(t, points, t[0], t[-1]))
                indices.append(window["source_window_index"])
            except ValueError:
                continue
        try:
            x = np.asarray(shapes)
            observed = session_features(x, indices)
            bad = {"repeated_template": session_features(np.tile(np.mean(x, axis=0), (len(x), 1)), indices),
                   "shuffled_order": session_features(x[rng.permutation(len(x))], indices),
                   "excess_noise": session_features(x + rng.normal(0, 1, x.shape), indices)}
            temporal.append({"map": map_id, "features": observed})
            for k, v in bad.items():
                temporal_controls[k].append({"map": map_id, "features": v})
        except ValueError as error:
            rejections[str(error)] += 1
        print("calibration", map_id[:10], flush=True)
    report = {"schema_version": "osi-v2-extended-calibration-v1", "confirmation_access": False,
              "validation_movement_access": False, "source_hashes": identities, "seed": SEED,
              "rejections": dict(rejections), "modes": {m: calibrate_feature_set(humans[m], controls[m], rng) for m in MODES},
              "temporal": calibrate_feature_set(temporal, temporal_controls, rng)}
    write(dest, report)
    return report


def score_vectors(human, generated, calibration):
    selected = {k: v for k, v in calibration.get("features", {}).items() if v["selected"]}
    if not selected:
        return None
    values = []
    for key, cal in selected.items():
        d = wasserstein_distance([r[key] for r in human], [r[key] for r in generated]) / cal["scale"]
        anchor = max(float(np.median(list(cal["control_distances"].values()))), 1.5 * cal["null_p95"], cal["null_median"] + .1)
        values.append(max(0, (d - cal["null_median"]) / (anchor - cal["null_median"])))
    return float(100 * np.exp(-np.mean(values)))


def evaluate(tag, cal):
    cases = OUT / "development-cases/validation" / tag
    manifest = json.loads((OUT / "development-batches/validation/window-manifest.json").read_text())
    expected = {r["map"] for r in manifest["records"]}
    if len({r["player"] for r in manifest["records"]}) != len(expected) or len({r["map_family"] for r in manifest["records"]}) != len(expected):
        raise ValueError("Validation maps are not independent")
    mode_windows = defaultdict(list)
    with gzip.open(OUT / "development-batches/validation/osi-v2-mode-windows.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            mode_windows[row["map"]].append(row)
    circles = defaultdict(list)
    for row in read_human("validation"):
        circles[row["map"]].append(row)
    maps = []
    failures = Counter()
    for map_id in sorted(expected):
        summary = json.loads((cases / f"{map_id}-seed42.json").read_text())
        trace = cases / f"{map_id}-seed42.npz"
        if sha(trace) != summary["trace_sha256"] or not summary["timestamp_and_key_identity"]:
            raise ValueError("Paired trace integrity failure")
        with np.load(trace) as arrays:
            times = arrays["times_ms"]
            traces = {arm: arrays[f"{arm}_positions"] for arm in ("math", "hybrid")}
        plan = load_map_plan(OUT / f"development-plans/{map_id}.map.ndjson.gz")
        event_lookup = {(m, i): (s, e, o) for m, i, s, e, o in events(plan)}
        values = {m: {a: [] for a in ("human", "math", "hybrid")} for m in MODES}
        for window in mode_windows[map_id]:
            mode = window["mode"]
            start, end, obj = event_lookup[(mode, window["object_index"])]
            try:
                generated = {arm: mode_features(mode, times, xy, start, end, obj) for arm, xy in traces.items()}
            except ValueError as error:
                failures[f"{mode}:{error}"] += 1
                continue
            values[mode]["human"].append(window["features"])
            for arm in traces:
                values[mode][arm].append(generated[arm])
        cells = {}
        for mode in MODES:
            if not values[mode]["human"]:
                cells[mode] = {"status": "N/A_no_common_support", "windows": 0}
                continue
            scored = {a: score_vectors(values[mode]["human"], values[mode][a], cal["modes"][mode]) for a in traces}
            cells[mode] = {"status": "supported" if all(v is not None for v in scored.values()) else "N/A_uncalibrated",
                           "windows": len(values[mode]["human"]), "scores": scored}
        shapes = {a: [] for a in ("human", "math", "hybrid")}
        indices = []
        for row in sorted(circles[map_id], key=lambda r: r["source_window_index"]):
            t, xy = np.asarray(row["raw_times_ms"]), np.asarray(row["raw_positions"])
            try:
                candidates = {"human": shape(t, xy, t[0], t[-1])}
                for arm, points in traces.items():
                    candidates[arm] = shape(times, points, t[0], t[-1])
            except ValueError:
                continue
            indices.append(row["source_window_index"])
            for arm, value in candidates.items():
                shapes[arm].append(value)
        try:
            features = {a: session_features(v, indices) for a, v in shapes.items()}
            scores = {a: score_vectors([features["human"]], [features[a]], cal["temporal"]) for a in traces}
            cells["temporal"] = {"status": "supported" if all(v is not None for v in scores.values()) else "N/A_uncalibrated",
                                 "windows": len(indices), "features": features, "scores": scores}
        except ValueError as error:
            cells["temporal"] = {"status": f"N/A_{error}"}
        maps.append({"map": map_id, "cells": cells})
        print("validation", map_id[:10], flush=True)
    rng = np.random.default_rng(SEED)
    aggregates = {}
    for mode in (*MODES, "temporal"):
        supported = [r["cells"][mode] for r in maps if r["cells"][mode]["status"] == "supported"]
        if not supported:
            aggregates[mode] = {"status": "N/A"}
            continue
        delta = np.asarray([r["scores"]["hybrid"] - r["scores"]["math"] for r in supported])
        boot = delta[rng.integers(len(delta), size=(20000, len(delta)))].mean(axis=1)
        aggregates[mode] = {"supported_maps": len(supported), "unsupported_maps": len(maps)-len(supported),
                            "math": float(np.mean([r["scores"]["math"] for r in supported])),
                            "hybrid": float(np.mean([r["scores"]["hybrid"] for r in supported])),
                            "paired_delta": float(delta.mean()), "paired_95": np.quantile(boot, [.025, .975]).tolist()}
    report = {"schema_version": "osi-v2-extended-development-audit-v1", "status": "opened_validation_not_confirmation",
              "confirmation_access": False, "full_osi_v2_score_available": False, "case_tag": tag,
              "source_sha256": sha(__file__), "calibration_sha256": sha(OUT / "development-batches/calibration/osi-v2-extended-calibration.json"),
              "paired_cases_report_sha256": sha(cases / "report.json"), "failures": dict(failures),
              "components": aggregates, "maps": maps}
    write(cases / "osi-v2-extended-audit.json", report)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-tag", required=True)
    args = parser.parse_args()
    if not args.case_tag or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.case_tag):
        raise ValueError("Invalid case tag")
    evaluate(args.case_tag, calibration())
