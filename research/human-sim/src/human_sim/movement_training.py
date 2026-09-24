"""Offline, CPU-first movement experiment. Never discovers local replay files.

Only explicitly listed, provenance-reviewed human demonstrations are admitted.
The pilot learns joint trajectory coefficients, not a real-time input policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from .dataset import _read_replay
from .io import load_map_plan, repository_git_commit

SKILL_GROUPS = ("beginner", "intermediate", "expert", "competitive")
SPLITS = ("train", "validation", "test")
AUTOMATION_MODS = {"HSR", "AT", "AP", "RX", "CN"}
FEATURES = ("log_duration_ms", "log_distance_px", "turn_cos", "turn_sin",
            "beginner", "intermediate", "expert", "competitive")
TAU = np.linspace(0.0, 1.0, 32)
BASE = 10 * TAU**3 - 15 * TAU**4 + 6 * TAU**5
# Residuals vanish with their first two derivatives at both ends, so later
# planner integration can preserve shared Hermite boundary derivatives.
BASIS = np.column_stack([64 * TAU**3 * (1 - TAU)**3 * (2 * TAU - 1)**i for i in range(4)])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus(manifest_path: str) -> tuple[list[dict], dict]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected corpus schema_version 1")
    records = manifest.get("replays", [])
    rows = []
    seen = set()
    entity_splits: dict[tuple[str, str], str] = {}
    inventory = []
    skipped = Counter()
    for record in records:
        if record.get("human_verified") is not True or not record.get("source"):
            raise ValueError("Every replay needs reviewed human provenance and a source")
        split, group = record["split"], record["skill_group"]
        if split not in SPLITS or group not in SKILL_GROUPS:
            raise ValueError("Invalid split or skill_group")
        if not record.get("skill_evidence"):
            raise ValueError("Record skill evidence; rank alone is not a calibrated skill percentile")
        replay_path = (path.parent / record["replay"]).resolve()
        map_path = (path.parent / record["map_plan"]).resolve()
        replay_hash = _digest(replay_path)
        if replay_hash in seen:
            raise ValueError("Duplicate replay content in corpus")
        seen.add(replay_hash)
        header, frames = _read_replay(replay_path)
        mods = {str(mod).upper() for mod in header.get("mods", [])}
        if header.get("synthetic") or mods & AUTOMATION_MODS:
            raise ValueError("Synthetic/automated replay is forbidden in the human corpus")
        plan = load_map_plan(str(map_path))
        # Pilot restriction avoids silently training on incorrect HR/rate transforms.
        if mods or plan.mods or abs(plan.clock_rate - 1.0) > 1e-9:
            raise ValueError("Movement pilot accepts NM only; mod transforms need separate validation")
        if header.get("beatmap_md5") != plan.beatmap_md5:
            raise ValueError("Replay and map plan do not match")
        player = str(header.get("player_hash", ""))
        if not player:
            raise ValueError("Missing player hash")
        for entity in (("player", player), ("map", plan.beatmap_md5)):
            previous = entity_splits.setdefault(entity, split)
            if previous != split:
                raise ValueError("Player/map leakage across corpus splits")
        times = np.array([float(frame["time_ms"]) for frame in frames])
        xy = np.array([[float(frame["x"]), float(frame["y"])] for frame in frames])
        if len(times) < 4 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(xy)):
            raise ValueError("Replay has insufficient or non-finite samples")
        if np.any(np.diff(times) < 0):
            raise ValueError("Replay timestamps go backwards")
        # Keep the last of equal-time frames; do not invent motion between them.
        keep = np.r_[np.diff(times) > 0, True]
        times, xy = times[keep], xy[keep]
        # Reject duplicate frame streams even when headers differ.
        stream_hash = hashlib.sha256(np.column_stack((times, xy)).tobytes()).hexdigest()
        if stream_hash in seen:
            raise ValueError("Duplicate movement stream in corpus")
        seen.add(stream_hash)
        inventory.append({"replay_sha256": replay_hash, "map_sha256": _digest(map_path),
                          "player": player, "map": plan.beatmap_md5, "split": split, "skill_group": group})
        for index in range(1, len(plan.objects)):
            left, right = plan.objects[index - 1:index + 1]
            if left.kind != "circle" or right.kind != "circle":
                skipped["non_circle_transition"] += 1
                continue
            begin, end = left.start_time_ms, right.start_time_ms
            duration = end - begin
            if not 20 <= duration <= 600:
                skipped["outside_pilot_duration"] += 1
                continue
            lo = max(0, int(np.searchsorted(times, begin)) - 1)
            hi = min(len(times), int(np.searchsorted(times, end, side="right")) + 1)
            if (begin < times[0] or end > times[-1] or hi - lo < 4
                    or np.max(np.diff(times[lo:hi])) > 75):
                skipped["sparse_or_missing_frames"] += 1
                continue
            a = np.array([left.position.x, left.position.y])
            b = np.array([right.position.x, right.position.y])
            direction = b - a
            distance = float(np.linalg.norm(direction))
            if distance < 8:
                skipped["stacked"] += 1
                continue
            direction /= distance
            rotation = np.column_stack((direction, [-direction[1], direction[0]]))
            sample_times = begin + TAU * duration
            positions = np.column_stack([np.interp(sample_times, times, xy[:, axis]) for axis in range(2)])
            # Retain imperfect movement, including misses. The endpoint residual
            # is separate from the within-segment shape in this pilot.
            local = (positions - positions[0]) @ rotation / distance
            residual = local - BASE[:, None] * local[-1]
            coefficients = np.linalg.lstsq(BASIS, residual, rcond=None)[0].ravel()
            following = plan.objects[index + 1] if index + 1 < len(plan.objects) else right
            outgoing = np.array([following.position.x, following.position.y]) - b
            outgoing /= max(np.linalg.norm(outgoing), 1e-9)
            features = [np.log1p(duration), np.log1p(distance), float(direction @ outgoing),
                        float(direction[0] * outgoing[1] - direction[1] * outgoing[0]),
                        *[float(group == label) for label in SKILL_GROUPS]]
            rows.append({"x": features, "y": coefficients, "split": split, "skill_group": group,
                         "player": player, "map": plan.beatmap_md5, "duration_ms": duration})
    report = {"schema_version": 1, "manifest_sha256": _digest(path), "replays": len(records),
              "windows": len(rows), "inventory": inventory, "skipped": dict(skipped),
              "coverage": {f"{split}/{group}": sum(r["split"] == split and r["skill_group"] == group for r in rows)
                           for split in SPLITS for group in SKILL_GROUPS},
              "status": "ready_for_coverage_check" if rows else "awaiting_human_replays"}
    return rows, report


def train(manifest_path: str, output: str, seed: int = 42) -> dict:
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor

    destination = Path(output)
    model_path = destination / "movement.joblib"
    if model_path.exists():
        raise ValueError("Use a new versioned output directory; never overwrite a trained candidate")
    rows, report = load_corpus(manifest_path)
    for split in SPLITS:
        for group in SKILL_GROUPS:
            subset = [r for r in rows if r["split"] == split and r["skill_group"] == group]
            minimum = 100 if split == "train" else 25
            if len(subset) < minimum or len({r["player"] for r in subset}) < 2:
                raise ValueError(f"Insufficient {split}/{group}: need {minimum} windows from at least two players")
    training = [r for r in rows if r["split"] == "train"]
    x = np.array([r["x"] for r in training])
    y = np.array([r["y"] for r in training])
    # Balance skill groups and prolific players instead of letting the largest
    # download source define the default movement style.
    counts = Counter((r["skill_group"], r["player"]) for r in training)
    players_per_group = Counter(group for group, _ in counts)
    weights = np.array([1 / (counts[r["skill_group"], r["player"]] * players_per_group[r["skill_group"]]) for r in training])
    model = ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, max_depth=16,
                               random_state=seed, n_jobs=2)
    model.fit(x, y, sample_weight=weights)
    metrics = {}
    for split in ("validation", "test"):
        for group in SKILL_GROUPS:
            subset = [r for r in rows if r["split"] == split and r["skill_group"] == group]
            truth = np.array([r["y"] for r in subset]).reshape(-1, 4, 2)
            predicted = model.predict(np.array([r["x"] for r in subset])).reshape(-1, 4, 2)
            error = np.einsum("tk,nkd->ntd", BASIS, predicted - truth)
            baseline = np.einsum("tk,nkd->ntd", BASIS, truth)
            metrics[f"{split}/{group}"] = {
                "windows": len(subset), "normalized_shape_rmse": float(np.sqrt(np.mean(error**2))),
                "minimum_jerk_shape_rmse": float(np.sqrt(np.mean(baseline**2)))}
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 1, "model": model, "features": FEATURES,
                 "basis": "endpoint_c2_polynomial_4", "pilot": "NM circle 20-600ms",
                 "training_manifest_sha256": report["manifest_sha256"]}, model_path)
    report.update({"status": "experimental_not_promoted", "seed": seed, "git_commit": repository_git_commit(),
                   "model_sha256": _digest(model_path), "metrics": metrics,
                   "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                   "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}},
                   "limitation": "Shape-regression baseline only; not integrated into planner, not a realism certification."})
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "train"))
    parser.add_argument("manifest")
    parser.add_argument("output", help="Audit JSON path or new training run directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "audit":
        _, report = load_corpus(args.manifest)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        report = train(args.manifest, args.output, args.seed)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
