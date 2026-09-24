"""Human-only execution-phase corpus, training, and candidate registration.

The manifest is the only source of replay paths.  This module never searches a
directory, never admits synthetic traces, and never trains on the simulator's
own output.  Its labels are normalized phase increments along a supplied
mathematical route; no observed XY path is a model target.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset import _read_replay
from .execution import (
    EXECUTION_FEATURE_VERSION,
    EXECUTION_NORMALIZATION_VERSION,
    FrozenRoute,
    canonical_sha256,
)
from .io import load_map_plan, repository_git_commit


SKILL_GROUPS = ("beginner", "intermediate", "expert", "competitive")
SPLITS = ("train", "validation", "test")
AUTOMATION_MODS = frozenset({"HSR", "AT", "AP", "RX", "CN"})
PHASE_POINTS = 32
PHASE_INCREMENT_COUNT = PHASE_POINTS - 1
FEATURE_NAMES = (
    "log_duration_ms",
    "log_distance_px",
    "turn_cos",
    "turn_sin",
    "entry_speed_px_s",
    "exit_speed_px_s",
    "mode_circle",
    "effort_level",
    "skill_beginner",
    "skill_intermediate",
    "skill_expert",
    "skill_competitive",
)


class TrainingBlocked(ValueError):
    """Raised when the corpus is not admissible for a human-data candidate."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("Human execution training is blocked by corpus readiness checks")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_manifest_path(manifest: Path, relative: str, field: str) -> Path:
    value = Path(str(relative))
    if value.is_absolute():
        raise ValueError(f"{field} must be relative to the corpus manifest")
    resolved = (manifest.parent / value).resolve()
    root = manifest.parent.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes the corpus directory") from error
    return resolved


def _turn_features(plan: Any, index: int) -> tuple[float, float]:
    left = plan.objects[index - 1]
    right = plan.objects[index]
    following = plan.objects[index + 1] if index + 1 < len(plan.objects) else right
    incoming = np.asarray([right.position.x - left.position.x, right.position.y - left.position.y], dtype=float)
    outgoing = np.asarray([following.position.x - right.position.x, following.position.y - right.position.y], dtype=float)
    incoming /= max(float(np.linalg.norm(incoming)), 1e-12)
    outgoing /= max(float(np.linalg.norm(outgoing)), 1e-12)
    return float(np.dot(incoming, outgoing)), float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])


def transition_route(plan: Any, index: int, *, skill_group: str = "beginner", record_id: str = "") -> FrozenRoute:
    """Build one frozen mathematical circle route for the strict pilot."""
    left = plan.objects[index - 1]
    right = plan.objects[index]
    if left.kind != "circle" or right.kind != "circle":
        raise ValueError("The strict execution pilot only accepts circle transitions")
    duration = right.start_time_ms - left.start_time_ms
    if duration <= 0:
        raise ValueError("transition duration must be positive")
    distance = float(np.hypot(right.position.x - left.position.x, right.position.y - left.position.y))
    if distance < 8:
        raise ValueError("stacked transition has no usable route distance")
    turn_cos, turn_sin = _turn_features(plan, index)
    route_id = record_id or f"{plan.beatmap_md5}:{left.index}->{right.index}"
    return FrozenRoute.from_points(
        route_id,
        [(left.position.x, left.position.y), (right.position.x, right.position.y)],
        mode="circle",
        duration_ms=duration,
        start_time_ms=left.start_time_ms,
        skill_level={"beginner": 25.0, "intermediate": 50.0, "expert": 80.0, "competitive": 98.0}[skill_group],
        metadata={
            "beatmap_md5": plan.beatmap_md5,
            "object_index": right.index,
            "turn_cos": turn_cos,
            "turn_sin": turn_sin,
            "skill_group": skill_group,
            "source": "mathematical_map_plan",
            "geometry_source": "two_point_chord",
            "benchmark_scope": "execution_phase_mechanism_only",
        },
    )


def feature_vector(route: FrozenRoute, skill_group: str | None = None) -> np.ndarray:
    metadata = route.metadata
    group = str(skill_group or metadata.get("skill_group", "beginner"))
    if group not in SKILL_GROUPS:
        raise ValueError(f"Unsupported skill group: {group}")
    values = [
        np.log1p(route.duration_ms),
        np.log1p(route.length_px),
        float(metadata.get("turn_cos", 1.0)),
        float(metadata.get("turn_sin", 0.0)),
        float(np.linalg.norm(route.entry_state.velocity)),
        float(np.linalg.norm(route.exit_state.velocity)),
        float(route.mode == "circle"),
        float(route.effort_level / 100.0),
        *[float(group == label) for label in SKILL_GROUPS],
    ]
    result = np.asarray(values, dtype=float)
    if result.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("execution feature vector is invalid")
    return result


def _projection_label(route: FrozenRoute, times: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, float, float]:
    start = np.asarray(route.points[0], dtype=float)
    end = np.asarray(route.points[-1], dtype=float)
    direction = end - start
    length = max(float(np.linalg.norm(direction)), 1e-12)
    unit = direction / length
    relative = positions - start
    projected = np.clip(relative @ unit / length, 0.0, 1.0)
    cross_track = relative[:, 0] * unit[1] - relative[:, 1] * unit[0]
    differences = np.diff(projected)
    reverse_distance = float(np.sum(np.maximum(-differences, 0.0)))
    # A small correction is retained through the monotone envelope; a loop or
    # large backtrack is rejected instead of being turned into a fake phase jump.
    if reverse_distance > 0.30 or (len(differences) and float(np.min(differences)) < -0.35):
        raise ValueError("ambiguous_backtrack")
    monotone = np.maximum.accumulate(projected)
    if monotone[-1] - monotone[0] < 0.05:
        raise ValueError("no_usable_motion")
    return monotone, float(np.sqrt(np.mean(cross_track**2))), reverse_distance * length


def _extract_phase_label_arrays(
    route: FrozenRoute,
    times: np.ndarray,
    positions: np.ndarray,
    *,
    max_gap_ms: float = 75.0,
) -> dict[str, Any]:
    """Project pre-parsed arrays onto a route without reparsing the frame stream."""
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if len(times) < 4 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError("sparse_or_nonfinite_frames")
    if np.any(np.diff(times) < 0):
        raise ValueError("timestamps_go_backwards")
    keep = np.r_[np.diff(times) > 0, True]
    times, positions = times[keep], positions[keep]
    if len(times) < 4:
        raise ValueError("sparse_or_nonfinite_frames")
    route_start = route.start_time_ms
    route_end = route.start_time_ms + route.duration_ms
    lo = max(0, int(np.searchsorted(times, route_start, side="left")) - 1)
    hi = min(len(times), int(np.searchsorted(times, route_end, side="right")) + 1)
    times = times[lo:hi]
    positions = positions[lo:hi]
    if len(times) < 4 or times[0] > route_start or times[-1] < route_end:
        raise ValueError("window_not_covered")
    gaps = np.diff(times)
    if len(gaps) and float(np.max(gaps)) > max_gap_ms:
        raise ValueError("source_gap_too_large")
    observed_phase, cross_track_rmse, correction_distance = _projection_label(route, times, positions)
    initial = float(observed_phase[0])
    completion = float(np.clip(observed_phase[-1] - initial, 0.0, 1.0))
    normalized = (observed_phase - initial) / max(completion, 1e-9)
    sample_times = route_start + np.linspace(0.0, route.duration_ms, PHASE_POINTS)
    sampled = np.interp(sample_times, times, normalized)
    sampled = np.maximum.accumulate(np.clip(sampled, 0.0, 1.0))
    increments = np.diff(sampled)
    increments = np.maximum(increments, 1e-6)
    increments /= float(increments.sum())
    median_gap = float(np.median(gaps)) if len(gaps) else 0.0
    source_gap_mask = []
    for left_time, right_time in zip(sample_times[:-1], sample_times[1:]):
        relevant = gaps[(times[:-1] <= right_time) & (times[1:] >= left_time)]
        source_gap_mask.append(bool(len(relevant) and median_gap > 0 and float(np.max(relevant)) > median_gap * 1.5))
    return {
        "phase_increments": increments,
        "log_phase_increments": np.log(increments),
        "completion_fraction": completion,
        "cross_track_rmse_px": cross_track_rmse,
        "correction_distance_px": correction_distance,
        "source_gap_fraction": float(np.mean(gaps > max_gap_ms)) if len(gaps) else 0.0,
        "source_gap_mask": source_gap_mask,
        "source_cadence_median_ms": median_gap,
        "source_sample_count": int(len(times)),
        "observed_start_phase": initial,
        "observed_end_phase": float(observed_phase[-1]),
    }


def extract_phase_label(route: FrozenRoute, frames: Sequence[Mapping[str, Any]], *, max_gap_ms: float = 75.0) -> dict[str, Any]:
    """Project an observed execution onto a route and derive phase increments."""
    times = np.asarray([float(frame["time_ms"]) for frame in frames], dtype=float)
    positions = np.asarray([[float(frame["x"]), float(frame["y"])] for frame in frames], dtype=float)
    return _extract_phase_label_arrays(route, times, positions, max_gap_ms=max_gap_ms)


def _stream_hash(times: np.ndarray, positions: np.ndarray) -> str:
    return hashlib.sha256(np.column_stack((times, positions)).tobytes()).hexdigest()


def _empty_report(manifest: Path, records: int = 0) -> dict[str, Any]:
    coverage = {f"{split}/{group}": 0 for split in SPLITS for group in SKILL_GROUPS}
    credentials = bool(os.environ.get("OSU_CLIENT_ID")) and bool(os.environ.get("OSU_CLIENT_SECRET"))
    return {
        "schema_version": 1,
        "manifest_sha256": _digest(manifest),
        "records": records,
        "windows": 0,
        "coverage": coverage,
        "skipped": {},
        "inventory": [],
        "status": "awaiting_human_replays",
        "human_data_training": "blocked_no_verified_human_data",
        "credentials_configured": credentials,
        "forbidden_sources": ["local HSR runs", "human.parquet from generated runs", "synthetic hybrid output"],
        "next_command": "human-sim execution-train research/human-sim/training/corpus.json research/human-sim/output/training/candidate-001 --seed 42",
        "player_hash_salt_id_present": False,
    }


def load_execution_corpus(manifest_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only explicitly listed, provenance-reviewed human executions."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Expected execution corpus schema_version 1")
    records = list(manifest.get("replays", []))
    report = _empty_report(manifest_path, len(records))
    report["player_hash_salt_id_present"] = bool(str(manifest.get("player_hash_salt_id", "")))
    if not records:
        return [], report
    salt_id = str(manifest.get("player_hash_salt_id", ""))
    if not salt_id:
        raise ValueError("non-empty human corpus must declare one stable player_hash_salt_id")
    report["player_hash_salt_id_present"] = True
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen_replays: set[str] = set()
    seen_streams: set[str] = set()
    entity_splits: dict[tuple[str, str], str] = {}
    inventory: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if record.get("human_verified") is not True or not record.get("source"):
            raise ValueError(f"record {record_index} lacks reviewed human provenance")
        split = str(record.get("split", ""))
        group = str(record.get("skill_group", ""))
        if split not in SPLITS or group not in SKILL_GROUPS:
            raise ValueError(f"record {record_index} has invalid split or skill_group")
        if not record.get("skill_evidence"):
            raise ValueError(f"record {record_index} lacks dated skill evidence")
        if str(record.get("source_type", "human_replay")).lower() in {"synthetic", "generated", "hsr"}:
            raise ValueError("generated or synthetic source is forbidden in the human corpus")
        replay_path = _resolve_manifest_path(manifest_path, record.get("replay", ""), "replay")
        map_path = _resolve_manifest_path(manifest_path, record.get("map_plan", ""), "map_plan")
        replay_hash = _digest(replay_path)
        map_hash = _digest(map_path)
        if replay_hash in seen_replays:
            raise ValueError("duplicate replay content in corpus")
        seen_replays.add(replay_hash)
        header, frames = _read_replay(replay_path)
        mods = {str(mod).upper() for mod in header.get("mods", [])}
        if bool(header.get("synthetic")) or mods & AUTOMATION_MODS:
            raise ValueError("Synthetic/automated replay is forbidden in the human corpus")
        plan = load_map_plan(map_path)
        replay_map = str(header.get("beatmap_md5", "")).lower()
        if not replay_map or replay_map != plan.beatmap_md5.lower():
            raise ValueError("Replay and map plan do not match")
        if plan.mods or abs(plan.clock_rate - 1.0) > 1e-9 or mods:
            skipped["unsupported_mod_transform"] += 1
            continue
        player = str(header.get("player_hash") or record.get("player_hash") or "")
        if not player:
            raise ValueError("human replay is missing the salted player hash")
        if record.get("player_hash") and str(record["player_hash"]) != player:
            raise ValueError("manifest player hash does not match replay header")
        for entity in (("player", player), ("map", plan.beatmap_md5)):
            previous = entity_splits.setdefault(entity, split)
            if previous != split:
                raise ValueError("Player/map leakage across corpus splits")
        times = np.asarray([float(frame["time_ms"]) for frame in frames], dtype=float)
        positions = np.asarray([[float(frame["x"]), float(frame["y"])] for frame in frames], dtype=float)
        if len(times) < 4 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
            raise ValueError("replay has insufficient or non-finite samples")
        if np.any(np.diff(times) < 0):
            raise ValueError("replay timestamps go backwards")
        keep = np.r_[np.diff(times) > 0, True]
        times, positions = times[keep], positions[keep]
        stream_hash = _stream_hash(times, positions)
        if stream_hash in seen_streams:
            raise ValueError("duplicate replay/frame stream in corpus")
        seen_streams.add(stream_hash)
        inventory.append({
            "replay_sha256": replay_hash,
            "map_sha256": map_hash,
            "player_hash": player,
            "beatmap_md5": plan.beatmap_md5,
            "split": split,
            "skill_group": group,
            "source": str(record["source"]),
        })
        for index in range(1, len(plan.objects)):
            left, right = plan.objects[index - 1:index + 1]
            duration = right.start_time_ms - left.start_time_ms
            if left.kind != "circle" or right.kind != "circle":
                skipped["unsupported_mode"] += 1
                continue
            if duration < 20 or duration > 600:
                skipped["outside_pilot_duration"] += 1
                continue
            try:
                route = transition_route(plan, index, skill_group=group, record_id=f"{plan.beatmap_md5}:{index}")
                label = extract_phase_label(route, frames)
            except ValueError as error:
                skipped[str(error)] += 1
                continue
            row = {
                "x": feature_vector(route, group),
                "y": np.asarray(label["log_phase_increments"], dtype=float),
                "phase_increments": np.asarray(label["phase_increments"], dtype=float),
                "completion_fraction": float(label["completion_fraction"]),
                "cross_track_rmse_px": float(label["cross_track_rmse_px"]),
                "correction_distance_px": float(label["correction_distance_px"]),
                "source_gap_fraction": float(label["source_gap_fraction"]),
                "source_gap_mask": list(label["source_gap_mask"]),
                "source_cadence_median_ms": float(label["source_cadence_median_ms"]),
                "source_sample_count": int(label["source_sample_count"]),
                "split": split,
                "skill_group": group,
                "player": player,
                "map": plan.beatmap_md5,
                "route_id": route.route_id,
                "duration_ms": duration,
                "distance_px": route.length_px,
                "mode": route.mode,
            }
            rows.append(row)
    report.update({
        "windows": len(rows),
        "coverage": {f"{split}/{group}": sum(row["split"] == split and row["skill_group"] == group for row in rows)
                      for split in SPLITS for group in SKILL_GROUPS},
        "skipped": dict(skipped),
        "inventory": inventory,
        "status": "ready_for_coverage_check" if rows else "awaiting_human_replays",
        "human_data_training": "ready_for_coverage_check" if rows else "blocked_no_verified_human_data",
    })
    return rows, report


def audit_corpus(manifest_path: str | Path) -> dict[str, Any]:
    _, report = load_execution_corpus(manifest_path)
    return report


@dataclass(frozen=True, slots=True)
class ExecutionPhaseModel:
    schema_version: int
    model_version: str
    feature_version: str
    normalization_version: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    phase_points: int
    training_manifest_sha256: str
    seed: int
    supported_modes: tuple[str, ...] = ("circle",)
    model_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.feature_version != EXECUTION_FEATURE_VERSION:
            raise ValueError("unsupported execution model contract")
        if self.normalization_version != EXECUTION_NORMALIZATION_VERSION:
            raise ValueError("unsupported execution normalization")
        if len(self.feature_names) != len(self.means) or len(self.means) != len(self.scales):
            raise ValueError("model feature normalization shape mismatch")
        if self.phase_points < 3 or len(self.coefficients) != self.phase_points - 1:
            raise ValueError("model phase coefficient shape mismatch")
        if any(len(row) != len(self.feature_names) + 1 for row in self.coefficients):
            raise ValueError("model coefficient rows must include an intercept")
        if any(float(scale) <= 0 or not math_is_finite(scale) for scale in self.scales):
            raise ValueError("model scales must be positive and finite")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "normalization_version": self.normalization_version,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": [list(row) for row in self.coefficients],
            "phase_points": self.phase_points,
            "training_manifest_sha256": self.training_manifest_sha256,
            "seed": self.seed,
            "supported_modes": list(self.supported_modes),
        }

    def as_dict(self) -> dict[str, Any]:
        value = self.payload()
        value["model_sha256"] = self.model_sha256 or canonical_sha256(value)
        return value

    def predict_phase_weights(self, route: FrozenRoute, sample_count: int) -> np.ndarray:
        if route.mode not in self.supported_modes:
            raise ValueError(f"unsupported mode: {route.mode}")
        return self.predict_phase_weights_from_features(feature_vector(route), sample_count)

    def predict_phase_weights_from_features(self, raw: Sequence[float], sample_count: int) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (len(self.feature_names),) or not np.all(np.isfinite(raw)):
            raise ValueError("invalid execution feature vector")
        normalized = (raw - np.asarray(self.means)) / np.asarray(self.scales)
        if np.any(np.abs(normalized) > 8.0):
            raise ValueError("out_of_distribution_features")
        augmented = np.concatenate(([1.0], normalized))
        log_weights = np.asarray(self.coefficients, dtype=float) @ augmented
        if sample_count != self.phase_points:
            old_x = np.linspace(0.0, 1.0, len(log_weights))
            new_x = np.linspace(0.0, 1.0, max(sample_count - 1, 1))
            log_weights = np.interp(new_x, old_x, log_weights)
        return np.exp(np.clip(log_weights, -20.0, 20.0))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionPhaseModel":
        payload = dict(value)
        model_hash = str(payload.pop("model_sha256", ""))
        model = cls(
            schema_version=int(payload["schema_version"]),
            model_version=str(payload["model_version"]),
            feature_version=str(payload["feature_version"]),
            normalization_version=str(payload["normalization_version"]),
            feature_names=tuple(payload["feature_names"]),
            means=tuple(float(item) for item in payload["means"]),
            scales=tuple(float(item) for item in payload["scales"]),
            coefficients=tuple(tuple(float(item) for item in row) for row in payload["coefficients"]),
            phase_points=int(payload["phase_points"]),
            training_manifest_sha256=str(payload["training_manifest_sha256"]),
            seed=int(payload["seed"]),
            supported_modes=tuple(payload.get("supported_modes", ["circle"])),
            model_sha256=model_hash,
        )
        expected = canonical_sha256(model.payload())
        if model_hash and model_hash != expected:
            raise ValueError("execution model content hash mismatch")
        return model


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def _coverage_gaps(rows: Sequence[Mapping[str, Any]], minimums: Mapping[str, int] | None = None) -> list[str]:
    minimums = minimums or {"train": 100, "validation": 25, "test": 25}
    missing = []
    for split in SPLITS:
        for group in SKILL_GROUPS:
            subset = [row for row in rows if row["split"] == split and row["skill_group"] == group]
            required = int(minimums[split])
            players = len({row["player"] for row in subset})
            if len(subset) < required or players < 2:
                missing.append(f"{split}/{group}: {len(subset)} windows, {players} players; need {required} windows and 2 players")
    return missing


def _blocked_report(manifest_path: str | Path, report: Mapping[str, Any], reasons: Sequence[str], output: str | Path) -> dict[str, Any]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    value = dict(report)
    value.update({
        "status": "blocked",
        "human_data_training": "blocked",
        "blockers": list(reasons),
        "output_candidate": str(destination),
        "next_command": f"human-sim execution-train {manifest_path} {destination} --seed 42",
    })
    (destination / "blocked-training.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _fit_linear(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    manifest_sha256: str,
    *,
    model_version: str = "execution-phase-ridge-v1",
    ridge_alpha: float = 1e-3,
) -> ExecutionPhaseModel:
    if not model_version:
        raise ValueError("model_version must be non-empty")
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive and finite")
    x = np.asarray([row["x"] for row in rows], dtype=float)
    y = np.asarray([row["y"] for row in rows], dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-9] = 1.0
    normalized = (x - means) / scales
    design = np.column_stack((np.ones(len(normalized)), normalized))
    regularizer = np.eye(design.shape[1]) * ridge_alpha
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ y).T
    model = ExecutionPhaseModel(
        schema_version=1,
        model_version=model_version,
        feature_version=EXECUTION_FEATURE_VERSION,
        normalization_version=EXECUTION_NORMALIZATION_VERSION,
        feature_names=FEATURE_NAMES,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
        phase_points=PHASE_POINTS,
        training_manifest_sha256=manifest_sha256,
        seed=int(seed),
    )
    return ExecutionPhaseModel(**{**model.__dict__, "model_sha256": canonical_sha256(model.payload())}) if hasattr(model, "__dict__") else ExecutionPhaseModel(
        schema_version=model.schema_version,
        model_version=model.model_version,
        feature_version=model.feature_version,
        normalization_version=model.normalization_version,
        feature_names=model.feature_names,
        means=model.means,
        scales=model.scales,
        coefficients=model.coefficients,
        phase_points=model.phase_points,
        training_manifest_sha256=model.training_manifest_sha256,
        seed=model.seed,
        supported_modes=model.supported_modes,
        model_sha256=canonical_sha256(model.payload()),
    )


def _phase_metrics(rows: Sequence[Mapping[str, Any]], model: ExecutionPhaseModel) -> dict[str, Any]:
    if not rows:
        return {
            "windows": 0,
            "model_evaluated_windows": 0,
            "out_of_distribution_rows": 0,
            "phase_rmse": None,
            "minimum_jerk_phase_rmse": None,
            "completion_mae": None,
            "full_population_baseline": {"windows": 0},
            "delivered_fallback": {"windows": 0, "fallback_share": 0.0, "reason_counts": {}},
        }
    errors = []
    baselines = []
    full_population_baselines = []
    completion = []
    out_of_distribution_rows = 0
    baseline = np.diff(10 * np.linspace(0.0, 1.0, PHASE_POINTS) ** 3
                       - 15 * np.linspace(0.0, 1.0, PHASE_POINTS) ** 4
                       + 6 * np.linspace(0.0, 1.0, PHASE_POINTS) ** 5)
    baseline = np.log(np.maximum(baseline, 1e-9))
    baseline -= np.logaddexp.reduce(baseline)
    for row in rows:
        truth = np.array(row["y"], dtype=float, copy=True)
        truth -= np.logaddexp.reduce(truth)
        full_population_baselines.append(baseline - truth)
        try:
            predicted = np.log(model.predict_phase_weights_from_features(row["x"], PHASE_POINTS))
        except ValueError as error:
            if str(error) != "out_of_distribution_features":
                raise
            out_of_distribution_rows += 1
            continue
        predicted -= np.logaddexp.reduce(predicted)
        errors.append(predicted - truth)
        baselines.append(baseline - truth)
        completion.append(float(row["completion_fraction"]))
    return {
        "windows": len(rows),
        "model_evaluated_windows": len(errors),
        "out_of_distribution_rows": out_of_distribution_rows,
        "supported_subset": {
            "windows": len(errors),
            "metrics_are_not_full_population": True,
        },
        "phase_rmse": float(np.sqrt(np.mean(np.asarray(errors) ** 2))) if errors else None,
        "minimum_jerk_phase_rmse": float(np.sqrt(np.mean(np.asarray(baselines) ** 2))) if baselines else None,
        "completion_mean": float(np.mean(completion)) if completion else None,
        "full_population_baseline": {
            "windows": len(rows),
            "phase_rmse": float(np.sqrt(np.mean(np.asarray(full_population_baselines) ** 2))),
            "minimum_jerk_phase_rmse": float(np.sqrt(np.mean(np.asarray(full_population_baselines) ** 2))),
        },
        "delivered_fallback": {
            "windows": out_of_distribution_rows,
            "fallback_share": float(out_of_distribution_rows / len(rows)),
            "reason_counts": {"out_of_distribution_features": out_of_distribution_rows} if out_of_distribution_rows else {},
        },
    }


def save_model(model: ExecutionPhaseModel, destination: Path) -> dict[str, Any]:
    destination.write_text(json.dumps(model.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"model_sha256": model.model_sha256, "file_sha256": _digest(destination)}


def load_phase_model(path: str | Path) -> ExecutionPhaseModel:
    source = Path(path)
    if source.is_dir():
        source = source / "model.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    return ExecutionPhaseModel.from_dict(value)


def train_candidate(
    manifest_path: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    minimums: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    rows, corpus_report = load_execution_corpus(manifest_path)
    missing = _coverage_gaps(rows, minimums)
    destination = Path(output)
    if missing:
        return _blocked_report(manifest_path, corpus_report, missing, destination)
    if destination.exists():
        existing = list(destination.iterdir())
        if any(path.name != "blocked-training.json" for path in existing):
            raise FileExistsError("candidate directories are immutable; choose a new output directory")
        if existing:
            (destination / "blocked-training.json").unlink()
    destination.mkdir(parents=True, exist_ok=True)
    model = _fit_linear([row for row in rows if row["split"] == "train"], seed, corpus_report["manifest_sha256"])
    model_hashes = save_model(model, destination / "model.json")
    validation_metrics = {
        f"{split}/{group}": _phase_metrics(
            [row for row in rows if row["split"] == split and row["skill_group"] == group], model
        )
        for split in ("train", "validation")
        for group in SKILL_GROUPS
    }
    report = {
        "schema_version": 1,
        "status": "experimental_not_promoted",
        "human_data_training": "trained_on_verified_human_replays",
        "model_version": model.model_version,
        "model_sha256": model.model_sha256,
        "model_file_sha256": model_hashes["file_sha256"],
        "training_manifest_sha256": corpus_report["manifest_sha256"],
        "feature_version": EXECUTION_FEATURE_VERSION,
        "normalization_version": EXECUTION_NORMALIZATION_VERSION,
        "seed": seed,
        "git_commit": repository_git_commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ("numpy",)},
        },
        "coverage": corpus_report["coverage"],
        "metrics": validation_metrics,
        "test_set": "sealed; evaluate explicitly at a release milestone",
        "geometry_contract": "phase-only; x(t)=route.position_at_phase(s(t)); no XY residuals",
        "limitation": "compact deterministic CPU baseline; promotion requires held-out human and visual review",
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "candidate.lock").write_text(json.dumps({"immutable": True, "model_sha256": model.model_sha256}, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_candidate(manifest_path: str | Path, candidate: str | Path, *, split: str = "test") -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"invalid evaluation split: {split}")
    rows, corpus_report = load_execution_corpus(manifest_path)
    model = load_phase_model(candidate)
    metrics = {
        f"{split}/{group}": _phase_metrics(
            [row for row in rows if row["split"] == split and row["skill_group"] == group], model
        )
        for group in SKILL_GROUPS
    }
    return {
        "schema_version": 1,
        "status": "evaluation_only",
        "split": split,
        "model_sha256": model.model_sha256,
        "manifest_sha256": corpus_report["manifest_sha256"],
        "metrics": metrics,
        "human_data_training": corpus_report["human_data_training"],
    }


def inspect_candidate(candidate: str | Path) -> dict[str, Any]:
    model = load_phase_model(candidate)
    report_path = Path(candidate) / "report.json" if Path(candidate).is_dir() else None
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path and report_path.exists() else {}
    return {
        "schema_version": model.schema_version,
        "model_version": model.model_version,
        "model_sha256": model.model_sha256,
        "feature_version": model.feature_version,
        "normalization_version": model.normalization_version,
        "supported_modes": list(model.supported_modes),
        "phase_points": model.phase_points,
        "training_manifest_sha256": model.training_manifest_sha256,
        "report_status": report.get("status", "unknown"),
    }


def retrain_candidate(
    manifest_path: str | Path,
    previous_candidate: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    report = train_candidate(manifest_path, output, seed=seed)
    if report.get("status") == "blocked":
        report["retraining"] = "blocked_before_candidate_creation"
        return report
    previous = load_phase_model(previous_candidate)
    rows, _ = load_execution_corpus(manifest_path)
    current = load_phase_model(Path(output) / "model.json")
    previous_metrics = _phase_metrics([row for row in rows if row["split"] == "validation"], previous)
    current_metrics = _phase_metrics([row for row in rows if row["split"] == "validation"], current)
    recommendation = "review_required"
    if current_metrics.get("phase_rmse") is not None and previous_metrics.get("phase_rmse") is not None:
        recommendation = "candidate_better_on_validation" if current_metrics["phase_rmse"] < previous_metrics["phase_rmse"] else "retain_previous"
    report.update({
        "retraining": "full_retrain_old_plus_new_human_data",
        "previous_model_sha256": previous.model_sha256,
        "validation_previous": previous_metrics,
        "validation_current": current_metrics,
        "promotion_recommendation": recommendation,
    })
    (Path(output) / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
