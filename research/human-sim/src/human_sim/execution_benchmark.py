"""Paired pure-math versus execution-only hybrid benchmarking."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .execution import (
    ExecutionTrace,
    FrozenRoute,
    HybridExecutionPlanner,
    ExecutionDiagnostics,
    KeyEvent,
    MathExecutionPlanner,
    _sample_times,
    _smooth_positive,
    _trace_from_phases,
    interval_bucket,
    key_schedule_from_trace,
    warp_math_trace,
)
from .execution_training import load_phase_model
from .planner import PLANNER_VERSION, HumanTracePlanner
from .schemas import HumanProfile


@dataclass(frozen=True, slots=True)
class PairedRouteCase:
    case_id: str
    map_id: str
    skill_group: str
    seed: int
    route: FrozenRoute

    @property
    def interval_bucket(self) -> str:
        return interval_bucket(self.route.duration_ms)


@dataclass(frozen=True, slots=True)
class RuntimeTraceCase:
    """One complete map trace frozen from the current runtime planner."""

    case_id: str
    map_id: str
    skill_group: str
    seed: int
    route: FrozenRoute
    source_frames: tuple[Any, ...]
    segment_boundaries_ms: tuple[float, ...]
    segment_modes: tuple[str, ...]
    planner_version: str


def _runtime_skill_percentile(skill_group: str) -> float:
    return {
        "beginner": 50.0,
        "intermediate": 75.0,
        "expert": 90.0,
        "competitive": 99.0,
    }.get(skill_group, 75.0)


def _runtime_segment_modes(plan: Any, timeline_start_ms: float) -> tuple[tuple[float, ...], tuple[str, ...]]:
    boundaries = [0.0]
    boundaries.extend(float(obj.start_time_ms - timeline_start_ms) for obj in plan.objects)
    boundaries.append(float(plan.objects[-1].end_time_ms - timeline_start_ms + 100.0))
    modes = ["break_free_roam"]
    for left, right in zip(plan.objects, plan.objects[1:]):
        if left.kind == "circle" and right.kind == "circle":
            modes.append("circle")
        elif left.kind == "slider" or right.kind == "slider":
            modes.append("slider")
        elif left.kind == "spinner" or right.kind == "spinner":
            modes.append("spinner")
        else:
            modes.append("break_free_roam")
    modes.append("break_free_roam")
    return tuple(boundaries), tuple(modes)


def runtime_trace_cases_from_map_plan(
    plan: Any,
    *,
    skill_group: str = "intermediate",
    effort_level: float = 100.0,
    seed: int = 42,
    sample_rate_hz: int = 500,
) -> tuple[RuntimeTraceCase, ...]:
    """Freeze samples, keys, and object-mode boundaries from the current planner."""
    percentile = _runtime_skill_percentile(skill_group)
    profile = HumanProfile(
        percentile=percentile,
        seed=seed,
        sample_rate_hz=sample_rate_hz,
        skill_level=percentile,
        effort_level=effort_level,
    )
    frames = tuple(HumanTracePlanner(plan, profile).generate())
    if len(frames) < 2:
        return ()
    frame_origin_ms = float(frames[0].time_us) / 1000.0
    times_ms = tuple(float(frame.time_us) / 1000.0 - frame_origin_ms for frame in frames)
    points = tuple((float(frame.x), float(frame.y)) for frame in frames)
    timeline_start_ms = float(plan.objects[0].start_time_ms - 1500.0)
    boundaries, modes = _runtime_segment_modes(plan, timeline_start_ms)
    route_id = f"runtime:{plan.beatmap_sha256}:{seed}"
    route = FrozenRoute.from_timed_samples(
        route_id,
        times_ms,
        points,
        mode="mixed",
        key_schedule=key_schedule_from_trace(frames),
        metadata={
            "map_id": plan.beatmap_md5,
            "geometry_source": "current_runtime_planner_samples",
            "key_source": "current_runtime_planner_trace",
            "planner_version": PLANNER_VERSION,
            "sample_rate_hz": sample_rate_hz,
            "segment_boundaries_ms": list(boundaries),
            "segment_modes": list(modes),
            "benchmark_scope": "runtime_planner_trace_execution_validation",
        },
    )
    return (
        RuntimeTraceCase(
            case_id=route_id,
            map_id=plan.beatmap_md5,
            skill_group=skill_group,
            seed=seed,
            route=route,
            source_frames=frames,
            segment_boundaries_ms=boundaries,
            segment_modes=modes,
            planner_version=PLANNER_VERSION,
        ),
    )


def _key_schedule(duration_ms: float, key: int) -> tuple[KeyEvent, ...]:
    return (KeyEvent(0.0, key, True), KeyEvent(float(duration_ms), key, False))


def cases_from_map_plan(
    plan: Any,
    *,
    skill_group: str = "intermediate",
    effort_level: float = 100.0,
    seed: int = 42,
) -> tuple[PairedRouteCase, ...]:
    """Build chord/object-geometry routes for a diagnostic benchmark only.

    These routes are not samples from :class:`HumanTracePlanner` and must not
    be used as runtime ORI evidence or human-data substitutes.
    """
    cases: list[PairedRouteCase] = []
    for index in range(1, len(plan.objects)):
        left, right = plan.objects[index - 1:index + 1]
        duration = float(right.start_time_ms - left.start_time_ms)
        if duration <= 0:
            continue
        if left.kind == "circle" and right.kind == "circle":
            points = [(left.position.x, left.position.y), (right.position.x, right.position.y)]
            mode = "circle"
        elif left.kind == "slider" or right.kind == "slider":
            points = [(left.position.x, left.position.y)]
            if left.path_samples:
                points.extend((point.x, point.y) for point in left.path_samples)
            points.append((left.end_position.x, left.end_position.y))
            points.append((right.position.x, right.position.y))
            mode = "slider"
        elif right.kind == "spinner":
            points = [(left.position.x, left.position.y), (right.position.x + 1.0, right.position.y)]
            mode = "spinner"
        else:
            points = [(left.position.x, left.position.y), (right.position.x, right.position.y)]
            mode = "idle"
        if np.linalg.norm(np.asarray(points[-1]) - np.asarray(points[0])) <= 1e-9:
            continue
        incoming = np.asarray(points[-1], dtype=float) - np.asarray(points[0], dtype=float)
        incoming /= max(float(np.linalg.norm(incoming)), 1e-12)
        following = plan.objects[index + 1] if index + 1 < len(plan.objects) else right
        outgoing = np.asarray([following.position.x - right.position.x, following.position.y - right.position.y], dtype=float)
        outgoing /= max(float(np.linalg.norm(outgoing)), 1e-12)
        route = FrozenRoute.from_points(
            f"{plan.beatmap_md5}:{left.index}->{right.index}",
            points,
            mode=mode,
            duration_ms=duration,
            start_time_ms=left.start_time_ms,
            key_schedule=_key_schedule(duration, index % 2),
            skill_level={"beginner": 25.0, "intermediate": 50.0, "expert": 80.0, "competitive": 98.0}.get(skill_group, 50.0),
            effort_level=effort_level,
            metadata={
                "map_id": plan.beatmap_md5,
                "left_index": left.index,
                "right_index": right.index,
                "target_order": index,
                "skill_group": skill_group,
                "turn_cos": float(np.dot(incoming, outgoing)),
                "turn_sin": float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]),
                "geometry_source": "map_plan_object_geometry",
                "benchmark_scope": "object_geometry_mechanism_diagnostic",
            },
        )
        cases.append(PairedRouteCase(route.route_id, plan.beatmap_md5, skill_group, seed, route))
    return tuple(cases)


def _metric_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}


def trace_metrics(trace: ExecutionTrace) -> dict[str, Any]:
    samples = trace.samples
    times = np.asarray([sample.time_ms for sample in samples], dtype=float) / 1000.0
    phases = np.asarray(trace.phases, dtype=float)
    positions = np.asarray(trace.positions, dtype=float)
    dt = np.diff(times)
    velocity = np.diff(positions, axis=0) / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0) / dt[1:, None] if len(velocity) > 1 else np.zeros((0, 2))
    jerk = np.diff(acceleration, axis=0) / dt[2:, None] if len(acceleration) > 1 else np.zeros((0, 2))
    velocity_jumps = np.linalg.norm(np.diff(velocity, axis=0), axis=1) if len(velocity) > 1 else np.zeros(0)
    reversals = np.sum(np.sum(velocity[:-1] * velocity[1:], axis=1) < 0) if len(velocity) > 1 else 0
    route_positions = np.asarray([trace.route.position_at_phase(value) for value in phases], dtype=float)
    on_route_error = np.linalg.norm(positions - route_positions, axis=1)
    early = np.flatnonzero(phases >= 0.999999)
    early_arrival_fraction = float((times[early[0]] / times[-1]) if len(early) and times[-1] > 0 else 0.0)
    pause_threshold = max(8.0, trace.route.length_px / max(trace.route.duration_ms / 1000.0, 1e-9) * 0.02)
    pause_fraction = float(np.mean(speed <= pause_threshold)) if len(speed) else 0.0
    event_time_errors = []
    for event in trace.key_schedule:
        event_time_errors.append(float(np.min(np.abs(np.asarray([sample.time_ms for sample in samples]) - event.time_ms))))
    return {
        "route_sha256": trace.route_sha256,
        "position_on_route_error_px": _metric_stats(on_route_error.tolist()),
        "position_on_route_error_max_px": float(np.max(on_route_error)),
        "phase_min_increment": float(np.min(np.diff(phases))) if len(phases) > 1 else 0.0,
        "phase_monotone": bool(np.all(np.diff(phases) >= -1e-9)),
        "phase_endpoint": float(phases[-1]),
        "target_completed": bool(phases[-1] >= 1.0 - 1e-6),
        "miss": bool(phases[-1] < 1.0 - 1e-6),
        "speed_px_s": _metric_stats(speed.tolist()),
        "acceleration_px_s2": _metric_stats(np.linalg.norm(acceleration, axis=1).tolist()),
        "jerk_px_s3": _metric_stats(np.linalg.norm(jerk, axis=1).tolist()),
        "velocity_vector_jump_px_s": _metric_stats(velocity_jumps.tolist()),
        "reversal_count": int(reversals),
        "reversal_frequency_per_second": float(reversals / max(times[-1] - times[0], 1e-9)),
        "early_arrival_fraction": early_arrival_fraction,
        "pause_fraction": pause_fraction,
        "event_time_interpolation_error_ms": _metric_stats(event_time_errors),
        "timing": {
            "event_count": len(event_time_errors),
            "event_error_ms": _metric_stats(event_time_errors),
        },
        "fallback": trace.diagnostics.fallback,
        "fallback_reason": trace.diagnostics.fallback_reason,
        "segment_count": trace.diagnostics.segment_count,
        "learned_segment_count": trace.diagnostics.learned_segment_count,
        "fallback_segment_count": trace.diagnostics.fallback_segment_count,
        "projection_fraction": trace.diagnostics.projection_fraction,
        "max_projection": trace.diagnostics.max_projection,
        "boundary": {
            "position_start": list(samples[0].position),
            "position_end": list(samples[-1].position),
            "velocity_start_px_s": list(velocity[0]) if len(velocity) else [0.0, 0.0],
            "velocity_end_px_s": list(velocity[-1]) if len(velocity) else [0.0, 0.0],
            "acceleration_start_px_s2": list(acceleration[0]) if len(acceleration) else [0.0, 0.0],
            "acceleration_end_px_s2": list(acceleration[-1]) if len(acceleration) else [0.0, 0.0],
        },
    }


def _paired_pair(case: PairedRouteCase, math_trace: ExecutionTrace, hybrid_trace: ExecutionTrace) -> dict[str, Any]:
    math_metrics = trace_metrics(math_trace)
    hybrid_metrics = trace_metrics(hybrid_trace)
    return {
        "case_id": case.case_id,
        "map_id": case.map_id,
        "skill_group": case.skill_group,
        "effort_level": case.route.effort_level,
        "seed": case.seed,
        "mode": case.route.mode,
        "duration_ms": case.route.duration_ms,
        "distance_px": case.route.length_px,
        "corner_angle_deg": float(
            np.degrees(np.arccos(np.clip(float(case.route.metadata.get("turn_cos", 1.0)), -1.0, 1.0)))
        ),
        "interval_bucket": case.interval_bucket,
        "route_sha256": case.route.route_sha256,
        "route_identity_equal": math_trace.route_sha256 == hybrid_trace.route_sha256 == case.route.route_sha256,
        "target_order": case.route.metadata.get("target_order"),
        "math_only": math_metrics,
        "hybrid": hybrid_metrics,
        "execution_delta": {
            "speed_p95": _difference(math_metrics["speed_px_s"]["p95"], hybrid_metrics["speed_px_s"]["p95"]),
            "acceleration_p95": _difference(math_metrics["acceleration_px_s2"]["p95"], hybrid_metrics["acceleration_px_s2"]["p95"]),
            "jerk_p95": _difference(math_metrics["jerk_px_s3"]["p95"], hybrid_metrics["jerk_px_s3"]["p95"]),
            "reversal_frequency": _difference(math_metrics["reversal_frequency_per_second"], hybrid_metrics["reversal_frequency_per_second"]),
            "pause_fraction": _difference(math_metrics["pause_fraction"], hybrid_metrics["pause_fraction"]),
        },
    }


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(right - left)


def _safe_filename(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", value)


def _aggregate(pairs: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    metrics = [pair[side] for pair in pairs]
    if not metrics:
        return {"pairs": 0}
    fallback_reasons: Counter[str] = Counter()
    for item in metrics:
        reason = item.get("fallback_reason")
        if not reason:
            continue
        for part in str(reason).split(";"):
            fallback_reasons[part] += 1
    return {
        "pairs": len(metrics),
        "route_position_error_max_px": float(max(item["position_on_route_error_max_px"] for item in metrics)),
        "phase_monotone_share": float(np.mean([item["phase_monotone"] for item in metrics])),
        "completion_share": float(np.mean([item["target_completed"] for item in metrics])),
        "miss_share": float(np.mean([item["miss"] for item in metrics])),
        "speed_p95": _metric_stats([item["speed_px_s"]["p95"] for item in metrics]),
        "acceleration_p95": _metric_stats([item["acceleration_px_s2"]["p95"] for item in metrics]),
        "jerk_p95": _metric_stats([item["jerk_px_s3"]["p95"] for item in metrics]),
        "velocity_jump_p95": _metric_stats([item["velocity_vector_jump_px_s"]["p95"] for item in metrics]),
        "reversal_frequency": _metric_stats([item["reversal_frequency_per_second"] for item in metrics]),
        "pause_fraction": _metric_stats([item["pause_fraction"] for item in metrics]),
        "fallback_share": float(np.mean([item["fallback"] for item in metrics])),
        "trace_fallback_share": float(np.mean([item["fallback"] for item in metrics])),
        "trace_fallback_share_definition": "fraction of complete traces containing at least one fallback segment",
        "fallback_reason_counts": dict(fallback_reasons),
        "segment_count_total": int(sum(item.get("segment_count", 1) for item in metrics)),
        "learned_segment_count_total": int(sum(item.get("learned_segment_count", 0) for item in metrics)),
        "fallback_segment_count_total": int(sum(item.get("fallback_segment_count", 0) for item in metrics)),
        "projection_fraction": float(np.mean([item["projection_fraction"] for item in metrics])),
    }


def _human_distribution_status(human_rows: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    if not human_rows:
        return {
            "status": "not_available_no_verified_human_data",
            "reason": "The manifest is empty or does not have admitted held-out windows; generated local runs are forbidden.",
        }
    by_group: dict[str, list[float]] = {}
    for row in human_rows:
        by_group.setdefault(str(row["skill_group"]), []).append(float(row.get("completion_fraction", 0.0)))
    return {"status": "available", "completion_fraction_by_tier": {key: _metric_stats(value) for key, value in by_group.items()}}


def run_paired_benchmark(
    cases: Sequence[PairedRouteCase],
    *,
    model: Any | None = None,
    blend: float = 1.0,
    sample_rate_hz: int = 500,
    export_dir: str | Path | None = None,
    human_rows: Sequence[Mapping[str, Any]] | None = None,
    model_name: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    math_planner = MathExecutionPlanner()
    hybrid_planner = HybridExecutionPlanner(model, blend=blend)
    pairs = []
    export_paths: list[str] = []
    export_root = Path(export_dir) if export_dir else None
    if export_root:
        export_root.mkdir(parents=True, exist_ok=True)
    for case in cases:
        math_trace = math_planner.plan(case.route, sample_rate_hz=sample_rate_hz)
        hybrid_trace = hybrid_planner.plan(case.route, sample_rate_hz=sample_rate_hz)
        pair = _paired_pair(case, math_trace, hybrid_trace)
        pairs.append(pair)
        if export_root:
            safe_case_id = _safe_filename(case.case_id)
            math_path = export_root / f"{safe_case_id}.math.json"
            hybrid_path = export_root / f"{safe_case_id}.hybrid.json"
            math_path.write_text(json.dumps(math_trace.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            hybrid_path.write_text(json.dumps(hybrid_trace.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            export_paths.extend([str(math_path), str(hybrid_path)])
    reproducible = True
    for case in cases:
        first = hybrid_planner.plan(case.route, sample_rate_hz=sample_rate_hz)
        second = hybrid_planner.plan(case.route, sample_rate_hz=sample_rate_hz)
        reproducible = reproducible and first.phases == second.phases and first.positions == second.positions
    buckets: dict[str, dict[str, Any]] = {}
    for bucket in sorted({pair["interval_bucket"] for pair in pairs}):
        selected = [pair for pair in pairs if pair["interval_bucket"] == bucket]
        buckets[bucket] = {"pairs": len(selected), "math_only": _aggregate(selected, "math_only"), "hybrid": _aggregate(selected, "hybrid")}
    route_hashes = [case.route.route_sha256 for case in cases]
    route_geometry_sources = sorted({str(case.route.metadata.get("geometry_source", "unspecified")) for case in cases})
    benchmark_scopes = sorted(
        {
            str(case.route.metadata.get("benchmark_scope", "supplied_frozen_route_execution_validation"))
            for case in cases
        }
    )
    diagnostic_only = "object_geometry_mechanism_diagnostic" in benchmark_scopes
    return {
        "schema_version": 1,
        "status": "complete" if pairs else "no_routes",
        "benchmark": "paired_execution_only_v1",
        "sample_rate_hz": sample_rate_hz,
        "blend": blend,
        "model_name": model_name,
        "model_id": model_id,
        "model_sha256": getattr(model, "model_sha256", "none"),
        "route_identity": {
            "all_pairs_shared_frozen_route": bool(all(pair["route_identity_equal"] for pair in pairs)),
            "route_hashes": route_hashes,
            "unique_route_hashes": len(set(route_hashes)),
        },
        "route_geometry_sources": route_geometry_sources,
        "benchmark_scope": "object_geometry_mechanism_diagnostic" if diagnostic_only else benchmark_scopes,
        "reproducible": reproducible,
        "pairs": pairs,
        "aggregates": {"math_only": _aggregate(pairs, "math_only"), "hybrid": _aggregate(pairs, "hybrid")},
        "interval_buckets": buckets,
        "human_execution_distribution": _human_distribution_status(human_rows),
        "runtime_ms": float((time.perf_counter() - started) * 1000.0),
        "visual_review": {
            "status": "exports_ready_for_blind_review" if export_paths else "not_exported",
            "paired_trace_exports": export_paths,
            "equal_route_and_schedule": True,
        },
        "scope_note": (
            "Diagnostic-only object-geometry routes. These chord/plan routes are not current runtime-planner traces and must not be used as ORI evidence."
            if diagnostic_only
            else "This report compares supplied frozen routes. Route planning, target selection, and key scheduling are excluded; geometry provenance is recorded explicitly."
        ),
    }


def _frame_control_signature(frames: Sequence[Any]) -> tuple[tuple[int, bool, bool], ...]:
    return tuple((int(frame.time_us), bool(frame.k1), bool(frame.k2)) for frame in frames)


def _bandwidth_limited_series(
    trace: ExecutionTrace,
    *,
    window_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    positions = np.asarray(trace.positions, dtype=float)
    times_ms = np.asarray([sample.time_ms for sample in trace.samples], dtype=float)
    window = max(3, int(window_samples) | 1)
    if len(times_ms) < 4:
        zeros = np.zeros(len(positions), dtype=float)
        return times_ms, zeros, zeros, zeros, window
    half = window // 2
    padded = np.pad(positions, ((half, half), (0, 0)), mode="edge")
    smoothed = np.column_stack(
        [np.convolve(padded[:, axis], np.ones(window) / float(window), mode="valid") for axis in range(positions.shape[1])]
    )
    times_s = times_ms / 1000.0
    velocity = np.gradient(smoothed, times_s, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, times_s, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, times_s, axis=0, edge_order=1)
    return times_ms, np.linalg.norm(velocity, axis=1), np.linalg.norm(acceleration, axis=1), np.linalg.norm(jerk, axis=1), window


def _bandwidth_limited_metrics(trace: ExecutionTrace, *, window_samples: int = 5) -> dict[str, Any]:
    """Estimate matched-bandwidth kinematics with one shared boxcar filter.

    This is a diagnostic comparison at the benchmark sampling rate.  It is
    deliberately not a human flicker metric or a calibrated perceptual gate.
    """
    times_ms, speed, acceleration, jerk, window = _bandwidth_limited_series(trace, window_samples=window_samples)
    if len(times_ms) < 4:
        empty = _metric_stats([])
        return {
            "method": "edge_padded_centered_boxcar",
            "window_samples": window,
            "sample_count": len(times_ms),
            "speed_px_s": empty,
            "acceleration_px_s2": empty,
            "jerk_px_s3": empty,
        }
    dt_ms = np.diff(times_ms)
    return {
        "method": "edge_padded_centered_boxcar",
        "window_samples": window,
        "sample_count": len(times_ms),
        "median_sample_interval_ms": float(np.median(dt_ms)) if len(dt_ms) else None,
        "speed_px_s": _metric_stats(speed.tolist()),
        "acceleration_px_s2": _metric_stats(acceleration.tolist()),
        "jerk_px_s3": _metric_stats(jerk.tolist()),
    }


def _matched_bandwidth_pair(math_trace: ExecutionTrace, hybrid_trace: ExecutionTrace) -> dict[str, Any]:
    math_metrics = _bandwidth_limited_metrics(math_trace)
    hybrid_metrics = _bandwidth_limited_metrics(hybrid_trace)
    return {
        "method": math_metrics["method"],
        "window_samples": math_metrics["window_samples"],
        "baseline_math_only": math_metrics,
        "candidate_delivered_hybrid": hybrid_metrics,
        "delta_candidate_minus_baseline": {
            "speed_p95": _difference(math_metrics["speed_px_s"]["p95"], hybrid_metrics["speed_px_s"]["p95"]),
            "acceleration_p95": _difference(math_metrics["acceleration_px_s2"]["p95"], hybrid_metrics["acceleration_px_s2"]["p95"]),
            "jerk_p95": _difference(math_metrics["jerk_px_s3"]["p95"], hybrid_metrics["jerk_px_s3"]["p95"]),
        },
        "interpretation": "Matched-bandwidth diagnostic only; not a calibrated human flicker score.",
    }


def _tail_row(
    math_trace: ExecutionTrace,
    hybrid_trace: ExecutionTrace,
    math_matched_jerk: np.ndarray,
    hybrid_matched_jerk: np.ndarray,
    start_ms: float,
    end_ms: float,
    *,
    index: int,
    label: str,
) -> dict[str, Any] | None:
    math_times = np.asarray([sample.time_ms for sample in math_trace.samples], dtype=float)
    hybrid_times = np.asarray([sample.time_ms for sample in hybrid_trace.samples], dtype=float)
    math_mask = (math_times >= start_ms) & (math_times <= end_ms)
    hybrid_mask = (hybrid_times >= start_ms) & (hybrid_times <= end_ms)
    math_indices = np.flatnonzero(math_mask)
    hybrid_indices = np.flatnonzero(hybrid_mask)
    if not len(math_indices) or not len(hybrid_indices):
        return None
    math_dt = np.diff(math_times)
    hybrid_dt = np.diff(hybrid_times)
    math_median_dt = float(np.median(math_dt)) if len(math_dt) else None
    hybrid_median_dt = float(np.median(hybrid_dt)) if len(hybrid_dt) else None

    def _support(times: np.ndarray, indices: np.ndarray, median_dt: float | None) -> dict[str, Any]:
        local_dt = np.diff(times[indices])
        if median_dt is None:
            native = np.ones(len(local_dt), dtype=bool)
        else:
            native = (local_dt >= median_dt * 0.5) & (local_dt <= median_dt * 1.5)
        left = min(2, len(indices) // 2)
        right = min(2, max((len(indices) - left) // 2, 0))
        interior = indices[left:len(indices) - right] if left + right < len(indices) else indices
        return {
            "sample_count": int(len(indices)),
            "interior_sample_count": int(len(interior)),
            "edge_excluded_sample_count": int(len(indices) - len(interior)),
            "gap_count": int(np.sum(~native)),
            "native_interval_count": int(np.sum(native)),
            "native_cadence_share": float(np.mean(native)) if len(native) else 1.0,
            "median_interval_ms": median_dt,
            "indices": interior,
        }

    math_support = _support(math_times, math_indices, math_median_dt)
    hybrid_support = _support(hybrid_times, hybrid_indices, hybrid_median_dt)
    math_inner = math_support.pop("indices")
    hybrid_inner = hybrid_support.pop("indices")
    math_raw = np.asarray([math_trace.samples[int(index)].jerk_px_s3 for index in math_inner], dtype=float)
    hybrid_raw = np.asarray([hybrid_trace.samples[int(index)].jerk_px_s3 for index in hybrid_inner], dtype=float)
    math_matched = math_matched_jerk[math_inner]
    hybrid_matched = hybrid_matched_jerk[hybrid_inner]
    if not len(math_raw) or not len(hybrid_raw):
        return None

    def _tail(values: np.ndarray) -> dict[str, Any]:
        return {
            "support_count": int(len(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    return {
        "index": index,
        "label": label,
        "start_time_ms": float(start_ms),
        "end_time_ms": float(end_ms),
        "baseline_support": math_support,
        "candidate_support": hybrid_support,
        "baseline_raw_jerk": _tail(math_raw),
        "candidate_raw_jerk": _tail(hybrid_raw),
        "baseline_matched_jerk": _tail(math_matched),
        "candidate_matched_jerk": _tail(hybrid_matched),
        "raw_jerk_p95_delta": float(np.percentile(hybrid_raw, 95) - np.percentile(math_raw, 95)),
        "matched_jerk_p95_delta": float(np.percentile(hybrid_matched, 95) - np.percentile(math_matched, 95)),
        "raw_jerk_max_delta": float(np.max(hybrid_raw) - np.max(math_raw)),
        "matched_jerk_max_delta": float(np.max(hybrid_matched) - np.max(math_matched)),
    }


def _tail_distribution(rows: Sequence[Mapping[str, Any]], *, interpretation: str) -> dict[str, Any]:
    if not rows:
        return {
            "support_count": 0,
            "sample_support": {"baseline_interior_samples": 0, "candidate_interior_samples": 0},
            "native_cadence": {"baseline_gap_count": 0, "candidate_gap_count": 0},
            "edge_mask": {"baseline_excluded_samples": 0, "candidate_excluded_samples": 0},
            "interpretation": interpretation,
        }
    raw_deltas = [float(row["raw_jerk_p95_delta"]) for row in rows]
    matched_deltas = [float(row["matched_jerk_p95_delta"]) for row in rows]
    return {
        "support_count": len(rows),
        "sample_support": {
            "baseline_interior_samples": sum(int(row["baseline_support"]["interior_sample_count"]) for row in rows),
            "candidate_interior_samples": sum(int(row["candidate_support"]["interior_sample_count"]) for row in rows),
        },
        "native_cadence": {
            "baseline_gap_count": sum(int(row["baseline_support"]["gap_count"]) for row in rows),
            "candidate_gap_count": sum(int(row["candidate_support"]["gap_count"]) for row in rows),
            "baseline_native_cadence_share": float(np.mean([row["baseline_support"]["native_cadence_share"] for row in rows])),
            "candidate_native_cadence_share": float(np.mean([row["candidate_support"]["native_cadence_share"] for row in rows])),
        },
        "edge_mask": {
            "baseline_excluded_samples": sum(int(row["baseline_support"]["edge_excluded_sample_count"]) for row in rows),
            "candidate_excluded_samples": sum(int(row["candidate_support"]["edge_excluded_sample_count"]) for row in rows),
            "rule": "exclude up to two samples at each region edge before tail estimates",
        },
        "baseline_raw_jerk_p95": _metric_stats([row["baseline_raw_jerk"]["p95"] for row in rows]),
        "candidate_raw_jerk_p95": _metric_stats([row["candidate_raw_jerk"]["p95"] for row in rows]),
        "delta_raw_jerk_p95": _metric_stats(raw_deltas),
        "baseline_raw_jerk_max": _metric_stats([row["baseline_raw_jerk"]["max"] for row in rows]),
        "candidate_raw_jerk_max": _metric_stats([row["candidate_raw_jerk"]["max"] for row in rows]),
        "delta_raw_jerk_max": _metric_stats([float(row["raw_jerk_max_delta"]) for row in rows]),
        "baseline_matched_jerk_p95": _metric_stats([row["baseline_matched_jerk"]["p95"] for row in rows]),
        "candidate_matched_jerk_p95": _metric_stats([row["candidate_matched_jerk"]["p95"] for row in rows]),
        "delta_matched_jerk_p95": _metric_stats(matched_deltas),
        "baseline_matched_jerk_max": _metric_stats([row["baseline_matched_jerk"]["max"] for row in rows]),
        "candidate_matched_jerk_max": _metric_stats([row["candidate_matched_jerk"]["max"] for row in rows]),
        "delta_matched_jerk_max": _metric_stats([float(row["matched_jerk_max_delta"]) for row in rows]),
        "candidate_raw_p95_exceedance_count": sum(delta > 0.0 for delta in raw_deltas),
        "candidate_matched_p95_exceedance_count": sum(delta > 0.0 for delta in matched_deltas),
        "top_worst_raw_joins_or_segments": sorted(rows, key=lambda row: float(row["raw_jerk_p95_delta"]), reverse=True)[:5],
        "interpretation": interpretation,
    }


def _join_local_tail_summary(
    math_trace: ExecutionTrace,
    hybrid_trace: ExecutionTrace,
    boundaries_ms: Sequence[float],
    *,
    window_ms: float = 50.0,
) -> dict[str, Any]:
    """Compare jerk tails in equal windows around every object join."""
    _, _, _, math_matched_jerk, _ = _bandwidth_limited_series(math_trace)
    _, _, _, hybrid_matched_jerk, _ = _bandwidth_limited_series(hybrid_trace)
    rows: list[dict[str, Any]] = []
    for boundary_index, boundary in enumerate(boundaries_ms[1:-1], start=1):
        row = _tail_row(
            math_trace,
            hybrid_trace,
            math_matched_jerk,
            hybrid_matched_jerk,
            float(boundary) - window_ms,
            float(boundary) + window_ms,
            index=boundary_index,
            label="adjacent_object_join",
        )
        if row is not None:
            rows.append(row)
    summary = _tail_distribution(
        rows,
        interpretation="Equal-window join-local tail diagnostic at matched sampling; not a calibrated human flicker score.",
    )
    summary.update(
        {
            "window_ms": window_ms,
            "joins_requested": max(len(boundaries_ms) - 2, 0),
            "joins_evaluated": len(rows),
            "candidate_jerk_p95_exceedance_count": summary.get("candidate_raw_p95_exceedance_count", 0),
            "candidate_jerk_p95_exceedance_share": (
                float(summary.get("candidate_raw_p95_exceedance_count", 0)) / max(len(rows), 1)
            ),
            "baseline_join_jerk_p95": summary.get("baseline_raw_jerk_p95"),
            "candidate_join_jerk_p95": summary.get("candidate_raw_jerk_p95"),
            "delta_join_jerk_p95": summary.get("delta_raw_jerk_p95"),
            "worst_join_delta_candidate_minus_baseline": summary["top_worst_raw_joins_or_segments"][0]
            if summary["top_worst_raw_joins_or_segments"]
            else None,
            "top_worst_joins": summary["top_worst_raw_joins_or_segments"],
        }
    )
    return summary


def _learned_segment_tail_summary(
    math_trace: ExecutionTrace,
    hybrid_trace: ExecutionTrace,
    hybrid_exposure: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare tails on the exact segments delivered by the learned path."""
    _, _, _, math_matched_jerk, _ = _bandwidth_limited_series(math_trace)
    _, _, _, hybrid_matched_jerk, _ = _bandwidth_limited_series(hybrid_trace)
    learned_details = [
        detail for detail in hybrid_exposure.get("segment_details", []) if detail.get("category") == "learned"
    ]
    rows: list[dict[str, Any]] = []
    for detail in learned_details:
        row = _tail_row(
            math_trace,
            hybrid_trace,
            math_matched_jerk,
            hybrid_matched_jerk,
            float(detail.get("start_time_ms", 0.0)),
            float(detail.get("end_time_ms", 0.0)),
            index=int(detail.get("segment_index", 0)),
            label="learned_segment",
        )
        if row is not None:
            rows.append(row)
    summary = _tail_distribution(
        rows,
        interpretation="Paired learned-segment tail diagnostic at native cadence with explicit edge/gap support; not a calibrated human flicker score.",
    )
    summary.update(
        {
            "learned_segments_requested": len(learned_details),
            "learned_segments_evaluated": len(rows),
            "candidate_learned_segment_count": hybrid_exposure.get("learned_segment_count", 0),
        }
    )
    return summary


def _segment_exposure(trace: ExecutionTrace, *, blend: float) -> dict[str, Any]:
    details = list(trace.diagnostics.segment_diagnostics)
    compact: list[dict[str, Any]] = []
    learned_duration = 0.0
    fallback_duration = 0.0
    unclassified_detail_duration = 0.0
    total_duration = 0.0
    reason_counts: Counter[str] = Counter()
    for detail in details:
        mode = str(detail.get("mode", "unknown"))
        duration = float(detail.get("duration_ms", 0.0) or 0.0)
        fallback = bool(detail.get("fallback", False))
        learned = bool(blend > 0.0 and mode == "circle" and not fallback)
        category = "learned" if learned else "fallback" if fallback else "unclassified"
        if learned:
            learned_duration += duration
        if fallback:
            fallback_duration += duration
            reason = detail.get("fallback_reason")
            if reason:
                reason_text = str(reason)
                if ":" in reason_text:
                    reason_text = reason_text.split(":", 1)[1]
                reason_counts[reason_text.split(":", 1)[0]] += 1
        if category == "unclassified":
            unclassified_detail_duration += duration
        total_duration += duration
        compact.append(
            {
                "segment_index": detail.get("segment_index"),
                "mode": mode,
                "start_time_ms": detail.get("start_time_ms"),
                "end_time_ms": detail.get("end_time_ms"),
                "duration_ms": duration,
                "category": category,
                "learned": learned,
                "fallback": fallback,
                "fallback_reason": detail.get("fallback_reason"),
                "projected_steps": detail.get("projected_steps", 0),
                "projection_fraction": detail.get("projection_fraction", 0.0),
                "max_projection": detail.get("max_projection", 0.0),
            }
        )
    segment_count = int(trace.diagnostics.segment_count)
    learned_count = int(trace.diagnostics.learned_segment_count)
    fallback_count = int(trace.diagnostics.fallback_segment_count)
    calculated_learned_count = sum(bool(item.get("learned")) for item in compact)
    calculated_fallback_count = sum(bool(item.get("fallback")) for item in compact)
    if learned_count != calculated_learned_count or fallback_count != calculated_fallback_count:
        raise AssertionError("segment exposure counters disagree with segment diagnostics")
    if len(compact) > segment_count or learned_count + fallback_count > segment_count:
        raise AssertionError("segment exposure categories exceed the segment denominator")
    skipped_segment_count = max(segment_count - len(compact), 0)
    unclassified_count = max(segment_count - learned_count - fallback_count, 0)
    unclassified_duration = unclassified_detail_duration + max(float(trace.route.duration_ms) - total_duration, 0.0)
    eligible_count = learned_count + fallback_count
    eligible_duration = learned_duration + fallback_duration
    return {
        "segment_count": segment_count,
        "evaluated_segment_count": len(compact),
        "skipped_segment_count": skipped_segment_count,
        "learned_segment_count": learned_count,
        "fallback_segment_count": fallback_count,
        "unclassified_segment_count": unclassified_count,
        "category_counts_are_mutually_exclusive": learned_count + fallback_count + unclassified_count == segment_count,
        "segment_category_formula": "total = learned + fallback + unclassified; eligible = learned + fallback",
        "eligible_segment_count": eligible_count,
        "learned_segment_share": learned_count / max(segment_count, 1),
        "fallback_segment_share": fallback_count / max(segment_count, 1),
        "learned_segment_share_all_evaluated": learned_count / max(segment_count, 1),
        "fallback_segment_share_all_evaluated": fallback_count / max(segment_count, 1),
        "learned_segment_share_eligible": learned_count / max(eligible_count, 1),
        "fallback_segment_share_eligible": fallback_count / max(eligible_count, 1),
        "trace_fallback": bool(trace.diagnostics.fallback),
        "trace_fallback_definition": "fraction of complete traces containing at least one fallback segment",
        "learned_duration_ms": learned_duration,
        "fallback_duration_ms": fallback_duration,
        "unclassified_duration_ms": unclassified_duration,
        "evaluated_segment_duration_ms": total_duration,
        "eligible_duration_ms": eligible_duration,
        "duration_denominator": "route_duration_ms for all-evaluated shares; eligible_duration_ms for eligible-only shares",
        "route_duration_ms": float(trace.route.duration_ms),
        "learned_duration_share_all_evaluated": learned_duration / max(float(trace.route.duration_ms), 1e-12),
        "fallback_duration_share_all_evaluated": fallback_duration / max(float(trace.route.duration_ms), 1e-12),
        "learned_duration_share_eligible": learned_duration / max(eligible_duration, 1e-12),
        "fallback_duration_share_eligible": fallback_duration / max(eligible_duration, 1e-12),
        "projection_fraction": trace.diagnostics.projection_fraction,
        "max_projection": trace.diagnostics.max_projection,
        "fallback_reason_counts": dict(reason_counts),
        "segment_details": compact,
    }


def _aggregate_segment_exposure(pairs: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    exposures = [pair["segment_exposure"][side] for pair in pairs if pair.get("segment_exposure", {}).get(side)]
    if not exposures:
        return {
            "pairs": 0,
            "segment_count_total": 0,
            "learned_segment_count_total": 0,
            "fallback_segment_count_total": 0,
            "unclassified_segment_count_total": 0,
            "learned_segment_share": 0.0,
            "fallback_segment_share": 0.0,
            "unclassified_segment_share": 0.0,
            "category_counts_are_mutually_exclusive": True,
            "segment_category_formula": "total = learned + fallback + unclassified; eligible = learned + fallback",
            "eligible_segment_count_total": 0,
            "trace_fallback_share": 0.0,
            "learned_duration_ms": _metric_stats([]),
            "fallback_duration_ms": _metric_stats([]),
            "unclassified_duration_ms": _metric_stats([]),
            "eligible_duration_ms": _metric_stats([]),
            "projection_fraction": _metric_stats([]),
            "fallback_reason_counts": {},
        }
    segment_count = sum(int(item.get("segment_count", 0)) for item in exposures)
    learned_count = sum(int(item.get("learned_segment_count", 0)) for item in exposures)
    fallback_count = sum(int(item.get("fallback_segment_count", 0)) for item in exposures)
    unclassified_count = sum(int(item.get("unclassified_segment_count", 0)) for item in exposures)
    eligible_count = learned_count + fallback_count
    reasons: Counter[str] = Counter()
    for item in exposures:
        reasons.update({str(key): int(value) for key, value in item.get("fallback_reason_counts", {}).items()})
    return {
        "pairs": len(exposures),
        "segment_count_total": segment_count,
        "learned_segment_count_total": learned_count,
        "fallback_segment_count_total": fallback_count,
        "unclassified_segment_count_total": unclassified_count,
        "category_counts_are_mutually_exclusive": learned_count + fallback_count + unclassified_count == segment_count,
        "segment_category_formula": "total = learned + fallback + unclassified; eligible = learned + fallback",
        "eligible_segment_count_total": eligible_count,
        "learned_segment_share": learned_count / max(segment_count, 1),
        "fallback_segment_share": fallback_count / max(segment_count, 1),
        "unclassified_segment_share": unclassified_count / max(segment_count, 1),
        "learned_segment_share_eligible": learned_count / max(eligible_count, 1),
        "fallback_segment_share_eligible": fallback_count / max(eligible_count, 1),
        "trace_fallback_share": float(np.mean([bool(item.get("trace_fallback")) for item in exposures])),
        "trace_fallback_share_definition": "fraction of complete traces containing at least one fallback segment",
        "learned_duration_ms": _metric_stats([float(item.get("learned_duration_ms", 0.0)) for item in exposures]),
        "fallback_duration_ms": _metric_stats([float(item.get("fallback_duration_ms", 0.0)) for item in exposures]),
        "unclassified_duration_ms": _metric_stats([float(item.get("unclassified_duration_ms", 0.0)) for item in exposures]),
        "eligible_duration_ms": _metric_stats([float(item.get("eligible_duration_ms", 0.0)) for item in exposures]),
        "learned_duration_share_eligible": _metric_stats([
            float(item.get("learned_duration_share_eligible", 0.0)) for item in exposures
        ]),
        "fallback_duration_share_eligible": _metric_stats([
            float(item.get("fallback_duration_share_eligible", 0.0)) for item in exposures
        ]),
        "projection_fraction": _metric_stats([float(item.get("projection_fraction", 0.0)) for item in exposures]),
        "fallback_reason_counts": dict(reasons),
    }


def run_runtime_trace_paired_benchmark(
    cases: Sequence[RuntimeTraceCase],
    *,
    model: Any | None = None,
    blend: float = 1.0,
    sample_rate_hz: int = 500,
    model_name: str | None = None,
    model_id: str | None = None,
    export_dir: str | Path | None = None,
    math_trace_cache: dict[str, dict[str, Any]] | None = None,
    verify_reproducibility: bool = True,
    include_tail_diagnostics: bool = True,
    progress_label: str | None = None,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare execution warping on frozen samples from the current runtime planner.

    The mathematical side is produced by the same segmented adapter with
    ``blend=0``; the hybrid side uses the requested model and blend. Both sides
    therefore inherit exactly the same runtime timestamps, key states, object
    boundaries, and route samples. Unsupported segments remain explicit
    math-only fallbacks inside ``warp_math_trace``.
    """
    started = time.perf_counter()
    pairs: list[dict[str, Any]] = []
    export_paths: list[str] = []
    export_root = Path(export_dir) if export_dir else None
    if export_root:
        export_root.mkdir(parents=True, exist_ok=True)
    for case_index, case in enumerate(cases, 1):
        cache_key = f"{case.route.route_sha256}:{sample_rate_hz}"
        cached_math = math_trace_cache.get(cache_key) if math_trace_cache is not None else None
        if cached_math is None:
            math_frames, math_trace = warp_math_trace(
                case.source_frames,
                model=None,
                blend=0.0,
                sample_rate_hz=sample_rate_hz,
                route_id=case.route.route_id,
                mode="mixed",
                segment_boundaries_ms=case.segment_boundaries_ms,
                segment_modes=case.segment_modes,
                frozen_route=case.route,
            )
            math_metrics = trace_metrics(math_trace)
            math_controls = _frame_control_signature(math_frames)
            math_exposure = _segment_exposure(math_trace, blend=0.0)
            if math_trace_cache is not None:
                math_trace_cache[cache_key] = {
                    "frames": math_frames,
                    "trace": math_trace,
                    "metrics": math_metrics,
                    "controls": math_controls,
                    "exposure": math_exposure,
                }
        else:
            math_frames = cached_math["frames"]
            math_trace = cached_math["trace"]
            math_metrics = cached_math["metrics"]
            math_controls = cached_math["controls"]
            math_exposure = cached_math["exposure"]
        if blend <= 0.0:
            hybrid_frames, hybrid_trace = math_frames, math_trace
            hybrid_metrics = math_metrics
            hybrid_controls = math_controls
            hybrid_exposure = math_exposure
        else:
            hybrid_frames, hybrid_trace = warp_math_trace(
                case.source_frames,
                model=model,
                blend=blend,
                sample_rate_hz=sample_rate_hz,
                route_id=case.route.route_id,
                mode="mixed",
                segment_boundaries_ms=case.segment_boundaries_ms,
                segment_modes=case.segment_modes,
                frozen_route=case.route,
            )
            hybrid_metrics = trace_metrics(hybrid_trace)
            hybrid_controls = _frame_control_signature(hybrid_frames)
        source_controls = _frame_control_signature(case.source_frames)
        key_schedule_equal = math_trace.key_schedule == hybrid_trace.key_schedule == case.route.key_schedule
        if blend > 0.0:
            hybrid_exposure = _segment_exposure(hybrid_trace, blend=blend)
        pair = {
            "case_id": case.case_id,
            "map_id": case.map_id,
            "skill_group": case.skill_group,
            "seed": case.seed,
            "planner_version": case.planner_version,
            "route_sha256": case.route.route_sha256,
            "route_identity_equal": math_trace.route_sha256 == hybrid_trace.route_sha256 == case.route.route_sha256,
            "key_schedule_identity_equal": key_schedule_equal,
            "frame_timestamps_and_key_states_equal": math_controls == hybrid_controls == source_controls,
            "segment_boundaries_ms": list(case.segment_boundaries_ms),
            "segment_modes": list(case.segment_modes),
            "source_frame_count": len(case.source_frames),
            "duration_ms": case.route.duration_ms,
            "math_only": math_metrics,
            "hybrid": hybrid_metrics,
            "segment_exposure": {
                "math_only": math_exposure,
                "candidate_delivered_hybrid": hybrid_exposure,
            },
            "matched_bandwidth": _matched_bandwidth_pair(math_trace, hybrid_trace)
            if include_tail_diagnostics else {"status": "omitted_for_validation_ablation"},
            "learned_segment_tails": _learned_segment_tail_summary(math_trace, hybrid_trace, hybrid_exposure)
            if include_tail_diagnostics else {"status": "omitted_for_validation_ablation"},
            "join_local_tails": _join_local_tail_summary(math_trace, hybrid_trace, case.segment_boundaries_ms)
            if include_tail_diagnostics else {"status": "omitted_for_validation_ablation"},
            "sampling_limits": {
                "source_sample_rate_hz": sample_rate_hz,
                "source_frames_are_synthetic": True,
                "matched_bandwidth_is_diagnostic": True,
                "human_flicker_calibration_available": False,
            },
            "execution_delta": {
                "speed_p95": _difference(math_metrics["speed_px_s"]["p95"], hybrid_metrics["speed_px_s"]["p95"]),
                "acceleration_p95": _difference(math_metrics["acceleration_px_s2"]["p95"], hybrid_metrics["acceleration_px_s2"]["p95"]),
                "jerk_p95": _difference(math_metrics["jerk_px_s3"]["p95"], hybrid_metrics["jerk_px_s3"]["p95"]),
                "reversal_frequency": _difference(math_metrics["reversal_frequency_per_second"], hybrid_metrics["reversal_frequency_per_second"]),
                "pause_fraction": _difference(math_metrics["pause_fraction"], hybrid_metrics["pause_fraction"]),
            },
        }
        pairs.append(pair)
        if progress_path:
            progress_file = Path(progress_path)
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            progress_file.write_text(
                json.dumps({
                    "schema_version": 1,
                    "status": "in_progress",
                    "blend": blend,
                    "model_sha256": getattr(model, "model_sha256", "none"),
                    "completed_cases": case_index,
                    "total_cases": len(cases),
                    "pairs": pairs,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if progress_label:
            print(f"[benchmark {progress_label}] case={case_index}/{len(cases)} blend={blend}", flush=True)
        if export_root:
            safe_case_id = _safe_filename(case.case_id)
            math_path = export_root / f"{safe_case_id}.math.json"
            hybrid_path = export_root / f"{safe_case_id}.hybrid.json"
            math_path.write_text(json.dumps(math_trace.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            hybrid_path.write_text(json.dumps(hybrid_trace.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            export_paths.extend([str(math_path), str(hybrid_path)])
    reproducible: bool | None = True if verify_reproducibility else None
    if verify_reproducibility:
        for case in cases:
            first_frames, first_trace = warp_math_trace(
                case.source_frames,
                model=model,
                blend=blend,
                sample_rate_hz=sample_rate_hz,
                route_id=case.route.route_id,
                mode="mixed",
                segment_boundaries_ms=case.segment_boundaries_ms,
                segment_modes=case.segment_modes,
                frozen_route=case.route,
            )
            second_frames, second_trace = warp_math_trace(
                case.source_frames,
                model=model,
                blend=blend,
                sample_rate_hz=sample_rate_hz,
                route_id=case.route.route_id,
                mode="mixed",
                segment_boundaries_ms=case.segment_boundaries_ms,
                segment_modes=case.segment_modes,
                frozen_route=case.route,
            )
            reproducible = reproducible and _frame_control_signature(first_frames) == _frame_control_signature(second_frames)
            reproducible = reproducible and first_trace.positions == second_trace.positions and first_trace.phases == second_trace.phases
    segment_mode_counts = Counter(mode for case in cases for mode in case.segment_modes)
    aggregates = {"math_only": _aggregate(pairs, "math_only"), "hybrid": _aggregate(pairs, "hybrid")}
    segment_exposure = {
        "math_only": _aggregate_segment_exposure(pairs, "math_only"),
        "candidate_delivered_hybrid": _aggregate_segment_exposure(pairs, "candidate_delivered_hybrid"),
    }
    result = {
        "schema_version": 1,
        "status": "complete" if pairs else "no_routes",
        "benchmark": "paired_runtime_planner_trace_execution_v1",
        "sample_rate_hz": sample_rate_hz,
        "blend": blend,
        "model_name": model_name,
        "model_id": model_id,
        "model_sha256": getattr(model, "model_sha256", "none"),
        "reproducible": reproducible,
        "pairs": pairs,
        "aggregates": aggregates,
        "segment_exposure": segment_exposure,
        "route_identity": {
            "all_pairs_shared_frozen_route": bool(all(pair["route_identity_equal"] for pair in pairs)),
            "all_pairs_shared_key_schedule": bool(all(pair["key_schedule_identity_equal"] for pair in pairs)),
            "all_pairs_shared_timestamps_and_key_states": bool(all(pair["frame_timestamps_and_key_states_equal"] for pair in pairs)),
            "route_hashes": [case.route.route_sha256 for case in cases],
            "unique_route_hashes": len({case.route.route_sha256 for case in cases}),
        },
        "route_geometry_sources": sorted({str(case.route.metadata.get("geometry_source", "unspecified")) for case in cases}),
        "runtime_planner": {
            "planner_version": PLANNER_VERSION,
            "source_geometry": "current_runtime_planner_samples",
            "source_key_states": "current_runtime_planner_trace",
            "case_count": len(cases),
            "source_frame_count": int(sum(len(case.source_frames) for case in cases)),
            "duration_ms": _metric_stats([case.route.duration_ms for case in cases]),
            "segment_mode_counts": dict(segment_mode_counts),
            "segment_boundaries_are_per_case": True,
            "mixed_map_time_warping": False,
        },
        "fallback": {
            "unsupported_modes_are_explicit": True,
            "reason_counts": dict(aggregates["hybrid"].get("fallback_reason_counts", {})),
            "delivered_fallback_segments": segment_exposure["candidate_delivered_hybrid"].get("fallback_segment_count_total", 0),
            "delivered_learned_segments": segment_exposure["candidate_delivered_hybrid"].get("learned_segment_count_total", 0),
            "trace_fallback_share": segment_exposure["candidate_delivered_hybrid"].get("trace_fallback_share", 0.0),
            "trace_fallback_share_definition": "fraction of complete traces containing at least one fallback segment",
            "segment_fallback_share": segment_exposure["candidate_delivered_hybrid"].get("fallback_segment_share", 0.0),
        },
        "human_execution_distribution": _human_distribution_status(None),
        "runtime_ms": float((time.perf_counter() - started) * 1000.0),
        "visual_review": {
            "status": "exports_ready_for_blind_review" if export_paths else "not_exported",
            "paired_trace_exports": export_paths,
            "equal_route_and_schedule": bool(
                all(pair["route_identity_equal"] for pair in pairs)
                and all(pair["key_schedule_identity_equal"] for pair in pairs)
                and all(pair["frame_timestamps_and_key_states_equal"] for pair in pairs)
            ),
        },
        "scope_note": "Execution-only validation on frozen samples emitted by the current synthetic runtime planner. This is not live game-client or human calibration evidence; route geometry, timestamps, keys, and object-mode boundaries are inherited per map case.",
    }
    if progress_path:
        progress_file = Path(progress_path)
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(
            json.dumps({
                "schema_version": 1,
                "status": "complete",
                "blend": blend,
                "model_sha256": getattr(model, "model_sha256", "none"),
                "completed_cases": len(cases),
                "total_cases": len(cases),
                "result": result,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def raw_learned_proposal_benchmark(
    cases: Sequence[PairedRouteCase],
    *,
    model: Any,
    sample_rate_hz: int = 500,
    model_name: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Measure the model proposal before safety projection or fallback.

    This is diagnostic-only.  The returned traces are never passed to the
    planner or gameplay runner; delivered execution must use
    :func:`run_paired_benchmark`, which retains all guards.
    """
    started = time.perf_counter()
    metrics: list[dict[str, Any]] = []
    errors: list[str] = []
    route_hashes: list[str] = []
    for case in cases:
        route_hashes.append(case.route.route_sha256)
        times = _sample_times(case.route, sample_rate_hz)
        math_trace = MathExecutionPlanner().plan(case.route, times_ms=times)
        try:
            proposal = np.asarray(model.predict_phase_weights(case.route, len(times)), dtype=float)
            if proposal.shape != (len(times) - 1,) or not np.all(np.isfinite(proposal)) or np.any(proposal <= 0):
                raise ValueError("invalid_phase_proposal")
            proposal = _smooth_positive(proposal)
            proposal *= math_trace.diagnostics.endpoint_phase / max(float(proposal.sum()), 1e-12)
            phases = np.concatenate(([0.0], np.cumsum(proposal)))
            diagnostics = ExecutionDiagnostics(
                planner="unconstrained-learned-proposal",
                model_version=getattr(model, "model_version", "unknown"),
                model_sha256=getattr(model, "model_sha256", "unknown"),
                blend=1.0,
                endpoint_phase=float(phases[-1]),
            )
            trace = _trace_from_phases(case.route, times, phases, diagnostics)
            metrics.append(trace_metrics(trace))
        except Exception as error:  # diagnostic failure is recorded, never promoted to execution
            errors.append(f"{case.case_id}:{type(error).__name__}:{error}")
    return {
        "status": "complete" if metrics else "no_routes",
        "diagnostic_only": True,
        "sample_rate_hz": sample_rate_hz,
        "model_name": model_name,
        "model_id": model_id,
        "model_sha256": getattr(model, "model_sha256", "none"),
        "pairs": len(metrics),
        "errors": errors,
        "route_identity": {
            "all_pairs_shared_frozen_route": bool(len(metrics) == len(cases)),
            "route_hashes": route_hashes,
            "unique_route_hashes": len(set(route_hashes)),
        },
        "aggregates": {"raw_learned": _aggregate([{"raw_learned": item} for item in metrics], "raw_learned") if metrics else {"pairs": 0}},
        "runtime_ms": float((time.perf_counter() - started) * 1000.0),
        "scope_note": "Raw proposal metrics are diagnostic only; no raw proposal is delivered to gameplay.",
    }


def run_paired_benchmark_for_map(
    map_plan_path: str | Path,
    *,
    model_path: str | Path | None = None,
    blend: float = 1.0,
    sample_rate_hz: int = 500,
    skill_group: str = "intermediate",
    effort_level: float = 100.0,
    seed: int = 42,
    export_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the public benchmark on the actual current runtime-planner trace.

    The older :func:`cases_from_map_plan` route builder remains available for
    object-geometry mechanism diagnostics only.  It is deliberately not used
    by this public entrypoint so its chord routes cannot be mistaken for
    runtime ORI evidence.
    """
    from .io import load_map_plan

    plan = load_map_plan(map_plan_path)
    cases = runtime_trace_cases_from_map_plan(
        plan,
        skill_group=skill_group,
        effort_level=effort_level,
        seed=seed,
        sample_rate_hz=sample_rate_hz,
    )
    model = load_phase_model(model_path) if model_path else None
    report = run_runtime_trace_paired_benchmark(
        cases,
        model=model,
        blend=blend,
        sample_rate_hz=sample_rate_hz,
        model_name=Path(model_path).name if model_path else None,
        model_id=getattr(model, "model_version", None),
        export_dir=export_dir,
    )
    report["public_entrypoint"] = "run_paired_benchmark_for_map"
    report["benchmark_entrypoint_scope"] = "runtime_planner_trace_execution_validation"
    return report


def write_paired_report(report: Mapping[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
