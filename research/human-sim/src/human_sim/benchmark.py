from __future__ import annotations

"""Reusable statistical benchmark for offline planner realism checks."""

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .context import build_contexts
from .io import load_map_plan, repository_git_commit
from .matching import match_press_events, rising_press_events
from .planner import PLANNER_VERSION, HumanTracePlanner
from .schemas import HumanProfile, MapPlan, TraceFrame


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_PROFILE = (50.0, 80.0)
MONOTONICITY_PROFILES = (
    (30.0, 80.0),
    (50.0, 80.0),
    (70.0, 80.0),
    (50.0, 40.0),
    (50.0, 100.0),
)


@dataclass(frozen=True)
class BenchmarkGates:
    """Broad artifact gates; stochastic distributions are not snapshots."""

    max_cross_seed_abs_correlation: float = 0.35
    max_lag_abs_correlation: float = 0.35
    max_rim_share: float = 0.20
    min_angular_entropy: float = 0.60
    entropy_min_samples: int = 32
    min_successful_landings: int = 1
    continuity_min_samples: int = 16
    max_flow_severe_stop_share: float = 0.15
    min_flow_median_carry_ratio: float = 0.50
    max_base_boundary_position_error_px: float = 1e-4
    max_base_shared_velocity_error_px_s: float = 1e-5
    max_base_shared_acceleration_error_px_s2: float = 1e-3
    max_reversal_carry_ratio: float = 2.5
    max_tangent_reset_excess_p95_deg: float = 90.0
    max_kinematic_speed_px_s: float = 20_000.0
    max_kinematic_acceleration_p95_px_s2: float = 250_000.0
    max_kinematic_lateral_acceleration_p95_px_s2: float = 250_000.0
    max_kinematic_jerk_p95_px_s3: float = 100_000_000.0
    approach_min_samples: int = 64
    min_approach_undershoot_share: float = 0.42
    max_approach_undershoot_share: float = 0.78
    # Press-time errors include timing/trajectory contamination on hard maps;
    # keep this artifact ceiling broad while the direct landing sampler's
    # tighter reference range remains visible in each run's diagnostics.
    max_approach_anisotropy: float = 2.00
    max_approach_abs_axis_correlation: float = 0.55
    min_approach_sector_entropy: float = 0.70
    max_approach_wedge_share: float = 0.42
    # The sampled landing model is evaluated separately from press-time
    # errors.  The latter also contain timing/late-arrival contamination on
    # maps that are beyond the selected skill profile.
    min_planned_approach_undershoot_share: float = 0.40
    max_planned_approach_undershoot_share: float = 0.75
    max_planned_approach_anisotropy: float = 1.75
    max_planned_approach_abs_axis_correlation: float = 0.40
    min_planned_approach_sector_entropy: float = 0.75
    max_planned_approach_wedge_share: float = 0.32
    approach_press_min_success_rate: float = 0.85
    approach_press_max_timing_p95_ms: float = 60.0
    max_trace_flick_share: float = 0.20


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "timing-tests" / "benchmark").is_dir():
            return parent
    return Path.cwd()


def default_map_paths(scope: str = "compact", repository_root: str | Path | None = None) -> list[Path]:
    """Return the checked-in map set for a compact/default/full run."""
    if scope not in {"compact", "default", "full"}:
        raise ValueError("scope must be compact, default, or full")
    root = Path(repository_root) if repository_root else _repository_root()
    benchmark_root = root / "timing-tests" / "benchmark"
    named = {
        "fixture-2x": benchmark_root / "fixture-2x.map.ndjson.gz",
        "kingborn-nm": benchmark_root / "kingborn-nm.map.ndjson.gz",
        "heavenly-hrdt": benchmark_root / "heavenly-hrdt.map.ndjson.gz",
        "centipede-generated": benchmark_root / "centipede-generated.map.ndjson.gz",
        "taitoki": benchmark_root / "taitoki.map.ndjson.gz",
        "vanquished-hrdt": benchmark_root / "vanquished-hrdt.map.ndjson.gz",
    }
    if scope == "compact":
        selected = [named["fixture-2x"], named["kingborn-nm"]]
    elif scope == "default":
        selected = [named[name] for name in ("kingborn-nm", "heavenly-hrdt", "centipede-generated")]
    else:
        selected = sorted(benchmark_root.glob("*.map.ndjson.gz"))
    existing = [path for path in selected if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"No benchmark map plans found under {benchmark_root}")
    return existing


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": int(len(array)),
        "min": round(float(np.min(array)), 6),
        "mean": round(float(np.mean(array)), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _kinematic_summary(frames: Sequence[TraceFrame], sample_rate_hz: int = 500) -> dict[str, Any]:
    """Summarize motion after resampling onto the configured cursor cadence.

    Planner traces contain exact event frames between ordinary sample ticks.
    Differentiating those variable intervals directly turns a legitimate
    position correction into a huge, sample-spacing-dependent jerk.  The
    interpolation below makes the derivative statistics comparable across
    maps and seeds while retaining the original trace for all other metrics.
    """
    empty = _summary([])
    if len(frames) < 2 or sample_rate_hz <= 0:
        return {
            "sampling_rate_hz": sample_rate_hz,
            "source_frames": len(frames),
            "resampled_frames": 0,
            "velocity_px_s": empty,
            "acceleration_px_s2": empty,
            "lateral_acceleration_px_s2": empty,
            "jerk_px_s3": empty,
        }

    times = np.asarray([frame.time_us for frame in frames], dtype=float) / 1_000_000.0
    positions = np.asarray([[frame.x, frame.y] for frame in frames], dtype=float)
    strictly_increasing = np.concatenate(([True], np.diff(times) > 0.0))
    times = times[strictly_increasing]
    positions = positions[strictly_increasing]
    step_s = 1.0 / float(sample_rate_hz)
    if len(times) < 2 or times[-1] - times[0] < step_s:
        return {
            "sampling_rate_hz": sample_rate_hz,
            "source_frames": len(frames),
            "resampled_frames": 0,
            "velocity_px_s": empty,
            "acceleration_px_s2": empty,
            "lateral_acceleration_px_s2": empty,
            "jerk_px_s3": empty,
        }

    uniform_times = np.arange(times[0], times[-1] + step_s * 0.5, step_s)
    uniform_positions = np.column_stack(
        [np.interp(uniform_times, times, positions[:, axis]) for axis in range(positions.shape[1])]
    )
    # Event frames are retained in the trace for exact key/cursor dispatch,
    # but a single off-grid event must not become a multi-million px/s²
    # derivative.  A short triangular physical resampling kernel represents
    # the configured cursor cadence while preserving the underlying path.
    filter_width = 5 if len(uniform_positions) >= 5 else 1
    if filter_width > 1:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float) / 9.0
        padded = np.pad(uniform_positions, ((filter_width // 2, filter_width // 2), (0, 0)), mode="edge")
        uniform_positions = np.column_stack(
            [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(positions.shape[1])]
        )
    velocity = np.diff(uniform_positions, axis=0) / step_s
    acceleration = np.diff(velocity, axis=0) / step_s if len(velocity) >= 2 else np.empty((0, 2))
    lateral_acceleration = np.empty(0)
    if len(acceleration) and len(velocity) >= len(acceleration):
        speed = np.linalg.norm(velocity[: len(acceleration)], axis=1)
        cross_product = np.abs(
            velocity[: len(acceleration), 0] * acceleration[:, 1]
            - velocity[: len(acceleration), 1] * acceleration[:, 0]
        )
        lateral_acceleration = cross_product / np.maximum(speed, 1.0)
    jerk = np.diff(acceleration, axis=0) / step_s if len(acceleration) >= 2 else np.empty((0, 2))
    return {
        "sampling_rate_hz": sample_rate_hz,
        "source_frames": len(frames),
        "resampled_frames": len(uniform_times),
        "resampling_filter": "triangular-5-sample" if filter_width > 1 else "none",
        "velocity_px_s": _summary(np.linalg.norm(velocity, axis=1)),
        "acceleration_px_s2": _summary(np.linalg.norm(acceleration, axis=1)),
        "lateral_acceleration_px_s2": _summary(lateral_acceleration),
        "jerk_px_s3": _summary(np.linalg.norm(jerk, axis=1)),
    }


def _normalized_entropy(angles: Sequence[float], bins: int = 16) -> float:
    if not angles:
        return 0.0
    values = (np.asarray(angles, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
    histogram, _ = np.histogram(values, bins=bins, range=(-math.pi, math.pi))
    probabilities = histogram[histogram > 0] / float(len(values))
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy / math.log(bins))


def _approach_aligned_summary(samples: Sequence[Sequence[float]] | np.ndarray) -> dict[str, Any]:
    """Summarize errors in the approach-aligned, radius-normalized frame.

    The summary intentionally uses the actual one-to-one matched press error,
    not a screen-space aggregate or a re-used press.  ``anisotropy`` is the
    square root of the covariance eigenvalue ratio, matching the convention
    used by the replay comparison analysis.  Skewness is the standardized
    third central moment and kurtosis is excess kurtosis.
    """
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return {
            "samples": 0,
            "longitudinal_mean": 0.0,
            "longitudinal_std": 0.0,
            "lateral_mean": 0.0,
            "lateral_std": 0.0,
            "covariance_eigen_anisotropy": 0.0,
            "undershoot_share": 0.0,
            "longitudinal_skewness": 0.0,
            "longitudinal_excess_kurtosis": 0.0,
            "lateral_skewness": 0.0,
            "lateral_excess_kurtosis": 0.0,
            "abs_axis_correlation": 0.0,
            "sector_entropy": 0.0,
            "tail_share_gt_1_0": 0.0,
            "rim_share_gt_0_8": 0.0,
            "wedge_share": 0.0,
        }
    values = values.reshape((-1, 2))
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values):
        return _approach_aligned_summary(np.empty((0, 2)))
    longitudinal = values[:, 0]
    lateral = values[:, 1]

    def standardized_moment(axis: np.ndarray, order: int, excess: bool = False) -> float:
        scale = float(np.std(axis))
        if scale <= 1e-9:
            return 0.0
        moment = float(np.mean(((axis - float(np.mean(axis))) / scale) ** order))
        return moment - 3.0 if excess else moment

    if len(values) >= 2:
        covariance = np.cov(values.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        anisotropy = math.sqrt(max(float(eigenvalues[-1]), 1e-12) / max(float(eigenvalues[0]), 1e-12))
    else:
        anisotropy = 0.0
    absolute_correlation = _correlation(np.abs(longitudinal), np.abs(lateral))
    angles = np.arctan2(lateral, longitudinal).tolist()
    radius = np.linalg.norm(values, axis=1)
    return {
        "samples": int(len(values)),
        "longitudinal_mean": round(float(np.mean(longitudinal)), 6),
        "longitudinal_std": round(float(np.std(longitudinal)), 6),
        "lateral_mean": round(float(np.mean(lateral)), 6),
        "lateral_std": round(float(np.std(lateral)), 6),
        "covariance_eigen_anisotropy": round(float(anisotropy), 6),
        "undershoot_share": round(float(np.mean(longitudinal < 0.0)), 6),
        "longitudinal_skewness": round(standardized_moment(longitudinal, 3), 6),
        "longitudinal_excess_kurtosis": round(standardized_moment(longitudinal, 4, excess=True), 6),
        "lateral_skewness": round(standardized_moment(lateral, 3), 6),
        "lateral_excess_kurtosis": round(standardized_moment(lateral, 4, excess=True), 6),
        "abs_axis_correlation": round(float(absolute_correlation or 0.0), 6),
        "sector_entropy": round(_normalized_entropy(angles), 6),
        "tail_share_gt_1_0": round(float(np.mean(radius > 1.0)), 6),
        "rim_share_gt_0_8": round(float(np.mean(radius > 0.8)), 6),
        "wedge_share": round(float(np.mean(np.abs(lateral) < 0.35 * np.maximum(np.abs(longitudinal), 0.03))), 6),
    }


def _trace_continuity_summary(
    frames: Sequence[TraceFrame],
    planner: HumanTracePlanner,
    timeline_start_ms: float,
) -> dict[str, Any]:
    """Measure carried speed at planned circle boundaries from the trace.

    This deliberately samples a fixed millisecond neighbourhood around each
    event instead of differentiating exact event-frame intervals.  It catches
    planner-generated stalls while remaining insensitive to the later cursor
    cadence used by the runtime dispatcher.
    """
    if len(frames) < 4:
        return {"source": "trace-fixed-window", "flow_0_45": {"samples": 0}, "turn_45_120": {"samples": 0}, "reversal_gt_120": {"samples": 0}, "tangent_reset_excess_deg": 0.0, "tangent_reset_excess_p95_deg": 0.0}
    frame_times = np.asarray([timeline_start_ms + frame.time_us / 1000.0 for frame in frames], dtype=float)
    positions = np.asarray([[frame.x, frame.y] for frame in frames], dtype=float)
    ordered = np.concatenate(([True], np.diff(frame_times) > 0.0))
    frame_times = frame_times[ordered]
    positions = positions[ordered]

    def position_at(time_ms: float) -> np.ndarray:
        return np.array(
            [
                np.interp(time_ms, frame_times, positions[:, axis])
                for axis in range(positions.shape[1])
            ],
            dtype=float,
        )

    def speed_at(time_ms: float, half_window_ms: float) -> float:
        left = position_at(time_ms - half_window_ms)
        right = position_at(time_ms + half_window_ms)
        return float(np.linalg.norm(right - left) / max(2.0 * half_window_ms / 1000.0, 1e-6))

    groups: dict[str, list[float]] = {"flow_0_45": [], "turn_45_120": [], "reversal_gt_120": []}
    tangent_excess: list[float] = []
    segment_indices = {
        int(segment["object_index"])
        for segment in planner.motion_segments
        if bool(segment.get("realized", True)) and bool(segment.get("target_reached", True))
    }
    segment_by_index = {
        int(segment["object_index"]): segment
        for segment in planner.motion_segments
        if bool(segment.get("realized", True))
    }
    map_objects = planner.map.objects

    def is_stacked_repeat(object_index: int) -> bool:
        if object_index <= 0 or object_index + 1 >= len(map_objects):
            return False
        current = map_objects[object_index]
        previous = map_objects[object_index - 1]
        following = map_objects[object_index + 1]
        if current.kind != "circle" or previous.kind != "circle" or following.kind != "circle":
            return True
        incoming = math.hypot(
            current.position.x - previous.end_position.x,
            current.position.y - previous.end_position.y,
        )
        outgoing = math.hypot(
            following.position.x - current.end_position.x,
            following.position.y - current.end_position.y,
        )
        stack_radius = max(float(current.radius), float(previous.radius), float(following.radius)) * 0.5
        return min(incoming, outgoing) < max(4.0, stack_radius)

    for segment in planner.motion_segments:
        if (
            not bool(segment.get("realized", True))
            or not bool(segment.get("target_reached", True))
            or bool(segment.get("break_before", False))
        ):
            continue
        object_index = int(segment["object_index"])
        if object_index - 1 not in segment_indices or object_index + 1 not in segment_indices:
            continue
        if is_stacked_repeat(object_index):
            continue
        angle = float(segment["corner_angle_deg"])
        hit_time = float(segment["end_time_ms"])
        if hit_time - frame_times[0] < 10.0 or hit_time + 10.0 > frame_times[-1]:
            continue
        local_window = max(2.0, min(8.0, 1000.0 / planner.profile.sample_rate_hz * 2.0))
        previous_gap = hit_time - float(segment_by_index[object_index - 1]["end_time_ms"])
        following_gap = float(segment_by_index[object_index + 1]["end_time_ms"]) - hit_time
        # Timing overlaps and stacked repeats do not provide a meaningful
        # before/after movement window; retain them in the object/timing
        # diagnostics but exclude them from continuity calibration.
        if min(previous_gap, following_gap) < max(3.0 * local_window, 12.0):
            continue
        boundary_speed = speed_at(hit_time, local_window)
        surrounding = [
            speed_at(hit_time - 4.0 * local_window, local_window),
            speed_at(hit_time + 4.0 * local_window, local_window),
        ]
        # Do not turn a near-idle neighbourhood into an artificial 10x
        # restart ratio.  250 px/s is below the profile's ordinary movement
        # envelope but above interpolation noise and deliberate idle drift.
        reference_speed = max(float(np.median(surrounding)), 250.0)
        carry = float(boundary_speed / reference_speed)
        if angle <= 45.0:
            groups["flow_0_45"].append(carry)
        elif angle <= 120.0:
            groups["turn_45_120"].append(carry)
        else:
            groups["reversal_gt_120"].append(carry)
        incoming = position_at(hit_time) - position_at(hit_time - 2.0 * local_window)
        outgoing = position_at(hit_time + 2.0 * local_window) - position_at(hit_time)
        if min(float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing))) < 1.5:
            continue
        incoming_unit = incoming / max(float(np.linalg.norm(incoming)), 1e-9)
        outgoing_unit = outgoing / max(float(np.linalg.norm(outgoing)), 1e-9)
        actual_turn = math.degrees(math.acos(float(np.clip(np.dot(incoming_unit, outgoing_unit), -1.0, 1.0))))
        tangent_excess.append(max(0.0, actual_turn - angle))

    def summarize(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        return {
            "samples": int(len(array)),
            "severe_stop_share": round(float(np.mean(array < 0.2)), 6) if len(array) else None,
            "median_carry_ratio": round(float(np.median(array)), 6) if len(array) else None,
            "p95_carry_ratio": round(float(np.percentile(array, 95)), 6) if len(array) else None,
            "max_carry_ratio": round(float(np.max(array)), 6) if len(array) else None,
        }

    return {
        "source": "trace-fixed-window",
        "flow_0_45": summarize(groups["flow_0_45"]),
        "turn_45_120": summarize(groups["turn_45_120"]),
        "reversal_gt_120": summarize(groups["reversal_gt_120"]),
        "tangent_reset_excess_deg": round(float(max(tangent_excess, default=0.0)), 6),
        "tangent_reset_excess_p95_deg": round(float(np.percentile(tangent_excess, 95)), 6) if tangent_excess else 0.0,
    }


def _trace_motion_summary(
    plan: MapPlan,
    frames: Sequence[TraceFrame],
    planner: HumanTracePlanner,
    timeline_start_ms: float,
) -> dict[str, Any]:
    """Measure sampled motion allocation and unnecessary compressed moves.

    Motion is measured on a uniform cursor cadence.  The trace may contain
    exact event frames for atomic key/cursor dispatch, but those off-grid
    frames are not allowed to manufacture a tiny derivative interval.  A
    transition is a flick candidate only when its map gap has ample time at
    the profile speed ceiling and the observed 10%-to-90% travel is both
    unusually compressed and unusually fast relative to that available gap.
    """
    empty = _summary([])
    if len(frames) < 3 or not plan.objects:
        return {
            "source": "trace-uniform-cadence",
            "sampling_rate_hz": planner.profile.sample_rate_hz,
            "transitions": 0,
            "launch_delay_share": empty,
            "transit_share": empty,
            "active_motion_share": empty,
            "speed_px_s": empty,
            "speed_over_gap_required": empty,
            "speed_over_ceiling": empty,
            "flick_count": 0,
            "flick_share": 0.0,
        }
    frame_times = np.asarray(
        [timeline_start_ms + frame.time_us / 1000.0 for frame in frames],
        dtype=float,
    )
    positions = np.asarray([[frame.x, frame.y] for frame in frames], dtype=float)
    ordered = np.concatenate(([True], np.diff(frame_times) > 0.0))
    frame_times = frame_times[ordered]
    positions = positions[ordered]
    step_ms = 1000.0 / max(1, planner.profile.sample_rate_hz)
    if len(frame_times) < 3 or frame_times[-1] - frame_times[0] < step_ms * 2.0:
        return {
            "source": "trace-uniform-cadence",
            "sampling_rate_hz": planner.profile.sample_rate_hz,
            "transitions": 0,
            "launch_delay_share": empty,
            "transit_share": empty,
            "active_motion_share": empty,
            "speed_px_s": empty,
            "speed_over_gap_required": empty,
            "speed_over_ceiling": empty,
            "flick_count": 0,
            "flick_share": 0.0,
        }
    uniform_times = np.arange(frame_times[0], frame_times[-1] + step_ms * 0.5, step_ms)
    uniform_positions = np.column_stack(
        [np.interp(uniform_times, frame_times, positions[:, axis]) for axis in range(2)]
    )
    velocity = np.diff(uniform_positions, axis=0) / max(step_ms / 1000.0, 1e-6)
    speed = np.linalg.norm(velocity, axis=1)
    velocity_times = uniform_times[1:]
    launch_delay: list[float] = []
    transit: list[float] = []
    active: list[float] = []
    transition_speeds: list[float] = []
    gap_speed_ratios: list[float] = []
    ceiling_ratios: list[float] = []
    flick_count = 0
    candidate_count = 0
    for object_index in range(1, len(plan.objects)):
        previous = plan.objects[object_index - 1]
        current = plan.objects[object_index]
        if previous.kind != "circle" or current.kind != "circle":
            continue
        start_time = float(previous.start_time_ms)
        end_time = float(current.start_time_ms)
        gap_ms = end_time - start_time
        vector = np.array(
            [current.position.x - previous.position.x, current.position.y - previous.position.y],
            dtype=float,
        )
        distance = float(np.linalg.norm(vector))
        if gap_ms <= max(3.0 * step_ms, 12.0) or distance < 20.0:
            continue
        mask = (uniform_times >= start_time) & (uniform_times <= end_time)
        local_times = uniform_times[mask]
        local_positions = uniform_positions[mask]
        if len(local_times) < 5:
            continue
        projection = (local_positions - np.array([previous.position.x, previous.position.y])) @ vector / max(
            distance * distance,
            1e-9,
        )
        above_10 = np.flatnonzero(projection >= 0.10)
        above_90 = np.flatnonzero(projection >= 0.90)
        if not len(above_10) or not len(above_90):
            continue
        time_10 = float(local_times[above_10[0]])
        time_90 = float(local_times[above_90[0]])
        if time_90 < time_10:
            continue
        candidate_count += 1
        gap_seconds = max(gap_ms / 1000.0, 1e-6)
        launch_share = float(np.clip((time_10 - start_time) / max(gap_ms, 1e-6), 0.0, 1.0))
        transit_share = float(np.clip((time_90 - time_10) / max(gap_ms, 1e-6), 0.0, 1.0))
        launch_delay.append(launch_share)
        transit.append(transit_share)
        local_speed_mask = (velocity_times >= start_time) & (velocity_times <= end_time)
        local_speeds = speed[local_speed_mask]
        if not len(local_speeds):
            continue
        active.append(float(np.mean(local_speeds > 100.0)))
        local_p95 = float(np.percentile(local_speeds, 95))
        required_speed = distance / gap_seconds
        ceiling = max(1.0, planner._speed_ceiling(distance))
        transition_speeds.append(local_p95)
        gap_speed_ratios.append(local_p95 / max(required_speed, 1.0))
        ceiling_ratios.append(local_p95 / ceiling)
        spare_gap = gap_ms >= 2.30 * distance / ceiling * 1000.0
        compressed = transit_share < 0.42
        unusually_fast = local_p95 > max(2.45 * required_speed, 1.28 * ceiling)
        if spare_gap and compressed and unusually_fast:
            flick_count += 1

    return {
        "source": "trace-uniform-cadence",
        "sampling_rate_hz": planner.profile.sample_rate_hz,
        "transitions": int(candidate_count),
        "launch_delay_share": _summary(launch_delay),
        "transit_share": _summary(transit),
        "active_motion_share": _summary(active),
        "speed_px_s": _summary(transition_speeds),
        "speed_over_gap_required": _summary(gap_speed_ratios),
        "speed_over_ceiling": _summary(ceiling_ratios),
        "flick_count": int(flick_count),
        "flick_share": round(flick_count / max(1, candidate_count), 6),
    }


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(right) < 3:
        return None
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if np.std(left_array) <= 1e-9 or np.std(right_array) <= 1e-9:
        return None
    value = float(np.corrcoef(left_array, right_array)[0, 1])
    return value if math.isfinite(value) else None


def _run_metrics(
    plan: MapPlan,
    frames: Sequence[TraceFrame],
    planner: HumanTracePlanner | None = None,
) -> dict[str, Any]:
    timeline_start = plan.objects[0].start_time_ms - 1500.0
    events = rising_press_events(frames, timeline_start_ms=timeline_start)
    assignments = match_press_events(plan.objects, events, minimum_window_ms=180.0)
    contexts = build_contexts(plan)
    radial: list[float] = []
    press_radial: list[float] = []
    angles: list[float] = []
    timing: list[float] = []
    approach_samples: list[tuple[float, float]] = []
    planned_approach_samples: list[tuple[float, float]] = []
    successful = 0
    judged_objects = 0
    matched = 0
    rows: list[dict[str, Any]] = []
    for object_position, obj in enumerate(plan.objects):
        if obj.kind == "spinner":
            continue
        judged_objects += 1
        planned_offset = planner.landing_offsets.get(object_position) if planner is not None else None
        if object_position:
            previous = plan.objects[object_position - 1]
            approach_vector = np.array(
                [obj.position.x - previous.end_position.x, obj.position.y - previous.end_position.y],
                dtype=float,
            )
        else:
            approach_vector = np.array([obj.position.x - 256.0, obj.position.y - 192.0], dtype=float)
        approach_length = float(np.linalg.norm(approach_vector))
        if approach_length > 1e-9:
            direction = approach_vector / approach_length
            normal = np.array([-direction[1], direction[0]], dtype=float)
            if planned_offset is not None:
                planned_vector = np.asarray(planned_offset, dtype=float)
                planned_aligned = (
                    float(np.dot(planned_vector, direction) / max(float(obj.radius), 1e-9)),
                    float(np.dot(planned_vector, normal) / max(float(obj.radius), 1e-9)),
                )
                planned_approach_samples.append(planned_aligned)
            else:
                planned_aligned = (0.0, 0.0)
        else:
            direction = np.zeros(2)
            normal = np.zeros(2)
            planned_aligned = (0.0, 0.0)
        press_position = assignments.get(object_position)
        if press_position is None:
            rows.append(
                {
                    "object_position": object_position,
                    "matched": False,
                    "success": False,
                    "planned_aligned_u": planned_aligned[0],
                    "planned_aligned_v": planned_aligned[1],
                }
            )
            continue
        matched += 1
        event = events[press_position]
        frame = frames[event.frame_index]
        timing_error = float(event.time_ms - obj.start_time_ms)
        offset_x = float(frame.x - obj.position.x)
        offset_y = float(frame.y - obj.position.y)
        aim_error = math.hypot(offset_x, offset_y)
        press_normalized_radius = aim_error / max(float(obj.radius), 1e-9)
        if planned_offset is None:
            offset = (offset_x, offset_y)
        else:
            offset = planned_offset
        normalized_radius = math.hypot(offset[0], offset[1]) / max(float(obj.radius), 1e-9)
        success = abs(timing_error) <= obj.hit_windows.meh_ms and aim_error <= obj.radius
        radial.append(normalized_radius)
        press_radial.append(press_normalized_radius)
        timing.append(timing_error)
        if approach_length > 1e-9:
            aligned = (
                float(np.dot(np.array([offset_x, offset_y]), direction) / max(float(obj.radius), 1e-9)),
                float(np.dot(np.array([offset_x, offset_y]), normal) / max(float(obj.radius), 1e-9)),
            )
            approach_samples.append(aligned)
        else:
            aligned = (0.0, 0.0)
        if math.hypot(*offset) > 1e-9:
            angles.append(math.atan2(offset[1], offset[0]))
        successful += int(success)
        rows.append(
            {
                "object_position": object_position,
                "matched": True,
                "success": success,
                "radial": normalized_radius,
                "press_radial": press_normalized_radius,
                "planned_offset_x": float(offset[0]),
                "planned_offset_y": float(offset[1]),
                "timing_error_ms": timing_error,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "aligned_u": aligned[0],
                "aligned_v": aligned[1],
                "planned_aligned_u": planned_aligned[0],
                "planned_aligned_v": planned_aligned[1],
                "context": contexts[object_position].label,
            }
        )

    radial_array = np.asarray(radial, dtype=float)
    timing_array = np.asarray(timing, dtype=float)
    context_counts: dict[str, int] = {}
    for context in contexts:
        if context.object_kind == "spinner":
            continue
        context_counts[context.label] = context_counts.get(context.label, 0) + 1
    context_total = max(1, sum(context_counts.values()))
    context_shares = {label: round(count / context_total, 6) for label, count in context_counts.items()}
    lag_correlations: dict[str, float] = {}
    for lag in range(2, 9):
        if len(radial_array) > lag:
            value = _correlation(radial_array[:-lag], radial_array[lag:])
            if value is not None:
                lag_correlations[str(lag)] = round(abs(value), 6)
    continuity = planner.continuity_stats if planner is not None else None
    if planner is not None:
        continuity = dict(continuity)
        continuity["trace"] = _trace_continuity_summary(frames, planner, timeline_start)
        continuity["motion"] = _trace_motion_summary(plan, frames, planner, timeline_start)
    return {
        "objects": len(plan.objects),
        "judged_objects": judged_objects,
        "presses": len(events),
        "matched_objects": matched,
        "successful_landings": successful,
        "matching_one_to_one": len(set(assignments.values())) == len(assignments),
        "radial": {
            "mean": round(float(np.mean(radial_array)), 6) if len(radial_array) else 0.0,
            "p95": round(float(np.percentile(radial_array, 95)), 6) if len(radial_array) else 0.0,
            "rim_share_gt_0_8": round(float(np.mean(radial_array > 0.8)), 6) if len(radial_array) else 0.0,
            "samples": int(len(radial_array)),
        },
        "press_radial": {
            "mean": round(float(np.mean(press_radial)), 6) if press_radial else 0.0,
            "p95": round(float(np.percentile(press_radial, 95)), 6) if press_radial else 0.0,
            "rim_share_gt_0_8": round(float(np.mean(np.asarray(press_radial) > 0.8)), 6) if press_radial else 0.0,
            "samples": len(press_radial),
        },
        "angular_entropy": round(_normalized_entropy(angles), 6),
        "angular_samples": len(angles),
        "approach_aligned": _approach_aligned_summary(approach_samples),
        "planned_approach_aligned": _approach_aligned_summary(planned_approach_samples),
        "lag_abs_correlation": lag_correlations,
        "timing": {
            "mean_signed_ms": round(float(np.mean(timing_array)), 6) if len(timing_array) else 0.0,
            "std_ms": round(float(np.std(timing_array)), 6) if len(timing_array) else 0.0,
            "p95_abs_ms": round(float(np.percentile(np.abs(timing_array), 95)), 6) if len(timing_array) else 0.0,
            "early_share": round(float(np.mean(timing_array < 0.0)), 6) if len(timing_array) else 0.0,
            "samples": int(len(timing_array)),
        },
        "context_distribution": {"counts": context_counts, "shares": context_shares},
        "continuity": continuity,
        "kinematics": _kinematic_summary(
            frames,
            sample_rate_hz=planner.profile.sample_rate_hz if planner is not None else 500,
        ),
        "rows": rows,
    }


def _cross_seed_metrics(run_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correlations: list[float] = []
    deltas: list[float] = []
    common_sample_count = 0
    pair_common_sample_counts: list[int] = []
    for left, right in itertools.combinations(run_records, 2):
        left_rows = {row["object_position"]: row for row in left["metrics"]["rows"] if row["matched"]}
        right_rows = {row["object_position"]: row for row in right["metrics"]["rows"] if row["matched"]}
        common = sorted(set(left_rows) & set(right_rows))
        if len(common) < 3:
            continue
        common_sample_count += len(common)
        pair_common_sample_counts.append(len(common))
        left_radial = [left_rows[index]["radial"] for index in common]
        right_radial = [right_rows[index]["radial"] for index in common]
        correlation = _correlation(left_radial, right_radial)
        if correlation is not None:
            correlations.append(abs(correlation))
        deltas.extend(
            math.hypot(
                left_rows[index].get("planned_offset_x", left_rows[index]["offset_x"])
                - right_rows[index].get("planned_offset_x", right_rows[index]["offset_x"]),
                left_rows[index].get("planned_offset_y", left_rows[index]["offset_y"])
                - right_rows[index].get("planned_offset_y", right_rows[index]["offset_y"]),
            )
            for index in common
        )
    return {
        "absolute_correlation": _summary(correlations),
        "landing_delta_px": _summary(deltas),
        "common_object_samples": common_sample_count,
        "minimum_pair_common_samples": min(pair_common_sample_counts, default=0),
        "checked": bool(pair_common_sample_counts) and min(pair_common_sample_counts) >= 32,
    }


def _aggregate_run_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    radial = [record["metrics"]["radial"]["mean"] for record in records]
    timing = [record["metrics"]["timing"]["p95_abs_ms"] for record in records]
    rim = [record["metrics"]["radial"]["rim_share_gt_0_8"] for record in records]
    entropy = [record["metrics"]["angular_entropy"] for record in records]
    jump_share = [record["metrics"]["context_distribution"]["shares"].get("jump", 0.0) for record in records]
    success = [
        record["metrics"]["successful_landings"] / max(1, record["metrics"]["judged_objects"])
        for record in records
    ]
    lag_values = [
        value
        for record in records
        for value in record["metrics"]["lag_abs_correlation"].values()
    ]
    kinematic_values = [record["metrics"]["kinematics"] for record in records]
    continuity_values = [record["metrics"].get("continuity") for record in records]
    continuity_values = [value for value in continuity_values if value is not None]
    trace_values = [value.get("trace") for value in continuity_values if value.get("trace") is not None]
    motion_values = [value.get("motion") for value in continuity_values if value.get("motion") is not None]
    approach_samples = [
        [row["aligned_u"], row["aligned_v"]]
        for record in records
        for row in record["metrics"]["rows"]
        if row.get("matched") and "aligned_u" in row and "aligned_v" in row
    ]
    planned_approach_samples = [
        [row["planned_aligned_u"], row["planned_aligned_v"]]
        for record in records
        for row in record["metrics"]["rows"]
        if "planned_aligned_u" in row and "planned_aligned_v" in row
    ]
    base_flow_values = [value["flow_0_45"] for value in continuity_values]
    base_turn_values = [value["turn_45_120"] for value in continuity_values]
    base_reversal_values = [value["reversal_gt_120"] for value in continuity_values]
    flow_values = [value["flow_0_45"] for value in trace_values] or base_flow_values
    turn_values = [value["turn_45_120"] for value in trace_values] or base_turn_values
    reversal_values = [value["reversal_gt_120"] for value in trace_values] or base_reversal_values
    flow_carries = [float(value["median_carry_ratio"]) for value in flow_values if value["median_carry_ratio"] is not None]
    flow_stops = [float(value["severe_stop_share"]) for value in flow_values if value["severe_stop_share"] is not None]
    turn_carries = [float(value["median_carry_ratio"]) for value in turn_values if value["median_carry_ratio"] is not None]
    return {
        "runs": len(records),
        "radial_mean": _summary(radial),
        "timing_p95_abs_ms": _summary(timing),
        "rim_share_gt_0_8": _summary(rim),
        "angular_entropy": _summary(entropy),
        "context_jump_share": _summary(jump_share),
        "angular_sample_count": int(sum(record["metrics"]["angular_samples"] for record in records)),
        "success_rate": _summary(success),
        "approach_aligned": _approach_aligned_summary(approach_samples),
        "planned_approach_aligned": _approach_aligned_summary(planned_approach_samples),
        "lag_abs_correlation_max": round(float(max(lag_values)), 6) if lag_values else 0.0,
        "successful_landings_total": int(sum(record["metrics"]["successful_landings"] for record in records)),
        "kinematics": {
            "max_speed_px_s": max((float(value["velocity_px_s"]["max"]) for value in kinematic_values), default=0.0),
            "max_acceleration_p95_px_s2": max((float(value["acceleration_px_s2"]["p95"]) for value in kinematic_values), default=0.0),
            "max_lateral_acceleration_p95_px_s2": max((float(value["lateral_acceleration_px_s2"]["p95"]) for value in kinematic_values), default=0.0),
            "max_jerk_p95_px_s3": max((float(value["jerk_px_s3"]["p95"]) for value in kinematic_values), default=0.0),
        },
        "motion": {
            "runs": len(motion_values),
            "transitions": int(sum(int(value.get("transitions", 0)) for value in motion_values)),
            "launch_delay_share": _summary(
                [float(value["launch_delay_share"]["mean"]) for value in motion_values]
            ),
            "transit_share": _summary(
                [float(value["transit_share"]["mean"]) for value in motion_values]
            ),
            "active_motion_share": _summary(
                [float(value["active_motion_share"]["mean"]) for value in motion_values]
            ),
            "speed_p95_px_s": _summary(
                [float(value["speed_px_s"]["p95"]) for value in motion_values]
            ),
            "speed_over_gap_required_p95": _summary(
                [float(value["speed_over_gap_required"]["p95"]) for value in motion_values]
            ),
            "speed_over_ceiling_p95": _summary(
                [float(value["speed_over_ceiling"]["p95"]) for value in motion_values]
            ),
            "flick_count": int(sum(int(value.get("flick_count", 0)) for value in motion_values)),
            "flick_share": _summary([float(value.get("flick_share", 0.0)) for value in motion_values]),
        },
        "continuity": {
            "runs": len(continuity_values),
            "flow_samples": int(sum(value["samples"] for value in flow_values)),
            "flow_severe_stop_share": _summary(flow_stops),
            "flow_median_carry_ratio": _summary(flow_carries),
            "turn_median_carry_ratio": _summary(turn_carries),
            "reversal_max_carry_ratio": max(
                (float(value["max_carry_ratio"]) for value in reversal_values if value["max_carry_ratio"] is not None),
                default=0.0,
            ),
            "base_boundary_position_error_px": max(
                (float(value["base_boundary_position_error_px"]) for value in continuity_values),
                default=0.0,
            ),
            "base_shared_position_error_px": max(
                (float(value["base_shared_position_error_px"]) for value in continuity_values),
                default=0.0,
            ),
            "base_shared_velocity_error_px_s": max(
                (float(value["base_shared_velocity_error_px_s"]) for value in continuity_values),
                default=0.0,
            ),
            "base_shared_acceleration_error_px_s2": max(
                (float(value["base_shared_acceleration_error_px_s2"]) for value in continuity_values),
                default=0.0,
            ),
            "tangent_reset_excess_deg": max(
                (float(value.get("trace", value)["tangent_reset_excess_deg"]) for value in continuity_values),
                default=0.0,
            ),
            "tangent_reset_excess_p95_deg": max(
                (float(value.get("trace", value).get("tangent_reset_excess_p95_deg", 0.0)) for value in continuity_values),
                default=0.0,
            ),
            "base_flow_median_carry_ratio": _summary(
                [float(value["median_carry_ratio"]) for value in base_flow_values if value["median_carry_ratio"] is not None]
            ),
            "trace": {
                "runs": len(trace_values),
                "flow_samples": int(sum(value["flow_0_45"]["samples"] for value in trace_values)),
                "flow_severe_stop_share": _summary(
                    [float(value["flow_0_45"]["severe_stop_share"]) for value in trace_values if value["flow_0_45"].get("severe_stop_share") is not None]
                ),
                "flow_median_carry_ratio": _summary(
                    [float(value["flow_0_45"]["median_carry_ratio"]) for value in trace_values if value["flow_0_45"].get("median_carry_ratio") is not None]
                ),
            },
        },
    }


def _profile_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = record["metrics"]
        rows.append(
            {
                "map": record["map"],
                "seed": record["seed"],
                "skill": record["skill"],
                "effort": record["effort"],
                "success_rate": metrics["successful_landings"] / max(1, metrics["judged_objects"]),
                "radial_mean": metrics["radial"]["mean"],
                "timing_p95_abs_ms": metrics["timing"]["p95_abs_ms"],
            }
        )
    return rows


def _monotonicity(rows: Sequence[dict[str, Any]], axis: str) -> dict[str, Any]:
    """Compare profile levels after aggregating paired multi-seed runs."""
    comparisons: list[dict[str, Any]] = []
    groups: dict[tuple[str, float], dict[float, list[dict[str, Any]]]] = {}
    for row in rows:
        fixed_value = float(row["effort"] if axis == "skill" else row["skill"])
        axis_value = float(row[axis])
        groups.setdefault((row["map"], fixed_value), {}).setdefault(axis_value, []).append(row)

    def aggregate(observations: Sequence[dict[str, Any]]) -> dict[str, float]:
        count = max(1, len(observations))
        return {
            "success_rate": sum(row["success_rate"] for row in observations) / count,
            "radial_mean": sum(row["radial_mean"] for row in observations) / count,
            "timing_p95_abs_ms": sum(row["timing_p95_abs_ms"] for row in observations) / count,
        }

    def by_seed(observations: Sequence[dict[str, Any]]) -> dict[Any, dict[str, float]]:
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for observation in observations:
            grouped.setdefault(observation.get("seed", 0), []).append(observation)
        return {seed: aggregate(seed_rows) for seed, seed_rows in grouped.items()}

    def comparison_metrics(left: dict[str, float], right: dict[str, float]) -> tuple[float, float, float, bool]:
        success_delta = right["success_rate"] - left["success_rate"]
        radial_ratio = right["radial_mean"] / max(left["radial_mean"], 1e-9)
        timing_ratio = right["timing_p95_abs_ms"] / max(left["timing_p95_abs_ms"], 1e-9)
        # Timing is noisier than the aim distribution; success and radial
        # error remain the stronger signals for a consistent regression.
        expected_improvement = success_delta >= -0.12 and radial_ratio <= 1.35 and timing_ratio <= 1.60
        return success_delta, radial_ratio, timing_ratio, expected_improvement

    for (map_name, fixed_value), values in groups.items():
        ordered_values = sorted(values)
        for left_value, right_value in zip(ordered_values, ordered_values[1:]):
            left = aggregate(values[left_value])
            right = aggregate(values[right_value])
            success_delta, radial_ratio, timing_ratio, aggregate_pass = comparison_metrics(left, right)
            left_by_seed = by_seed(values[left_value])
            right_by_seed = by_seed(values[right_value])
            common_seeds = sorted(set(left_by_seed) & set(right_by_seed), key=str)
            seed_failures = 0
            for seed in common_seeds:
                if not comparison_metrics(left_by_seed[seed], right_by_seed[seed])[3]:
                    seed_failures += 1
            seed_limit = max(1, len(common_seeds) // 3)
            passed = aggregate_pass and seed_failures <= seed_limit
            consistent_failure = not passed and (
                seed_failures > len(common_seeds) / 2.0 if common_seeds else not aggregate_pass
            )
            comparisons.append(
                {
                    "map": map_name,
                    "fixed_" + ("effort" if axis == "skill" else "skill"): fixed_value,
                    "from": left_value,
                    "to": right_value,
                    "success_delta": round(success_delta, 6),
                    "radial_ratio": round(radial_ratio, 6),
                    "timing_ratio": round(timing_ratio, 6),
                    "seed_count": len(common_seeds),
                    "seed_failures": seed_failures,
                    "consistent_failure": consistent_failure,
                    "pass": passed,
                }
            )
    failures = sum(not comparison["pass"] for comparison in comparisons)
    consistent_failures = sum(comparison["consistent_failure"] for comparison in comparisons)
    return {
        "axis": axis,
        "comparisons": comparisons,
        "failures": failures,
        "consistent_failures": consistent_failures,
        "pass": consistent_failures == 0,
    }


def _gate_report(
    *,
    same_seed: bool,
    map_reports: Sequence[dict[str, Any]],
    profile_report: dict[str, Any],
    gates: BenchmarkGates,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, value: Any, limit: Any, note: str) -> None:
        checks.append({"name": name, "pass": bool(passed), "value": value, "limit": limit, "note": note})

    add("same_seed_exact", same_seed, same_seed, True, "same map/profile/seed must reproduce exact trace frames")
    matching_ok = all(
        run["metrics"]["matching_one_to_one"]
        for report in map_reports
        for run in report["runs"]
    )
    add("one_to_one_matching", matching_ok, matching_ok, True, "a press may match at most one object")
    cross_values = [
        report["cross_seed"]["absolute_correlation"]["max"]
        for report in map_reports
        if report["cross_seed"]["checked"]
    ]
    cross_max = max(cross_values, default=0.0)
    add("cross_seed_correlation", cross_max <= gates.max_cross_seed_abs_correlation, round(cross_max, 6), gates.max_cross_seed_abs_correlation, "broad ceiling; checked only when every seed pair has at least 32 common object samples")
    lag_max = max((report["aggregate"]["lag_abs_correlation_max"] for report in map_reports), default=0.0)
    add("lag_periodicity", lag_max <= gates.max_lag_abs_correlation, round(lag_max, 6), gates.max_lag_abs_correlation, "lag-2..8 absolute correlation ceiling")
    rim_max = max((report["aggregate"]["rim_share_gt_0_8"]["max"] for report in map_reports), default=0.0)
    add("rim_share", rim_max <= gates.max_rim_share, round(rim_max, 6), gates.max_rim_share, "normalized landing radius > 0.8 must remain a minority")
    entropy_values = [
        report["aggregate"]["angular_entropy"]["min"]
        for report in map_reports
        if report["aggregate"]["angular_sample_count"] >= gates.entropy_min_samples
    ]
    entropy_min = min(entropy_values, default=None)
    add(
        "angular_entropy",
        entropy_min is None or entropy_min >= gates.min_angular_entropy,
        None if entropy_min is None else round(entropy_min, 6),
        gates.min_angular_entropy,
        f"checked only when there are at least {gates.entropy_min_samples} angular samples",
    )
    approach_reports = []
    planned_approach_reports = []
    approach_press_exclusions: list[dict[str, str]] = []
    for report in map_reports:
        aggregate = report["aggregate"]
        actual = aggregate.get("approach_aligned", {})
        planned = aggregate.get("planned_approach_aligned", {})
        if int(planned.get("samples", 0)) >= gates.approach_min_samples:
            planned_approach_reports.append(planned)
        if int(actual.get("samples", 0)) < gates.approach_min_samples:
            continue
        success_rate = float(aggregate.get("success_rate", {}).get("mean", 0.0))
        timing_p95 = float(aggregate.get("timing_p95_abs_ms", {}).get("mean", 0.0))
        if success_rate >= gates.approach_press_min_success_rate and timing_p95 <= gates.approach_press_max_timing_p95_ms:
            approach_reports.append(actual)
        else:
            reasons = []
            if success_rate < gates.approach_press_min_success_rate:
                reasons.append(f"success_rate<{gates.approach_press_min_success_rate:.2f}")
            if timing_p95 > gates.approach_press_max_timing_p95_ms:
                reasons.append(f"timing_p95_ms>{gates.approach_press_max_timing_p95_ms:.1f}")
            approach_press_exclusions.append({"map": str(report["map"]), "reason": ", ".join(reasons)})
    approach_undershoot = [float(value.get("undershoot_share", 0.0)) for value in approach_reports]
    approach_anisotropy = [float(value.get("covariance_eigen_anisotropy", 0.0)) for value in approach_reports]
    approach_correlation = [float(value.get("abs_axis_correlation", 0.0)) for value in approach_reports]
    approach_sector_entropy = [float(value.get("sector_entropy", 0.0)) for value in approach_reports]
    approach_wedge = [float(value.get("wedge_share", 0.0)) for value in approach_reports]
    add(
        "approach_undershoot_share",
        not approach_reports
        or (
            min(approach_undershoot) >= gates.min_approach_undershoot_share
            and max(approach_undershoot) <= gates.max_approach_undershoot_share
        ),
        None if not approach_reports else [round(min(approach_undershoot), 6), round(max(approach_undershoot), 6)],
        [gates.min_approach_undershoot_share, gates.max_approach_undershoot_share],
        "clean press-time maps retain a mild undershoot bias; excluded hard-map diagnostics remain in JSON",
    )
    add(
        "approach_aligned_anisotropy",
        not approach_reports or max(approach_anisotropy) <= gates.max_approach_anisotropy,
        None if not approach_reports else round(max(approach_anisotropy), 6),
        gates.max_approach_anisotropy,
        f"clean press-time maps with >= {gates.approach_min_samples} samples, success >= {gates.approach_press_min_success_rate:.2f}, timing p95 <= {gates.approach_press_max_timing_p95_ms:.1f} ms",
    )
    add(
        "approach_abs_axis_correlation",
        not approach_reports or max(approach_correlation) <= gates.max_approach_abs_axis_correlation,
        None if not approach_reports else round(max(approach_correlation), 6),
        gates.max_approach_abs_axis_correlation,
        "clean press-time lateral magnitude must not be a fixed fraction of longitudinal magnitude",
    )
    add(
        "approach_sector_entropy",
        not approach_reports or min(approach_sector_entropy) >= gates.min_approach_sector_entropy,
        None if not approach_reports else round(min(approach_sector_entropy), 6),
        gates.min_approach_sector_entropy,
        "clean press-time errors should occupy a cloud of sectors rather than a narrow cone",
    )
    add(
        "approach_wedge_share",
        not approach_reports or max(approach_wedge) <= gates.max_approach_wedge_share,
        None if not approach_reports else round(max(approach_wedge), 6),
        gates.max_approach_wedge_share,
        "clean press-time fraction inside the approach-axis wedge must remain a minority",
    )
    planned_undershoot = [float(value.get("undershoot_share", 0.0)) for value in planned_approach_reports]
    planned_anisotropy = [float(value.get("covariance_eigen_anisotropy", 0.0)) for value in planned_approach_reports]
    planned_correlation = [float(value.get("abs_axis_correlation", 0.0)) for value in planned_approach_reports]
    planned_sector_entropy = [float(value.get("sector_entropy", 0.0)) for value in planned_approach_reports]
    planned_wedge = [float(value.get("wedge_share", 0.0)) for value in planned_approach_reports]
    add(
        "planned_approach_undershoot_share",
        not planned_approach_reports
        or (
            min(planned_undershoot) >= gates.min_planned_approach_undershoot_share
            and max(planned_undershoot) <= gates.max_planned_approach_undershoot_share
        ),
        None if not planned_approach_reports else [round(min(planned_undershoot), 6), round(max(planned_undershoot), 6)],
        [gates.min_planned_approach_undershoot_share, gates.max_planned_approach_undershoot_share],
        "the sampled landing model must retain a mild undershoot bias without becoming one-sided",
    )
    add(
        "planned_approach_aligned_anisotropy",
        not planned_approach_reports or max(planned_anisotropy) <= gates.max_planned_approach_anisotropy,
        None if not planned_approach_reports else round(max(planned_anisotropy), 6),
        gates.max_planned_approach_anisotropy,
        "the sampled landing covariance must remain approximately elliptical",
    )
    add(
        "planned_approach_abs_axis_correlation",
        not planned_approach_reports or max(planned_correlation) <= gates.max_planned_approach_abs_axis_correlation,
        None if not planned_approach_reports else round(max(planned_correlation), 6),
        gates.max_planned_approach_abs_axis_correlation,
        "the sampled landing lateral magnitude must not be a fixed fraction of longitudinal magnitude",
    )
    add(
        "planned_approach_sector_entropy",
        not planned_approach_reports or min(planned_sector_entropy) >= gates.min_planned_approach_sector_entropy,
        None if not planned_approach_reports else round(min(planned_sector_entropy), 6),
        gates.min_planned_approach_sector_entropy,
        "the sampled landing cloud must occupy local-frame sectors rather than a narrow cone",
    )
    add(
        "planned_approach_wedge_share",
        not planned_approach_reports or max(planned_wedge) <= gates.max_planned_approach_wedge_share,
        None if not planned_approach_reports else round(max(planned_wedge), 6),
        gates.max_planned_approach_wedge_share,
        "the sampled landing cloud must not collapse into an approach-axis wedge",
    )
    total_success = sum(report["aggregate"]["successful_landings_total"] for report in map_reports)
    add("successful_landings", total_success >= gates.min_successful_landings, total_success, gates.min_successful_landings, "benchmark must produce at least one judged landing")
    kinematic_aggregates = [report["aggregate"].get("kinematics", {}) for report in map_reports]
    max_speed = max((float(value.get("max_speed_px_s", 0.0)) for value in kinematic_aggregates), default=0.0)
    max_acceleration_p95 = max(
        (float(value.get("max_acceleration_p95_px_s2", 0.0)) for value in kinematic_aggregates),
        default=0.0,
    )
    max_lateral_acceleration_p95 = max(
        (float(value.get("max_lateral_acceleration_p95_px_s2", 0.0)) for value in kinematic_aggregates),
        default=0.0,
    )
    max_jerk_p95 = max((float(value.get("max_jerk_p95_px_s3", 0.0)) for value in kinematic_aggregates), default=0.0)
    add("kinematic_speed_bound", max_speed <= gates.max_kinematic_speed_px_s, round(max_speed, 6), gates.max_kinematic_speed_px_s, "uniform-cadence trace speed bound")
    add("kinematic_acceleration_bound", max_acceleration_p95 <= gates.max_kinematic_acceleration_p95_px_s2, round(max_acceleration_p95, 6), gates.max_kinematic_acceleration_p95_px_s2, "uniform-cadence acceleration p95 bound")
    add("kinematic_lateral_acceleration_bound", max_lateral_acceleration_p95 <= gates.max_kinematic_lateral_acceleration_p95_px_s2, round(max_lateral_acceleration_p95, 6), gates.max_kinematic_lateral_acceleration_p95_px_s2, "uniform-cadence lateral acceleration p95 bound")
    add("kinematic_jerk_bound", max_jerk_p95 <= gates.max_kinematic_jerk_p95_px_s3, round(max_jerk_p95, 6), gates.max_kinematic_jerk_p95_px_s3, "uniform-cadence jerk p95 bound")
    motion_aggregates = [report["aggregate"].get("motion", {}) for report in map_reports]
    motion_flick_values = [
        float(value.get("flick_share", {}).get("max", 0.0))
        for value in motion_aggregates
        if int(value.get("transitions", 0)) >= gates.continuity_min_samples
    ]
    max_flick_share = max(motion_flick_values, default=0.0)
    add(
        "trace_flick_share",
        max_flick_share <= gates.max_trace_flick_share,
        round(max_flick_share, 6),
        gates.max_trace_flick_share,
        "ordinary transitions with spare time must not be compressed into unnecessary flicks",
    )
    continuity_reports = [report["aggregate"].get("continuity", {}) for report in map_reports]
    flow_reports = [
        report
        for report in continuity_reports
        if int(report.get("flow_samples", 0)) >= gates.continuity_min_samples
    ]
    flow_stop_max = max(
        (float(report["flow_severe_stop_share"]["max"]) for report in flow_reports),
        default=0.0,
    )
    flow_carry_min = min(
        (float(report["flow_median_carry_ratio"]["min"]) for report in flow_reports),
        default=None,
    )
    turn_carry_values = [
        float(report["turn_median_carry_ratio"]["mean"])
        for report in continuity_reports
        if report.get("turn_median_carry_ratio", {}).get("n", 0) > 0
    ]
    reversal_max = max(
        (float(report.get("reversal_max_carry_ratio", 0.0)) for report in continuity_reports),
        default=0.0,
    )
    add(
        "flow_severe_stop_share",
        not flow_reports or flow_stop_max <= gates.max_flow_severe_stop_share,
        round(flow_stop_max, 6),
        gates.max_flow_severe_stop_share,
        f"checked only when a map has at least {gates.continuity_min_samples} shallow-flow waypoints",
    )
    add(
        "flow_median_carry_ratio",
        not flow_reports or (flow_carry_min is not None and flow_carry_min >= gates.min_flow_median_carry_ratio),
        None if flow_carry_min is None else round(flow_carry_min, 6),
        gates.min_flow_median_carry_ratio,
        "ordinary 0-45 degree flow must retain substantial endpoint speed",
    )
    turn_below_flow = all(
        float(report["turn_median_carry_ratio"]["mean"]) <= float(report["flow_median_carry_ratio"]["mean"]) + 0.05
        for report in continuity_reports
        if report.get("flow_median_carry_ratio", {}).get("n", 0) > 0
        and report.get("turn_median_carry_ratio", {}).get("n", 0) > 0
    )
    add(
        "turn_carry_below_flow",
        turn_below_flow,
        turn_carry_values,
        "45-120 degree median carry should not exceed shallow flow by > 0.05",
        "geometry-derived carry should decrease with turn angle within stochastic trace tolerance",
    )
    add(
        "reversal_restart_spike",
        reversal_max <= gates.max_reversal_carry_ratio,
        round(reversal_max, 6),
        gates.max_reversal_carry_ratio,
        "near-reversal knots may slow down but must not create a restart-speed spike",
    )
    tangent_reset_p95_max = max(
        (float(report.get("tangent_reset_excess_p95_deg", 0.0)) for report in continuity_reports),
        default=0.0,
    )
    add(
        "tangent_reset_excess",
        tangent_reset_p95_max <= gates.max_tangent_reset_excess_p95_deg,
        round(tangent_reset_p95_max, 6),
        gates.max_tangent_reset_excess_p95_deg,
        "the p95 trace tangent change should not greatly exceed map corner geometry",
    )
    base_position_max = max(
        (float(report.get("base_boundary_position_error_px", 0.0)) for report in continuity_reports),
        default=0.0,
    )
    base_velocity_max = max(
        (float(report.get("base_shared_velocity_error_px_s", 0.0)) for report in continuity_reports),
        default=0.0,
    )
    base_acceleration_max = max(
        (float(report.get("base_shared_acceleration_error_px_s2", 0.0)) for report in continuity_reports),
        default=0.0,
    )
    add(
        "base_boundary_position_continuity",
        base_position_max <= gates.max_base_boundary_position_error_px,
        round(base_position_max, 9),
        gates.max_base_boundary_position_error_px,
        "quintic base path must arrive at every ordinary waypoint",
    )
    add(
        "base_velocity_continuity",
        base_velocity_max <= gates.max_base_shared_velocity_error_px_s,
        round(base_velocity_max, 9),
        gates.max_base_shared_velocity_error_px_s,
        "adjacent segments share the exact waypoint velocity",
    )
    add(
        "base_acceleration_continuity",
        base_acceleration_max <= gates.max_base_shared_acceleration_error_px_s2,
        round(base_acceleration_max, 9),
        gates.max_base_shared_acceleration_error_px_s2,
        "adjacent segments share the exact waypoint acceleration",
    )
    add(
        "skill_monotonicity",
        profile_report["skill"]["pass"],
        profile_report["skill"]["consistent_failures"],
        0,
        "higher skill should not consistently worsen error or success across selected maps and seeds",
    )
    add(
        "effort_monotonicity",
        profile_report["effort"]["pass"],
        profile_report["effort"]["consistent_failures"],
        0,
        "higher effort should not consistently worsen error or success across selected maps and seeds",
    )
    return {
        "pass": all(check["pass"] for check in checks),
        "checks": checks,
        "diagnostics": {
            "planned_approach_maps": len(planned_approach_reports),
            "clean_press_approach_maps": len(approach_reports),
            "excluded_press_approach_maps": approach_press_exclusions,
        },
    }


def run_benchmark(
    *,
    scope: str = "compact",
    map_paths: Iterable[str | Path] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    skill: float = DEFAULT_PROFILE[0],
    effort: float = DEFAULT_PROFILE[1],
    sample_rate_hz: int = 500,
    monotonicity_profiles: Sequence[tuple[float, float]] = MONOTONICITY_PROFILES,
    monotonicity_seeds: Sequence[int] | None = None,
    gates: BenchmarkGates | None = None,
    classification: str = "planner-only/not-runtime-validated",
    runtime_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a deterministic multi-map/multi-seed benchmark in memory."""
    selected_paths = [Path(path) for path in map_paths] if map_paths is not None else default_map_paths(scope)
    selected_paths = [path.resolve() for path in selected_paths]
    if not selected_paths or any(not path.is_file() for path in selected_paths):
        missing = [str(path) for path in selected_paths if not path.is_file()]
        raise FileNotFoundError(f"Benchmark map plan missing: {', '.join(missing)}")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one benchmark seed is required")
    monotonicity_seed_values = (
        tuple(int(seed) for seed in monotonicity_seeds)
        if monotonicity_seeds is not None
        else seed_values[: min(3, len(seed_values))]
    )
    if not monotonicity_seed_values:
        raise ValueError("at least one monotonicity seed is required")
    gates = gates or BenchmarkGates()

    map_reports: list[dict[str, Any]] = []
    same_seed_results: list[dict[str, Any]] = []
    for map_path in selected_paths:
        plan = load_map_plan(map_path)
        map_name = map_path.stem.removesuffix(".map.ndjson")
        records: list[dict[str, Any]] = []
        for seed in seed_values:
            profile = HumanProfile(
                percentile=99.5,
                seed=seed,
                sample_rate_hz=sample_rate_hz,
                skill_level=float(skill),
                effort_level=float(effort),
            )
            planner = HumanTracePlanner(plan, profile)
            frames = planner.generate()
            metrics = _run_metrics(plan, frames, planner)
            record = {"map": map_name, "path": str(map_path), "seed": seed, "skill": float(skill), "effort": float(effort), "metrics": metrics}
            records.append(record)

        repeat_profile = HumanProfile(99.5, seed_values[0], sample_rate_hz, skill_level=float(skill), effort_level=float(effort))
        first_frames = HumanTracePlanner(plan, repeat_profile).generate()
        second_frames = HumanTracePlanner(plan, repeat_profile).generate()
        first_signature = [(frame.time_us, frame.x, frame.y, frame.k1, frame.k2) for frame in first_frames]
        second_signature = [(frame.time_us, frame.x, frame.y, frame.k1, frame.k2) for frame in second_frames]
        same_seed_results.append({"map": map_name, "seed": seed_values[0], "exact": first_signature == second_signature, "frames": len(first_frames)})
        map_reports.append(
            {
                "map": map_name,
                "path": str(map_path),
                "aggregate": _aggregate_run_metrics(records),
                "cross_seed": _cross_seed_metrics(records),
                "runs": records,
            }
        )

    monotonicity_records: list[dict[str, Any]] = []
    for map_path in selected_paths:
        plan = load_map_plan(map_path)
        map_name = map_path.stem.removesuffix(".map.ndjson")
        for monotonicity_seed in monotonicity_seed_values:
            for profile_skill, profile_effort in monotonicity_profiles:
                profile = HumanProfile(99.5, monotonicity_seed, sample_rate_hz, skill_level=float(profile_skill), effort_level=float(profile_effort))
                planner = HumanTracePlanner(plan, profile)
                frames = planner.generate()
                monotonicity_records.append(
                    {
                        "map": map_name,
                        "seed": monotonicity_seed,
                        "skill": float(profile_skill),
                        "effort": float(profile_effort),
                        "metrics": _run_metrics(plan, frames, planner),
                    }
                )
    profile_rows = _profile_rows(monotonicity_records)
    profile_report = {
        "skill": _monotonicity(profile_rows[:], "skill"),
        "effort": _monotonicity(profile_rows[:], "effort"),
        "maps": [path.stem.removesuffix(".map.ndjson") for path in selected_paths],
        "seeds": list(monotonicity_seed_values),
        "rows": profile_rows,
    }
    same_seed = all(result["exact"] for result in same_seed_results)
    gate_report = _gate_report(same_seed=same_seed, map_reports=map_reports, profile_report=profile_report, gates=gates)
    git_commit = repository_git_commit()
    return {
        "schema_version": 1,
        "benchmark": "human-sim-statistical-regression",
        "planner_version": PLANNER_VERSION,
        "git_commit": git_commit,
        "build_identity": f"human-sim-python:{PLANNER_VERSION}:{git_commit}",
        "scope": scope,
        "classification": classification,
        "runtime_quality": runtime_quality,
        "configuration": {
            "maps": [str(path) for path in selected_paths],
            "seeds": list(seed_values),
            "monotonicity_seeds": list(monotonicity_seed_values),
            "skill": float(skill),
            "effort": float(effort),
            "sample_rate_hz": sample_rate_hz,
            "gates": gates.__dict__,
        },
        "same_seed": {"all_exact": same_seed, "runs": same_seed_results},
        "maps": map_reports,
        "monotonicity": profile_report,
        "gates": gate_report,
    }


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Human-sim benchmark: {report['scope']} / {report['classification']}",
        f"planner={report['planner_version']} git={report.get('git_commit', 'unknown')} maps={len(report['maps'])} seeds={report['configuration']['seeds']}",
        f"same-seed-exact={'PASS' if report['same_seed']['all_exact'] else 'FAIL'}",
    ]
    for map_report in report["maps"]:
        aggregate = map_report["aggregate"]
        cross = map_report["cross_seed"]["absolute_correlation"]
        cross_value = f"{cross['max']:.3f}" if map_report["cross_seed"]["checked"] else "n/a"
        continuity = aggregate.get("continuity", {})
        flow_carry = continuity.get("flow_median_carry_ratio", {}).get("mean")
        flow_stop = continuity.get("flow_severe_stop_share", {}).get("mean")
        approach = aggregate.get("approach_aligned", {})
        planned_approach = aggregate.get("planned_approach_aligned", {})
        motion = aggregate.get("motion", {})
        continuity_text = (
            f" flow_carry={flow_carry:.3f} flow_stop={flow_stop:.3f}"
            if flow_carry is not None and flow_stop is not None
            else ""
        )
        lines.append(
            f"{map_report['map']}: success={aggregate['success_rate']['mean']:.3f} "
            f"radial_mean={aggregate['radial_mean']['mean']:.3f} "
            f"rim={aggregate['rim_share_gt_0_8']['mean']:.3f} "
            f"entropy={aggregate['angular_entropy']['mean']:.3f} "
            f"jump_share={aggregate['context_jump_share']['mean']:.3f} "
            f"timing_p95={aggregate['timing_p95_abs_ms']['mean']:.2f}ms "
            f"cross_seed_abs_corr_max={cross_value} "
            f"aligned=(u:{approach.get('longitudinal_mean', 0.0):.3f},"
            f"v:{approach.get('lateral_mean', 0.0):.3f},"
            f"aniso:{approach.get('covariance_eigen_anisotropy', 0.0):.3f},"
            f"wedge:{approach.get('wedge_share', 0.0):.3f}) "
            f"planned=(aniso:{planned_approach.get('covariance_eigen_anisotropy', 0.0):.3f},"
            f"wedge:{planned_approach.get('wedge_share', 0.0):.3f}) "
            f"motion=(transit:{motion.get('transit_share', {}).get('mean', 0.0):.3f},"
            f"flick:{motion.get('flick_share', {}).get('mean', 0.0):.3f})"
            f"{continuity_text}"
        )
    lines.append(f"skill-monotonicity={'PASS' if report['monotonicity']['skill']['pass'] else 'FAIL'} effort-monotonicity={'PASS' if report['monotonicity']['effort']['pass'] else 'FAIL'}")
    lines.append(f"gates={'PASS' if report['gates']['pass'] else 'FAIL'}")
    for check in report["gates"]["checks"]:
        if not check["pass"]:
            lines.append(f"  FAIL {check['name']}: {check['value']} ({check['note']})")
    if report.get("runtime_quality"):
        quality = report["runtime_quality"]
        lines.append(f"runtime-quality={quality.get('status')} ({quality.get('classification')})")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: str | Path, summary_output: str | Path | None = None) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary = format_summary(report) + "\n"
    if summary_output:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8")
