"""Development implementation of OSI V2 circle-window human features.

All continuous descriptors use the same fixed 20 Hz low-pass and 50 Hz scored
cadence for human and generated traces.  Native replay samples are retained
only for gap and flicker-tail diagnostics.  This file has no final score until
calibration, mode cells and exclusions are frozen.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from human_sim.scoped_realism import (
    OUTPUT_STEP_MS, clip_transition_samples, common_bandwidth_positions,
    movement_descriptors,
)


def _point(point: Any) -> np.ndarray:
    if hasattr(point, "x") and hasattr(point, "y"):
        return np.asarray([float(point.x), float(point.y)], dtype=float)
    return np.asarray(point, dtype=float)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else 0.0


def circle_window_features(
    times_ms: Sequence[float],
    positions: Sequence[Sequence[float]],
    *,
    event_start_ms: float,
    event_end_ms: float,
    origin: Any,
    target: Any,
    target_radius_px: float,
) -> dict[str, Any]:
    """Extract map-relative descriptors on one predeclared circle transition."""
    if event_end_ms <= event_start_ms or target_radius_px <= 0:
        raise ValueError("invalid event window or target radius")
    origin_xy = _point(origin)
    target_xy = _point(target)
    route = target_xy - origin_xy
    route_length = float(np.linalg.norm(route))
    if route_length < 1e-3:
        raise ValueError("degenerate route distance")
    raw_t, raw_xy = clip_transition_samples(times_ms, positions, event_start_ms, event_end_ms)
    filtered_t, filtered_xy, bandwidth = common_bandwidth_positions(raw_t, raw_xy)
    movement, movement_support = movement_descriptors(raw_t, raw_xy)
    duration_s = (event_end_ms - event_start_ms) / 1000.0
    unit = route / route_length
    relative = filtered_xy - origin_xy
    projected = relative @ unit
    lateral = relative[:, 0] * unit[1] - relative[:, 1] * unit[0]
    velocity = np.diff(filtered_xy, axis=0) / (OUTPUT_STEP_MS / 1000.0)
    speed = np.linalg.norm(velocity, axis=1)
    scaled_speed = speed * duration_s / route_length
    normalized_progress = projected / route_length
    peak_index = int(np.argmax(speed)) if len(speed) else 0
    time_to_peak = float(peak_index / max(len(speed) - 1, 1))
    contact_xy = np.asarray([np.interp(event_end_ms, raw_t, raw_xy[:, axis]) for axis in range(2)])
    contact_error_px = float(np.linalg.norm(contact_xy - target_xy))
    projected_contact_error = float(np.dot(contact_xy - target_xy, unit) / route_length)

    raw_dt = np.diff(raw_t)
    raw_step = np.linalg.norm(np.diff(raw_xy, axis=0), axis=1)
    positive = raw_dt > 0
    native_speed = raw_step[positive] / (raw_dt[positive] / 1000.0)
    native_projected_step = np.diff((raw_xy - origin_xy) @ unit)
    native_reversals = np.count_nonzero(native_projected_step < -0.5)
    velocity_delta = np.linalg.norm(np.diff(velocity, axis=0), axis=1)
    lower_speed = scaled_speed < 0.05
    stop_restart = int(np.count_nonzero((~lower_speed[:-1]) & lower_speed[1:])) if len(lower_speed) > 1 else 0
    late_start = max(int(np.floor(0.75 * len(speed))), 0)
    late_path_fraction = float(np.sum(speed[late_start:]) / max(np.sum(speed), 1e-9))
    # Candidate correction descriptor. The 20 Hz filter suppresses native
    # replay cadence noise; calibration must decide whether this separates
    # human corrections from deliberate corruptions before it receives weight.
    local_peaks = np.flatnonzero((speed[1:-1] > speed[:-2]) & (speed[1:-1] >= speed[2:])) + 1
    peak_threshold = 0.2 * max(float(np.max(speed)) if len(speed) else 0.0, 1e-9)
    secondary_peaks = local_peaks[(local_peaks > peak_index + 1) & (speed[local_peaks] >= peak_threshold)]

    curvature = np.zeros(max(len(velocity) - 1, 0), dtype=float)
    if len(curvature):
        cross = velocity[:-1, 0] * velocity[1:, 1] - velocity[:-1, 1] * velocity[1:, 0]
        dot = np.sum(velocity[:-1] * velocity[1:], axis=1)
        curvature = np.arctan2(cross, dot)
    speed_turn_corr = 0.0
    if len(curvature) >= 3 and np.std(speed[1:]) > 1e-9 and np.std(np.abs(curvature)) > 1e-9:
        speed_turn_corr = float(np.corrcoef(speed[1:], np.abs(curvature))[0, 1])

    return {
        "speed_phase": {
            "peak_time_fraction": time_to_peak,
            "log_speed_p50": movement["log_speed_p50"],
            "log_speed_p90": movement["log_speed_p90"],
            "stop_fraction": movement["stop_fraction"],
            "early_progress_fraction": float(np.interp(0.25, np.linspace(0, 1, len(normalized_progress)), normalized_progress)),
            "late_progress_fraction": float(np.interp(0.75, np.linspace(0, 1, len(normalized_progress)), normalized_progress)),
            "late_path_fraction": late_path_fraction,
            "secondary_filtered_speed_peaks": float(len(secondary_peaks)),
        },
        "spatial_contact": {
            "log_path_length_ratio": movement["log_path_length_ratio"],
            "lateral_rms_over_route": float(np.sqrt(np.mean(lateral**2)) / route_length),
            "lateral_p95_over_route": _quantile(np.abs(lateral), 0.95) / route_length,
            "contact_error_over_radius": contact_error_px / target_radius_px,
            "contact_legal": float(contact_error_px <= target_radius_px),
            "signed_contact_overshoot": projected_contact_error,
            "reversal_distance_ratio": movement["log1p_reversal_distance_ratio"],
        },
        "continuity": {
            "log_acceleration_p95": movement["log_acceleration_p95"],
            "log_jerk_p95": movement["log_jerk_p95"],
            "stop_restart_count": float(stop_restart),
            "high_frequency_power_share": movement["high_frequency_power_share"],
            "filtered_velocity_jump_p95_px_s": _quantile(velocity_delta, 0.95),
        },
        "coupled_motor": {
            "speed_turn_correlation": speed_turn_corr,
            "contact_error_times_speed_p90": (contact_error_px / target_radius_px) * float(np.exp(movement["log_speed_p90"])),
            "turning_rms_radians": float(np.sqrt(np.mean(curvature**2))) if len(curvature) else 0.0,
        },
        "support": {
            "route_distance_px": route_length,
            "target_diameter_px": 2.0 * target_radius_px,
            "fitts_like_difficulty_covariate": float(np.log2(1.0 + route_length / (2.0 * target_radius_px))),
            "route_angle_radians": float(np.arctan2(route[1], route[0])),
            "raw_samples": len(raw_t),
            "native_nonpositive_gaps": int(np.count_nonzero(raw_dt <= 0)),
            "native_gaps_over_40ms": int(np.count_nonzero(raw_dt > 40)),
            "native_max_gap_ms": float(np.max(raw_dt)) if len(raw_dt) else None,
            "filtered_samples": len(filtered_t),
            "bandwidth": bandwidth,
            "movement": movement_support,
        },
        "native_diagnostics_unscored": {
            "step_p99_px": _quantile(raw_step, 0.99),
            "step_max_px": float(np.max(raw_step)) if len(raw_step) else 0.0,
            "speed_p99_px_s": _quantile(native_speed, 0.99),
            "reversal_count": float(native_reversals),
        },
    }


def _event_samples(
    times_ms: Sequence[float],
    positions: Sequence[Sequence[float]],
    start_ms: float,
    end_ms: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if end_ms <= start_ms:
        raise ValueError("nonpositive event duration")
    raw_t, raw_xy = clip_transition_samples(times_ms, positions, start_ms, end_ms)
    filtered_t, filtered_xy, support = common_bandwidth_positions(raw_t, raw_xy)
    # The shared filter returns times relative to its input window start.
    return filtered_t + start_ms, filtered_xy, support


def slider_window_features(
    times_ms: Sequence[float],
    positions: Sequence[Sequence[float]],
    slider: Any,
) -> dict[str, Any]:
    """Path-relative slider following at the common effective bandwidth."""
    if getattr(slider, "kind", None) != "slider" or len(slider.path_samples) < 2:
        raise ValueError("slider path samples unavailable")
    start_ms, end_ms = float(slider.start_time_ms), float(slider.end_time_ms)
    times, cursor, support = _event_samples(times_ms, positions, start_ms, end_ms)
    path = np.asarray([_point(point) for point in slider.path_samples], dtype=float)
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if lengths[-1] <= 1e-6:
        raise ValueError("degenerate slider path")
    spans = int(slider.repeat_count) + 1
    progress = np.clip((times - start_ms) / (end_ms - start_ms) * spans, 0.0, float(spans))
    span_index = np.minimum(np.floor(progress).astype(int), spans - 1)
    within = np.where(span_index % 2 == 0, progress - span_index, 1.0 - (progress - span_index))
    target_distance = within * lengths[-1]
    target = np.column_stack([np.interp(target_distance, lengths, path[:, axis]) for axis in range(2)])
    error = np.linalg.norm(cursor - target, axis=1)
    radius = float(slider.radius)
    if radius <= 0:
        raise ValueError("invalid slider radius")
    cursor_velocity = np.diff(cursor, axis=0) / (OUTPUT_STEP_MS / 1000.0)
    target_velocity = np.diff(target, axis=0) / (OUTPUT_STEP_MS / 1000.0)
    tangent = target_velocity / np.maximum(np.linalg.norm(target_velocity, axis=1)[:, None], 1e-9)
    signed_lag = np.sum((cursor[:-1] - target[:-1]) * tangent, axis=1) / radius
    tail_xy = np.asarray([np.interp(end_ms, np.asarray(times_ms, dtype=float), np.asarray(positions, dtype=float)[:, axis]) for axis in range(2)])
    tail_error = float(np.linalg.norm(tail_xy - _point(slider.end_position)) / radius)
    return {
        "mode": "slider",
        "features": {
            "follow_error_median_radius": _quantile(error / radius, 0.5),
            "follow_error_p95_radius": _quantile(error / radius, 0.95),
            "inside_ball_fraction": float(np.mean(error <= radius)),
            "signed_tangential_lag_median_radius": _quantile(signed_lag, 0.5),
            "tangential_lag_p95_radius": _quantile(np.abs(signed_lag), 0.95),
            "tail_error_radius": tail_error,
            "cursor_target_speed_ratio": float(np.sum(np.linalg.norm(cursor_velocity, axis=1)) / max(np.sum(np.linalg.norm(target_velocity, axis=1)), 1e-9)),
        },
        "repeat_count": int(slider.repeat_count),
        "support": support,
    }


def spinner_window_features(
    times_ms: Sequence[float],
    positions: Sequence[Sequence[float]],
    spinner: Any,
) -> dict[str, Any]:
    """Rotation organization during a spinner, independent of exact phase."""
    if getattr(spinner, "kind", None) != "spinner":
        raise ValueError("object is not a spinner")
    times, cursor, support = _event_samples(times_ms, positions, spinner.start_time_ms, spinner.end_time_ms)
    relative = cursor - _point(spinner.position)
    radius = np.linalg.norm(relative, axis=1)
    angle = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    angular_velocity = np.diff(angle) / (OUTPUT_STEP_MS / 1000.0)
    absolute_speed = np.abs(angular_velocity)
    direction = np.sign(angular_velocity)
    direction_reversal = int(np.count_nonzero(direction[1:] * direction[:-1] < 0))
    return {
        "mode": "spinner",
        "features": {
            "angular_speed_median_rad_s": _quantile(absolute_speed, 0.5),
            "angular_speed_p95_rad_s": _quantile(absolute_speed, 0.95),
            "radial_distance_median_px": _quantile(radius, 0.5),
            "radial_distance_iqr_px": _quantile(radius, 0.75) - _quantile(radius, 0.25),
            "rotation_direction_reversals": float(direction_reversal),
            "net_over_total_rotation": float(abs(np.sum(angular_velocity)) / max(np.sum(absolute_speed), 1e-9)),
        },
        "support": support,
    }


def break_window_features(
    times_ms: Sequence[float],
    positions: Sequence[Sequence[float]],
    *,
    break_start_ms: float,
    break_end_ms: float,
    next_target: Any,
    next_radius_px: float,
) -> dict[str, Any]:
    """Cursor occupancy, pause and return timing on an actual map break."""
    if break_end_ms - break_start_ms < 1000 or next_radius_px <= 0:
        raise ValueError("break shorter than one second or invalid next radius")
    times, cursor, support = _event_samples(times_ms, positions, break_start_ms, break_end_ms)
    speed = np.linalg.norm(np.diff(cursor, axis=0), axis=1) / (OUTPUT_STEP_MS / 1000.0)
    distance_to_next = np.linalg.norm(cursor - _point(next_target), axis=1)
    radius = next_radius_px
    inside = distance_to_next <= radius
    if np.any(inside):
        return_fraction = float((times[np.argmax(inside)] - break_start_ms) / (break_end_ms - break_start_ms))
    else:
        return_fraction = 1.0
    return {
        "mode": "break_free_roam",
        "features": {
            "pause_fraction_speed_lt_15_px_s": float(np.mean(speed < 15.0)),
            "wander_path_px": float(np.sum(np.linalg.norm(np.diff(cursor, axis=0), axis=1))),
            "in_playfield_fraction": float(np.mean((cursor[:, 0] >= 0) & (cursor[:, 0] <= 512) & (cursor[:, 1] >= 0) & (cursor[:, 1] <= 384))),
            "return_onset_fraction": return_fraction,
            "end_distance_to_next_radius": float(distance_to_next[-1] / radius),
        },
        "support": support,
    }
