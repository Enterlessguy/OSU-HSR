"""Scoped human-reference similarity for route-conditioned circle execution.

This is deliberately not ORI and its score is not a probability or a
"percentage human".  It compares normalized phase profiles on a fixed 32-point
time grid.  Calibration uses only independently grouped human validation rows;
candidate evaluation is a separate operation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .execution_training import PHASE_POINTS, _extract_phase_label_arrays, transition_route
from .execution_exploratory import _read_decoded
from .io import load_map_plan


SCOPED_REALISM_ID = "hsr-scoped-human-execution-similarity-v1"
DESCRIPTORS = (
    "phase_at_25",
    "phase_at_50",
    "phase_at_75",
    "peak_time",
    "dwell_fraction",
    "accel_decel_asymmetry",
    "log_phase_jerk_rms",
    "speed_entropy",
)
MOVEMENT_DESCRIPTORS = (
    "log_speed_p50",
    "log_speed_p90",
    "log_acceleration_p95",
    "log_jerk_p95",
    "stop_fraction",
    "high_frequency_power_share",
    "log_path_length_ratio",
    "log1p_cross_track_rms_ratio",
    "log1p_reversal_distance_ratio",
    "log1p_endpoint_velocity_difference",
)
WORKING_STEP_MS = 2.0
OUTPUT_STEP_MS = 20.0
LOWPASS_HZ = 20.0
FILTER_SIGMA_MS = 1000.0 * np.sqrt(np.log(2.0)) / (2.0 * np.pi * LOWPASS_HZ)
FILTER_EDGE_MS = 20.0
MAX_SOURCE_GAP_MS = 40.0
MIN_FILTERED_SAMPLES = 6
SEVERITIES = (0.25, 0.5, 0.75)
NEGATIVE_ANCHOR_CORRUPTIONS = ("late_snap", "early_snap", "pause_restart", "log_increment_noise")
TEMPLATE_CONTROLS = ("linear_template", "minimum_jerk_template")
CORRUPTIONS = NEGATIVE_ANCHOR_CORRUPTIONS + TEMPLATE_CONTROLS


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_increments(values: Sequence[float]) -> np.ndarray:
    increments = np.asarray(values, dtype=float)
    if increments.shape != (PHASE_POINTS - 1,) or not np.all(np.isfinite(increments)):
        raise ValueError(f"phase profile must have {PHASE_POINTS - 1} finite increments")
    increments = np.maximum(increments, 1e-9)
    return increments / float(np.sum(increments))


def _profile_shape(values: Sequence[float]) -> dict[str, float | np.ndarray]:
    increments = _normalize_increments(values)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    peak_index = int(np.argmax(increments))
    changes = np.diff(increments)
    before = changes[:peak_index]
    after = changes[peak_index:]
    acceleration = float(np.sum(np.maximum(before, 0.0)))
    deceleration = float(np.sum(np.maximum(-after, 0.0)))
    asymmetry = (acceleration - deceleration) / max(acceleration + deceleration, 1e-9)
    jerk = np.diff(increments, n=2) * float((PHASE_POINTS - 1) ** 3)
    jerk_rms = float(np.sqrt(np.mean(jerk**2))) if len(jerk) else 0.0
    positive = increments[increments > 0]
    entropy = float(-np.sum(positive * np.log(positive)) / np.log(len(increments)))
    return {
        "increments": increments,
        "cumulative": cumulative,
        "peak_time": peak_index / float(PHASE_POINTS - 2),
        "dwell_fraction": float(np.mean(increments < (0.1 / (PHASE_POINTS - 1)))),
        "asymmetry": asymmetry,
        "log_jerk_rms": float(np.log1p(jerk_rms)),
        "speed_entropy": entropy,
    }


def profile_descriptors(values: Sequence[float]) -> dict[str, float]:
    profile = _profile_shape(values)
    cumulative = np.asarray(profile["cumulative"], dtype=float)
    q25 = int(round(0.25 * (PHASE_POINTS - 1)))
    q50 = int(round(0.50 * (PHASE_POINTS - 1)))
    q75 = int(round(0.75 * (PHASE_POINTS - 1)))
    return {
        "phase_at_25": float(cumulative[q25]),
        "phase_at_50": float(cumulative[q50]),
        "phase_at_75": float(cumulative[q75]),
        "peak_time": float(profile["peak_time"]),
        "dwell_fraction": float(profile["dwell_fraction"]),
        "accel_decel_asymmetry": float(profile["asymmetry"]),
        "log_phase_jerk_rms": float(profile["log_jerk_rms"]),
        "speed_entropy": float(profile["speed_entropy"]),
    }


def profile_distance_components(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    """Return complementary, dimensionless phase-profile differences."""
    a = _profile_shape(left)
    b = _profile_shape(right)
    ac = np.asarray(a["cumulative"], dtype=float)
    bc = np.asarray(b["cumulative"], dtype=float)
    ai = np.asarray(a["increments"], dtype=float)
    bi = np.asarray(b["increments"], dtype=float)
    q25 = int(round(0.25 * (PHASE_POINTS - 1)))
    q75 = int(round(0.75 * (PHASE_POINTS - 1)))
    return {
        "curve_rmse": float(np.sqrt(np.mean((ac - bc) ** 2))),
        "increment_w1": float(np.mean(np.abs(np.sort(ai) - np.sort(bi)))),
        "peak_time_error": abs(float(a["peak_time"]) - float(b["peak_time"])),
        "phase_at_25_error": abs(float(ac[q25]) - float(bc[q25])),
        "phase_at_75_error": abs(float(ac[q75]) - float(bc[q75])),
        "dwell_fraction_error": abs(float(a["dwell_fraction"]) - float(b["dwell_fraction"])),
        "accel_decel_asymmetry_error": abs(float(a["asymmetry"]) - float(b["asymmetry"])),
        "log_phase_jerk_rms_error": abs(float(a["log_jerk_rms"]) - float(b["log_jerk_rms"])),
    }


def _context(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row["context"], dtype=float)


def nearest_independent_pairs(rows: Sequence[Mapping[str, Any]], *, caliper_quantile: float = 0.95) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Match each row to nearest different-player and different-family row."""
    if not rows:
        return [], {"eligible": 0, "accepted": 0}
    contexts = np.asarray([_context(row) for row in rows], dtype=float)
    center = np.median(contexts, axis=0)
    scale = np.subtract(*np.percentile(contexts, [75, 25], axis=0))
    scale = np.maximum(scale, 1e-6)
    standardized = (contexts - center) / scale
    proposed: list[tuple[int, int, float]] = []
    for index, row in enumerate(rows):
        allowed = [
            other for other, candidate in enumerate(rows)
            if other != index
            and candidate["player"] != row["player"]
            and candidate["map_family"] != row["map_family"]
        ]
        if not allowed:
            continue
        distances = np.linalg.norm(standardized[allowed] - standardized[index], axis=1)
        selected_position = int(np.argmin(distances))
        proposed.append((index, allowed[selected_position], float(distances[selected_position])))
    if not proposed:
        return [], {"eligible": 0, "accepted": 0}
    threshold = float(np.quantile([item[2] for item in proposed], caliper_quantile))
    accepted = [(left, right) for left, right, distance in proposed if distance <= threshold + 1e-12]
    return accepted, {
        "eligible": len(proposed),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(proposed),
        "caliper_quantile": caliper_quantile,
        "caliper_distance": threshold,
        "nearest_distance_p50": float(np.median([item[2] for item in proposed])),
        "nearest_distance_p95": float(np.quantile([item[2] for item in proposed], 0.95)),
        "context_fields": ["log_duration_ms", "log_distance_px", "turn_cos", "turn_sin"],
        "matching_rule": "nearest robust-IQR-standardized context; different player and map family",
    }


def _template(name: str, original: np.ndarray, severity: float, seed: int) -> np.ndarray:
    count = len(original)
    if name == "late_snap":
        target = np.full(count, 1e-9)
        target[max(count - max(1, count // 10), 0):] = 1.0
    elif name == "early_snap":
        target = np.full(count, 1e-9)
        target[:max(1, count // 10)] = 1.0
    elif name == "pause_restart":
        target = np.ones(count)
        target[count // 3: 2 * count // 3] = 1e-9
    elif name == "linear_template":
        target = np.ones(count)
    elif name == "minimum_jerk_template":
        time = np.linspace(0.0, 1.0, count + 1)
        phase = 10 * time**3 - 15 * time**4 + 6 * time**5
        target = np.diff(phase)
    elif name == "log_increment_noise":
        rng = np.random.default_rng(seed)
        target = original * np.exp(rng.normal(0.0, 4.0, count))
    else:
        raise ValueError(f"unknown corruption: {name}")
    target = _normalize_increments(target)
    return _normalize_increments((1.0 - severity) * original + severity * target)


def _wasserstein_equal_weight(left: Sequence[float], right: Sequence[float]) -> float:
    """Numerical W1 for finite, equal-weight empirical scalar samples."""
    a = np.sort(np.asarray(left, dtype=float))
    b = np.sort(np.asarray(right, dtype=float))
    if not len(a) or not len(b):
        raise ValueError("Wasserstein samples cannot be empty")
    quantiles = (np.arange(512, dtype=float) + 0.5) / 512.0
    aq = np.quantile(a, quantiles)
    bq = np.quantile(b, quantiles)
    return float(np.mean(np.abs(aq - bq)))


def _validation_scale(values: Sequence[float], *, floor: float, rule: str = "robust_tail") -> float:
    """Apply one named validation spread rule uniformly to a descriptor set."""
    array = np.asarray(values, dtype=float)
    p05, p25, p75, p95 = np.percentile(array, [5, 25, 75, 95])
    if rule == "robust_tail":
        return max(float(p75 - p25), 0.1 * float(p95 - p05), floor)
    if rule == "iqr_floor_sensitivity":
        return max(float(p75 - p25), floor)
    raise ValueError(f"unknown validation scale rule: {rule}")


def population_distance(
    left_profiles: Sequence[Sequence[float]],
    right_profiles: Sequence[Sequence[float]],
    scales: Mapping[str, float],
    names: Sequence[str] = DESCRIPTORS,
) -> tuple[float, dict[str, float]]:
    left = [profile_descriptors(profile) for profile in left_profiles]
    right = [profile_descriptors(profile) for profile in right_profiles]
    distances = {
        name: _wasserstein_equal_weight([row[name] for row in left], [row[name] for row in right])
        for name in names
    }
    total = float(np.mean([distances[name] / float(scales[name]) for name in names]))
    return total, distances


def _gaussian_kernel(sigma_samples: float) -> np.ndarray:
    radius = max(1, int(np.ceil(4.0 * sigma_samples)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    return kernel / float(np.sum(kernel))


def common_bandwidth_positions(
    times_ms: Sequence[float],
    positions: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Symmetrically low-pass and resample one cursor transition."""
    times = np.asarray(times_ms, dtype=float)
    points = np.asarray(positions, dtype=float)
    if len(times) < 4 or points.shape != (len(times), 2):
        raise ValueError("sparse cursor transition")
    keep = np.r_[True, np.diff(times) > 0]
    times, points = times[keep], points[keep]
    if len(times) < 4 or times[-1] - times[0] < 2 * FILTER_EDGE_MS + OUTPUT_STEP_MS * (MIN_FILTERED_SAMPLES - 1):
        raise ValueError("transition too short for common-bandwidth features")
    relative = times - times[0]
    working = np.arange(0.0, relative[-1] + WORKING_STEP_MS * 0.25, WORKING_STEP_MS)
    interpolated = np.column_stack([
        np.interp(working, relative, points[:, axis]) for axis in range(2)
    ])
    kernel = _gaussian_kernel(FILTER_SIGMA_MS / WORKING_STEP_MS)
    # ``same`` convolution implicitly zero-pads and therefore makes the score
    # depend on the absolute screen origin.  Reflecting the signal preserves
    # translation invariance while retaining the predeclared symmetric filter.
    radius = len(kernel) // 2
    filtered = np.column_stack([
        np.convolve(np.pad(interpolated[:, axis], radius, mode="reflect"), kernel, mode="valid")
        for axis in range(2)
    ])
    output = np.arange(FILTER_EDGE_MS, relative[-1] - FILTER_EDGE_MS + OUTPUT_STEP_MS * 0.25, OUTPUT_STEP_MS)
    sampled = np.column_stack([np.interp(output, working, filtered[:, axis]) for axis in range(2)])
    valid = np.ones(len(output), dtype=bool)
    gaps = np.diff(relative)
    for gap_index in np.flatnonzero(gaps > MAX_SOURCE_GAP_MS):
        gap_start = relative[gap_index] - FILTER_EDGE_MS
        gap_end = relative[gap_index + 1] + FILTER_EDGE_MS
        valid &= ~((output >= gap_start) & (output <= gap_end))
    output, sampled = output[valid], sampled[valid]
    if len(output) < MIN_FILTERED_SAMPLES or np.any(np.diff(output) > OUTPUT_STEP_MS * 1.01):
        raise ValueError("insufficient contiguous common-bandwidth support")
    return output, sampled, {
        "working_step_ms": WORKING_STEP_MS,
        "output_step_ms": OUTPUT_STEP_MS,
        "lowpass_minus_3db_hz": LOWPASS_HZ,
        "gaussian_sigma_ms": FILTER_SIGMA_MS,
        "filter_edge_trim_ms": FILTER_EDGE_MS,
        "source_gap_mask_threshold_ms": MAX_SOURCE_GAP_MS,
        "source_samples": len(times),
        "output_samples": len(output),
        "source_median_dt_ms": float(np.median(np.diff(times))),
        "source_max_dt_ms": float(np.max(np.diff(times))),
    }


def clip_transition_samples(
    times_ms: Sequence[float],
    positions: Sequence[Sequence[float]],
    start_ms: float,
    end_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate exact task boundaries and retain only interior samples.

    Human and synthetic streams therefore enter the common-bandwidth path on
    the identical closed time interval, independent of their native cadence.
    """
    times = np.asarray(times_ms, dtype=float)
    points = np.asarray(positions, dtype=float)
    if points.shape != (len(times), 2) or len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(points)):
        raise ValueError("invalid transition samples")
    keep = np.r_[True, np.diff(times) > 0]
    times, points = times[keep], points[keep]
    if end_ms <= start_ms or times[0] > start_ms or times[-1] < end_ms:
        raise ValueError("samples do not span exact transition boundaries")
    interior = (times > start_ms) & (times < end_ms)
    clipped_times = np.concatenate(([float(start_ms)], times[interior], [float(end_ms)]))
    clipped_points = np.vstack((
        [np.interp(start_ms, times, points[:, axis]) for axis in range(2)],
        points[interior],
        [np.interp(end_ms, times, points[:, axis]) for axis in range(2)],
    ))
    return clipped_times, clipped_points


def movement_descriptors(times_ms: Sequence[float], positions: Sequence[Sequence[float]]) -> tuple[dict[str, float], dict[str, Any]]:
    times, points, support = common_bandwidth_positions(times_ms, positions)
    duration_s = (times[-1] - times[0]) / 1000.0
    chord_vector = points[-1] - points[0]
    chord = float(np.linalg.norm(chord_vector))
    if chord < 1e-3 or duration_s <= 0:
        raise ValueError("stationary or degenerate scored transition")
    dt_s = OUTPUT_STEP_MS / 1000.0
    velocity = np.diff(points, axis=0) / dt_s
    acceleration = np.diff(velocity, axis=0) / dt_s
    jerk = np.diff(acceleration, axis=0) / dt_s
    speed = np.linalg.norm(velocity, axis=1) * duration_s / chord
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1) * duration_s**2 / chord
    jerk_magnitude = np.linalg.norm(jerk, axis=1) * duration_s**3 / chord
    displacements = np.linalg.norm(np.diff(points, axis=0), axis=1)
    unit = chord_vector / chord
    projected_steps = np.diff((points - points[0]) @ unit)
    cross_track = (points[:, 0] - points[0, 0]) * unit[1] - (points[:, 1] - points[0, 1]) * unit[0]
    centered_speed = speed - float(np.mean(speed))
    frequencies = np.fft.rfftfreq(len(centered_speed), d=dt_s)
    power = np.abs(np.fft.rfft(centered_speed)) ** 2
    high = float(np.sum(power[(frequencies >= 10.0) & (frequencies <= LOWPASS_HZ)]))
    non_dc = float(np.sum(power[frequencies > 0]))
    endpoint_difference = float(np.linalg.norm(velocity[-1] - velocity[0]) * duration_s / chord)
    descriptors = {
        "log_speed_p50": float(np.log(max(float(np.median(speed)), 1e-9))),
        "log_speed_p90": float(np.log(max(float(np.quantile(speed, 0.90)), 1e-9))),
        "log_acceleration_p95": float(np.log(max(float(np.quantile(acceleration_magnitude, 0.95)), 1e-9))),
        "log_jerk_p95": float(np.log(max(float(np.quantile(jerk_magnitude, 0.95)), 1e-9))),
        "stop_fraction": float(np.mean(speed < 0.05)),
        "high_frequency_power_share": high / max(non_dc, 1e-12),
        "log_path_length_ratio": float(np.log(max(float(np.sum(displacements) / chord), 1e-9))),
        "log1p_cross_track_rms_ratio": float(np.log1p(np.sqrt(np.mean(cross_track**2)) / chord)),
        "log1p_reversal_distance_ratio": float(np.log1p(np.sum(np.maximum(-projected_steps, 0.0)) / chord)),
        "log1p_endpoint_velocity_difference": float(np.log1p(endpoint_difference)),
    }
    return descriptors, support


def movement_population_distance(
    left: Sequence[Mapping[str, Any] | None],
    right: Sequence[Mapping[str, Any] | None],
    scales: Mapping[str, float],
    names: Sequence[str] = MOVEMENT_DESCRIPTORS,
) -> tuple[float, dict[str, float]]:
    # Candidate-dependent degeneracy is an intention-to-treat failure, not an
    # eligibility filter.  Human calibration support is selected before any
    # candidate is opened; candidate evaluators retain each eligible row and
    # represent a stationary/non-finite output as ``None`` here.
    if any(row is None for row in left) or any(row is None for row in right):
        return float("inf"), {name: float("inf") for name in names}
    distances = {
        name: _wasserstein_equal_weight([row[name] for row in left], [row[name] for row in right])
        for name in names
    }
    total = float(np.mean([distances[name] / float(scales[name]) for name in names]))
    return total, distances


def _corrupt_positions(
    times_ms: Sequence[float],
    positions: Sequence[Sequence[float]],
    corruption: str,
    severity: float,
    seed: int,
) -> np.ndarray:
    times = np.asarray(times_ms, dtype=float)
    points = np.asarray(positions, dtype=float)
    normalized_time = (times - times[0]) / max(float(times[-1] - times[0]), 1e-9)
    count = PHASE_POINTS - 1
    original_increments = np.full(count, 1.0 / count)
    if corruption == "position_noise":
        chord = max(float(np.linalg.norm(points[-1] - points[0])), 1.0)
        chord_vector = points[-1] - points[0]
        unit = chord_vector / chord
        normal = np.asarray([-unit[1], unit[0]])
        # A 12.5 Hz cross-track oscillation lies inside the frozen 20 Hz
        # analysis bandwidth.  Endpoint tapering keeps task endpoints fixed.
        taper = np.sin(np.pi * normalized_time) ** 2
        oscillation = np.sin(2.0 * np.pi * 12.5 * (times - times[0]) / 1000.0)
        return points + (severity * 0.25 * chord * taper * oscillation)[:, None] * normal
    template_name = corruption if corruption in CORRUPTIONS else "pause_restart"
    increments = _template(template_name, original_increments, severity, seed)
    template_time = np.linspace(0.0, 1.0, PHASE_POINTS)
    template_phase = np.concatenate(([0.0], np.cumsum(increments)))
    phase = np.interp(normalized_time, template_time, template_phase)
    # Time-warp the observed path instead of replacing it with a straight
    # chord; this isolates execution timing while preserving spatial style.
    target = np.column_stack([
        np.interp(phase, normalized_time, points[:, axis]) for axis in range(2)
    ])
    return (1.0 - severity) * points + severity * target


def _player_pair_matches(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedy no-reuse context matches for each unordered pair of players."""
    contexts = np.asarray([_context(row) for row in rows], dtype=float)
    center = np.median(contexts, axis=0)
    scale = np.maximum(np.subtract(*np.percentile(contexts, [75, 25], axis=0)), 1e-6)
    standardized = (contexts - center) / scale
    players = sorted({str(row["player"]) for row in rows})
    nearest = []
    for index, row in enumerate(rows):
        allowed = [
            other for other, candidate in enumerate(rows)
            if candidate["player"] != row["player"] and candidate["map_family"] != row["map_family"]
        ]
        if allowed:
            nearest.append(float(np.min(np.linalg.norm(standardized[allowed] - standardized[index], axis=1))))
    caliper = float(np.quantile(nearest, 0.95))
    groups = []
    total_selected_rows: Counter[int] = Counter()
    for left_position, left_player in enumerate(players):
        for right_player in players[left_position + 1:]:
            left_indices = [index for index, row in enumerate(rows) if row["player"] == left_player]
            right_indices = [index for index, row in enumerate(rows) if row["player"] == right_player]
            candidates = []
            for left_index in left_indices:
                for right_index in right_indices:
                    if rows[left_index]["map_family"] == rows[right_index]["map_family"]:
                        continue
                    distance = float(np.linalg.norm(standardized[left_index] - standardized[right_index]))
                    if distance <= caliper + 1e-12:
                        candidates.append((distance, left_index, right_index))
            used_left: set[int] = set()
            used_right: set[int] = set()
            matched = []
            for distance, left_index, right_index in sorted(candidates):
                if left_index in used_left or right_index in used_right:
                    continue
                used_left.add(left_index)
                used_right.add(right_index)
                matched.append((left_index, right_index, distance))
                total_selected_rows[left_index] += 1
                total_selected_rows[right_index] += 1
            if len(matched) >= 5:
                groups.append({
                    "left_player": left_player,
                    "right_player": right_player,
                    "pairs": matched,
                    "left_families": sorted({rows[left]["map_family"] for left, _, _ in matched}),
                    "right_families": sorted({rows[right]["map_family"] for _, right, _ in matched}),
                })
    return groups, {
        "player_pair_comparisons": len(groups),
        "unique_players": len(players),
        "unique_map_families": len({row["map_family"] for row in rows}),
        "unique_rows_used": len(total_selected_rows),
        "total_row_uses": int(sum(total_selected_rows.values())),
        "maximum_row_reuse": int(max(total_selected_rows.values(), default=0)),
        "caliper_quantile": 0.95,
        "caliper_distance": caliper,
        "matching_rule": "within each unordered player pair, greedy no-reuse robust-IQR context matches; same map family forbidden",
        "context_fields": ["log_duration_ms", "log_distance_px", "turn_cos", "turn_sin"],
    }


def calibrate(validation_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze distributional scales and negative anchors without candidate output."""
    groups, matching = _player_pair_matches(validation_rows)
    if len(groups) < 3:
        raise ValueError("fewer than three player-pair population comparisons")
    descriptor_rows = [profile_descriptors(row["increments"]) for row in validation_rows]
    scales: dict[str, float] = {}
    for name in DESCRIPTORS:
        values = np.asarray([item[name] for item in descriptor_rows], dtype=float)
        scales[name] = _validation_scale(values, floor=1e-4)
    human_distances = []
    for group in groups:
        left_profiles = [validation_rows[left]["increments"] for left, _, _ in group["pairs"]]
        right_profiles = [validation_rows[right]["increments"] for _, right, _ in group["pairs"]]
        distance, _ = population_distance(left_profiles, right_profiles, scales)
        human_distances.append(distance)
    human_distances = np.asarray(human_distances, dtype=float)
    corruption: dict[str, Any] = {}
    severe_negative_distances: list[float] = []
    negative_monotonic = True
    for corruption_name in CORRUPTIONS:
        medians: list[float] = []
        by_severity: list[dict[str, float]] = []
        for severity in SEVERITIES:
            distances = []
            for group_index, group in enumerate(groups):
                damaged = []
                reference = []
                for pair_index, (left, right, _) in enumerate(group["pairs"]):
                    original = _normalize_increments(validation_rows[left]["increments"])
                    damaged.append(_template(corruption_name, original, severity, seed=17_000 + group_index * 1_000 + pair_index))
                    reference.append(validation_rows[right]["increments"])
                distance, _ = population_distance(damaged, reference, scales)
                distances.append(distance)
            median = float(np.median(distances))
            medians.append(median)
            by_severity.append({"severity": severity, "median_distance": median, "p05": float(np.quantile(distances, 0.05)), "p95": float(np.quantile(distances, 0.95))})
            if severity == SEVERITIES[-1] and corruption_name in NEGATIVE_ANCHOR_CORRUPTIONS:
                severe_negative_distances.extend(distances)
        family_monotonic = all(right > left for left, right in zip(medians, medians[1:]))
        if corruption_name in NEGATIVE_ANCHOR_CORRUPTIONS:
            negative_monotonic = negative_monotonic and family_monotonic
        corruption[corruption_name] = {"by_severity": by_severity, "strictly_monotonic_median": family_monotonic}
    anchor = float(np.median(severe_negative_distances))
    human_p95 = float(np.quantile(human_distances, 0.95))
    severe_family_medians = {
        name: float(details["by_severity"][-1]["median_distance"])
        for name, details in corruption.items()
    }
    negative_separation = all(severe_family_medians[name] > human_p95 for name in NEGATIVE_ANCHOR_CORRUPTIONS)
    human_median = float(np.median(human_distances))
    template_specificity = all(severe_family_medians[name] >= human_median for name in TEMPLATE_CONTROLS)
    payload = {
        "schema_version": 1,
        "benchmark_id": SCOPED_REALISM_ID,
        "phase_points": PHASE_POINTS,
        "features": list(DESCRIPTORS),
        "feature_weights": {name: 1.0 / len(DESCRIPTORS) for name in DESCRIPTORS},
        "scales": scales,
        "matching": matching,
        "human_human": {
            "population_comparisons": len(human_distances),
            "distance_p05": float(np.quantile(human_distances, 0.05)),
            "distance_p50": float(np.median(human_distances)),
            "distance_p95": human_p95,
        },
        "corruptions": corruption,
        "severe_corruption_anchor_b": anchor,
        "controls": {
            "identity_distance": 0.0,
            "negative_anchor_corruptions": list(NEGATIVE_ANCHOR_CORRUPTIONS),
            "template_specificity_controls": list(TEMPLATE_CONTROLS),
            "negative_corruption_medians_strictly_monotonic": negative_monotonic,
            "every_severe_negative_family_median_above_human_p95": negative_separation,
            "template_controls_not_better_than_human_median": template_specificity,
            "severe_family_medians": severe_family_medians,
        },
        "score_definition": "100 * clip(1 - raw_scaled_distance / severe_corruption_anchor_b, 0, 1)",
        "score_interpretation": "100 is exact descriptor identity; 0 is the pooled median severe-corruption distance or worse; not percent-human and not ORI",
        "status": "calibrated" if negative_monotonic and negative_separation and template_specificity and anchor > human_p95 else "failed_controls",
    }
    payload["calibration_sha256"] = canonical_sha256(payload)
    return payload


def calibrate_full(
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    scale_rule: str = "robust_tail",
) -> dict[str, Any]:
    """Calibrate the frozen 40/60 phase-plus-movement composite."""
    usable: list[dict[str, Any]] = []
    movement_rejections: Counter[str] = Counter()
    for row in validation_rows:
        try:
            movement, support = movement_descriptors(row["raw_times_ms"], row["raw_positions"])
        except ValueError as error:
            movement_rejections[str(error)] += 1
            continue
        enriched = dict(row)
        enriched["movement_descriptors"] = movement
        enriched["movement_support"] = support
        usable.append(enriched)
    groups, matching = _player_pair_matches(usable)
    if len(groups) < 3:
        raise ValueError("fewer than three player-pair comparisons with movement support")

    phase_rows = [profile_descriptors(row["increments"]) for row in usable]
    movement_rows = [row["movement_descriptors"] for row in usable]
    phase_spans = {
        name: float(np.subtract(*np.percentile([row[name] for row in phase_rows], [95, 5]))) for name in DESCRIPTORS
    }
    phase_active = [name for name in DESCRIPTORS if phase_spans[name] > 1e-4]
    phase_diagnostic = [name for name in DESCRIPTORS if name not in phase_active]
    phase_scales = {}
    for name in phase_active:
        values = np.asarray([row[name] for row in phase_rows], dtype=float)
        phase_scales[name] = _validation_scale(values, floor=1e-4, rule=scale_rule)
    movement_floor = lambda name: 1e-4 if name in {"stop_fraction", "high_frequency_power_share"} else 1e-3
    movement_spans = {
        name: float(np.subtract(*np.percentile([row[name] for row in movement_rows], [95, 5]))) for name in MOVEMENT_DESCRIPTORS
    }
    movement_active = [name for name in MOVEMENT_DESCRIPTORS if movement_spans[name] > movement_floor(name)]
    movement_diagnostic = [name for name in MOVEMENT_DESCRIPTORS if name not in movement_active]
    movement_scales = {}
    for name in movement_active:
        values = np.asarray([row[name] for row in movement_rows], dtype=float)
        floor = movement_floor(name)
        movement_scales[name] = _validation_scale(values, floor=floor, rule=scale_rule)

    clean_rows = []
    for group in groups:
        left_indices = [left for left, _, _ in group["pairs"]]
        right_indices = [right for _, right, _ in group["pairs"]]
        phase_distance, phase_features = population_distance(
            [usable[index]["increments"] for index in left_indices],
            [usable[index]["increments"] for index in right_indices],
            phase_scales,
            phase_active,
        )
        movement_distance, movement_features = movement_population_distance(
            [usable[index]["movement_descriptors"] for index in left_indices],
            [usable[index]["movement_descriptors"] for index in right_indices],
            movement_scales,
            movement_active,
        )
        clean_rows.append({
            "left_player": group["left_player"],
            "right_player": group["right_player"],
            "matched_windows": len(left_indices),
            "phase_distance": phase_distance,
            "movement_distance": movement_distance,
            "composite_distance": 0.4 * phase_distance + 0.6 * movement_distance,
            "phase_feature_distances": phase_features,
            "movement_feature_distances": movement_features,
        })
    clean = np.asarray([row["composite_distance"] for row in clean_rows], dtype=float)
    human_p95 = float(np.quantile(clean, 0.95))
    human_median = float(np.median(clean))

    full_negative = NEGATIVE_ANCHOR_CORRUPTIONS + ("position_noise",)
    corruption_results = {}
    severe_negative: list[float] = []
    negative_monotonic = True
    for corruption in CORRUPTIONS + ("position_noise",):
        by_severity = []
        medians = []
        for severity in SEVERITIES:
            group_distances = []
            for group_index, group in enumerate(groups):
                damaged_profiles = []
                reference_profiles = []
                damaged_movement = []
                reference_movement = []
                for pair_index, (left, right, _) in enumerate(group["pairs"]):
                    seed = 71_000 + group_index * 1_000 + pair_index
                    original_profile = _normalize_increments(usable[left]["increments"])
                    damaged_profiles.append(
                        original_profile if corruption == "position_noise"
                        else _template(corruption, original_profile, severity, seed)
                    )
                    reference_profiles.append(usable[right]["increments"])
                    damaged_positions = _corrupt_positions(
                        usable[left]["raw_times_ms"], usable[left]["raw_positions"], corruption, severity, seed
                    )
                    damaged_descriptor, _ = movement_descriptors(usable[left]["raw_times_ms"], damaged_positions)
                    damaged_movement.append(damaged_descriptor)
                    reference_movement.append(usable[right]["movement_descriptors"])
                phase_distance, _ = population_distance(damaged_profiles, reference_profiles, phase_scales, phase_active)
                movement_distance, _ = movement_population_distance(damaged_movement, reference_movement, movement_scales, movement_active)
                group_distances.append(0.4 * phase_distance + 0.6 * movement_distance)
            median = float(np.median(group_distances))
            medians.append(median)
            by_severity.append({
                "severity": severity,
                "median_distance": median,
                "p05": float(np.quantile(group_distances, 0.05)),
                "p95": float(np.quantile(group_distances, 0.95)),
            })
            if severity == SEVERITIES[-1] and corruption in full_negative:
                severe_negative.extend(group_distances)
        monotonic = all(right > left for left, right in zip(medians, medians[1:]))
        if corruption in full_negative:
            negative_monotonic = negative_monotonic and monotonic
        corruption_results[corruption] = {"by_severity": by_severity, "strictly_monotonic_median": monotonic}

    severe_medians = {name: rows["by_severity"][-1]["median_distance"] for name, rows in corruption_results.items()}
    negative_separation = all(severe_medians[name] > human_p95 for name in full_negative)
    negative_median_separation = all(severe_medians[name] > human_median for name in full_negative)
    template_specificity = all(severe_medians[name] >= human_median for name in TEMPLATE_CONTROLS)
    anchor = float(np.median(severe_negative))
    exploratory_valid = negative_monotonic and negative_median_separation and template_specificity and anchor > human_median
    status = "calibrated" if exploratory_valid and negative_separation and anchor > human_p95 else ("exploratory_limited_separation" if exploratory_valid else "failed_controls")
    result = {
        "schema_version": 1,
        "benchmark_id": SCOPED_REALISM_ID,
        "status": status,
        "channel_weights": {"phase": 0.4, "movement": 0.6},
        "phase": {
            "descriptors": phase_active,
            "diagnostic_only_descriptors": phase_diagnostic,
            "scales": phase_scales,
            "scale_rule": scale_rule,
        },
        "movement": {
            "descriptors": movement_active,
            "diagnostic_only_descriptors": movement_diagnostic,
            "applicability_rule": "scored only when validation p95-p05 span exceeds the descriptor numerical floor",
            "scales": movement_scales,
            "scale_rule": scale_rule,
            "common_bandwidth": {
                "working_step_ms": WORKING_STEP_MS,
                "output_step_ms": OUTPUT_STEP_MS,
                "lowpass_minus_3db_hz": LOWPASS_HZ,
                "gaussian_sigma_ms": FILTER_SIGMA_MS,
                "filter_edge_trim_ms": FILTER_EDGE_MS,
                "source_gap_mask_threshold_ms": MAX_SOURCE_GAP_MS,
                "minimum_filtered_samples": MIN_FILTERED_SAMPLES,
            },
            "eligible_validation_windows": len(usable),
            "rejected_validation_windows": dict(movement_rejections),
        },
        "matching": matching,
        "human_human": {
            "population_comparisons": len(clean_rows),
            "distance_p05": float(np.quantile(clean, 0.05)),
            "distance_p50": human_median,
            "distance_p95": human_p95,
            "comparisons": clean_rows,
        },
        "corruptions": corruption_results,
        "severe_negative_anchor_b": anchor,
        "controls": {
            "identity_distance": 0.0,
            "negative_anchor_corruptions": list(full_negative),
            "template_specificity_controls": list(TEMPLATE_CONTROLS),
            "negative_corruption_medians_strictly_monotonic": negative_monotonic,
            "every_severe_negative_family_median_above_human_p95": negative_separation,
            "every_severe_negative_family_median_above_human_median": negative_median_separation,
            "template_controls_not_better_than_human_median": template_specificity,
            "severe_family_medians": severe_medians,
        },
        "score_definition": "100 * clip(1 - (0.4*D_phase + 0.6*D_movement) / severe_negative_anchor_b, 0, 1)",
        "score_interpretation": "100 is exact descriptor-distribution identity; 0 is pooled median severe negative-anchor distance or worse; not percent-human and not ORI",
        "calibration_warning": None if status == "calibrated" else "human reference range overlaps at least one severe negative control; descriptive population similarity only, not individual-trace or humanhood classification",
    }
    result["calibration_sha256"] = canonical_sha256(result)
    return result


def score_distance(distance: float, calibration: Mapping[str, Any]) -> float:
    anchor_key = "severe_negative_anchor_b" if "severe_negative_anchor_b" in calibration else "severe_corruption_anchor_b"
    anchor = float(calibration[anchor_key])
    if calibration.get("status") not in {"calibrated", "exploratory_limited_separation"} or anchor <= 0:
        raise ValueError("calibration controls did not pass")
    return float(100.0 * np.clip(1.0 - float(distance) / anchor, 0.0, 1.0))


def load_human_rows(
    manifest_path: str | Path,
    decoded_root: str | Path,
    map_plan_root: str | Path,
    *,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct the capped transition rows used by the frozen V2 fit."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    max_per_player = int(manifest["max_windows_per_player"])
    decoded_root = Path(decoded_root)
    map_plan_root = Path(map_plan_root)
    used_by_player: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for record in manifest["records"]:
        if record["split"] != split:
            continue
        player = str(record["player"])
        map_hash = str(record["map"])
        map_family = str(record["map_family"])
        plan = load_map_plan(map_plan_root / f"{map_hash}.map.ndjson.gz")
        _, frames = _read_decoded(decoded_root / f"{record['replay_sha256']}.ndjson")
        times = np.asarray([float(frame["time_ms"]) for frame in frames], dtype=float)
        positions = np.asarray([[float(frame["x"]), float(frame["y"])] for frame in frames], dtype=float)
        for object_index in range(1, len(plan.objects)):
            if used_by_player[player] >= max_per_player:
                break
            try:
                route = transition_route(plan, object_index, skill_group="intermediate", record_id=f"scoped:{record['replay_sha256']}:{object_index}")
                label = _extract_phase_label_arrays(route, times, positions)
            except ValueError as error:
                skipped[str(error)] += 1
                continue
            previous = plan.objects[object_index - 1]
            current = plan.objects[object_index]
            before = plan.objects[object_index - 2] if object_index >= 2 else None
            previous_point = np.asarray([previous.position.x, previous.position.y], dtype=float)
            before_point = (
                np.asarray([before.position.x, before.position.y], dtype=float)
                if before is not None else previous_point
            )
            current_point = np.asarray([current.position.x, current.position.y], dtype=float)
            incoming = previous_point - before_point
            outgoing = current_point - previous_point
            if np.linalg.norm(incoming) > 1e-9 and np.linalg.norm(outgoing) > 1e-9:
                incoming /= np.linalg.norm(incoming)
                outgoing /= np.linalg.norm(outgoing)
                turn_cos = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
                turn_sin = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
            else:
                turn_cos, turn_sin = 1.0, 0.0
            route_start = route.start_time_ms
            route_end = route.start_time_ms + route.duration_ms
            try:
                raw_times, raw_positions = clip_transition_samples(times, positions, route_start, route_end)
            except ValueError as error:
                skipped[str(error)] += 1
                continue
            rows.append({
                "player": player,
                "map": map_hash,
                "map_family": map_family,
                "replay_sha256": str(record["replay_sha256"]),
                "source_window_index": object_index,
                "duration_ms": route.duration_ms,
                "distance_px": route.length_px,
                "context": [float(np.log(max(route.duration_ms, 1.0))), float(np.log(max(route.length_px, 1e-3))), turn_cos, turn_sin],
                "increments": np.exp(np.asarray(label["log_phase_increments"], dtype=float)).tolist(),
                "source_cadence_median_ms": label["source_cadence_median_ms"],
                "source_sample_count": label["source_sample_count"],
                "raw_times_ms": raw_times.tolist(),
                "raw_positions": raw_positions.tolist(),
            })
            used_by_player[player] += 1
    inventory = {
        "split": split,
        "rows": len(rows),
        "players": len({row["player"] for row in rows}),
        "maps": len({row["map"] for row in rows}),
        "map_families": len({row["map_family"] for row in rows}),
        "per_player_cap": max_per_player,
        "skipped": dict(skipped),
    }
    return rows, inventory


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_inventory(paths: Iterable[str | Path], *, root: str | Path) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    result = []
    for raw_path in sorted((Path(path).resolve() for path in paths), key=lambda item: str(item).lower()):
        result.append({"path": raw_path.relative_to(root).as_posix(), "bytes": raw_path.stat().st_size, "sha256": file_sha256(raw_path)})
    return result
