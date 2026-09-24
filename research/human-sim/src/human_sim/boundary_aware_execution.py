"""Boundary-aware research execution adapter with honest three-arm telemetry.

The adapter repairs only explicitly selected circle-segment neighborhoods.  It
preserves timestamps, key states, object contacts, model weights and physical
limits.  Geometry repair and learned timing are reported separately.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, eye, kron, vstack
from scipy.sparse.linalg import spsolve

from .execution import ExecutionConstraints, FrozenRoute, _interval_kinematics


BOUNDARY_AWARE_VERSION = "boundary-aware-multisegment-execution-v2"
DEFAULT_MAX_DISPLACEMENT_PX = 4.0
DEFAULT_RMS_DISPLACEMENT_PX = 1.5
SOLVER_SAFETY_FACTOR = 0.995
LEARNED_RESERVE_FACTOR = 0.75


def _difference_operator(locations: np.ndarray) -> csr_matrix:
    spacing = np.diff(locations)
    rows = np.repeat(np.arange(len(spacing)), 2)
    cols = np.column_stack((np.arange(len(spacing)), np.arange(1, len(spacing) + 1))).ravel()
    data = np.column_stack((-1.0 / spacing, 1.0 / spacing)).ravel()
    return csr_matrix((data, (rows, cols)), shape=(len(spacing), len(locations)))


def derivative_operators(times_ms: np.ndarray) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    times_s = np.asarray(times_ms, dtype=float) / 1000.0
    if np.any(np.diff(times_s) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    d1 = _difference_operator(times_s)
    velocity_times = 0.5 * (times_s[:-1] + times_s[1:])
    d2 = _difference_operator(velocity_times) @ d1
    acceleration_times = 0.5 * (velocity_times[:-1] + velocity_times[1:])
    d3 = _difference_operator(acceleration_times) @ d2
    return d1.tocsr(), d2.tocsr(), d3.tocsr()


def exact_kinematic_summary(points: np.ndarray, times_ms: np.ndarray, constraints: ExecutionConstraints) -> dict[str, Any]:
    velocity, acceleration, jerk = _interval_kinematics(np.asarray(points, dtype=float), np.asarray(times_ms, dtype=float) / 1000.0)
    values = {
        "speed_px_s": float(np.max(np.linalg.norm(velocity, axis=1))) if len(velocity) else 0.0,
        "acceleration_px_s2": float(np.max(np.linalg.norm(acceleration, axis=1))) if len(acceleration) else 0.0,
        "jerk_px_s3": float(np.max(np.linalg.norm(jerk, axis=1))) if len(jerk) else 0.0,
    }
    limits = {
        "speed_px_s": constraints.max_speed_px_s,
        "acceleration_px_s2": constraints.max_acceleration_px_s2,
        "jerk_px_s3": constraints.max_jerk_px_s3,
    }
    failures = [name for name, value in values.items() if value > limits[name] + 1e-6]
    return {**values, "limits": limits, "failure_families": failures, "valid": not failures}


def _failing_supports(points: np.ndarray, times_ms: np.ndarray, constraints: ExecutionConstraints) -> list[tuple[int, int]]:
    velocity, acceleration, jerk = _interval_kinematics(points, times_ms / 1000.0)
    result = []
    for values, size, limit in (
        (velocity, 2, constraints.max_speed_px_s),
        (acceleration, 3, constraints.max_acceleration_px_s2),
        (jerk, 4, constraints.max_jerk_px_s3),
    ):
        result.extend((int(index), int(index + size - 1)) for index in np.flatnonzero(np.linalg.norm(values, axis=1) > limit + 1e-6))
    return result


def _merge(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _neighborhoods(spans: Sequence[tuple[int, int]], failures: Sequence[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    intervals = []
    for selected_start, selected_end in spans:
        hits = [(start, end) for start, end in failures if end >= selected_start - 3 and start <= selected_end + 3]
        intervals.append((min((x[0] for x in hits), default=selected_start), max((x[1] for x in hits), default=selected_end)))
    current = _merge(intervals)
    changed = True
    while changed:
        changed = False
        expanded = []
        for left, right in current:
            hits = [(start, end) for start, end in failures if end >= left - 3 and start <= right + 3]
            value = (max(3, min([left, *(x[0] for x in hits)])), min(count - 4, max([right, *(x[1] for x in hits)])))
            expanded.append(value)
            changed |= value != (left, right)
        current = _merge(expanded)
    return current


def _segment_specs(times_ms: np.ndarray, boundaries_ms: Sequence[float], modes: Sequence[str]) -> list[tuple[int, int, int, str]]:
    values = sorted({0.0, float(times_ms[-1]), *(max(0.0, min(float(times_ms[-1]), float(value))) for value in boundaries_ms)})
    indices = []
    for value in values:
        index = int(np.argmin(np.abs(times_ms - value)))
        if not indices or index > indices[-1]:
            indices.append(index)
    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != len(times_ms) - 1:
        indices.append(len(times_ms) - 1)
    return [(index, start, end, modes[index] if index < len(modes) else "mixed") for index, (start, end) in enumerate(zip(indices, indices[1:])) if end > start]


def _solve_anchor(
    frozen: np.ndarray,
    times_ms: np.ndarray,
    edit_start: int,
    edit_end: int,
    contacts: set[int],
    constraints: ExecutionConstraints,
    max_displacement_px: float,
    rms_displacement_px: float,
    max_function_evaluations: int,
) -> dict[str, Any]:
    validation_start, validation_end = edit_start - 3, edit_end + 3
    base = frozen[validation_start : validation_end + 1].copy()
    editable = [index - validation_start for index in range(edit_start, edit_end + 1) if index not in contacts]
    if not editable:
        return {"found": False, "fallback_reason": "no_editable_non_contact_samples"}
    lookup = {local: offset for offset, local in enumerate(editable)}
    operators = tuple(zip(derivative_operators(times_ms[validation_start : validation_end + 1]), (
        constraints.max_speed_px_s, constraints.max_acceleration_px_s2, constraints.max_jerk_px_s3
    )))

    def positions(values: np.ndarray) -> np.ndarray:
        result = base.copy()
        delta = values.reshape(-1, 2)
        for local, offset in lookup.items():
            result[local] += delta[offset]
        return result

    def residual(values: np.ndarray) -> np.ndarray:
        result = positions(values)
        delta = values.reshape(-1, 2)
        radial = np.linalg.norm(delta, axis=1)
        pieces = [0.03 * values / rms_displacement_px]
        pieces.append(100.0 * np.maximum(radial / max_displacement_px - 1.0, 0.0))
        pieces.append(np.asarray([100.0 * max(float(np.sqrt(np.mean(radial**2))) / rms_displacement_px - 1.0, 0.0)]))
        for operator, limit in operators:
            ratio = np.linalg.norm(operator @ result, axis=1) / float(limit * LEARNED_RESERVE_FACTOR)
            pieces.append(100.0 * np.maximum(ratio - 1.0, 0.0))
        return np.concatenate(pieces)

    variable_count = 2 * len(editable)
    patterns = [eye(variable_count, format="csr"), kron(eye(len(editable), format="csr"), np.ones((1, 2))), csr_matrix(np.ones((1, variable_count)))]
    selector = csr_matrix((np.ones(len(editable)), (editable, np.arange(len(editable)))), shape=(len(base), len(editable)))
    for operator, _ in operators:
        patterns.append(kron((abs(operator) @ selector).astype(bool).astype(float), np.ones((1, 2)), format="csr"))
    selector = selector.tocsr()

    # A deterministic sparse quadratic smoother supplies an interior warm start.
    # This is score-blind: it uses only the unchanged physical limits and fixed
    # contact/context samples.  The nonlinear solve remains the acceptance path.
    normal_penalty = csr_matrix((len(editable), len(editable)), dtype=float)
    right_penalty = np.zeros((len(editable), 2), dtype=float)
    for operator, limit in operators:
        scaled = (operator @ selector) / float(limit * LEARNED_RESERVE_FACTOR)
        frozen_term = (operator @ base) / float(limit * LEARNED_RESERVE_FACTOR)
        normal_penalty = normal_penalty + scaled.T @ scaled
        right_penalty -= scaled.T @ frozen_term
    warm_candidates = []
    for ridge_weight in np.logspace(-4.0, 6.0, 21):
        normal = eye(len(editable), format="csr") + ridge_weight * normal_penalty
        right = ridge_weight * right_penalty
        values = np.column_stack((spsolve(normal, right[:, 0]), spsolve(normal, right[:, 1]))).ravel()
        values = np.clip(values, -max_displacement_px, max_displacement_px)
        candidate_points = positions(values)
        displacement = np.linalg.norm(candidate_points - base, axis=1)
        ratios = [
            float(np.max(np.linalg.norm(operator @ candidate_points, axis=1)) / (limit * LEARNED_RESERVE_FACTOR))
            for operator, limit in operators
        ]
        budget_valid = float(np.max(displacement)) <= max_displacement_px + 1e-9 and float(np.sqrt(np.mean(displacement**2))) <= rms_displacement_px + 1e-9
        warm_candidates.append((
            (not budget_valid, max(max(ratios) - 1.0, 0.0), float(np.sqrt(np.mean(displacement**2)))),
            float(ridge_weight), values,
        ))
    _, selected_ridge_weight, warm = min(warm_candidates, key=lambda item: item[0])
    results = []
    for start in (warm, np.zeros(variable_count, dtype=float)):
        candidate = least_squares(
            residual, start, bounds=(-max_displacement_px, max_displacement_px),
            jac_sparsity=vstack(patterns, format="csr"), x_scale="jac", max_nfev=max_function_evaluations,
            ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        results.append(candidate)
        candidate_points = positions(candidate.x)
        candidate_displacement = np.linalg.norm(candidate_points - base, axis=1)
        if exact_kinematic_summary(
            candidate_points, times_ms[validation_start : validation_end + 1], constraints
        )["valid"] and float(np.max(candidate_displacement)) <= max_displacement_px + 1e-9 \
                and float(np.sqrt(np.mean(candidate_displacement**2))) <= rms_displacement_px + 1e-9:
            break
    result = min(results, key=lambda item: (not exact_kinematic_summary(
        positions(item.x), times_ms[validation_start : validation_end + 1], constraints
    )["valid"], float(np.linalg.norm(item.fun))))
    context = positions(result.x)
    displacement = np.linalg.norm(context - base, axis=1)
    exact = exact_kinematic_summary(context, times_ms[validation_start : validation_end + 1], constraints)
    budget_valid = float(np.max(displacement)) <= max_displacement_px + 1e-9 and float(np.sqrt(np.mean(displacement**2))) <= rms_displacement_px + 1e-9
    repaired = frozen.copy()
    repaired[validation_start : validation_end + 1] = context
    return {
        "found": exact["valid"] and budget_valid,
        "fallback_reason": None if exact["valid"] and budget_valid else "no_repaired_anchor_found_within_budget_and_evaluation_cap",
        "repaired": repaired,
        "validation_start": validation_start, "validation_end": validation_end,
        "edit_start": edit_start, "edit_end": edit_end,
        "function_evaluations": int(result.nfev), "total_function_evaluations": int(sum(item.nfev for item in results)),
        "solver_success": bool(result.success), "warm_start": "normalized_sparse_quadratic_smoother",
        "warm_start_ridge_weight": selected_ridge_weight,
        "exact_validation": exact, "budget_valid": budget_valid,
        "max_displacement_px": float(np.max(displacement)), "rms_displacement_px": float(np.sqrt(np.mean(displacement**2))),
        "editable_samples": len(editable),
    }


def _smooth_phases(model: Any, route: FrozenRoute, times_ms: np.ndarray, call_counter: list[int]) -> np.ndarray:
    call_counter[0] += 1
    weights = np.asarray(model.predict_phase_weights(route, 32), dtype=float)
    if weights.shape != (31,) or np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("invalid native-grid phase proposal")
    edges = np.linspace(0.0, 1.0, 32)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate = PchipInterpolator(np.concatenate(([0.0], centers, [1.0])), np.concatenate(([weights[0]], weights, [weights[-1]])) * 31, extrapolate=False)
    integral = rate.antiderivative()
    normalized = (times_ms - times_ms[0]) / max(float(times_ms[-1] - times_ms[0]), 1e-12)
    phases = np.asarray((integral(normalized) - integral(0.0)) / (integral(1.0) - integral(0.0)), dtype=float)
    phases[0], phases[-1] = 0.0, 1.0
    return np.maximum.accumulate(phases)


def _alpha_candidates() -> list[float]:
    return sorted({index / 1024 for index in range(1025)} | {2.0 ** -power for power in range(1, 25)}, reverse=True)


def boundary_aware_warp_math_trace(
    frames: Sequence[Any],
    *,
    model: Any,
    blend: float,
    segment_boundaries_ms: Sequence[float],
    segment_modes: Sequence[str],
    selected_segment_indices: Sequence[int],
    max_displacement_px: float = DEFAULT_MAX_DISPLACEMENT_PX,
    rms_displacement_px: float = DEFAULT_RMS_DISPLACEMENT_PX,
    max_function_evaluations: int = 250,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Return repaired-math, repaired-hybrid, and explicit candidate telemetry."""
    if not frames or len(frames) < 7:
        raise ValueError("boundary-aware execution requires at least seven frames")
    if not selected_segment_indices:
        raise ValueError("boundary-aware execution requires explicit selected segment indices")
    if not 0.0 < float(blend) <= 1.0:
        raise ValueError("boundary-aware execution blend must be in (0, 1]")
    times_ms = np.asarray([float(frame.time_us) / 1000.0 for frame in frames], dtype=float)
    times_ms -= times_ms[0]
    frozen = np.asarray([[float(frame.x), float(frame.y)] for frame in frames], dtype=float)
    specs = {index: (start, end, mode) for index, start, end, mode in _segment_specs(times_ms, segment_boundaries_ms, segment_modes)}
    selected = sorted({int(value) for value in selected_segment_indices})
    missing = [value for value in selected if value not in specs]
    unsupported = [value for value in selected if value in specs and specs[value][2] != "circle"]
    if missing or unsupported:
        raise ValueError(f"invalid boundary-aware segments missing={missing} unsupported={unsupported}")
    spans = [(specs[value][0], specs[value][1]) for value in selected]
    contacts = {int(np.argmin(np.abs(times_ms - float(value)))) for value in segment_boundaries_ms}
    neighborhoods = _neighborhoods(spans, _failing_supports(frozen, times_ms, ExecutionConstraints()), len(frozen))
    repaired = frozen.copy()
    records = []
    for index, (edit_start, edit_end) in enumerate(neighborhoods):
        solved = _solve_anchor(
            frozen, times_ms, edit_start, edit_end, contacts, ExecutionConstraints(),
            max_displacement_px, rms_displacement_px, max_function_evaluations,
        )
        covered = [value for value in selected if specs[value][1] >= edit_start and specs[value][0] <= edit_end]
        record = {"neighborhood_index": index, "selected_segment_indices": covered, **{key: value for key, value in solved.items() if key != "repaired"}}
        if solved["found"]:
            repaired[edit_start : edit_end + 1] = solved["repaired"][edit_start : edit_end + 1]
        records.append(record)

    hybrid = repaired.copy()
    call_counter = [0]
    for record in records:
        if not record["found"]:
            record.update({"learned_accepted": False, "effective_alpha": None, "learned_fallback_reason": "missing_feasible_repaired_anchor"})
            continue
        learned = []
        try:
            for segment_index in record["selected_segment_indices"]:
                start, end, _ = specs[segment_index]
                route = FrozenRoute.from_timed_samples(
                    f"boundary-aware:{segment_index}", times_ms[start : end + 1] - times_ms[start], repaired[start : end + 1], mode="circle",
                    metadata={"candidate": BOUNDARY_AWARE_VERSION, "segment_index": segment_index},
                )
                learned.append((start, end, route, np.asarray(route.reference_phases), _smooth_phases(model, route, times_ms[start : end + 1], call_counter)))
        except (TypeError, ValueError, FloatingPointError, RuntimeError, ArithmeticError) as error:
            record.update({
                "learned_accepted": False,
                "effective_alpha": None,
                "learned_fallback_reason": f"model_proposal_rejected:{type(error).__name__}",
                "meaningful_learned_samples": 0,
                "learned_sample_denominator": int(record["edit_end"] - record["edit_start"] + 1),
                "max_learned_delta_px": 0.0,
                "repaired_hybrid_exact_validation": record["exact_validation"],
            })
            continue
        accepted = repaired.copy()
        effective_alpha = 0.0
        checks = 0
        for alpha_fraction in _alpha_candidates():
            checks += 1
            proposal = repaired.copy()
            for start, end, route, anchor_phases, learned_phases in learned:
                phases = (1.0 - blend * alpha_fraction) * anchor_phases + blend * alpha_fraction * learned_phases
                proposal[start : end + 1] = np.asarray([route.position_at_phase(value) for value in phases])
            if exact_kinematic_summary(
                proposal[record["validation_start"] : record["validation_end"] + 1],
                times_ms[record["validation_start"] : record["validation_end"] + 1], ExecutionConstraints(),
            )["valid"]:
                accepted, effective_alpha = proposal, float(blend * alpha_fraction)
                break
        hybrid[record["edit_start"] : record["edit_end"] + 1] = accepted[record["edit_start"] : record["edit_end"] + 1]
        delta = np.linalg.norm(accepted[record["edit_start"] : record["edit_end"] + 1] - repaired[record["edit_start"] : record["edit_end"] + 1], axis=1)
        meaningful_samples = int(np.sum(delta > 1e-6))
        learned_accepted = effective_alpha > 0.0 and meaningful_samples > 0
        hybrid_exact = exact_kinematic_summary(
            accepted[record["validation_start"] : record["validation_end"] + 1],
            times_ms[record["validation_start"] : record["validation_end"] + 1], ExecutionConstraints(),
        )
        record.update({
            "learned_accepted": learned_accepted,
            "effective_alpha": effective_alpha,
            "alpha_candidates_checked": checks,
            "learned_fallback_reason": None if learned_accepted else (
                "nonzero_phase_alpha_without_coordinate_movement"
                if effective_alpha > 0.0 else "no_nonzero_learned_mixture_passes_exact_outer_join_validator"
            ),
            "meaningful_learned_samples": meaningful_samples,
            "learned_sample_denominator": int(len(delta)),
            "max_learned_delta_px": float(np.max(delta)),
            "repaired_hybrid_exact_validation": hybrid_exact,
        })

    anchor_frames = [type(frame)(frame.time_us, float(point[0]), float(point[1]), frame.k1, frame.k2) for frame, point in zip(frames, repaired)]
    hybrid_frames = [type(frame)(frame.time_us, float(point[0]), float(point[1]), frame.k1, frame.k2) for frame, point in zip(frames, hybrid)]
    fallback = [record for record in records if not record.get("learned_accepted")]
    telemetry = {
        "adapter": BOUNDARY_AWARE_VERSION,
        "requested_blend": float(blend),
        "selected_segment_indices": selected,
        "model_sha256": getattr(model, "model_sha256", "unknown"),
        "actual_model_calls": call_counter[0],
        "proposal_count": call_counter[0],
        "neighborhood_count": len(records),
        "learned_segment_count": sum(len(record["selected_segment_indices"]) for record in records if record.get("learned_accepted")),
        "fallback_segment_count": sum(len(record["selected_segment_indices"]) for record in fallback),
        "fallback": bool(fallback),
        "fallback_reason": ";".join(record.get("learned_fallback_reason") or record.get("fallback_reason") or "unknown" for record in fallback) or None,
        "meaningful_learned_samples": sum(int(record.get("meaningful_learned_samples", 0)) for record in records),
        "coordinate_movement_threshold_px": 1e-6,
        "frozen_math_full_trace_validity": exact_kinematic_summary(frozen, times_ms, ExecutionConstraints()),
        "repaired_math_full_trace_validity": exact_kinematic_summary(repaired, times_ms, ExecutionConstraints()),
        "repaired_hybrid_full_trace_validity": exact_kinematic_summary(hybrid, times_ms, ExecutionConstraints()),
        "whole_trace_scope_warning": "untouched invalid stencils remain outside selected neighborhoods",
        "max_displacement_budget_px": max_displacement_px,
        "rms_displacement_budget_px": rms_displacement_px,
        "solver_safety_factor": SOLVER_SAFETY_FACTOR,
        "learned_reserve_factor": LEARNED_RESERVE_FACTOR,
        "timestamps_preserved": all(left.time_us == right.time_us for left, right in zip(frames, hybrid_frames)),
        "keys_preserved": all((left.k1, left.k2) == (right.k1, right.k2) for left, right in zip(frames, hybrid_frames)),
        "neighborhoods": records,
    }
    return anchor_frames, hybrid_frames, telemetry
