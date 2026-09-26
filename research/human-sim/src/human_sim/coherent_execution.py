"""Experimental runtime adapter for the frozen grouped coherent trajectory model.

The learned model proposes only a smooth, zero-boundary trajectory delta over
runs of four consecutive circle contacts.  The existing planner remains the
route, timing, aim and key source.  Unsupported or infeasible windows preserve
the planner trace byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.polynomial import Legendre, Polynomial

from .boundary_aware_execution import exact_kinematic_summary
from .execution import ExecutionConstraints, canonical_sha256
from .schemas import MapPlan, TraceFrame


COHERENT_EXECUTION_VERSION = "math-residual-lateral-gated-runtime-v1"
LATERAL_RESIDUAL_GAIN = 1.5
STATE_TIME_SCALE_S = 0.1
DWELL_MS = 750.0
DWELL_SUBSPANS = 4
DETAIL_MODES = 4
MAX_NODES = 7
MAX_COEFFICIENTS = 45
FEATURES = 22
CONTROLLER_GRID = np.linspace(1.0, 0.0, 101)


@dataclass(frozen=True)
class CoherentTrajectoryModel:
    canonical_sha256: str
    file_sha256: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    max_nodes: int


def load_coherent_model(path: str | Path) -> CoherentTrajectoryModel:
    source = Path(path)
    if source.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("coherent model exceeds the 5 MiB research artifact limit")
    raw = source.read_bytes()
    bundle = json.loads(raw)
    declared = str(bundle.get("sha256") or "")
    unsigned = dict(bundle)
    unsigned.pop("sha256", None)
    if not declared or canonical_sha256(unsigned) != declared:
        raise ValueError("coherent model canonical SHA-256 mismatch")
    group_size = int(bundle.get("group_size", -1))
    if int(bundle.get("seed", -1)) != 101 or group_size not in (100, 200):
        raise ValueError("research runtime requires an order-101 group-100 or group-200 checkpoint")
    if len(bundle.get("component_ids", [])) != group_size or len(set(bundle["component_ids"])) != group_size:
        raise ValueError("model's independent component count does not match group size")
    joint = bundle.get("joint")
    if not isinstance(joint, dict):
        raise ValueError("coherent model lacks joint weights")
    mean = np.asarray(joint.get("feature_mean"), dtype=float)
    scale = np.asarray(joint.get("feature_scale"), dtype=float)
    weights = np.asarray(joint.get("weights"), dtype=float)
    max_nodes = int(joint.get("max_nodes", -1))
    if mean.shape != (FEATURES,) or scale.shape != (FEATURES,):
        raise ValueError("coherent model feature contract mismatch")
    if weights.shape != (MAX_COEFFICIENTS, FEATURES + 1, 2) or max_nodes != MAX_NODES:
        raise ValueError("coherent model coefficient contract mismatch")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or not np.all(np.isfinite(weights)):
        raise ValueError("coherent model contains non-finite values")
    if np.any(scale <= 0.0):
        raise ValueError("coherent model feature scales must be positive")
    return CoherentTrajectoryModel(
        canonical_sha256=declared,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        feature_mean=mean,
        feature_scale=scale,
        weights=weights,
        max_nodes=max_nodes,
    )


def _poly_derivative(coefficients: np.ndarray, order: int) -> np.ndarray:
    values = np.asarray(coefficients, dtype=float)
    for _ in range(order):
        values = np.asarray([power * values[power] for power in range(1, len(values))])
    return values


def _hermite_polynomials(order: int = 0) -> np.ndarray:
    base = np.asarray(
        [
            [1, 0, 0, -10, 15, -6],
            [0, 1, 0, -6, 8, -3],
            [0, 0, 0.5, -1.5, 1.5, -0.5],
            [0, 0, 0, 10, -15, 6],
            [0, 0, 0, -4, 7, -3],
            [0, 0, 0, 0.5, -1, 0.5],
        ],
        dtype=float,
    )
    derived = [_poly_derivative(row, order) for row in base]
    width = max(len(row) for row in derived)
    return np.asarray([np.pad(row, (0, width - len(row))) for row in derived])


def _state_design(times_ms: np.ndarray, nodes: np.ndarray, order: int = 0) -> np.ndarray:
    design = np.zeros((len(times_ms), 3 * len(nodes)))
    polynomials = _hermite_polynomials(order)
    for row, value in enumerate(times_ms):
        span = min(max(int(np.searchsorted(nodes, value, side="right") - 1), 0), len(nodes) - 2)
        duration_s = (nodes[span + 1] - nodes[span]) / 1000.0
        u = np.clip((value - nodes[span]) / (nodes[span + 1] - nodes[span]), 0.0, 1.0)
        powers = u ** np.arange(polynomials.shape[1])
        basis = polynomials @ powers / duration_s**order
        scales = np.tile(np.asarray([1.0, duration_s / STATE_TIME_SCALE_S, duration_s**2 / STATE_TIME_SCALE_S**2]), 2)
        basis *= scales
        design[row, 3 * span : 3 * span + 3] += basis[:3]
        design[row, 3 * (span + 1) : 3 * (span + 1) + 3] += basis[3:]
    return design


@lru_cache(maxsize=4)
def _detail_polynomials(modes: int) -> tuple[Polynomial, ...]:
    envelope = Polynomial([0, 0, 0, 1, -3, 3, -1])
    x, weights = np.polynomial.legendre.leggauss(24)
    u = (x + 1) / 2
    result: list[Polynomial] = []
    for degree in range(modes):
        polynomial = envelope * Legendre.basis(degree).convert(kind=Polynomial)(Polynomial([-1, 2]))
        for previous in result:
            polynomial -= previous * np.sum(weights * polynomial(u) * previous(u)) / 2
        polynomial /= np.sqrt(np.sum(weights * polynomial(u) ** 2) / 2)
        result.append(polynomial)
    return tuple(result)


def _design(times_ms: np.ndarray, nodes: np.ndarray, modes: int = DETAIL_MODES, order: int = 0) -> np.ndarray:
    backbone = _state_design(times_ms, nodes, order)
    detail = np.zeros((len(times_ms), (len(nodes) - 1) * modes))
    for row, value in enumerate(times_ms):
        span = min(max(int(np.searchsorted(nodes, value, side="right") - 1), 0), len(nodes) - 2)
        duration_s = (nodes[span + 1] - nodes[span]) / 1000.0
        u = np.clip((value - nodes[span]) / (nodes[span + 1] - nodes[span]), 0.0, 1.0)
        for mode, polynomial in enumerate(_detail_polynomials(modes)):
            detail[row, span * modes + mode] = polynomial.deriv(order)(u) / duration_s**order
    return np.column_stack((backbone, detail))


def _expand_nodes(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    nodes = [float(events[0])]
    contacts = [0]
    dwell = False
    for left, right in zip(events[:-1], events[1:]):
        if right - left >= DWELL_MS:
            dwell = True
            nodes.extend(float(left + fraction * (right - left)) for fraction in np.linspace(0.0, 1.0, DWELL_SUBSPANS + 1)[1:])
        else:
            nodes.append(float(right))
        contacts.append(len(nodes) - 1)
    return np.asarray(nodes), np.asarray(contacts, dtype=int), dwell


def _task_baseline(
    contacts: np.ndarray,
    events: np.ndarray,
    nodes: np.ndarray,
    contact_indices: np.ndarray,
    outer_velocity: np.ndarray,
    outer_acceleration: np.ndarray,
) -> np.ndarray:
    states = np.zeros((3 * len(nodes), 2))
    positions = np.column_stack([np.interp(nodes, events, contacts[:, axis]) for axis in range(2)])
    states[0::3] = positions
    states[3 * contact_indices] = contacts
    seconds = nodes / 1000.0
    for index in range(len(nodes)):
        if index == 0:
            velocity = (positions[1] - positions[0]) / (seconds[1] - seconds[0])
        elif index == len(nodes) - 1:
            velocity = (positions[-1] - positions[-2]) / (seconds[-1] - seconds[-2])
        else:
            velocity = (positions[index + 1] - positions[index - 1]) / (seconds[index + 1] - seconds[index - 1])
        states[3 * index + 1] = velocity * STATE_TIME_SCALE_S
    states[1] = outer_velocity[0] * STATE_TIME_SCALE_S
    states[-2] = outer_velocity[1] * STATE_TIME_SCALE_S
    states[2] = outer_acceleration[0] * STATE_TIME_SCALE_S**2
    states[-1] = outer_acceleration[1] * STATE_TIME_SCALE_S**2
    return np.pad(states, ((0, (len(nodes) - 1) * DETAIL_MODES), (0, 0)))


def _context(events: np.ndarray, contacts: np.ndarray, outer_v: np.ndarray, outer_a: np.ndarray) -> dict[str, Any]:
    nodes, contact_indices, dwell = _expand_nodes(events)
    if len(nodes) > MAX_NODES:
        raise ValueError("expanded_node_capacity_over_7")
    positions = np.column_stack([np.interp(nodes, events, contacts[:, axis]) for axis in range(2)])
    return {
        "nodes": nodes,
        "contact_indices": contact_indices,
        "centers": contacts,
        "positions": positions,
        "dwell": dwell,
        "baseline": _task_baseline(contacts, events, nodes, contact_indices, outer_v, outer_a),
        "event_times": events,
    }


def _geometry(context: dict[str, Any]) -> dict[str, Any]:
    centers = context["centers"]
    chord = centers[-1] - centers[0]
    if np.linalg.norm(chord) < 1e-8:
        candidates = np.diff(centers, axis=0)
        chord = candidates[np.argmax(np.linalg.norm(candidates, axis=1))]
    unit = chord / max(np.linalg.norm(chord), 1e-8)
    if np.linalg.norm(unit) < 1e-8:
        unit = np.asarray([1.0, 0.0])
    rotation = np.asarray([unit, [-unit[1], unit[0]]])
    length = max(float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))), 32.0)
    n = len(context["nodes"])
    mapping = np.r_[np.arange(3 * n), 3 * MAX_NODES + np.arange(DETAIL_MODES * (n - 1))]
    mask = np.zeros(MAX_COEFFICIENTS)
    mask[mapping] = 1.0
    mask[3 * context["contact_indices"]] = 0.0
    mask[:3] = 0.0
    return {"origin": centers[0], "rotation": rotation, "length": length, "mapping": mapping, "mask": mask}


def _features(context: dict[str, Any], incoming: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    positions = np.zeros((4, 2))
    positions[: len(context["centers"])] = (context["centers"] - geometry["origin"]) @ geometry["rotation"].T / geometry["length"]
    intervals = np.diff(context["event_times"]) / 1000.0
    duration = np.zeros(3)
    duration[: len(intervals)] = np.log(intervals / 0.1)
    present = np.zeros(3)
    present[: len(intervals)] = 1.0
    state = np.asarray(
        [
            (incoming[0] - geometry["origin"]) @ geometry["rotation"].T / geometry["length"],
            incoming[1] @ geometry["rotation"].T * 0.1 / geometry["length"],
            incoming[2] @ geometry["rotation"].T * 0.01 / geometry["length"],
        ]
    )
    return np.r_[positions.ravel(), duration, present, len(context["nodes"]) / 7.0, float(context["dwell"]), state.ravel()]


def _normalize(coefficients: np.ndarray, context: dict[str, Any], geometry: dict[str, Any]) -> np.ndarray:
    actual = coefficients.copy()
    actual[: 3 * len(context["nodes"]) : 3] -= geometry["origin"]
    actual = actual @ geometry["rotation"].T / geometry["length"]
    padded = np.zeros((MAX_COEFFICIENTS, 2))
    padded[geometry["mapping"]] = actual
    return padded


def _decode(padded: np.ndarray, context: dict[str, Any], geometry: dict[str, Any]) -> np.ndarray:
    actual = padded[geometry["mapping"]] @ geometry["rotation"] * geometry["length"]
    actual[: 3 * len(context["nodes"]) : 3] += geometry["origin"]
    return actual


def _predict(context: dict[str, Any], model: CoherentTrajectoryModel) -> np.ndarray:
    geometry = _geometry(context)
    baseline = context["baseline"]
    incoming = np.asarray([baseline[0], baseline[1] / 0.1, baseline[2] / 0.01])
    feature = _features(context, incoming, geometry)
    encoded = np.r_[1.0, (feature - model.feature_mean) / model.feature_scale]
    delta = np.einsum("ikd,k->id", model.weights, encoded) * geometry["mask"][:, None]
    result = _normalize(baseline, context, geometry) + delta
    signal = _decode(result, context, geometry)
    # The offline checkpoint did not establish safe joins to arbitrary planner
    # traces.  Runtime-test mode therefore admits learned shape only inside a
    # window and preserves both outer P/V/A states exactly.
    signal[:3] = baseline[:3]
    end = 3 * (len(context["nodes"]) - 1)
    signal[end : end + 3] = baseline[end : end + 3]
    return signal


def _frame_arrays(frames: Sequence[TraceFrame]) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([frame.time_us / 1000.0 for frame in frames], dtype=float)
    points = np.asarray([[frame.x, frame.y] for frame in frames], dtype=float)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(points)) or np.any(np.diff(times) <= 0):
        raise ValueError("coherent input requires finite coordinates and strictly increasing timestamps")
    return times, points


def _outer_states(times: np.ndarray, points: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seconds = times / 1000.0
    velocity = np.gradient(points, seconds, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, seconds, axis=0, edge_order=2)
    outer_v = np.column_stack([np.interp(events[[0, -1]], times, velocity[:, axis]) for axis in range(2)])
    outer_a = np.column_stack([np.interp(events[[0, -1]], times, acceleration[:, axis]) for axis in range(2)])
    return outer_v, outer_a


def _circle_windows(map_plan: MapPlan) -> list[list[int]]:
    result: list[list[int]] = []
    run: list[int] = []

    def append_run() -> None:
        for offset in range(0, max(0, len(run) - 3), 3):
            result.append(run[offset : offset + 4])

    for index, obj in enumerate(map_plan.objects):
        if obj.kind == "circle":
            # A long gap is a break, even when circles bound both sides.  The
            # learned execution may not rewrite the mathematical break motion.
            if run and obj.start_time_ms - map_plan.objects[run[-1]].end_time_ms >= DWELL_MS:
                append_run()
                run = []
            run.append(index)
            continue
        append_run()
        run = []
    append_run()
    return result


def apply_coherent_trajectory_model(
    frames: Sequence[TraceFrame],
    map_plan: MapPlan,
    model: CoherentTrajectoryModel,
    *,
    blend: float = 1.0,
) -> tuple[list[TraceFrame], dict[str, Any]]:
    """Apply bounded learned interior deltas to eligible four-circle windows."""
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("coherent blend must be between 0 and 1")
    if not frames or len(map_plan.objects) < 4 or blend == 0.0:
        return list(frames), {
            "adapter": COHERENT_EXECUTION_VERSION,
            "model_sha256": model.canonical_sha256,
            "model_file_sha256": model.file_sha256,
            "effective_mode": "math-only",
            "fallback": True,
            "fallback_reason": "no_supported_window_or_zero_blend",
            "segment_count": 0,
            "learned_segment_count": 0,
            "fallback_segment_count": 0,
            "windows": [],
        }

    times, points = _frame_arrays(frames)
    original = points.copy()
    timeline_start = map_plan.objects[0].start_time_ms - 1500.0
    records: list[dict[str, Any]] = []
    constraints = ExecutionConstraints()
    for object_indices in _circle_windows(map_plan):
        events = np.asarray([map_plan.objects[index].start_time_ms - timeline_start for index in object_indices], dtype=float)
        mask = (times >= events[0] - 1e-9) & (times <= events[-1] + 1e-9)
        if np.count_nonzero(mask) < 8 or np.any(np.diff(events) <= 0.0):
            records.append({"object_indices": object_indices, "accepted": False, "reason": "insufficient_or_invalid_time_support"})
            continue
        contacts = np.column_stack([np.interp(events, times, points[:, axis]) for axis in range(2)])
        try:
            outer_v, outer_a = _outer_states(times, points, events)
            context = _context(events, contacts, outer_v, outer_a)
            signal = _predict(context, model)
        except (ValueError, FloatingPointError) as error:
            records.append({"object_indices": object_indices, "accepted": False, "reason": str(error)})
            continue

        baseline = context["baseline"]
        delta_coefficients = signal - baseline
        # Amplify the learned component normal to this four-contact route.
        # Apply the transform to coefficients before both continuous and
        # sampled safety checks. This keeps every zero P/V/A join intact.
        route_normal = _geometry(context)["rotation"][1]
        delta_coefficients = delta_coefficients + (LATERAL_RESIDUAL_GAIN - 1.0) * np.outer(
            delta_coefficients @ route_normal, route_normal
        )
        local_times = times[mask]
        matrix = _design(local_times, context["nodes"])
        delta_points = matrix @ delta_coefficients
        if not np.all(np.isfinite(delta_points)):
            records.append({"object_indices": object_indices, "accepted": False, "reason": "nonfinite_learned_delta"})
            continue
        if not np.any(np.linalg.norm(delta_points, axis=1) > 1e-9):
            records.append({"object_indices": object_indices, "accepted": False, "reason": "zero_learned_delta"})
            continue
        base_points = points[mask].copy()
        base_summary = exact_kinematic_summary(base_points, local_times, constraints)
        accepted_alpha: float | None = None
        accepted_points: np.ndarray | None = None
        checked = 0
        analytic = None
        sampled = None
        for alpha_value in CONTROLLER_GRID:
            alpha = float(alpha_value * blend)
            # Research variant: preserve the actual planner path and apply only
            # the learned residual. The deployed adapter instead replaces the
            # planner path with the fitted task curve.
            candidate = base_points + alpha * delta_points
            checked += 1
            analytic = _continuous_envelope(alpha * delta_coefficients, context["nodes"])
            sampled = exact_kinematic_summary(candidate, local_times, constraints)
            sampled["relative_failure_families"] = [
                metric for metric, limit in sampled["limits"].items()
                if sampled[metric] > max(limit, 1.01 * base_summary[metric]) + 1e-6
            ]
            sampled["relative_valid"] = not sampled["relative_failure_families"]
            if analytic["valid"] and sampled["relative_valid"]:
                accepted_alpha = alpha
                accepted_points = candidate
                break
        if accepted_points is None or accepted_alpha is None or accepted_alpha <= 0.0:
            records.append(
                {
                    "object_indices": object_indices,
                    "accepted": False,
                    "reason": "no_positive_feasible_alpha",
                    "checks": checked,
                    "analytic": analytic,
                    "sampled": sampled,
                }
            )
            continue
        points[mask] = accepted_points
        learned_contribution = np.linalg.norm(accepted_alpha * delta_points, axis=1)
        total_replacement = np.linalg.norm(accepted_points - base_points, axis=1)
        records.append(
            {
                "object_indices": object_indices,
                "accepted": True,
                "alpha": accepted_alpha,
                "checks": checked,
                "samples": int(np.count_nonzero(mask)),
                "learned_contribution_rms_px": float(np.sqrt(np.mean(learned_contribution**2))),
                "learned_contribution_max_px": float(np.max(learned_contribution)),
                "total_replacement_rms_px": float(np.sqrt(np.mean(total_replacement**2))),
                "total_replacement_max_px": float(np.max(total_replacement)),
                "composition": "math_plus_learned_residual_research_variant",
                "lateral_residual_gain": LATERAL_RESIDUAL_GAIN,
                "analytic_residual_only": True,
                "baseline_sampled": base_summary,
                "outer_pva_delta_forced_zero": True,
                "analytic": analytic,
                "sampled": sampled,
            }
        )

    output = [
        TraceFrame(frame.time_us, float(point[0]), float(point[1]), frame.k1, frame.k2)
        for frame, point in zip(frames, points)
    ]
    accepted = sum(bool(record.get("accepted")) for record in records)
    changed = np.linalg.norm(points - original, axis=1)
    return output, {
        "adapter": COHERENT_EXECUTION_VERSION,
        "requested_mode": "coherent",
        "effective_mode": "coherent" if accepted else "math-only",
        "blend": blend,
        "model_sha256": model.canonical_sha256,
        "model_file_sha256": model.file_sha256,
        "fallback": accepted == 0,
        "fallback_reason": None if accepted else "no_supported_positive_feasible_window",
        "segment_count": len(records),
        "learned_segment_count": accepted,
        "fallback_segment_count": len(records) - accepted,
        "changed_sample_count": int(np.count_nonzero(changed > 1e-9)),
        "trace_contribution_rms_px": float(np.sqrt(np.mean(changed**2))),
        "trace_contribution_max_px": float(np.max(changed)),
        "outer_join_policy": "learned delta has zero P/V/A at every window boundary",
        "composition": "math_plus_learned_residual_research_variant",
        "analytic_envelope_limitation": "analytic envelope bounds learned residual; sampled gate allows at most one-percent relative worsening of the existing math envelope",
        "scope": "experimental test build; frozen model; no human-level or deployment claim",
        "windows": records,
    }


def _span_polynomial(coefficients: np.ndarray, nodes: np.ndarray, span: int) -> np.ndarray:
    state_count = 3 * len(nodes)
    states = coefficients[:state_count]
    duration_s = (nodes[span + 1] - nodes[span]) / 1000.0
    scaled = np.vstack(
        (
            states[3 * span],
            states[3 * span + 1] * duration_s / STATE_TIME_SCALE_S,
            states[3 * span + 2] * duration_s**2 / STATE_TIME_SCALE_S**2,
            states[3 * (span + 1)],
            states[3 * (span + 1) + 1] * duration_s / STATE_TIME_SCALE_S,
            states[3 * (span + 1) + 2] * duration_s**2 / STATE_TIME_SCALE_S**2,
        )
    )
    result = np.pad(_hermite_polynomials().T @ scaled, ((0, DETAIL_MODES), (0, 0)))
    detail = coefficients[state_count + span * DETAIL_MODES : state_count + (span + 1) * DETAIL_MODES]
    for mode, polynomial in enumerate(_detail_polynomials(DETAIL_MODES)):
        result[: len(polynomial.coef)] += polynomial.coef[:, None] * detail[mode]
    return result


def _continuous_envelope(coefficients: np.ndarray, nodes: np.ndarray) -> dict[str, Any]:
    limits = {
        "speed_px_s": 12_000.0,
        "acceleration_px_s2": 120_000.0,
        "jerk_px_s3": 2_500_000.0,
    }
    result: dict[str, Any] = {}
    for order, metric in ((1, "speed_px_s"), (2, "acceleration_px_s2"), (3, "jerk_px_s3")):
        candidates = []
        for span in range(len(nodes) - 1):
            duration_s = (nodes[span + 1] - nodes[span]) / 1000.0
            polynomial = _span_polynomial(coefficients, nodes, span)
            derivative = np.column_stack(
                [_poly_derivative(polynomial[:, axis], order) for axis in range(2)]
            ) / duration_s**order
            norm_squared = np.zeros(2 * derivative.shape[0] - 1)
            for axis in range(2):
                norm_squared += np.convolve(derivative[:, axis], derivative[:, axis])
            stationary = np.asarray([power * norm_squared[power] for power in range(1, len(norm_squared))])
            roots = [] if not np.any(np.abs(stationary) > 1e-12) else np.roots(stationary[::-1])
            values = [0.0, 1.0, *(float(root.real) for root in roots if abs(root.imag) < 1e-8 and 0.0 < root.real < 1.0)]
            for u in values:
                vector = np.asarray([np.polynomial.polynomial.polyval(u, derivative[:, axis]) for axis in range(2)])
                candidates.append(float(np.linalg.norm(vector)))
        result[metric] = max(candidates, default=0.0)
        result[f"{metric}_ratio"] = result[metric] / limits[metric]
    result["failure_families"] = [metric for metric, limit in limits.items() if result[metric] > limit + 1e-6]
    result["valid"] = not result["failure_families"]
    return result
