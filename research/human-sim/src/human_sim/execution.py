"""Execution-only route contracts and offline phase-law planning.

This module deliberately has no Windows input or target-selection code.  A
``FrozenRoute`` is created by the mathematical planner and is the only source
of geometry consumed by the execution model.  The learned component proposes
positive phase increments; it never returns XY points, targets, keys, or
routes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


EXECUTION_SCHEMA_VERSION = 1
EXECUTION_FEATURE_VERSION = "execution-context-v2"
EXECUTION_NORMALIZATION_VERSION = "phase-increment-log-v1"
EXECUTION_BASELINE_VERSION = "math-minimum-jerk-v1"
EXECUTION_HYBRID_VERSION = "execution-phase-hybrid-v1"
EXECUTION_TIME_AWARE_HYBRID_VERSION = "execution-phase-hybrid-time-aware-integration-v3"
INTERVAL_BUCKETS = ("<10", "10-25", "25-50", "50-100", "100-200", ">200")


def _finite_pair(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in pair):
        raise ValueError(f"{name} must be finite")
    return pair


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(_canonical(value), separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _key_event_dict(event: "KeyEvent") -> dict[str, Any]:
    return {"time_ms": event.time_ms, "key": event.key, "down": event.down}


@dataclass(frozen=True, slots=True)
class KinematicState:
    position: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    acceleration: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _finite_pair(self.position, "position"))
        object.__setattr__(self, "velocity", _finite_pair(self.velocity, "velocity"))
        object.__setattr__(self, "acceleration", _finite_pair(self.acceleration, "acceleration"))

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "velocity": list(self.velocity),
            "acceleration": list(self.acceleration),
        }


@dataclass(frozen=True, slots=True)
class ExecutionConstraints:
    """Mathematical feasibility envelope used after a model proposal."""

    max_speed_px_s: float = 12_000.0
    max_acceleration_px_s2: float = 120_000.0
    max_jerk_px_s3: float = 2_500_000.0
    min_phase_increment: float = 1e-10

    def __post_init__(self) -> None:
        for name in ("max_speed_px_s", "max_acceleration_px_s2", "max_jerk_px_s3"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        minimum = float(self.min_phase_increment)
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError("min_phase_increment must be finite and non-negative")
        object.__setattr__(self, "min_phase_increment", minimum)


@dataclass(frozen=True, slots=True)
class KeyEvent:
    time_ms: float
    key: int
    down: bool

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_ms)) or float(self.time_ms) < 0:
            raise ValueError("key event time must be finite and non-negative")
        if int(self.key) not in (0, 1):
            raise ValueError("key must be 0 or 1")
        object.__setattr__(self, "time_ms", float(self.time_ms))
        object.__setattr__(self, "key", int(self.key))


@dataclass(frozen=True, slots=True)
class FrozenRoute:
    """Immutable mathematical route and its timing/state contract.

    ``points`` are the route geometry.  ``position_at_phase`` interpolates the
    arc-length table, so a phase law can only move along this geometry.  The
    optional reference parameterization is used when a legacy mathematical
    trace is wrapped for an explicitly requested offline execution experiment;
    it does not change the route geometry.
    """

    route_id: str
    mode: str
    points: tuple[tuple[float, float], ...]
    arc_lengths: tuple[float, ...]
    duration_ms: float
    start_time_ms: float = 0.0
    entry_state: KinematicState = field(default_factory=KinematicState)
    exit_state: KinematicState = field(default_factory=KinematicState)
    constraints: ExecutionConstraints = field(default_factory=ExecutionConstraints)
    key_schedule: tuple[KeyEvent, ...] = ()
    skill_level: float = 50.0
    effort_level: float = 100.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_times_ms: tuple[float, ...] = ()
    reference_phases: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.route_id:
            raise ValueError("route_id must not be empty")
        if not self.mode:
            raise ValueError("mode must not be empty")
        points = tuple(_finite_pair(point, "route point") for point in self.points)
        arc_lengths = tuple(float(value) for value in self.arc_lengths)
        if len(points) < 2 or len(arc_lengths) != len(points):
            raise ValueError("a route needs at least two points and one arc length per point")
        if any(not math.isfinite(value) or value < 0 for value in arc_lengths):
            raise ValueError("arc lengths must be finite and non-negative")
        if any(right + 1e-9 < left for left, right in zip(arc_lengths, arc_lengths[1:])):
            raise ValueError("arc lengths must be non-decreasing")
        if arc_lengths[-1] <= 1e-9:
            raise ValueError("route geometry must have positive length")
        duration = float(self.duration_ms)
        start = float(self.start_time_ms)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("route duration must be positive and finite")
        if not math.isfinite(start):
            raise ValueError("route start time must be finite")
        if not 0 <= float(self.skill_level) <= 100 or not 0 <= float(self.effort_level) <= 100:
            raise ValueError("skill and effort must be between 0 and 100")
        schedule = tuple(sorted(self.key_schedule, key=lambda event: (event.time_ms, event.key, not event.down)))
        reference_times = tuple(float(value) for value in self.reference_times_ms)
        reference_phases = tuple(float(value) for value in self.reference_phases)
        if bool(reference_times) != bool(reference_phases) or len(reference_times) != len(reference_phases):
            raise ValueError("reference times and phases must be supplied together")
        if reference_times:
            if len(reference_times) < 2 or reference_times[0] < -1e-6 or reference_times[-1] > duration + 1e-6:
                raise ValueError("reference times must cover the route duration")
            if any(right <= left for left, right in zip(reference_times, reference_times[1:])):
                raise ValueError("reference times must be strictly increasing")
            if any(value < -1e-8 or value > 1.0 + 1e-8 for value in reference_phases):
                raise ValueError("reference phases must be in [0, 1]")
            if any(right + 1e-8 < left for left, right in zip(reference_phases, reference_phases[1:])):
                raise ValueError("reference phases must be monotone")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "arc_lengths", arc_lengths)
        object.__setattr__(self, "duration_ms", duration)
        object.__setattr__(self, "start_time_ms", start)
        object.__setattr__(self, "key_schedule", schedule)
        object.__setattr__(self, "skill_level", float(self.skill_level))
        object.__setattr__(self, "effort_level", float(self.effort_level))
        object.__setattr__(self, "reference_times_ms", reference_times)
        object.__setattr__(self, "reference_phases", reference_phases)
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))

    @classmethod
    def from_points(
        cls,
        route_id: str,
        points: Sequence[Sequence[float]],
        *,
        mode: str,
        duration_ms: float,
        start_time_ms: float = 0.0,
        entry_state: KinematicState | None = None,
        exit_state: KinematicState | None = None,
        constraints: ExecutionConstraints | None = None,
        key_schedule: Sequence[KeyEvent] = (),
        skill_level: float = 50.0,
        effort_level: float = 100.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FrozenRoute":
        normalized = tuple(_finite_pair(point, "route point") for point in points)
        if len(normalized) < 2:
            raise ValueError("a route needs at least two points")
        cumulative = [0.0]
        for left, right in zip(normalized, normalized[1:]):
            cumulative.append(cumulative[-1] + math.dist(left, right))
        return cls(
            route_id=route_id,
            mode=mode,
            points=normalized,
            arc_lengths=tuple(cumulative),
            duration_ms=duration_ms,
            start_time_ms=start_time_ms,
            entry_state=entry_state or KinematicState(position=normalized[0]),
            exit_state=exit_state or KinematicState(position=normalized[-1]),
            constraints=constraints or ExecutionConstraints(),
            key_schedule=tuple(key_schedule),
            skill_level=skill_level,
            effort_level=effort_level,
            metadata=metadata or {},
        )

    @classmethod
    def from_timed_samples(
        cls,
        route_id: str,
        times_ms: Sequence[float],
        points: Sequence[Sequence[float]],
        *,
        mode: str,
        key_schedule: Sequence[KeyEvent] = (),
        metadata: Mapping[str, Any] | None = None,
        constraints: ExecutionConstraints | None = None,
    ) -> "FrozenRoute":
        if len(times_ms) != len(points) or len(times_ms) < 2:
            raise ValueError("timed samples need matching time and point arrays")
        times = tuple(float(value) for value in times_ms)
        if times[0] < 0 or any(not math.isfinite(value) for value in times):
            raise ValueError("timed sample times must be finite and non-negative")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("timed sample times must be strictly increasing")
        normalized = tuple(_finite_pair(point, "route point") for point in points)
        cumulative = [0.0]
        for left, right in zip(normalized, normalized[1:]):
            cumulative.append(cumulative[-1] + math.dist(left, right))
        total = cumulative[-1]
        if total <= 1e-9:
            raise ValueError("timed samples have no geometric movement")
        phases = tuple(value / total for value in cumulative)
        return cls(
            route_id=route_id,
            mode=mode,
            points=normalized,
            arc_lengths=tuple(cumulative),
            duration_ms=times[-1],
            entry_state=KinematicState(position=normalized[0]),
            exit_state=KinematicState(position=normalized[-1]),
            constraints=constraints or ExecutionConstraints(),
            key_schedule=tuple(key_schedule),
            metadata=metadata or {},
            reference_times_ms=times,
            reference_phases=phases,
        )

    @property
    def length_px(self) -> float:
        return self.arc_lengths[-1]

    @property
    def route_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "route_id": self.route_id,
            "mode": self.mode,
            "points": [list(point) for point in self.points],
            "arc_lengths": list(self.arc_lengths),
            "duration_ms": self.duration_ms,
            "start_time_ms": self.start_time_ms,
            "entry_state": self.entry_state.as_dict(),
            "exit_state": self.exit_state.as_dict(),
            "constraints": dict(self.constraints.__dict__) if hasattr(self.constraints, "__dict__") else {
                "max_speed_px_s": self.constraints.max_speed_px_s,
                "max_acceleration_px_s2": self.constraints.max_acceleration_px_s2,
                "max_jerk_px_s3": self.constraints.max_jerk_px_s3,
                "min_phase_increment": self.constraints.min_phase_increment,
            },
            "key_schedule": [_key_event_dict(event) for event in self.key_schedule],
            "skill_level": self.skill_level,
            "effort_level": self.effort_level,
            "metadata": dict(self.metadata),
            "reference_times_ms": list(self.reference_times_ms),
            "reference_phases": list(self.reference_phases),
        }

    def position_at_phase(self, phase: float) -> tuple[float, float]:
        value = float(np.clip(float(phase), 0.0, 1.0))
        distance = value * self.length_px
        index = int(np.searchsorted(self.arc_lengths, distance, side="right") - 1)
        index = max(0, min(index, len(self.points) - 2))
        left_distance = self.arc_lengths[index]
        right_distance = self.arc_lengths[index + 1]
        if right_distance <= left_distance + 1e-12:
            return self.points[index + 1]
        amount = (distance - left_distance) / (right_distance - left_distance)
        left, right = self.points[index], self.points[index + 1]
        return (left[0] + amount * (right[0] - left[0]), left[1] + amount * (right[1] - left[1]))

    def tangent_at_phase(self, phase: float) -> tuple[float, float]:
        value = float(np.clip(float(phase), 0.0, 1.0))
        distance = value * self.length_px
        index = int(np.searchsorted(self.arc_lengths, distance, side="right") - 1)
        index = max(0, min(index, len(self.points) - 2))
        left, right = self.points[index], self.points[index + 1]
        dx, dy = right[0] - left[0], right[1] - left[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-12:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)


class PhaseExecutionModel(Protocol):
    """Small contract implemented by an offline execution model artifact."""

    model_sha256: str
    model_version: str
    feature_version: str

    def predict_phase_weights(self, route: FrozenRoute, sample_count: int) -> np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostics:
    planner: str
    model_version: str
    model_sha256: str
    fallback: bool = False
    fallback_reason: str | None = None
    blend: float = 0.0
    effective_blend: float | None = None
    projected_steps: int = 0
    projection_fraction: float = 0.0
    max_projection: float = 0.0
    endpoint_phase: float = 1.0
    supported_mode: bool = True
    configuration_sha256: str = ""
    segment_count: int = 1
    learned_segment_count: int = 0
    fallback_segment_count: int = 0
    segment_diagnostics: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "planner": self.planner,
            "model_version": self.model_version,
            "model_sha256": self.model_sha256,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "blend": self.blend,
            "effective_blend": self.effective_blend,
            "projected_steps": self.projected_steps,
            "projection_fraction": self.projection_fraction,
            "max_projection": self.max_projection,
            "endpoint_phase": self.endpoint_phase,
            "supported_mode": self.supported_mode,
            "configuration_sha256": self.configuration_sha256,
            "segment_count": self.segment_count,
            "learned_segment_count": self.learned_segment_count,
            "fallback_segment_count": self.fallback_segment_count,
            "segment_diagnostics": [dict(item) for item in self.segment_diagnostics],
        }


@dataclass(frozen=True, slots=True)
class ExecutionSample:
    time_ms: float
    phase: float
    position: tuple[float, float]
    speed_px_s: float
    acceleration_px_s2: float
    jerk_px_s3: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_ms", float(self.time_ms))
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(self, "position", _finite_pair(self.position, "sample position"))
        for name in ("speed_px_s", "acceleration_px_s2", "jerk_px_s3"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"sample {name} must be finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    route: FrozenRoute
    samples: tuple[ExecutionSample, ...]
    key_schedule: tuple[KeyEvent, ...]
    diagnostics: ExecutionDiagnostics

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if len(samples) < 2:
            raise ValueError("execution trace needs at least two samples")
        if any(right.time_ms <= left.time_ms for left, right in zip(samples, samples[1:])):
            raise ValueError("execution sample times must be strictly increasing")
        phases = [sample.phase for sample in samples]
        if any(value < -1e-7 or value > 1.0 + 1e-7 for value in phases):
            raise ValueError("execution phases must remain in [0, 1]")
        if any(right + 1e-8 < left for left, right in zip(phases, phases[1:])):
            raise ValueError("execution phase must be monotone")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "key_schedule", tuple(self.key_schedule))

    @property
    def route_sha256(self) -> str:
        return self.route.route_sha256

    @property
    def phases(self) -> tuple[float, ...]:
        return tuple(sample.phase for sample in self.samples)

    @property
    def positions(self) -> tuple[tuple[float, float], ...]:
        return tuple(sample.position for sample in self.samples)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "route_sha256": self.route_sha256,
            "route_id": self.route.route_id,
            "mode": self.route.mode,
            "route": self.route.as_dict(),
            "key_schedule": [_key_event_dict(event) for event in self.key_schedule],
            "diagnostics": self.diagnostics.as_dict(),
            "samples": [
                {
                    "time_ms": sample.time_ms,
                    "phase": sample.phase,
                    "position": list(sample.position),
                    "speed_px_s": sample.speed_px_s,
                    "acceleration_px_s2": sample.acceleration_px_s2,
                    "jerk_px_s3": sample.jerk_px_s3,
                }
                for sample in self.samples
            ],
        }


def interval_bucket(duration_ms: float) -> str:
    value = float(duration_ms)
    if value < 10:
        return "<10"
    if value < 25:
        return "10-25"
    if value < 50:
        return "25-50"
    if value < 100:
        return "50-100"
    if value < 200:
        return "100-200"
    return ">200"


def _finite_differences(values: np.ndarray, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(values) < 2:
        zeros = np.zeros(len(values), dtype=float)
        return zeros, zeros, zeros
    velocity = np.gradient(values, times_s, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, times_s, axis=0, edge_order=1) if len(values) >= 3 else np.zeros_like(velocity)
    jerk = np.gradient(acceleration, times_s, axis=0, edge_order=1) if len(values) >= 4 else np.zeros_like(acceleration)
    return velocity, acceleration, jerk


def _interval_kinematics(values: np.ndarray, times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward derivatives located at successive interval midpoints."""
    if len(values) < 2:
        empty = np.zeros((0,) + values.shape[1:], dtype=float)
        return empty, empty, empty
    dt = np.diff(times_s)
    if np.any(dt <= 0):
        raise ValueError("sample times must be strictly increasing")
    velocity = np.diff(values, axis=0) / dt.reshape((-1,) + (1,) * (values.ndim - 1))
    velocity_times = 0.5 * (times_s[:-1] + times_s[1:])
    if len(velocity) < 2:
        return velocity, np.zeros((0,) + values.shape[1:], dtype=float), np.zeros((0,) + values.shape[1:], dtype=float)
    acceleration_dt = np.diff(velocity_times)
    acceleration = np.diff(velocity, axis=0) / acceleration_dt.reshape((-1,) + (1,) * (values.ndim - 1))
    acceleration_times = 0.5 * (velocity_times[:-1] + velocity_times[1:])
    if len(acceleration) < 2:
        return velocity, acceleration, np.zeros((0,) + values.shape[1:], dtype=float)
    jerk_dt = np.diff(acceleration_times)
    jerk = np.diff(acceleration, axis=0) / jerk_dt.reshape((-1,) + (1,) * (values.ndim - 1))
    return velocity, acceleration, jerk


def _trace_from_phases(route: FrozenRoute, times_ms: np.ndarray, phases: np.ndarray, diagnostics: ExecutionDiagnostics) -> ExecutionTrace:
    phases = np.maximum.accumulate(np.clip(np.asarray(phases, dtype=float), 0.0, 1.0))
    phases = np.minimum(phases, diagnostics.endpoint_phase)
    points = np.asarray([route.position_at_phase(value) for value in phases], dtype=float)
    times_s = times_ms / 1000.0
    velocity, acceleration, jerk = _finite_differences(points, times_s)
    samples = tuple(
        ExecutionSample(
            time_ms=float(time),
            phase=float(phase),
            position=(float(point[0]), float(point[1])),
            speed_px_s=float(np.linalg.norm(velocity[index])),
            acceleration_px_s2=float(np.linalg.norm(acceleration[index])),
            jerk_px_s3=float(np.linalg.norm(jerk[index])),
        )
        for index, (time, phase, point) in enumerate(zip(times_ms, phases, points))
    )
    return ExecutionTrace(route, samples, route.key_schedule, diagnostics)


def _constraint_violation(trace: ExecutionTrace) -> str | None:
    """Return the first sampled kinematic envelope violation, if any."""
    if not trace.samples:
        return None
    route = trace.route
    times_s = np.asarray([sample.time_ms for sample in trace.samples], dtype=float) / 1000.0
    positions = np.asarray([sample.position for sample in trace.samples], dtype=float)
    dt = np.diff(times_s)
    if len(dt) == 0 or np.any(dt <= 0):
        return "invalid_sample_times"
    velocity, acceleration_values, jerk_values = _interval_kinematics(positions, times_s)
    speed = float(np.max(np.linalg.norm(velocity, axis=1))) if len(velocity) else 0.0
    acceleration = float(np.max(np.linalg.norm(acceleration_values, axis=1))) if len(acceleration_values) else 0.0
    jerk = float(np.max(np.linalg.norm(jerk_values, axis=1))) if len(jerk_values) else 0.0
    tolerance = 1e-6
    if speed > route.constraints.max_speed_px_s + tolerance:
        return f"speed:{speed:.6g}>{route.constraints.max_speed_px_s:.6g}"
    if acceleration > route.constraints.max_acceleration_px_s2 + tolerance:
        return f"acceleration:{acceleration:.6g}>{route.constraints.max_acceleration_px_s2:.6g}"
    if jerk > route.constraints.max_jerk_px_s3 + tolerance:
        return f"jerk:{jerk:.6g}>{route.constraints.max_jerk_px_s3:.6g}"
    return None


def _sample_times(route: FrozenRoute, sample_rate_hz: int) -> np.ndarray:
    if not 1 <= int(sample_rate_hz) <= 10_000:
        raise ValueError("sample_rate_hz must be between 1 and 10000")
    if route.reference_times_ms:
        return np.asarray(route.reference_times_ms, dtype=float)
    step = 1000.0 / float(sample_rate_hz)
    count = max(1, int(math.ceil(route.duration_ms / step)))
    values = np.arange(count + 1, dtype=float) * step
    values[-1] = route.duration_ms
    return np.unique(np.round(values, 9))


def _minimum_jerk_phase(route: FrozenRoute, times_ms: np.ndarray) -> tuple[np.ndarray, float]:
    if route.reference_phases:
        phases = np.interp(times_ms, route.reference_times_ms, route.reference_phases)
        return np.maximum.accumulate(phases), float(phases[-1])
    duration_s = route.duration_ms / 1000.0
    peak_factor = 1.875
    reachable = min(
        1.0,
        route.constraints.max_speed_px_s * duration_s / (peak_factor * route.length_px),
        route.constraints.max_acceleration_px_s2 * duration_s**2 / (5.7736 * route.length_px),
        route.constraints.max_jerk_px_s3 * duration_s**3 / (60.0 * route.length_px),
    )
    amount = np.clip(times_ms / route.duration_ms, 0.0, 1.0)
    smooth = 10 * amount**3 - 15 * amount**4 + 6 * amount**5
    return smooth * reachable, float(reachable)


class MathExecutionPlanner:
    """Deterministic minimum-jerk phase planner and control group."""

    model_version = EXECUTION_BASELINE_VERSION
    model_sha256 = "math-only"
    configuration_sha256 = canonical_sha256({"planner": EXECUTION_BASELINE_VERSION, "normalization": EXECUTION_NORMALIZATION_VERSION})

    def plan(self, route: FrozenRoute, sample_rate_hz: int = 500, times_ms: Sequence[float] | None = None) -> ExecutionTrace:
        times = np.asarray(times_ms, dtype=float) if times_ms is not None else _sample_times(route, sample_rate_hz)
        if len(times) < 2 or times[0] < -1e-9 or any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("execution times must be strictly increasing and non-negative")
        phases, endpoint = _minimum_jerk_phase(route, times)
        diagnostics = ExecutionDiagnostics(
            planner="math-only",
            model_version=self.model_version,
            model_sha256=self.model_sha256,
            endpoint_phase=endpoint,
            configuration_sha256=self.configuration_sha256,
        )
        return _trace_from_phases(route, times, phases, diagnostics)


def _smooth_positive(weights: np.ndarray) -> np.ndarray:
    if len(weights) < 3:
        return np.maximum(weights, 1e-12)
    smoothed = weights.copy()
    smoothed[1:-1] = 0.25 * weights[:-2] + 0.5 * weights[1:-1] + 0.25 * weights[2:]
    return np.maximum(smoothed, 1e-12)


def _project_increments(
    increments: np.ndarray,
    *,
    target_phase: float,
    route: FrozenRoute,
    times_ms: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Project positive increments into the mathematical speed envelope."""
    values = np.maximum(np.asarray(increments, dtype=float), route.constraints.min_phase_increment)
    if values.size == 0:
        return values, 0, 0.0
    values *= max(target_phase, 0.0) / max(float(values.sum()), 1e-12)
    dt_s = np.diff(times_ms) / 1000.0
    upper = route.constraints.max_speed_px_s * dt_s / route.length_px
    # The target phase is reduced instead of snapping to an impossible endpoint.
    reachable = min(float(target_phase), float(upper.sum()))
    values *= reachable / max(float(values.sum()), 1e-12)
    projected = 0
    maximum_delta = 0.0
    for _ in range(8):
        over = values > upper
        if not bool(over.any()):
            break
        before = values.copy()
        values[over] = upper[over]
        remainder = float(reachable - values.sum())
        free = ~over
        if remainder > 1e-12 and bool(free.any()):
            free_sum = float(values[free].sum())
            values[free] += remainder * values[free] / max(free_sum, 1e-12)
        projected += int(over.sum())
        maximum_delta = max(maximum_delta, float(np.max(np.abs(before - values))))
    values = np.maximum(values, route.constraints.min_phase_increment)
    values *= reachable / max(float(values.sum()), 1e-12)
    # Smooth phase-rate changes when the *sampled route* violates the
    # acceleration/jerk envelope.  Checking position derivatives here keeps
    # this projection contract aligned with the derivatives reported by
    # trace_metrics; phase-rate-only checks can miss boundary spikes from
    # interpolation and finite-difference stencils.
    for _ in range(32):
        phases = np.concatenate(([0.0], np.cumsum(values)))
        points = np.asarray([route.position_at_phase(value) for value in phases], dtype=float)
        times_s = times_ms / 1000.0
        velocity, acceleration, jerk = _interval_kinematics(points, times_s)
        speed_px = np.linalg.norm(velocity, axis=1)
        acceleration_px = np.linalg.norm(acceleration, axis=1)
        jerk_px = np.linalg.norm(jerk, axis=1)
        if (
            (not len(speed_px) or float(np.max(speed_px)) <= route.constraints.max_speed_px_s + 1e-6)
            and (not len(acceleration_px) or float(np.max(acceleration_px)) <= route.constraints.max_acceleration_px_s2 + 1e-6)
            and (not len(jerk_px) or float(np.max(jerk_px)) <= route.constraints.max_jerk_px_s3 + 1e-6)
        ):
            break
        before = values.copy()
        padded = np.pad(values, (1, 1), mode="edge")
        values = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
        values = np.maximum(values, route.constraints.min_phase_increment)
        values *= reachable / max(float(values.sum()), 1e-12)
        projected += 1
        maximum_delta = max(maximum_delta, float(np.max(np.abs(before - values))))
    return values, projected, maximum_delta


class HybridExecutionPlanner:
    """Offline phase-law hybrid with strict geometry preservation and fallback."""

    planner_name = EXECUTION_HYBRID_VERSION

    def __init__(self, model: PhaseExecutionModel | None = None, *, blend: float = 1.0, supported_modes: Sequence[str] = ("circle",)):
        if not 0.0 <= float(blend) <= 1.0:
            raise ValueError("blend must be between 0 and 1")
        self.model = model
        self.blend = float(blend)
        self.supported_modes = frozenset(str(mode) for mode in supported_modes)
        self.math = MathExecutionPlanner()
        self.configuration_sha256 = canonical_sha256(
            {
                "planner": self.planner_name,
                "blend": self.blend,
                "supported_modes": sorted(self.supported_modes),
                "feature_version": EXECUTION_FEATURE_VERSION,
                "normalization_version": EXECUTION_NORMALIZATION_VERSION,
            }
        )

    def _fallback(self, route: FrozenRoute, times: np.ndarray, reason: str, *, supported: bool = True) -> ExecutionTrace:
        math_trace = self.math.plan(route, times_ms=times)
        diagnostics = ExecutionDiagnostics(
            planner="math-fallback",
            model_version=self.model.model_version if self.model is not None else "none",
            model_sha256=getattr(self.model, "model_sha256", "none"),
            fallback=True,
            fallback_reason=reason,
            blend=self.blend,
            endpoint_phase=math_trace.diagnostics.endpoint_phase,
            supported_mode=supported,
            configuration_sha256=self.configuration_sha256,
        )
        return ExecutionTrace(route, math_trace.samples, route.key_schedule, diagnostics)

    def plan(self, route: FrozenRoute, sample_rate_hz: int = 500, times_ms: Sequence[float] | None = None) -> ExecutionTrace:
        times = np.asarray(times_ms, dtype=float) if times_ms is not None else _sample_times(route, sample_rate_hz)
        math_trace = self.math.plan(route, times_ms=times)
        if self.blend <= 0.0:
            return math_trace
        if route.mode not in self.supported_modes:
            return self._fallback(route, times, f"unsupported_mode:{route.mode}", supported=False)
        if self.model is None:
            return self._fallback(route, times, "missing_model")
        if getattr(self.model, "schema_version", EXECUTION_SCHEMA_VERSION) != EXECUTION_SCHEMA_VERSION:
            return self._fallback(route, times, "incompatible_schema_version")
        if getattr(self.model, "feature_version", EXECUTION_FEATURE_VERSION) != EXECUTION_FEATURE_VERSION:
            return self._fallback(route, times, "incompatible_feature_version")
        if getattr(self.model, "normalization_version", EXECUTION_NORMALIZATION_VERSION) != EXECUTION_NORMALIZATION_VERSION:
            return self._fallback(route, times, "incompatible_normalization_version")
        try:
            proposal = self._phase_proposal(route, times)
        except Exception as error:  # model rejection must never break math-only planning
            return self._fallback(route, times, f"model_error:{type(error).__name__}")
        if proposal.shape != (len(times) - 1,) or not np.all(np.isfinite(proposal)) or np.any(proposal <= 0):
            return self._fallback(route, times, "invalid_phase_proposal")
        target_phase = math_trace.diagnostics.endpoint_phase
        math_increments = np.diff(math_trace.phases)
        proposal = self._smooth_proposal(proposal)
        proposal *= target_phase / max(float(proposal.sum()), 1e-12)
        blended = (1.0 - self.blend) * math_increments + self.blend * proposal
        projected, count, max_delta, effective_blend = self._project_proposal(
            blended, math_increments=math_increments, target_phase=target_phase, route=route, times_ms=times
        )
        if effective_blend is not None and self.blend > 0.0 and effective_blend <= 1e-9:
            return self._fallback(route, times, "no_nonzero_learned_mixture_passes_constraints")
        phases = np.concatenate(([0.0], np.cumsum(projected)))
        phases = np.minimum(phases, target_phase)
        diagnostics = ExecutionDiagnostics(
            planner=self.planner_name,
            model_version=getattr(self.model, "model_version", "unknown"),
            model_sha256=getattr(self.model, "model_sha256", "unknown"),
            blend=self.blend,
            effective_blend=effective_blend,
            projected_steps=count,
            projection_fraction=count / max(len(projected), 1),
            max_projection=max_delta,
            endpoint_phase=float(phases[-1]),
            configuration_sha256=self.configuration_sha256,
        )
        trace = _trace_from_phases(route, times, phases, diagnostics)
        violation = _constraint_violation(trace)
        if violation is not None:
            return self._fallback(route, times, f"constraint_violation_after_projection:{violation}")
        return trace

    def _phase_proposal(self, route: FrozenRoute, times: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_phase_weights(route, len(times)), dtype=float)

    def _smooth_proposal(self, proposal: np.ndarray) -> np.ndarray:
        return _smooth_positive(proposal)

    def _project_proposal(
        self,
        proposal: np.ndarray,
        *,
        math_increments: np.ndarray,
        target_phase: float,
        route: FrozenRoute,
        times_ms: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float | None]:
        values, count, maximum = _project_increments(proposal, target_phase=target_phase, route=route, times_ms=times_ms)
        return values, count, maximum, None


class TimeAwareHybridExecutionPlanner(HybridExecutionPlanner):
    """Corrected integration of equal-time model weights onto real timestamps.

    V2 was trained on 32 equally spaced normalized-time points.  The legacy
    adapter instead requested one weight per runtime sample, which makes phase
    advance depend on sample density.  This adapter preserves the model,
    route, timestamps, keys, blend, projection and physical limits; it only
    maps the learned cumulative phase law through normalized time.
    """

    planner_name = EXECUTION_TIME_AWARE_HYBRID_VERSION

    def _phase_proposal(self, route: FrozenRoute, times: np.ndarray) -> np.ndarray:
        phase_points = int(getattr(self.model, "phase_points", 32))
        weights = np.asarray(self.model.predict_phase_weights(route, phase_points), dtype=float)
        if weights.shape != (phase_points - 1,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            return weights
        weights = _smooth_positive(weights)
        cumulative = np.concatenate(([0.0], np.cumsum(weights)))
        cumulative /= max(float(cumulative[-1]), 1e-12)
        normalized_time = np.clip(times / max(float(route.duration_ms), 1e-12), 0.0, 1.0)
        phases = np.interp(normalized_time, np.linspace(0.0, 1.0, phase_points), cumulative)
        return np.diff(phases)

    def _smooth_proposal(self, proposal: np.ndarray) -> np.ndarray:
        # Smoothing happened on the model's uniform-time grid before mapping.
        # Re-smoothing on the irregular runtime sample index would recreate the
        # timestamp-density bug this adapter corrects.
        return np.maximum(proposal, 1e-12)

    def _project_proposal(
        self,
        proposal: np.ndarray,
        *,
        math_increments: np.ndarray,
        target_phase: float,
        route: FrozenRoute,
        times_ms: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float | None]:
        """Retain the largest learned mixture that passes the exact validator.

        Interpolation between the frozen math law and learned time law occurs
        in phase space at the unchanged timestamps.  This avoids every
        sample-index smoothing or endpoint renormalization step.
        """
        proposal = np.maximum(np.asarray(proposal, dtype=float), route.constraints.min_phase_increment)
        proposal *= target_phase / max(float(proposal.sum()), 1e-12)

        def valid(values: np.ndarray) -> bool:
            phases = np.concatenate(([0.0], np.cumsum(values)))
            diagnostics = ExecutionDiagnostics(
                planner=self.planner_name,
                model_version=getattr(self.model, "model_version", "unknown"),
                model_sha256=getattr(self.model, "model_sha256", "unknown"),
                blend=self.blend,
                endpoint_phase=float(phases[-1]),
                configuration_sha256=self.configuration_sha256,
            )
            return _constraint_violation(_trace_from_phases(route, times_ms, phases, diagnostics)) is None

        if valid(proposal):
            return proposal, 0, 0.0, self.blend
        math_values = np.asarray(math_increments, dtype=float)
        if not valid(math_values):
            return proposal, 1, float(np.max(np.abs(proposal - math_values))), None
        low, high = 0.0, 1.0
        best = math_values
        for _ in range(24):
            alpha = 0.5 * (low + high)
            candidate = (1.0 - alpha) * math_values + alpha * proposal
            if valid(candidate):
                low, best = alpha, candidate
            else:
                high = alpha
        return best, 24, float(np.max(np.abs(proposal - best))), self.blend * low


def key_schedule_from_trace(frames: Sequence[Any]) -> tuple[KeyEvent, ...]:
    """Extract a relative immutable key schedule without retiming it."""
    if not frames:
        return ()
    origin_us = float(frames[0].time_us)
    events: list[KeyEvent] = []
    previous = (bool(frames[0].k1), bool(frames[0].k2))
    for key in (0, 1):
        if previous[key]:
            events.append(KeyEvent(0.0, key, True))
    for frame in frames[1:]:
        current = (bool(frame.k1), bool(frame.k2))
        time_ms = (float(frame.time_us) - origin_us) / 1000.0
        for key in (0, 1):
            if current[key] != previous[key]:
                events.append(KeyEvent(time_ms, key, current[key]))
        previous = current
    return tuple(events)


def _trace_from_positions(
    route: FrozenRoute,
    times_ms: np.ndarray,
    points: Sequence[Sequence[float]],
    diagnostics: ExecutionDiagnostics,
) -> ExecutionTrace:
    positions = np.asarray(points, dtype=float)
    times_s = np.asarray(times_ms, dtype=float) / 1000.0
    velocity, acceleration, jerk = _finite_differences(positions, times_s)
    if route.reference_phases and len(route.reference_phases) == len(positions):
        phases = np.asarray(route.reference_phases, dtype=float)
    else:
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(distances)))
        phases = cumulative / max(float(cumulative[-1]), 1e-12)
    samples = tuple(
        ExecutionSample(
            time_ms=float(time),
            phase=float(phase),
            position=(float(point[0]), float(point[1])),
            speed_px_s=float(np.linalg.norm(velocity[index])),
            acceleration_px_s2=float(np.linalg.norm(acceleration[index])),
            jerk_px_s3=float(np.linalg.norm(jerk[index])),
        )
        for index, (time, phase, point) in enumerate(zip(times_ms, phases, positions))
    )
    return ExecutionTrace(route, samples, route.key_schedule, diagnostics)


def warp_math_trace(
    frames: Sequence[Any],
    *,
    model: PhaseExecutionModel | None,
    blend: float,
    sample_rate_hz: int,
    route_id: str = "planner-trace",
    mode: str = "circle",
    segment_boundaries_ms: Sequence[float] | None = None,
    segment_modes: Sequence[str] | None = None,
    frozen_route: FrozenRoute | None = None,
    time_aware_phase: bool = False,
) -> tuple[list[Any], ExecutionTrace]:
    """Apply a phase law to already-generated math segments.

    This adapter is intentionally post-planner and offline.  It preserves every
    source timestamp and key state, so the hybrid cannot choose targets or key
    timing.  When segment boundaries are supplied, each segment is planned
    independently against the same frozen source geometry while the output
    keeps the original boundary samples and key schedule.  Only circle
    segments are currently supported; sliders, spinners, mixed joins and
    free-roam segments always remain on the mathematical path.
    """
    if len(frames) < 2:
        if not frames:
            return [], None  # type: ignore[return-value]
        fallback_route = FrozenRoute.from_points(
            route_id,
            [(frames[0].x, frames[0].y), (frames[0].x + 1.0, frames[0].y)],
            mode=mode,
            duration_ms=1.0,
        )
        return list(frames), MathExecutionPlanner().plan(fallback_route, sample_rate_hz=sample_rate_hz)
    times = np.asarray([float(frame.time_us) / 1000.0 for frame in frames], dtype=float)
    times -= times[0]
    points = [(float(frame.x), float(frame.y)) for frame in frames]
    try:
        if frozen_route is not None:
            if frozen_route.route_id != route_id:
                raise ValueError("frozen_route_id_mismatch")
            if frozen_route.duration_ms + 1e-6 < float(times[-1]):
                raise ValueError("frozen_route_duration_mismatch")
            if tuple(frozen_route.reference_times_ms) != tuple(float(value) for value in times):
                raise ValueError("frozen_route_timestamps_mismatch")
            if not np.array_equal(np.asarray(frozen_route.points, dtype=float), np.asarray(points, dtype=float)):
                raise ValueError("frozen_route_geometry_mismatch")
            if frozen_route.key_schedule != key_schedule_from_trace(frames):
                raise ValueError("frozen_route_key_schedule_mismatch")
            full_route = frozen_route
        else:
            full_route = FrozenRoute.from_timed_samples(
                route_id,
                times,
                points,
                mode=mode,
                key_schedule=key_schedule_from_trace(frames),
                metadata={"source": "mathematical_trace", "sample_rate_hz": sample_rate_hz},
            )
    except ValueError:
        if frozen_route is not None:
            raise
        fallback_route = FrozenRoute.from_points(
            route_id,
            [points[0], (points[0][0] + 1.0, points[0][1])],
            mode=mode,
            duration_ms=max(float(times[-1]), 1.0),
        )
        return list(frames), MathExecutionPlanner().plan(fallback_route, sample_rate_hz=sample_rate_hz)

    boundary_values = [0.0, float(times[-1])]
    if segment_boundaries_ms is not None:
        boundary_values.extend(float(value) for value in segment_boundaries_ms)
    boundary_values = sorted({max(0.0, min(float(times[-1]), value)) for value in boundary_values})
    boundary_indices: list[int] = []
    for boundary in boundary_values:
        index = int(np.argmin(np.abs(times - boundary)))
        if not boundary_indices or index > boundary_indices[-1]:
            boundary_indices.append(index)
    if boundary_indices[0] != 0:
        boundary_indices.insert(0, 0)
    if boundary_indices[-1] != len(frames) - 1:
        boundary_indices.append(len(frames) - 1)
    segment_count = max(len(boundary_indices) - 1, 1)
    requested_modes = tuple(str(value) for value in (segment_modes or (mode,)))
    output_points = list(points)
    segment_diagnostics: list[dict[str, Any]] = []
    fallback_reasons: list[str] = []
    projected_steps = 0
    learned_segments = 0
    fallback_segments = 0
    max_projection = 0.0
    for segment_index, (start_index, end_index) in enumerate(zip(boundary_indices, boundary_indices[1:])):
        if end_index <= start_index:
            continue
        segment_mode = requested_modes[segment_index] if segment_index < len(requested_modes) else mode
        segment_frames = frames[start_index:end_index + 1]
        segment_times = times[start_index:end_index + 1] - times[start_index]
        segment_points = [(float(frame.x), float(frame.y)) for frame in segment_frames]
        try:
            segment_route = FrozenRoute.from_timed_samples(
                f"{route_id}:segment:{segment_index}",
                segment_times,
                segment_points,
                mode=segment_mode,
                key_schedule=key_schedule_from_trace(segment_frames),
                metadata={
                    "source": "mathematical_trace_segment",
                    "parent_route_id": route_id,
                    "segment_index": segment_index,
                    "sample_rate_hz": sample_rate_hz,
                },
            )
            # Keep the supported-mode set explicit and narrow.  Passing the
            # requested segment mode here would accidentally bless sliders,
            # spinners or mixed joins as learned execution modes.
            planner_type = TimeAwareHybridExecutionPlanner if time_aware_phase else HybridExecutionPlanner
            segment_trace = planner_type(model, blend=blend, supported_modes=("circle",)).plan(
                segment_route,
                times_ms=segment_times,
            )
        except ValueError as error:
            reason = f"segment[{segment_index}]:route_rejected:{error}"
            fallback_reasons.append(reason)
            fallback_segments += 1
            segment_diagnostics.append(
                {
                    "segment_index": segment_index,
                    "mode": segment_mode,
                    "start_time_ms": float(segment_times[0]),
                    "end_time_ms": float(segment_times[-1]),
                    "duration_ms": float(segment_times[-1] - segment_times[0]),
                    "fallback": True,
                    "fallback_reason": reason,
                }
            )
            continue
        for output_index, sample in enumerate(segment_trace.samples, start=start_index):
            output_points[output_index] = sample.position
        diagnostics = segment_trace.diagnostics.as_dict()
        diagnostics.update(
            {
                "segment_index": segment_index,
                "mode": segment_mode,
                "start_time_ms": float(segment_times[0]),
                "end_time_ms": float(segment_times[-1]),
                "duration_ms": float(segment_times[-1] - segment_times[0]),
            }
        )
        segment_diagnostics.append(diagnostics)
        projected_steps += segment_trace.diagnostics.projected_steps
        max_projection = max(max_projection, segment_trace.diagnostics.max_projection)
        if blend > 0.0 and not segment_trace.diagnostics.fallback and segment_mode == "circle":
            learned_segments += 1
        if segment_trace.diagnostics.fallback:
            fallback_segments += 1
            fallback_reasons.append(f"segment[{segment_index}]:{segment_trace.diagnostics.fallback_reason}")

    if blend <= 0.0:
        planner_name = "math-only"
    elif fallback_segments == 0 and learned_segments > 0:
        planner_name = "execution-phase-segmented-hybrid"
    else:
        planner_name = "execution-phase-segmented-hybrid-with-fallback"
    full_diagnostics = ExecutionDiagnostics(
        planner=planner_name,
        model_version=getattr(model, "model_version", "none"),
        model_sha256=getattr(model, "model_sha256", "none"),
        fallback=bool(fallback_reasons),
        fallback_reason=";".join(fallback_reasons) if fallback_reasons else None,
        blend=float(blend),
        projected_steps=projected_steps,
        projection_fraction=projected_steps / max(len(output_points) - 1, 1),
        max_projection=max_projection,
        endpoint_phase=1.0,
        supported_mode=not any(item.get("mode") not in {"circle", "math-only"} for item in segment_diagnostics),
        configuration_sha256=full_route.route_sha256,
        segment_count=segment_count,
        learned_segment_count=learned_segments,
        fallback_segment_count=fallback_segments,
        segment_diagnostics=tuple(segment_diagnostics),
    )
    trace = _trace_from_positions(full_route, times, output_points, full_diagnostics)
    trace = ExecutionTrace(full_route, trace.samples, full_route.key_schedule, full_diagnostics)
    warped = [
        type(frame)(frame.time_us, point[0], point[1], frame.k1, frame.k2)
        for frame, point in zip(frames, output_points)
    ]
    return warped, trace
