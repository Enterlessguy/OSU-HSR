from __future__ import annotations

"""Shared, interpretable pattern features for map objects.

The planner, replay dataset extractor, fitted-model path, and offline analysis
all need to describe the same local map pattern.  Keeping that derivation here
prevents a small threshold change in one consumer from silently changing the
meaning of ``stream`` in another.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from typing import Any

from .schemas import MapObject, MapPlan


CONTEXT_LABELS = ("stream", "burst", "jump", "transition", "slider", "spinner")

# Stable names used by dataset/modeling.  They are intentionally flat so the
# resulting Parquet data remains easy to inspect and old callers can continue
# to use the original distance/interval/strain fields.
CONTEXT_FEATURES = [
    "distance",
    "interval_ms",
    "strain",
    "approach_velocity_px_s",
    "direction_change_angle_rad",
    "rhythm_ratio",
    "rhythm_continuation",
    "local_density",
    "local_density_objects_per_s",
    "local_object_count",
    "strain_history_mean",
    "strain_history_max",
    "radius",
    "distance_over_radius",
    "edge_proximity",
    "start_x_norm",
    "start_y_norm",
    "clock_rate",
    "hidden",
    "hard_rock",
    "double_time",
    "half_time",
    "flashlight",
    "map_approach_rate",
    "map_overall_difficulty",
    "map_circle_size",
    "map_drain_rate",
    "map_object_count",
    "context",
    "object_kind",
]


def _metadata_number(metadata: dict[str, Any], name: str, default: float = 0.0) -> float:
    """Read optional exporter metadata without making it a schema requirement."""
    normalized = {
        str(key).strip().lower().replace("-", "_").replace(" ", "_"): value
        for key, value in metadata.items()
    }
    aliases = {
        "approach_rate": ("approach_rate", "approachrate", "ar"),
        "overall_difficulty": ("overall_difficulty", "overalldifficulty", "od"),
        "circle_size": ("circle_size", "circlesize", "cs"),
        "drain_rate": ("drain_rate", "drainrate", "hp"),
        "object_count": ("object_count", "objectcount", "objects"),
    }
    value = next((normalized[key] for key in aliases.get(name, (name,)) if key in normalized), default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _position(point: Any) -> tuple[float, float]:
    return float(point.x), float(point.y)


def _movement(previous: MapObject, current: MapObject) -> tuple[float, float]:
    """Return the object-to-object approach vector using robust endpoints."""
    previous_end = _position(previous.end_position)
    current_start = _position(current.position)
    return current_start[0] - previous_end[0], current_start[1] - previous_end[1]


def _angle_between(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length <= 1e-9 or second_length <= 1e-9:
        return 0.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)
    return math.acos(float(max(-1.0, min(1.0, cosine))))


@dataclass(frozen=True)
class PatternContext:
    """Structured local context for one map object.

    ``label`` is deliberately small and backwards-compatible.  Consumers that
    need more signal should use :meth:`as_feature_dict` rather than rebuilding
    a second classifier from interval and distance.
    """

    label: str
    object_kind: str
    interval_ms: float
    distance: float
    strain: float
    approach_velocity_px_s: float
    direction_change_angle_rad: float
    rhythm_ratio: float
    rhythm_continuation: float
    local_density: float
    local_density_objects_per_s: float
    local_object_count: int
    strain_history_mean: float
    strain_history_max: float
    radius: float
    distance_over_radius: float
    edge_proximity: float
    start_x_norm: float
    start_y_norm: float
    clock_rate: float
    hidden: bool
    hard_rock: bool
    double_time: bool
    half_time: bool
    flashlight: bool
    map_approach_rate: float
    map_overall_difficulty: float
    map_circle_size: float
    map_drain_rate: float
    map_object_count: int

    @property
    def direction_change_angle_deg(self) -> float:
        return math.degrees(self.direction_change_angle_rad)

    def as_feature_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "interval_ms": self.interval_ms,
            "strain": self.strain,
            "approach_velocity_px_s": self.approach_velocity_px_s,
            "direction_change_angle_rad": self.direction_change_angle_rad,
            "rhythm_ratio": self.rhythm_ratio,
            "rhythm_continuation": self.rhythm_continuation,
            "local_density": self.local_density,
            "local_density_objects_per_s": self.local_density_objects_per_s,
            "local_object_count": self.local_object_count,
            "strain_history_mean": self.strain_history_mean,
            "strain_history_max": self.strain_history_max,
            "radius": self.radius,
            "distance_over_radius": self.distance_over_radius,
            "edge_proximity": self.edge_proximity,
            "start_x_norm": self.start_x_norm,
            "start_y_norm": self.start_y_norm,
            "clock_rate": self.clock_rate,
            "hidden": self.hidden,
            "hard_rock": self.hard_rock,
            "double_time": self.double_time,
            "half_time": self.half_time,
            "flashlight": self.flashlight,
            "map_approach_rate": self.map_approach_rate,
            "map_overall_difficulty": self.map_overall_difficulty,
            "map_circle_size": self.map_circle_size,
            "map_drain_rate": self.map_drain_rate,
            "map_object_count": self.map_object_count,
            "context": self.label,
            "object_kind": self.object_kind,
        }


def _classify(
    *,
    kind: str,
    index: int,
    interval_ms: float,
    distance: float,
    radius: float,
    direction_change_angle_rad: float,
    rhythm_continuation: float,
) -> str:
    if kind in {"slider", "spinner"}:
        return kind
    if index == 0:
        return "transition"

    # A jump needs both meaningful geometry and urgency.  Distance alone
    # labels regular, moderate-speed patterns such as a 150 px / 214 ms
    # alternating map as jumps, while angle alone labels compact reversals as
    # jumps.  Normalized distance and approach velocity keep the classifier
    # interpretable without making every 60-degree turn a new pattern family.
    distance_over_radius = distance / max(radius, 1e-6)
    rhythm_break = rhythm_continuation < 0.45
    urgent_move = (distance / max(16.0, interval_ms) * 1000.0) >= 900.0 or rhythm_break
    large_jump = distance > max(120.0, radius * 3.5) and urgent_move
    meaningful_turn = (
        math.degrees(direction_change_angle_rad) >= 60.0
        and distance >= max(80.0, radius * 2.5)
        and distance_over_radius >= 3.0
        and urgent_move
    )
    if large_jump or meaningful_turn:
        return "jump"
    if interval_ms < 130.0:
        return "burst" if rhythm_break else "stream"
    if interval_ms < 220.0:
        return "burst"
    return "jump" if rhythm_break and distance > 80.0 else "transition"


def build_contexts(plan: MapPlan, *, density_window_ms: float = 600.0) -> tuple[PatternContext, ...]:
    """Build deterministic contexts for every object in ``plan``.

    Optional map metadata is read defensively.  First objects and incomplete
    history receive neutral values rather than NaNs or special-case failures.
    """
    objects = plan.objects
    starts = [float(obj.start_time_ms) for obj in objects]
    intervals: list[float] = []
    distances: list[float] = []
    strains: list[float] = []
    movements: list[tuple[float, float]] = []
    for index, obj in enumerate(objects):
        if index == 0:
            interval = 1000.0
            distance = 0.0
            movement = (0.0, 0.0)
        else:
            previous = objects[index - 1]
            interval = max(0.0, float(obj.start_time_ms - previous.end_time_ms))
            movement = _movement(previous, obj)
            distance = math.hypot(*movement)
        effective_interval = max(16.0, interval)
        strain = float(max(0.0, min(1.0, (distance / effective_interval) / 2.2)))
        intervals.append(interval)
        distances.append(distance)
        strains.append(strain)
        movements.append(movement)

    metadata = dict(plan.metadata or {})
    map_object_count = int(_metadata_number(metadata, "object_count", len(objects)))
    map_object_count = max(0, map_object_count)
    mods = frozenset(str(mod).upper() for mod in plan.mods)
    metadata_numbers = {
        "map_approach_rate": _metadata_number(metadata, "approach_rate"),
        "map_overall_difficulty": _metadata_number(metadata, "overall_difficulty"),
        "map_circle_size": _metadata_number(metadata, "circle_size"),
        "map_drain_rate": _metadata_number(metadata, "drain_rate"),
    }

    contexts: list[PatternContext] = []
    for index, obj in enumerate(objects):
        interval = intervals[index]
        distance = distances[index]
        movement = movements[index]
        if index >= 2:
            direction_change = _angle_between(movements[index - 1], movement)
        else:
            direction_change = 0.0

        if index == 0:
            rhythm_ratio = 1.0
            rhythm_continuation = 0.0
        elif index == 1:
            # There is no prior inter-object interval yet.  Do not penalize
            # the second note in an otherwise regular stream for that missing
            # history.
            rhythm_ratio = 1.0
            rhythm_continuation = 1.0
        elif intervals[index - 1] <= 0.0 or interval <= 0.0:
            rhythm_ratio = 4.0
            rhythm_continuation = 0.25
        else:
            rhythm_ratio = float(max(0.25, min(4.0, interval / intervals[index - 1])))
            rhythm_continuation = float(math.exp(-abs(math.log(rhythm_ratio))))

        left = bisect_left(starts, starts[index] - density_window_ms / 2.0)
        right = bisect_right(starts, starts[index] + density_window_ms / 2.0)
        local_count = max(1, right - left)
        local_density_per_second = local_count / max(0.25, density_window_ms / 1000.0)
        local_density = float(max(0.0, min(1.0, local_count / 8.0)))
        history = strains[max(0, index - 4) : index]
        history_mean = float(sum(history) / len(history)) if history else 0.0
        history_max = float(max(history)) if history else 0.0
        radius = max(1e-6, float(obj.radius))
        x, y = _position(obj.position)
        edge_distance = min(x, 512.0 - x, y, 384.0 - y)
        edge_proximity = float(max(0.0, min(1.0, 1.0 - edge_distance / 128.0)))
        label = _classify(
            kind=obj.kind,
            index=index,
            interval_ms=interval,
            distance=distance,
            radius=radius,
            direction_change_angle_rad=direction_change,
            rhythm_continuation=rhythm_continuation,
        )
        contexts.append(
            PatternContext(
                label=label,
                object_kind=obj.kind,
                interval_ms=float(interval),
                distance=float(distance),
                strain=strains[index],
                approach_velocity_px_s=float(distance / max(16.0, interval) * 1000.0),
                direction_change_angle_rad=float(direction_change),
                rhythm_ratio=float(rhythm_ratio),
                rhythm_continuation=float(rhythm_continuation),
                local_density=local_density,
                local_density_objects_per_s=float(local_density_per_second),
                local_object_count=local_count,
                strain_history_mean=history_mean,
                strain_history_max=history_max,
                radius=radius,
                distance_over_radius=float(distance / radius),
                edge_proximity=edge_proximity,
                start_x_norm=float(max(0.0, min(1.0, x / 512.0))),
                start_y_norm=float(max(0.0, min(1.0, y / 384.0))),
                clock_rate=float(plan.clock_rate),
                hidden="HD" in mods,
                hard_rock="HR" in mods,
                double_time="DT" in mods,
                half_time="HT" in mods,
                flashlight="FL" in mods,
                map_object_count=map_object_count,
                **metadata_numbers,
            )
        )
    return tuple(contexts)


def context_for_object(plan: MapPlan, index: int) -> PatternContext:
    """Return one context, retaining a convenient API for analysis callers."""
    if not 0 <= index < len(plan.objects):
        raise IndexError(index)
    return build_contexts(plan)[index]


def context_label(plan: MapPlan, index: int) -> str:
    """Compatibility helper for code that still consumes only a label."""
    return context_for_object(plan, index).label


# Names that make the intended shared implementation discoverable to callers
# that used the roadmap terminology.
build_pattern_contexts = build_contexts
PatternFeatures = PatternContext
