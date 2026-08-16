from __future__ import annotations

"""Deterministic one-to-one press/object matching shared by analyses."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .schemas import MapObject


@dataclass(frozen=True)
class PressEvent:
    time_ms: float
    key: int = -1
    frame_index: int = -1


def _value(frame: Any, name: str, default: Any = None) -> Any:
    if isinstance(frame, dict):
        return frame.get(name, default)
    return getattr(frame, name, default)


def rising_press_events(frames: Iterable[Any], *, timeline_start_ms: float = 0.0) -> list[PressEvent]:
    """Extract every key rising edge exactly once from trace/replay frames."""
    events: list[PressEvent] = []
    previous = [False, False]
    for frame_index, frame in enumerate(frames):
        if _value(frame, "time_us") is not None:
            time_ms = timeline_start_ms + float(_value(frame, "time_us")) / 1000.0
        else:
            time_ms = timeline_start_ms + float(_value(frame, "time_ms", 0.0))
        state = [bool(_value(frame, "k1", False)), bool(_value(frame, "k2", False))]
        for key in range(2):
            if state[key] and not previous[key]:
                events.append(PressEvent(time_ms=time_ms, key=key, frame_index=frame_index))
        previous = state
    return events


def match_presses_to_objects(
    objects: Sequence[MapObject],
    press_times_ms: Sequence[float] | Sequence[PressEvent],
    *,
    minimum_window_ms: float = 0.0,
) -> dict[int, int]:
    """Return a one-to-one object-position -> press-position assignment.

    Candidate pairs are ranked globally by timing error, and each object and
    press can be consumed at most once.  The time-window lookup is indexed so
    large benchmark maps do not require an all-pairs scan.  Ties are resolved
    by map order then press order for reproducibility.
    """
    if not objects or not press_times_ms:
        return {}
    times = [
        float(value.time_ms if isinstance(value, PressEvent) else value)
        for value in press_times_ms
    ]
    if any(right < left for left, right in zip(times, times[1:])):
        # The caller's press index remains meaningful only for sorted inputs;
        # trace/replay extraction is ordered by time, so reject rather than
        # silently returning a mapping in a different coordinate system.
        raise ValueError("press times must be sorted in non-decreasing order")

    candidates: list[tuple[float, int, int]] = []
    for object_position, obj in enumerate(objects):
        window = max(float(minimum_window_ms), float(obj.hit_windows.miss_ms))
        left = bisect_left(times, float(obj.start_time_ms) - window)
        right = bisect_right(times, float(obj.start_time_ms) + window)
        for press_position in range(left, right):
            error = abs(times[press_position] - float(obj.start_time_ms))
            candidates.append((error, object_position, press_position))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    assigned_objects: set[int] = set()
    assigned_presses: set[int] = set()
    assignments: dict[int, int] = {}
    for _error, object_position, press_position in candidates:
        if object_position in assigned_objects or press_position in assigned_presses:
            continue
        assignments[object_position] = press_position
        assigned_objects.add(object_position)
        assigned_presses.add(press_position)
    return assignments


def match_press_events(
    objects: Sequence[MapObject],
    presses: Sequence[PressEvent],
    *,
    minimum_window_ms: float = 0.0,
) -> dict[int, int]:
    """Convenience wrapper for event records."""
    return match_presses_to_objects(objects, presses, minimum_window_ms=minimum_window_ms)
