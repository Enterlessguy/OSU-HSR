from __future__ import annotations

from human_sim.matching import PressEvent, match_press_events
from human_sim.schemas import HitWindows, MapObject, Point


def _objects() -> tuple[MapObject, ...]:
    return tuple(
        MapObject(
            index=index,
            kind="circle",
            start_time_ms=float(start),
            end_time_ms=float(start),
            position=Point(100 + index * 20, 100),
            end_position=Point(100 + index * 20, 100),
            radius=32.0,
            hit_windows=HitWindows(40, 80, 120, 150),
        )
        for index, start in enumerate((100, 200, 300))
    )


def test_matching_is_one_to_one_and_does_not_reuse_a_press():
    presses = [PressEvent(101, 0), PressEvent(299, 1), PressEvent(301, 0)]
    matched = match_press_events(_objects(), presses, minimum_window_ms=0)

    assert len(matched) == 3
    assert len(set(matched.values())) == len(matched)
    assert matched[0] == 0
    assert matched[2] in {1, 2}


def test_matching_skips_press_outside_object_window():
    presses = [PressEvent(500, 0)]
    assert match_press_events(_objects(), presses, minimum_window_ms=0) == {}
