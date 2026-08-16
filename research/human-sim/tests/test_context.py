from __future__ import annotations

import math

from human_sim.context import build_contexts
from human_sim.planner import HumanTracePlanner
from human_sim.schemas import HitWindows, HumanProfile, MapObject, MapPlan, Point


def _object(index: int, kind: str, start: float, x: float, y: float, end_x: float | None = None, end_y: float | None = None) -> MapObject:
    return MapObject(
        index=index,
        kind=kind,
        start_time_ms=start,
        end_time_ms=start if kind != "slider" else start + 40.0,
        position=Point(x, y),
        end_position=Point(end_x if end_x is not None else x, end_y if end_y is not None else y),
        radius=32.0,
        hit_windows=HitWindows(40.0, 80.0, 120.0, 150.0),
    )


def _plan() -> MapPlan:
    return MapPlan(
        schema_version=1,
        beatmap_sha256="a" * 64,
        beatmap_md5="b" * 32,
        clock_rate=1.5,
        mods=("HD", "HR"),
        metadata={"approach_rate": 9, "overall_difficulty": 8, "circle_size": 4, "drain_rate": 6},
        objects=(
            _object(0, "circle", 1000, 256, 192),
            _object(1, "circle", 1080, 440, 192),
            _object(2, "circle", 1160, 440, 350),
            _object(3, "circle", 1280, 300, 350),
            _object(4, "slider", 1600, 300, 350, 400, 200),
            _object(5, "spinner", 1800, 256, 192),
        ),
    )


def test_context_features_handle_first_history_and_optional_metadata():
    contexts = build_contexts(_plan())

    assert contexts[0].label == "transition"
    # A dense 184 px move is a jump even though its interval is stream-like.
    assert contexts[1].label == "jump"
    assert contexts[2].direction_change_angle_deg > 30.0
    assert contexts[4].label == "slider"
    assert contexts[5].label == "spinner"
    assert contexts[2].local_object_count >= 3
    assert contexts[2].rhythm_continuation > 0.5
    features = contexts[2].as_feature_dict()
    for name in ("approach_velocity_px_s", "direction_change_angle_rad", "rhythm_ratio", "local_density", "edge_proximity", "map_approach_rate", "hidden", "hard_rock"):
        assert name in features
        if isinstance(features[name], float):
            assert math.isfinite(features[name])
    assert all(math.isfinite(float(value)) for value in features.values() if isinstance(value, (int, float)))

    alias_plan = MapPlan(
        schema_version=1,
        beatmap_sha256="e" * 64,
        beatmap_md5="f" * 32,
        clock_rate=1.0,
        mods=(),
        objects=_plan().objects,
        metadata={"AR": "9", "OD": "8", "CS": "4", "HP": "6", "object count": "6"},
    )
    alias_context = build_contexts(alias_plan)[0]
    assert alias_context.map_approach_rate == 9.0
    assert alias_context.map_overall_difficulty == 8.0
    assert alias_context.map_circle_size == 4.0
    assert alias_context.map_drain_rate == 6.0
    assert alias_context.map_object_count == 6

    compact_reversal = MapPlan(
        schema_version=1,
        beatmap_sha256="1" * 64,
        beatmap_md5="2" * 32,
        clock_rate=1.0,
        mods=(),
        objects=(
            _object(0, "circle", 1000, 200, 200),
            _object(1, "circle", 1080, 260, 200),
            _object(2, "circle", 1160, 200, 200),
        ),
        metadata={},
    )
    compact_contexts = build_contexts(compact_reversal)
    assert compact_contexts[2].direction_change_angle_deg >= 170.0
    assert compact_contexts[2].label == "stream"


def test_missing_metadata_is_backward_compatible_and_planner_reuses_contexts():
    plan = _plan()
    plan = MapPlan(
        schema_version=plan.schema_version,
        beatmap_sha256=plan.beatmap_sha256,
        beatmap_md5=plan.beatmap_md5,
        clock_rate=plan.clock_rate,
        mods=(),
        objects=plan.objects,
        metadata={},
    )
    contexts = build_contexts(plan)
    planner = HumanTracePlanner(plan, HumanProfile(95.0, 11, 500))

    assert planner.contexts == contexts
    assert planner._context(1, plan.objects[1], contexts[1].strain) == contexts[1].label
    assert contexts[0].map_approach_rate == 0.0
    assert all(math.isfinite(context.strain) for context in contexts)
