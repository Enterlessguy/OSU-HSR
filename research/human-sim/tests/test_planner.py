from __future__ import annotations

import gzip
import json
import math

import numpy as np
import pytest

from human_sim.io import load_map_plan, write_trace
from human_sim.library_audit import discover_standard_beatmaps, verify_perfect_plan
from human_sim.planner import HumanTracePlanner
from human_sim.schemas import HumanProfile, TraceFrame
from human_sim.validation import validate_trace


def _write_map(path):
    value = {
        "schema_version": 1,
        "beatmap_sha256": "a" * 64,
        "beatmap_md5": "b" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "fixture"},
        "objects": [
            {
                "index": 0,
                "kind": "circle",
                "effective_start_time_ms": 2000,
                "effective_end_time_ms": 2000,
                "position": {"x": 128, "y": 96},
                "radius": 32,
            },
            {
                "index": 1,
                "kind": "slider",
                "effective_start_time_ms": 2500,
                "effective_end_time_ms": 3200,
                "position": {"x": 128, "y": 96},
                "end_position": {"x": 384, "y": 288},
                "radius": 32,
                "repeat_count": 0,
                "path_samples": [{"x": 128, "y": 96}, {"x": 256, "y": 220}, {"x": 384, "y": 288}],
            },
            {
                "index": 2,
                "kind": "spinner",
                "effective_start_time_ms": 3800,
                "effective_end_time_ms": 5000,
                "position": {"x": 256, "y": 192},
                "radius": 32,
            },
        ],
    }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")


def test_same_seed_is_byte_deterministic(tmp_path):
    map_path = tmp_path / "map.ndjson.gz"
    _write_map(map_path)
    plan = load_map_plan(map_path)
    profile = HumanProfile(95.0, 1234, 500)
    frames_a = HumanTracePlanner(plan, profile).generate()
    frames_b = HumanTracePlanner(plan, profile).generate()
    out_a = tmp_path / "a.trace.gz"
    out_b = tmp_path / "b.trace.gz"
    hash_a = write_trace(out_a, map_plan=plan, profile_percentile=95.0, seed=1234, sample_rate_hz=500, frames=frames_a)
    hash_b = write_trace(out_b, map_plan=plan, profile_percentile=95.0, seed=1234, sample_rate_hz=500, frames=frames_b)
    assert hash_a == hash_b
    with gzip.open(out_a, "rt", encoding="utf-8") as first, gzip.open(out_b, "rt", encoding="utf-8") as second:
        assert first.read() == second.read()


def test_trace_is_monotonic_and_releases_keys(tmp_path):
    map_path = tmp_path / "map.ndjson.gz"
    trace_path = tmp_path / "trace.ndjson.gz"
    _write_map(map_path)
    plan = load_map_plan(map_path)
    frames = HumanTracePlanner(plan, HumanProfile(99.5, 7, 500)).generate()
    write_trace(trace_path, map_plan=plan, profile_percentile=99.5, seed=7, sample_rate_hz=500, frames=frames)
    result = validate_trace(trace_path)
    assert result["status"] == "valid"
    assert result["frames"] > 1000


def test_repeat_slider_samples_are_not_repeated_twice(tmp_path):
    map_path = tmp_path / "repeat-slider-map.ndjson.gz"
    samples = []
    for index in range(101):
        progress = index / 100
        span_progress = 2 * progress if progress <= 0.5 else 2 * (1 - progress)
        samples.append({"x": 128 + 256 * span_progress, "y": 96})
    value = {
        "schema_version": 1,
        "beatmap_sha256": "c" * 64,
        "beatmap_md5": "d" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "repeat slider fixture"},
        "objects": [
            {
                "index": 0,
                "kind": "slider",
                "effective_start_time_ms": 2000,
                "effective_end_time_ms": 3000,
                "position": {"x": 128, "y": 96},
                "end_position": {"x": 128, "y": 96},
                "radius": 32,
                "repeat_count": 1,
                "path_samples": samples,
            }
        ],
    }
    with gzip.open(map_path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")

    frames = HumanTracePlanner(load_map_plan(map_path), HumanProfile(99.5, 7, 500)).generate()
    key_down_index = next(index for index, frame in enumerate(frames) if frame.k1)

    # Halfway through a two-span slider the cursor must be at the far repeat
    # point. The old implementation applied repeat_count to the already
    # repeated samples and incorrectly returned to the start at this point.
    assert frames[key_down_index + 250].x > 360
    assert frames[key_down_index + 500].x < 150


def test_slider_motion_stays_on_absolute_map_clock(tmp_path):
    map_path = tmp_path / "map.ndjson.gz"
    _write_map(map_path)
    plan = load_map_plan(map_path)
    planner = HumanTracePlanner(plan, HumanProfile(95.0, 1234, 500))
    slider = plan.objects[1]

    class ZeroNoise:
        @staticmethod
        def normal(_mean=0.0, _sigma=1.0, size=None):
            return np.zeros(size) if size is not None else 0.0

    planner.rng = ZeroNoise()
    points = [(slider.start_time_ms + 40.0, np.array([slider.position.x, slider.position.y]))]
    planner._append_slider(points, slider, slider.start_time_ms + 40.0, 0.0)

    # Even with a 40 ms late head hit, the final curve sample belongs to the
    # slider's real end time and reaches the exported tail. The old shifted
    # timeline did not reach the tail until 40 ms after the object ended.
    end_time, end_position = points[-1]
    assert end_time == slider.end_time_ms
    assert np.linalg.norm(end_position - np.array([slider.end_position.x, slider.end_position.y])) < 0.01


def test_slider_key_is_held_past_tail_without_explicit_break(tmp_path):
    map_path = tmp_path / "map.ndjson.gz"
    _write_map(map_path)
    plan = load_map_plan(map_path)
    frames = HumanTracePlanner(plan, HumanProfile(75.0, 42, 500)).generate()
    timeline_start = plan.objects[0].start_time_ms - 1500.0
    slider = plan.objects[1]
    end_index = round((slider.end_time_ms - timeline_start) / 2.0)

    assert frames[end_index].k1 or frames[end_index].k2
    assert frames[end_index + 2].k1 or frames[end_index + 2].k2


def test_perfect_baseline_hits_fixture_objects_exactly(tmp_path):
    map_path = tmp_path / "map.ndjson.gz"
    _write_map(map_path)
    plan = load_map_plan(map_path)
    profile = HumanProfile(99.5, 42, 500, perfect_baseline=True)
    frames = HumanTracePlanner(plan, profile).generate()
    timeline_start = plan.objects[0].start_time_ms - 1500.0

    circle_index = round((plan.objects[0].start_time_ms - timeline_start) / 2.0)
    slider_start_index = round((plan.objects[1].start_time_ms - timeline_start) / 2.0)
    slider_end_index = round((plan.objects[1].end_time_ms - timeline_start) / 2.0)

    assert frames[circle_index].k1
    assert (frames[circle_index].x, frames[circle_index].y) == (128.0, 96.0)
    assert frames[slider_start_index].k2
    assert (frames[slider_start_index].x, frames[slider_start_index].y) == (128.0, 96.0)
    assert frames[slider_end_index].k2
    assert abs(frames[slider_end_index].x - 384.0) < 0.01
    assert abs(frames[slider_end_index].y - 288.0) < 0.01


def test_perfect_baseline_inserts_exact_hit_frames_between_sample_ticks(tmp_path):
    map_path = tmp_path / "off-grid-map.ndjson.gz"
    value = {
        "schema_version": 1,
        "beatmap_sha256": "1" * 64,
        "beatmap_md5": "2" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "off-grid fixture"},
        "objects": [
            {
                "index": 0,
                "kind": "circle",
                "effective_start_time_ms": 2000.3,
                "effective_end_time_ms": 2000.3,
                "position": {"x": 64, "y": 64},
                "radius": 24,
            },
            {
                "index": 1,
                "kind": "circle",
                "effective_start_time_ms": 2501.0,
                "effective_end_time_ms": 2501.0,
                "position": {"x": 448, "y": 320},
                "radius": 24,
            },
        ],
    }
    with gzip.open(map_path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")

    plan = load_map_plan(map_path)
    frames = HumanTracePlanner(plan, HumanProfile(99.5, 42, 500, perfect_baseline=True)).generate()
    result = verify_perfect_plan(plan, frames, 500)

    assert result["max_hit_time_error_ms"] < 0.001
    assert result["max_head_position_error_osu_px"] < 0.001


def test_audit_finds_off_grid_slider_tail_by_timestamp(tmp_path):
    map_path = tmp_path / "off-grid-slider.ndjson.gz"
    value = {
        "schema_version": 1,
        "beatmap_sha256": "3" * 64,
        "beatmap_md5": "4" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "off-grid slider fixture"},
        "objects": [
            {
                "index": 0,
                "kind": "slider",
                "effective_start_time_ms": 2000.3,
                "effective_end_time_ms": 2501.1,
                "position": {"x": 96, "y": 96},
                "end_position": {"x": 416, "y": 288},
                "radius": 24,
                "repeat_count": 0,
                "path_samples": [{"x": 96, "y": 96}, {"x": 416, "y": 288}],
            }
        ],
    }
    with gzip.open(map_path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")

    plan = load_map_plan(map_path)
    frames = HumanTracePlanner(plan, HumanProfile(99.5, 42, 500, perfect_baseline=True)).generate()
    result = verify_perfect_plan(plan, frames, 500)

    assert result["slider_tail_holds"] == 1


def test_perfect_baseline_uses_negative_preroll_for_early_first_object(tmp_path):
    map_path = tmp_path / "early-first-object.ndjson.gz"
    trace_path = tmp_path / "early-first-object.trace.ndjson.gz"
    value = {
        "schema_version": 1,
        "beatmap_sha256": "5" * 64,
        "beatmap_md5": "6" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "early first object fixture"},
        "objects": [
            {
                "index": 0,
                "kind": "circle",
                "effective_start_time_ms": 7.0,
                "effective_end_time_ms": 7.0,
                "position": {"x": 42, "y": 80},
                "radius": 24,
            }
        ],
    }
    with gzip.open(map_path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")

    plan = load_map_plan(map_path)
    frames = HumanTracePlanner(plan, HumanProfile(99.5, 42, 500, perfect_baseline=True)).generate()
    first_key_down = next(frame for frame in frames if frame.k1 or frame.k2)
    write_trace(
        trace_path,
        map_plan=plan,
        profile_percentile=99.5,
        seed=42,
        sample_rate_hz=500,
        frames=frames,
        diagnostic_perfect=True,
    )

    assert first_key_down.time_us == 1_500_000
    assert (first_key_down.x, first_key_down.y) == (42.0, 80.0)
    assert validate_trace(trace_path)["max_speed_osu_px_s"] < 5000.0
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
    assert header["timeline_start_effective_ms"] == -1493.0
    assert header["clock_rate"] == 1.0


def test_perfect_baseline_rearms_keys_for_dense_stream(tmp_path):
    map_path = tmp_path / "dense-stream.ndjson.gz"
    objects = [
        {
            "index": index,
            "kind": "circle",
            "effective_start_time_ms": 2000 + index * 20,
            "effective_end_time_ms": 2000 + index * 20,
            "position": {"x": 96 + (index % 4) * 96, "y": 96 + (index % 2) * 160},
            "radius": 24,
        }
        for index in range(12)
    ]
    value = {
        "schema_version": 1,
        "beatmap_sha256": "e" * 64,
        "beatmap_md5": "f" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "dense stream fixture"},
        "objects": objects,
    }
    with gzip.open(map_path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")

    frames = HumanTracePlanner(
        load_map_plan(map_path), HumanProfile(99.5, 42, 500, perfect_baseline=True)
    ).generate()
    transitions = 0
    previous_k1 = previous_k2 = False
    for frame in frames:
        transitions += int(frame.k1 and not previous_k1)
        transitions += int(frame.k2 and not previous_k2)
        previous_k1, previous_k2 = frame.k1, frame.k2

    assert transitions == len(objects)


def test_perfect_trace_allows_expert_speed_without_weakening_profile_guard(tmp_path):
    map_path = tmp_path / "map.ndjson.gz"
    _write_map(map_path)
    plan = load_map_plan(map_path)
    frames = [
        TraceFrame(0, 100.0, 100.0, False, False),
        TraceFrame(2_000, 500.0, 100.0, False, False),
    ]

    perfect_path = tmp_path / "perfect.trace.gz"
    write_trace(
        perfect_path,
        map_plan=plan,
        profile_percentile=99.5,
        seed=42,
        sample_rate_hz=500,
        frames=frames,
        diagnostic_perfect=True,
    )
    assert validate_trace(perfect_path)["max_speed_osu_px_s"] == 200_000.0

    profile_path = tmp_path / "profile.trace.gz"
    write_trace(
        profile_path,
        map_plan=plan,
        profile_percentile=99.5,
        seed=42,
        sample_rate_hz=500,
        frames=frames,
        diagnostic_perfect=False,
    )
    with pytest.raises(ValueError, match="limit 100000"):
        validate_trace(profile_path)


def test_library_audit_verifies_perfect_fixture(tmp_path):
    map_path = tmp_path / "map.ndjson.gz"
    _write_map(map_path)
    plan = load_map_plan(map_path)
    frames = HumanTracePlanner(plan, HumanProfile(99.5, 42, 500, perfect_baseline=True)).generate()

    result = verify_perfect_plan(plan, frames, 500)

    assert result["key_downs"] == len(plan.objects)
    assert result["max_hit_time_error_ms"] <= 1.0
    assert result["max_head_position_error_osu_px"] < 0.01
    assert result["slider_tail_holds"] == 1


def test_library_discovery_filters_nonstandard_rulesets(tmp_path):
    standard = tmp_path / "standard"
    standard.write_text("osu file format v14\n[General]\nMode:0\n", encoding="utf-8")
    mania = tmp_path / "mania"
    mania.write_text("osu file format v14\n[General]\nMode:3\n", encoding="utf-8")
    asset = tmp_path / "asset"
    asset.write_bytes(b"not a beatmap")

    assert discover_standard_beatmaps(tmp_path) == [standard]


def _write_long_gap_map(path):
    value = {
        "schema_version": 1,
        "beatmap_sha256": "9" * 64,
        "beatmap_md5": "a" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "long gap fixture"},
        "objects": [
            {
                "index": 0,
                "kind": "circle",
                "effective_start_time_ms": 2000,
                "effective_end_time_ms": 2000,
                "position": {"x": 128, "y": 96},
                "radius": 32,
            },
            {
                "index": 1,
                "kind": "circle",
                "effective_start_time_ms": 9000,
                "effective_end_time_ms": 9000,
                "position": {"x": 384, "y": 288},
                "radius": 32,
            },
        ],
    }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")


def test_idle_wander_fills_long_gap_smoothly(tmp_path):
    map_path = tmp_path / "long-gap.ndjson.gz"
    trace_path = tmp_path / "long-gap.trace.ndjson.gz"
    _write_long_gap_map(map_path)
    plan = load_map_plan(map_path)
    frames = HumanTracePlanner(plan, HumanProfile(50.0, 1234, 500)).generate()

    # Trace-frame times are relative to the 500 ms pre-roll; the second
    # object's approach starts around trace-time 8100 (absolute ~8600), so the
    # idle window is safely inside the gap.
    xs = [frame.x for frame in frames if 2500.0 <= frame.time_us / 1000.0 <= 7800.0]
    ys = [frame.y for frame in frames if 2500.0 <= frame.time_us / 1000.0 <= 7800.0]
    assert np.std(xs) > 5.0 or np.std(ys) > 5.0

    # The doodle must stay smooth (no flicks/jerk) in the idle window.
    max_speed = 0.0
    previous = None
    for frame in frames:
        t = frame.time_us / 1000.0
        if 2500.0 <= t <= 7800.0:
            if previous is not None:
                delta_seconds = (t - previous[0]) / 1000.0
                if delta_seconds > 0:
                    distance = math.hypot(frame.x - previous[1], frame.y - previous[2])
                    max_speed = max(max_speed, distance / delta_seconds)
            previous = (t, frame.x, frame.y)
    assert max_speed < 2500.0

    # The wander envelope must converge onto the next target before the
    # approach: just before the second press (trace-time ~8500, absolute
    # ~9000) the cursor has to be near the second circle, otherwise a long
    # doodle could cause a late-arrival miss.
    late_window = [frame for frame in frames if 8350.0 <= frame.time_us / 1000.0 <= 8450.0]
    assert late_window
    target_x, target_y = 384.0, 288.0
    max_return_distance = max(
        math.hypot(frame.x - target_x, frame.y - target_y) for frame in late_window
    )
    assert max_return_distance < 45.0

    write_trace(
        trace_path,
        map_plan=plan,
        profile_percentile=50.0,
        seed=1234,
        sample_rate_hz=500,
        frames=frames,
    )
    assert validate_trace(trace_path)["status"] == "valid"


def test_perfect_baseline_does_not_doodle_during_long_gap(tmp_path):
    map_path = tmp_path / "long-gap.ndjson.gz"
    _write_long_gap_map(map_path)
    plan = load_map_plan(map_path)
    frames = HumanTracePlanner(plan, HumanProfile(99.5, 1234, 500, perfect_baseline=True)).generate()

    # Calibration traces stay exact and deterministic: no idle doodle. With
    # v2.7 flow motion the cursor glides straight from the first circle to the
    # second during the gap; assert the path stays on the connecting segment.
    positions = [
        (frame.x, frame.y) for frame in frames if 2500.0 <= frame.time_us / 1000.0 <= 7800.0
    ]
    assert positions
    ax, ay = 128.0, 96.0
    bx, by = 384.0, 288.0
    seg_dx, seg_dy = bx - ax, by - ay
    seg_len = math.hypot(seg_dx, seg_dy)
    for x, y in positions:
        progress = max(0.0, min(1.0, ((x - ax) * seg_dx + (y - ay) * seg_dy) / (seg_len * seg_len)))
        perpendicular = math.hypot((x - ax) - progress * seg_dx, (y - ay) - progress * seg_dy)
        assert perpendicular < 0.01
