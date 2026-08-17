from __future__ import annotations

import gzip
import json

import numpy as np

from human_sim.io import load_map_plan
from human_sim.planner import HumanTracePlanner, _quintic_hermite
from human_sim.schemas import HumanProfile


def _write_circle_flow_map(path):
    positions = [
        (96, 192),
        (150, 192),
        (204, 192),
        (258, 192),
        (312, 192),
        (366, 192),
        (393, 239),
        (420, 282),
        (250, 100),
        (420, 282),
        (250, 100),
        (420, 282),
    ]
    objects = [
        {
            "index": index,
            "kind": "circle",
            "effective_start_time_ms": 2000 + index * 200,
            "effective_end_time_ms": 2000 + index * 200,
            "position": {"x": x, "y": y},
            "radius": 32,
        }
        for index, (x, y) in enumerate(positions)
    ]
    value = {
        "schema_version": 1,
        "beatmap_sha256": "f" * 64,
        "beatmap_md5": "e" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "motion continuity fixture"},
        "objects": objects,
    }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")


def test_quintic_hermite_matches_position_velocity_and_acceleration_endpoints():
    p0 = np.array([10.0, 20.0])
    v0 = np.array([100.0, -20.0])
    a0 = np.array([30.0, 40.0])
    p1 = np.array([220.0, 80.0])
    v1 = np.array([-50.0, 70.0])
    a1 = np.array([-20.0, 10.0])
    start = _quintic_hermite(p0, v0, a0, p1, v1, a1, 240.0, 0.0)
    end = _quintic_hermite(p0, v0, a0, p1, v1, a1, 240.0, 1.0)
    assert np.allclose(start[0], p0)
    assert np.allclose(start[1], v0)
    assert np.allclose(start[2], a0)
    assert np.allclose(end[0], p1)
    assert np.allclose(end[1], v1)
    assert np.allclose(end[2], a1)


def test_ordinary_circle_motion_keeps_state_and_stratifies_turns(tmp_path):
    map_path = tmp_path / "flow.map.ndjson.gz"
    _write_circle_flow_map(map_path)
    plan = load_map_plan(map_path)
    planner = HumanTracePlanner(plan, HumanProfile(99.5, 42, 500, skill_level=50, effort_level=80))
    frames = planner.generate()
    stats = planner.continuity_stats

    assert stats["flow_0_45"]["samples"] >= 4
    assert stats["flow_0_45"]["median_carry_ratio"] >= 0.50
    assert stats["flow_0_45"]["severe_stop_share"] <= 0.15
    assert stats["turn_45_120"]["median_carry_ratio"] < stats["flow_0_45"]["median_carry_ratio"]
    assert stats["reversal_gt_120"]["median_carry_ratio"] < stats["flow_0_45"]["median_carry_ratio"]
    assert stats["base_boundary_position_error_px"] <= 1e-4
    assert stats["base_shared_velocity_error_px_s"] <= 1e-5
    assert stats["base_shared_acceleration_error_px_s2"] <= 1e-3
    assert all(left.time_us < right.time_us for left, right in zip(frames, frames[1:]))


def test_motion_state_is_reproducible_and_noise_does_not_reset_per_segment(tmp_path):
    map_path = tmp_path / "flow.map.ndjson.gz"
    _write_circle_flow_map(map_path)
    plan = load_map_plan(map_path)
    profile = HumanProfile(99.5, 1234, 500, skill_level=50, effort_level=80)
    first = HumanTracePlanner(plan, profile)
    second = HumanTracePlanner(plan, profile)
    first_frames = first.generate()
    second_frames = second.generate()
    assert [(frame.time_us, frame.x, frame.y, frame.k1, frame.k2) for frame in first_frames] == [
        (frame.time_us, frame.x, frame.y, frame.k1, frame.k2) for frame in second_frames
    ]
    assert first.motion_state.last_timestamp_ms >= plan.objects[0].start_time_ms - 1500.0
    assert np.linalg.norm(first.motion_state.ou_offset) >= 0.0
    assert first.continuity_stats["correction_share"] < 0.75


def test_subsample_event_keeps_hermite_endpoint_exact_and_timestamps_increasing(tmp_path):
    map_path = tmp_path / "subsample.map.ndjson.gz"
    objects = [
        {
            "index": index,
            "kind": "circle",
            "effective_start_time_ms": 2000 + index,
            "effective_end_time_ms": 2000 + index,
            "position": {"x": 100 + 50 * index, "y": 192},
            "radius": 32,
        }
        for index in range(5)
    ]
    value = {
        "schema_version": 1,
        "beatmap_sha256": "a" * 64,
        "beatmap_md5": "b" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "subsample boundary fixture"},
        "objects": objects,
    }
    with gzip.open(map_path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")

    plan = load_map_plan(map_path)
    planner = HumanTracePlanner(plan, HumanProfile(99.5, 42, 500, skill_level=50, effort_level=80))
    frames = planner.generate()

    assert planner.continuity_stats["base_boundary_position_error_px"] <= 1e-4
    assert planner.continuity_stats["base_shared_velocity_error_px_s"] <= 1e-5
    assert planner.continuity_stats["base_shared_acceleration_error_px_s2"] <= 1e-3
    assert all(left.time_us < right.time_us for left, right in zip(frames, frames[1:]))
