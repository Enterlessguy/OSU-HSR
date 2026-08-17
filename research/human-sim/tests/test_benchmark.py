from __future__ import annotations

import gzip
import json
import math
from types import SimpleNamespace

from human_sim.benchmark import (
    _approach_aligned_summary,
    _kinematic_summary,
    _monotonicity,
    _spinner_idle_summary,
    _trace_motion_summary,
    format_summary,
    run_benchmark,
    write_report,
)
from human_sim.schemas import MapPlan, TraceFrame


def _write_map(path):
    objects = []
    for index in range(40):
        objects.append(
            {
                "index": index,
                "kind": "circle",
                "effective_start_time_ms": 2000 + index * 170,
                "effective_end_time_ms": 2000 + index * 170,
                "position": {"x": 80 + (index % 8) * 48, "y": 70 + (index % 5) * 55},
                "radius": 32,
            }
        )
    value = {
        "schema_version": 1,
        "beatmap_sha256": "c" * 64,
        "beatmap_md5": "d" * 32,
        "clock_rate": 1.0,
        "mods": [],
        "metadata": {"title": "benchmark fixture", "object_count": len(objects)},
        "objects": objects,
    }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(value) + "\n")


def test_benchmark_reports_reproducibility_distributions_and_json(tmp_path):
    map_path = tmp_path / "benchmark.map.ndjson.gz"
    output_path = tmp_path / "benchmark.json"
    summary_path = tmp_path / "benchmark.txt"
    _write_map(map_path)

    report = run_benchmark(
        map_paths=[map_path],
        seeds=(42, 43, 44),
        monotonicity_profiles=((40, 80), (50, 80), (60, 80), (50, 60), (50, 80), (50, 90)),
    )
    write_report(report, output_path, summary_path)

    metrics = report["maps"][0]["runs"][0]["metrics"]
    assert report["same_seed"]["all_exact"]
    assert metrics["matching_one_to_one"]
    assert "press_radial" in metrics
    assert "planned_offset_x" in metrics["rows"][0]
    assert "context_distribution" in metrics
    assert "continuity" in metrics
    assert "flow_0_45" in metrics["continuity"]
    assert metrics["approach_aligned"]["samples"] >= 32
    assert "covariance_eigen_anisotropy" in metrics["approach_aligned"]
    assert "wedge_share" in metrics["approach_aligned"]
    assert "planned_approach_aligned" in metrics
    assert "covariance_eigen_anisotropy" in metrics["planned_approach_aligned"]
    assert "motion" in metrics["continuity"]
    assert "flick_share" in metrics["continuity"]["motion"]
    assert report["monotonicity"]["maps"] == ["benchmark"]
    assert report["monotonicity"]["seeds"] == [42, 43, 44]
    assert "acceleration_px_s2" in metrics["kinematics"]
    assert "lateral_acceleration_px_s2" in metrics["kinematics"]
    assert "spinner_idle_motion" in metrics
    assert metrics["kinematics"]["sampling_rate_hz"] == 500
    assert report["classification"] == "planner-only/not-runtime-validated"
    assert "gates=" in format_summary(report)
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert "same-seed-exact=PASS" in summary_path.read_text(encoding="utf-8")


def test_monotonicity_rejects_consistent_multi_seed_inversion():
    rows = []
    for seed in (1, 2, 3):
        rows.extend(
            [
                {"map": "mixed", "seed": seed, "skill": 50.0, "effort": 80.0, "success_rate": 0.9, "radial_mean": 0.2, "timing_p95_abs_ms": 10.0},
                {"map": "mixed", "seed": seed, "skill": 70.0, "effort": 80.0, "success_rate": 0.5, "radial_mean": 0.5, "timing_p95_abs_ms": 20.0},
            ]
        )
    result = _monotonicity(rows, "skill")
    assert not result["pass"]
    assert result["consistent_failures"] == 1


def test_kinematics_resample_variable_event_frames():
    frames = [
        TraceFrame(0, 0.0, 0.0, False, False),
        TraceFrame(700, 10.0, 0.0, False, False),
        TraceFrame(2000, 20.0, 0.0, True, False),
        TraceFrame(2600, 30.0, 0.0, False, False),
    ]
    result = _kinematic_summary(frames, sample_rate_hz=500)
    assert result["resampled_frames"] == 2
    assert result["jerk_px_s3"]["n"] == 0
    assert "raw_acceleration_px_s2" in result


def test_trace_motion_detects_slider_handoff_overshoot_and_return():
    plan = MapPlan.from_dict(
        {
            "schema_version": 1,
            "beatmap_sha256": "a" * 64,
            "beatmap_md5": "b" * 32,
            "clock_rate": 1.0,
            "mods": [],
            "metadata": {"title": "yank fixture"},
            "objects": [
                {
                    "index": 0,
                    "kind": "slider",
                    "effective_start_time_ms": 1000,
                    "effective_end_time_ms": 1200,
                    "position": {"x": 100, "y": 100},
                    "end_position": {"x": 100, "y": 100},
                    "path_samples": [{"x": 100, "y": 100}, {"x": 100, "y": 100}],
                    "radius": 32,
                },
                {
                    "index": 1,
                    "kind": "circle",
                    "effective_start_time_ms": 1400,
                    "effective_end_time_ms": 1400,
                    "position": {"x": 200, "y": 100},
                    "radius": 32,
                },
            ],
        }
    )
    timeline_start_ms = -500.0
    samples = [
        (900.0, 100.0, 100.0),
        (1200.0, 100.0, 100.0),
        (1260.0, 100.0, 250.0),
        (1320.0, 100.0, 300.0),
        (1400.0, 200.0, 100.0),
        (1500.0, 200.0, 100.0),
    ]
    frames = [
        TraceFrame(int(round((time_ms - timeline_start_ms) * 1000.0)), x, y, False, False)
        for time_ms, x, y in samples
    ]
    planner = SimpleNamespace(
        profile=SimpleNamespace(sample_rate_hz=500),
        _speed_ceiling=lambda _distance: 4_000.0,
    )

    result = _trace_motion_summary(plan, frames, planner, timeline_start_ms)

    assert result["slider_handoffs_expected"] == 1
    assert result["slider_handoffs_evaluated"] == 1
    assert result["yank_count"] == 1
    assert result["corridor_deviation_ratio"]["max"] > 1.25


def test_approach_summary_detects_a_wedge_and_accepts_a_cloud():
    angles = [0.0, 0.45, 1.1, 2.0, 2.8, -0.8, -1.7, -2.6]
    cloud = [[0.18 * math.cos(angle), 0.16 * math.sin(angle)] for angle in angles]
    summary = _approach_aligned_summary(cloud)
    assert summary["sector_entropy"] > 0.5
    assert summary["abs_axis_correlation"] < 0.8
    assert summary["samples"] == len(cloud)

    wedge = [[0.4, 0.04], [0.6, -0.05], [0.8, 0.03], [-0.5, 0.04], [-0.7, -0.06]]
    wedge_summary = _approach_aligned_summary(wedge)
    assert wedge_summary["wedge_share"] > 0.6


def test_spinner_idle_summary_does_not_hide_wrong_direction_or_hovering():
    plan = MapPlan.from_dict(
        {
            "schema_version": 1,
            "beatmap_sha256": "e" * 64,
            "beatmap_md5": "f" * 32,
            "clock_rate": 1.0,
            "mods": [],
            "metadata": {"title": "mode-specific fixture"},
            "objects": [
                {"index": 0, "kind": "spinner", "effective_start_time_ms": 1000, "effective_end_time_ms": 2000, "position": {"x": 256, "y": 192}, "radius": 32},
                {"index": 1, "kind": "circle", "effective_start_time_ms": 5000, "effective_end_time_ms": 5000, "position": {"x": 384, "y": 288}, "radius": 32},
            ],
        }
    )
    timeline_start = -500.0
    frames = []
    for time_ms in range(1000, 5001, 10):
        if time_ms <= 2000:
            angle = (time_ms - 1000) / 1000.0 * math.tau * 4.0
            x = 256.0 + 80.0 * math.cos(angle)
            y = 192.0 + 80.0 * math.sin(angle)
        else:
            x, y = 384.0, 288.0
        frames.append(TraceFrame(int((time_ms - timeline_start) * 1000), x, y, False, False))

    summary = _spinner_idle_summary(plan, frames, timeline_start, 100)

    assert summary["spinner_windows"] == 1
    assert summary["spinner_counter_clockwise_share"] < 0.05
    assert summary["idle_windows"] == 1
    assert summary["idle_target_hover_share_under_45_px"] > 0.95
