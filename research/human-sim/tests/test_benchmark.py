from __future__ import annotations

import gzip
import json
import math

from human_sim.benchmark import (
    _approach_aligned_summary,
    _kinematic_summary,
    _monotonicity,
    format_summary,
    run_benchmark,
    write_report,
)
from human_sim.schemas import TraceFrame


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
