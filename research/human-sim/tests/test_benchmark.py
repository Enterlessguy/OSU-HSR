from __future__ import annotations

import gzip
import json

from human_sim.benchmark import format_summary, run_benchmark, write_report


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
    assert "acceleration_px_s2" in metrics["kinematics"]
    assert report["classification"] == "planner-only/not-runtime-validated"
    assert "gates=" in format_summary(report)
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert "same-seed-exact=PASS" in summary_path.read_text(encoding="utf-8")
