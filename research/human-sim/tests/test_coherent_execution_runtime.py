from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from human_sim.cli import _resolve_auto_execution, main
from human_sim.coherent_execution import (
    CoherentTrajectoryModel,
    _circle_windows,
    apply_coherent_trajectory_model,
    load_coherent_model,
)
from human_sim.execution import canonical_sha256
from human_sim.schemas import HitWindows, MapObject, MapPlan, Point, TraceFrame


def _bundle() -> dict:
    value = {
        "seed": 101,
        "group_size": 200,
        "component_ids": [str(index) for index in range(200)],
        "joint": {
            "feature_mean": [0.0] * 22,
            "feature_scale": [1.0] * 22,
            "weights": np.zeros((45, 23, 2)).tolist(),
            "max_nodes": 7,
        },
    }
    value["sha256"] = canonical_sha256(value)
    return value


def _plan() -> MapPlan:
    windows = HitWindows(40, 80, 120, 150)
    objects = tuple(
        MapObject(index, "circle", time, time, Point(x, y), Point(x, y), 32, hit_windows=windows)
        for index, (time, x, y) in enumerate(
            ((2000.0, 100.0, 100.0), (2200.0, 180.0, 130.0), (2400.0, 260.0, 110.0), (2600.0, 340.0, 150.0))
        )
    )
    return MapPlan(1, "a" * 64, "b" * 32, 1.0, (), objects, {})


def test_loader_verifies_frozen_contract_and_hash(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_bundle()), encoding="utf-8")
    model = load_coherent_model(path)
    assert model.canonical_sha256 == _bundle()["sha256"]
    assert model.weights.shape == (45, 23, 2)

    changed = _bundle()
    changed["joint"]["weights"][0][0][0] = 1.0
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical SHA-256 mismatch"):
        load_coherent_model(path)


def test_zero_delta_preserves_planner_trace_and_records_adapter():
    plan = _plan()
    frames = [
        TraceFrame(time_us, 40.0 + time_us / 10_000.0, 90.0 + time_us / 40_000.0, False, False)
        for time_us in range(0, 2_800_001, 2000)
    ]
    model = CoherentTrajectoryModel(
        canonical_sha256="c" * 64,
        file_sha256="d" * 64,
        feature_mean=np.zeros(22),
        feature_scale=np.ones(22),
        weights=np.zeros((45, 23, 2)),
        max_nodes=7,
    )
    output, diagnostics = apply_coherent_trajectory_model(frames, plan, model)
    assert np.allclose([[frame.x, frame.y] for frame in output], [[frame.x, frame.y] for frame in frames], atol=1e-10)
    assert [(frame.time_us, frame.k1, frame.k2) for frame in output] == [
        (frame.time_us, frame.k1, frame.k2) for frame in frames
    ]
    assert diagnostics["segment_count"] == 1
    assert diagnostics["learned_segment_count"] == 0
    assert diagnostics["fallback_segment_count"] == 1
    assert diagnostics["windows"][0]["reason"] == "zero_learned_delta"
    assert diagnostics["changed_sample_count"] == 0
    assert diagnostics["model_sha256"] == "c" * 64


def test_zero_blend_is_explicit_fallback():
    model = CoherentTrajectoryModel("c" * 64, "d" * 64, np.zeros(22), np.ones(22), np.zeros((45, 23, 2)), 7)
    frames = [TraceFrame(0, 0.0, 0.0, False, False)]
    output, diagnostics = apply_coherent_trajectory_model(frames, _plan(), model, blend=0.0)
    assert output == frames
    assert diagnostics["effective_mode"] == "math-only"
    assert diagnostics["fallback"] is True


@pytest.mark.parametrize("frames", [
    [TraceFrame(0, float("nan"), 0, False, False), TraceFrame(2000, 1, 1, False, False)],
    [TraceFrame(2000, 1, 1, False, False), TraceFrame(0, 2, 2, False, False)],
    [TraceFrame(0, 1, 1, False, False), TraceFrame(0, 2, 2, False, False)],
])
def test_invalid_input_fails_before_trajectory_composition(frames):
    model = CoherentTrajectoryModel("c" * 64, "d" * 64, np.zeros(22), np.ones(22), np.zeros((45, 23, 2)), 7)
    with pytest.raises(ValueError, match="strictly increasing timestamps"):
        apply_coherent_trajectory_model(frames, _plan(), model)


def test_default_auto_run_resolves_pinned_gated_model():
    root = Path(__file__).resolve().parents[3]
    mode, blend, model_path = _resolve_auto_execution(root, None, None, None)
    assert mode == "coherent" and blend == 1.0
    assert model_path is not None
    assert load_coherent_model(model_path).canonical_sha256 == (
        "302e4fe996664366f498ac74301cd294ee39b4736dfb812e4851281183c4fce1"
    )
    with pytest.raises(ValueError, match="positive blend"):
        _resolve_auto_execution(root, "coherent", model_path, 0.0)


def test_planner_refuses_all_math_coherent_output(tmp_path, monkeypatch):
    model_path = tmp_path / "zero-model.json"
    model_path.write_text(json.dumps(_bundle()), encoding="utf-8")
    map_path = tmp_path / "map-plan.ndjson"
    map_path.write_text(
        json.dumps({
            "schema_version": 1,
            "beatmap_sha256": "a" * 64,
            "beatmap_md5": "b" * 32,
            "clock_rate": 1,
            "mods": [],
            "objects": [
                {"index": index, "kind": "circle", "effective_start_time_ms": time,
                 "effective_end_time_ms": time, "position": [x, y], "radius": 32}
                for index, (time, x, y) in enumerate(
                    ((2000, 100, 100), (2200, 180, 130), (2400, 260, 110), (2600, 340, 150))
                )
            ],
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "trace.ndjson.gz"
    monkeypatch.setattr("sys.argv", ["human-sim", "plan", str(map_path), str(output),
                                  "--percentile", "99.5", "--execution-mode", "coherent",
                                  "--execution-model", str(model_path), "--execution-blend", "1"])
    with pytest.raises(ValueError, match="no delivered learned movement"):
        main()
    assert not output.exists()


def test_long_gap_splits_circle_windows_and_preserves_break_motion():
    windows = HitWindows(40, 80, 120, 150)
    times = (2000.0, 2200.0, 2400.0, 4000.0, 4200.0, 4400.0)
    objects = tuple(
        MapObject(index, "circle", time, time, Point(80.0 + index * 50.0, 100.0),
                  Point(80.0 + index * 50.0, 100.0), 32, hit_windows=windows)
        for index, time in enumerate(times)
    )
    plan = MapPlan(1, "a" * 64, "b" * 32, 1.0, (), objects, {})
    assert _circle_windows(plan) == []
    model = CoherentTrajectoryModel(
        "c" * 64, "d" * 64, np.zeros(22), np.ones(22), np.zeros((45, 23, 2)), 7
    )
    frames = [
        TraceFrame(time_us, 40.0 + time_us / 10_000.0, 90.0, False, False)
        for time_us in range(0, 4_501_000, 2000)
    ]
    output, diagnostics = apply_coherent_trajectory_model(frames, plan, model)
    assert output == frames
    assert diagnostics["learned_segment_count"] == 0
    assert diagnostics["fallback"] is True
