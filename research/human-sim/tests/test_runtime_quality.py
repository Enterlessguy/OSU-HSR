from __future__ import annotations

import json

from human_sim.runtime_quality import RuntimeQualityConfig, assess_runtime_quality


def _clean(**overrides):
    value = {
        "schema_version": 1,
        "kind": "runtime_telemetry",
        "status": "completed",
        "dispatch_p95_us": 120,
        "dispatch_p99_us": 300,
        "dispatch_max_us": 900,
        "key_down_p95_us": 180,
        "send_input_p95_us": 80,
        "send_input_max_us": 400,
        "deadline_coalesced_frames": 0,
        "delivered_frames": 1000,
        "heartbeat_count": 100,
        "heartbeat_max_gap_ms": 60,
    }
    value.update(overrides)
    return value


def test_clean_runtime_telemetry_is_accepted():
    assessment = assess_runtime_quality(_clean())
    assert assessment["status"] == "clean"
    assert assessment["classification"] == "runtime-validated"
    assert assessment["accepted"]


def test_inline_json_runtime_telemetry_is_accepted():
    assessment = assess_runtime_quality(json.dumps(_clean()))
    assert assessment["status"] == "clean"


def test_x11_dispatch_is_not_marked_calibrated_by_windows_thresholds():
    assessment = assess_runtime_quality(_clean(input_backend="x11-xtest"))
    assert assessment["status"] == "not_validated"
    assert assessment["classification"] == "runtime-not-calibrated"
    assert not assessment["accepted"]
    assert any("no platform-specific timing calibration" in reason for reason in assessment["reasons"])


def test_latency_outlier_is_degraded_and_raw_telemetry_is_preserved():
    assessment = assess_runtime_quality(_clean(dispatch_max_us=300_000))
    assert assessment["status"] == "degraded"
    assert not assessment["accepted"]
    assert assessment["raw_diagnostics"]["parsed_telemetry"]["dispatch_max_us"] == 300_000
    assert any("dispatch max" in reason for reason in assessment["threshold_reasons"])


def test_focus_or_heartbeat_failure_is_invalid():
    log = """Dispatch lateness (us): all p50=100, p95=200, p99=300, max=500; key-down p95=200, max=400; SendInput p95=100, max=300; cadence-skipped=0, deadline-coalesced=0, delivered=1000.\nResearch client lost focus; run aborted.\n"""
    assessment = assess_runtime_quality(log)
    assert assessment["status"] == "invalid"
    assert any("focus" in reason for reason in assessment["integrity_reasons"])
    assert "lost focus" in assessment["raw_diagnostics"]["raw_text"]
    assert assess_runtime_quality("Research client lost focus; run aborted.")["status"] == "invalid"
    assert assess_runtime_quality("Research window moved or resized; run aborted.")["status"] == "invalid"


def test_missing_runtime_telemetry_is_not_falsely_clean():
    assessment = assess_runtime_quality("Synthetic planner benchmark completed without a runner log.")
    assert assessment["status"] == "not_validated"
    assert assessment["classification"] == "planner-only/not-runtime-validated"
    assert not assessment["accepted"]


def test_coalescing_threshold_can_be_overridden():
    assessment = assess_runtime_quality(
        _clean(deadline_coalesced_frames=30),
        config=RuntimeQualityConfig(max_deadline_coalesced_fraction=0.05),
    )
    assert assessment["status"] == "clean"
