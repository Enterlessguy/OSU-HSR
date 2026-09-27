from __future__ import annotations

"""Quality gate for HumanSim.Runner telemetry and benchmark ingestion."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeQualityConfig:
    """Default thresholds for a host whose normal dispatch is sub-millisecond."""

    max_dispatch_p95_us: float = 1_000.0
    max_dispatch_p99_us: float = 5_000.0
    max_dispatch_max_us: float = 100_000.0
    max_key_down_p95_us: float = 2_500.0
    max_send_input_p95_us: float = 2_500.0
    max_send_input_max_us: float = 100_000.0
    max_deadline_coalesced_fraction: float = 0.02
    max_heartbeat_gap_ms: float = 2_000.0
    minimum_heartbeat_count: int = 1


_NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
_HUMAN_TELEMETRY_RE = re.compile(
    rf"Dispatch lateness \(us\): all p50={_NUMBER}, p95={_NUMBER}, p99={_NUMBER}, max={_NUMBER}; "
    rf"key-down p95={_NUMBER}, max={_NUMBER}; SendInput p95={_NUMBER}, max={_NUMBER}; "
    rf"cadence-skipped=([0-9,]+), deadline-coalesced=([0-9,]+), delivered=([0-9,]+)\.",
    re.IGNORECASE,
)
_EXECUTING_RE = re.compile(r"Executing a [0-9]+ Hz trace through a [0-9]+ Hz cursor cadence: ([0-9,]+) dispatch frames from ([0-9,]+) trace frames", re.IGNORECASE)
_HEARTBEAT_GAP_RE = re.compile(r"heartbeat(?:\s+max(?:imum)?\s+gap)?[^0-9]*(%s)\s*ms" % _NUMBER, re.IGNORECASE)


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and result not in {float("inf"), float("-inf")} else default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _read_source(source: str | Path | Mapping[str, Any] | None) -> tuple[str, dict[str, Any], str | None]:
    if source is None:
        return "", {}, None
    if isinstance(source, Mapping):
        return "", dict(source), "mapping"
    # The assessment API also accepts an in-memory log string, which is useful
    # for CI and keeps tests independent of the Windows runner.
    inline = str(source)
    inline_markers = (
        "runtime telemetry",
        "dispatch lateness",
        "lost focus",
        "focus loss",
        "window ownership changed",
        "window moved or resized",
        "client size or screen position changed",
        "virtual screen size or layout changed",
        "dpi changed",
        "run aborted",
        "heartbeat",
        "protocol mismatch",
        "pipe disconnected",
        "synthetic research trace completed",
    )
    if "\n" in inline or inline.lstrip().startswith("{") or any(marker in inline.lower() for marker in inline_markers):
        return inline, {}, "inline"
    path = Path(source)
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace"), {}, str(path)
    return "", {}, str(path)


def parse_runtime_telemetry(source: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    """Parse structured runner telemetry, retaining human-readable fallbacks."""
    raw_text, mapping, source_label = _read_source(source)
    telemetry: dict[str, Any] = {}
    if mapping:
        telemetry.update(mapping)
    stripped = raw_text.strip()
    if stripped.startswith("{"):
        try:
            parsed_file = json.loads(stripped)
        except json.JSONDecodeError:
            parsed_file = None
        if isinstance(parsed_file, dict):
            telemetry.update(parsed_file)

    for line in raw_text.splitlines():
        if "runtime telemetry" in line.lower():
            marker = line.lower().find("runtime telemetry")
            candidate = line[marker + len("runtime telemetry") :]
            candidate = candidate.lstrip(" :\t")
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                telemetry.update(parsed)
        match = _HUMAN_TELEMETRY_RE.search(line)
        if match:
            groups = match.groups()
            telemetry.update(
                {
                    "dispatch_p50_us": float(groups[0]),
                    "dispatch_p95_us": float(groups[1]),
                    "dispatch_p99_us": float(groups[2]),
                    "dispatch_max_us": float(groups[3]),
                    "key_down_p95_us": float(groups[4]),
                    "key_down_max_us": float(groups[5]),
                    "send_input_p95_us": float(groups[6]),
                    "send_input_max_us": float(groups[7]),
                    "cadence_skipped_frames": _integer(groups[8]),
                    "deadline_coalesced_frames": _integer(groups[9]),
                    "delivered_frames": _integer(groups[10]),
                }
            )
        executing = _EXECUTING_RE.search(line)
        if executing:
            telemetry.setdefault("delivered_frames_expected", _integer(executing.group(1)))
            telemetry.setdefault("trace_frames", _integer(executing.group(2)))
        heartbeat_gap = _HEARTBEAT_GAP_RE.search(line)
        if heartbeat_gap:
            telemetry.setdefault("heartbeat_max_gap_ms", float(heartbeat_gap.group(1)))

    # New runners name this duration for the selected backend. Keep reading
    # historical SendInput fields from earlier Windows-only telemetry.
    telemetry.setdefault("dispatch_backend_p95_us", _get(telemetry, "send_input_p95_us"))
    telemetry.setdefault("dispatch_backend_max_us", _get(telemetry, "send_input_max_us"))

    return {
        "source": source_label,
        "raw_text": raw_text,
        "telemetry": telemetry,
    }


def _get(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
        camel = name.split("_")
        camel_name = camel[0] + "".join(part.title() for part in camel[1:])
        if camel_name in data:
            return data[camel_name]
    return None


def assess_runtime_quality(
    source: str | Path | Mapping[str, Any] | None,
    *,
    config: RuntimeQualityConfig | None = None,
) -> dict[str, Any]:
    """Classify a run as clean, degraded, invalid, or not runtime-validated.

    No telemetry is never reported as a clean runtime result.  The original
    source and parsed values are included so a rejected run remains auditable.
    """
    config = config or RuntimeQualityConfig()
    parsed = parse_runtime_telemetry(source)
    raw_text = parsed["raw_text"]
    telemetry = parsed["telemetry"]
    raw_lower = raw_text.lower()
    reasons: list[str] = []
    integrity_reasons: list[str] = []
    threshold_reasons: list[str] = []
    input_backend = _get(telemetry, "input_backend")
    if input_backend == "research-client":
        planned = _integer(_get(telemetry, "planned_frames"), 0)
        consumed = _integer(_get(telemetry, "client_consumed_frames"), 0)
        invalid = (
            _get(telemetry, "status") != "completed"
            or planned <= 0 or consumed != planned
            or _integer(_get(telemetry, "delivered_frames"), 0) != planned
            or _integer(_get(telemetry, "heartbeat_count"), 0) < config.minimum_heartbeat_count
            or _number(_get(telemetry, "heartbeat_max_gap_ms"), float("inf")) > config.max_heartbeat_gap_ms
            or any(marker in raw_lower for marker in ("lost focus", "pipe disconnected", "run aborted", "digest mismatch"))
        )
        return {
            "schema_version": 1,
            "status": "invalid" if invalid else "not_validated",
            "classification": "runtime-invalid" if invalid else "client-timeline-complete/not-os-dispatch-calibrated",
            "accepted": False,
            "reasons": ["client frame accounting or integrity check failed"] if invalid else
                       ["client timeline delivery is complete; OS dispatch latency thresholds do not apply to this backend"],
            "threshold_reasons": [],
            "integrity_reasons": ["client delivery mismatch"] if invalid else [],
            "config": asdict(config),
            "raw_diagnostics": {"source": parsed["source"], "raw_text": raw_text, "parsed_telemetry": telemetry},
        }
    backend_timing_unverified = input_backend == "x11-xtest" and not bool(_get(telemetry, "timing_only"))
    if backend_timing_unverified:
        reasons.append("X11 XTest dispatch has no platform-specific timing calibration")

    metric_names = (
        "dispatch_p95_us",
        "dispatch_p99_us",
        "dispatch_max_us",
        "key_down_p95_us",
        "dispatch_backend_p95_us",
        "dispatch_backend_max_us",
        "deadline_coalesced_frames",
        "delivered_frames",
    )
    has_latency_telemetry = any(_get(telemetry, name) is not None for name in metric_names)
    has_structured_telemetry = bool(_get(telemetry, "kind") == "runtime_telemetry" or _get(telemetry, "schema_version") is not None)
    has_completion = bool(_get(telemetry, "status") == "completed" or "synthetic research trace completed" in raw_lower)
    required_telemetry = (
        "dispatch_p95_us",
        "dispatch_p99_us",
        "dispatch_max_us",
        "key_down_p95_us",
        "dispatch_backend_p95_us",
        "dispatch_backend_max_us",
        "deadline_coalesced_frames",
        "delivered_frames",
        "heartbeat_count",
        "heartbeat_max_gap_ms",
    )
    missing_telemetry = [name for name in required_telemetry if _get(telemetry, name) is None]

    if not has_latency_telemetry:
        reasons.append("runtime latency telemetry is missing")
    elif missing_telemetry:
        reasons.append("runtime telemetry fields are missing: " + ", ".join(missing_telemetry))
    focus_patterns = (
        "lost focus",
        "focus loss",
        "not foreground",
        "window changed",
        "window ownership changed",
        "window moved or resized",
        "client size or screen position changed",
        "virtual screen size or layout changed",
        "dpi change",
        "window or dpi",
    )
    if any(pattern in raw_lower for pattern in focus_patterns):
        integrity_reasons.append("focus/window/DPI integrity failure reported by runner")
    heartbeat_patterns = ("heartbeat timed out", "heartbeat clock sample is invalid", "pipe disconnected", "heartbeat gap")
    if any(pattern in raw_lower for pattern in heartbeat_patterns):
        integrity_reasons.append("heartbeat integrity failure reported by runner")
    if "research message token mismatch" in raw_lower or "protocol mismatch" in raw_lower:
        integrity_reasons.append("runner/client protocol integrity failure")
    if _get(telemetry, "status") in {"aborted", "failed", "invalid"} or (raw_text and "run aborted" in raw_lower):
        integrity_reasons.append("runner did not complete the run")
    if has_latency_telemetry and not has_completion and _get(telemetry, "status") != "completed":
        integrity_reasons.append("completion marker is missing")

    if has_latency_telemetry:
        checks = (
            ("dispatch p95", _get(telemetry, "dispatch_p95_us"), config.max_dispatch_p95_us),
            ("dispatch p99", _get(telemetry, "dispatch_p99_us"), config.max_dispatch_p99_us),
            ("dispatch max", _get(telemetry, "dispatch_max_us"), config.max_dispatch_max_us),
            ("key-down p95", _get(telemetry, "key_down_p95_us"), config.max_key_down_p95_us),
            ("input backend p95", _get(telemetry, "dispatch_backend_p95_us"), config.max_send_input_p95_us),
            ("input backend max", _get(telemetry, "dispatch_backend_max_us"), config.max_send_input_max_us),
        )
        for label, value, limit in checks:
            numeric = _number(value)
            if numeric is None:
                continue
            if numeric > limit:
                threshold_reasons.append(f"{label} {numeric:g} us exceeds {limit:g} us")
        delivered = _integer(_get(telemetry, "delivered_frames"), 0)
        coalesced = _integer(_get(telemetry, "deadline_coalesced_frames"), 0)
        if delivered <= 0:
            integrity_reasons.append("runtime telemetry reports no delivered dispatch frames")
        if delivered > 0:
            fraction = coalesced / delivered
            if fraction > config.max_deadline_coalesced_fraction:
                threshold_reasons.append(
                    f"deadline coalescing {fraction:.3%} exceeds {config.max_deadline_coalesced_fraction:.3%}"
                )
        heartbeat_count = _integer(_get(telemetry, "heartbeat_count"), 0)
        heartbeat_gap = _number(_get(telemetry, "heartbeat_max_gap_ms"))
        timing_only = bool(_get(telemetry, "timing_only"))
        if has_structured_telemetry and heartbeat_count < config.minimum_heartbeat_count:
            integrity_reasons.append(f"heartbeat count {heartbeat_count} is below {config.minimum_heartbeat_count}")
        if heartbeat_gap is not None and heartbeat_gap > config.max_heartbeat_gap_ms:
            integrity_reasons.append(f"heartbeat gap {heartbeat_gap:g} ms exceeds {config.max_heartbeat_gap_ms:g} ms")
        if timing_only and _get(telemetry, "dispatch_backend_p95_us") is None:
            # A timing-only harness intentionally does not dispatch input; its
            # absence is valid and is retained in the raw telemetry.
            pass

    reasons.extend(threshold_reasons)
    reasons.extend(integrity_reasons)
    if integrity_reasons:
        status = "invalid"
        classification = "runtime-invalid"
        accepted = False
    elif not has_latency_telemetry or missing_telemetry:
        status = "not_validated"
        classification = "planner-only/not-runtime-validated"
        accepted = False
    elif threshold_reasons:
        status = "degraded"
        classification = "runtime-degraded"
        accepted = False
    elif backend_timing_unverified:
        status = "not_validated"
        classification = "runtime-not-calibrated"
        accepted = False
    else:
        status = "clean"
        classification = "runtime-validated"
        accepted = True

    return {
        "schema_version": 1,
        "status": status,
        "classification": classification,
        "accepted": accepted,
        "reasons": reasons,
        "threshold_reasons": threshold_reasons,
        "integrity_reasons": integrity_reasons,
        "config": asdict(config),
        "raw_diagnostics": {
            "source": parsed["source"],
            "raw_text": raw_text,
            "parsed_telemetry": telemetry,
        },
    }


def quality_gate_passes(assessment: Mapping[str, Any]) -> bool:
    """Small helper for benchmark/CI callers."""
    return bool(assessment.get("accepted", False))


# Compatibility aliases for callers that prefer the roadmap terminology.
assess_run_quality = assess_runtime_quality
