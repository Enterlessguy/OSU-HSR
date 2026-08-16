from __future__ import annotations

"""Reusable statistical benchmark for offline planner realism checks."""

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .context import build_contexts
from .io import load_map_plan
from .matching import match_press_events, rising_press_events
from .planner import PLANNER_VERSION, HumanTracePlanner
from .schemas import HumanProfile, MapPlan, TraceFrame


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_PROFILE = (50.0, 80.0)
MONOTONICITY_PROFILES = (
    (30.0, 80.0),
    (50.0, 80.0),
    (70.0, 80.0),
    (50.0, 40.0),
    (50.0, 80.0),
    (50.0, 100.0),
)


@dataclass(frozen=True)
class BenchmarkGates:
    """Broad artifact gates; stochastic distributions are not snapshots."""

    max_cross_seed_abs_correlation: float = 0.60
    max_lag_abs_correlation: float = 0.65
    max_rim_share: float = 0.20
    min_angular_entropy: float = 0.60
    entropy_min_samples: int = 32
    min_successful_landings: int = 1


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "timing-tests" / "benchmark").is_dir():
            return parent
    return Path.cwd()


def default_map_paths(scope: str = "compact", repository_root: str | Path | None = None) -> list[Path]:
    """Return the checked-in map set for a compact/default/full run."""
    if scope not in {"compact", "default", "full"}:
        raise ValueError("scope must be compact, default, or full")
    root = Path(repository_root) if repository_root else _repository_root()
    benchmark_root = root / "timing-tests" / "benchmark"
    named = {
        "fixture-2x": benchmark_root / "fixture-2x.map.ndjson.gz",
        "kingborn-nm": benchmark_root / "kingborn-nm.map.ndjson.gz",
        "heavenly-hrdt": benchmark_root / "heavenly-hrdt.map.ndjson.gz",
        "centipede-generated": benchmark_root / "centipede-generated.map.ndjson.gz",
        "taitoki": benchmark_root / "taitoki.map.ndjson.gz",
        "vanquished-hrdt": benchmark_root / "vanquished-hrdt.map.ndjson.gz",
    }
    if scope == "compact":
        selected = [named["fixture-2x"], named["kingborn-nm"]]
    elif scope == "default":
        selected = [named[name] for name in ("kingborn-nm", "heavenly-hrdt", "centipede-generated")]
    else:
        selected = sorted(benchmark_root.glob("*.map.ndjson.gz"))
    existing = [path for path in selected if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"No benchmark map plans found under {benchmark_root}")
    return existing


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": int(len(array)),
        "min": round(float(np.min(array)), 6),
        "mean": round(float(np.mean(array)), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _kinematic_summary(frames: Sequence[TraceFrame]) -> dict[str, dict[str, float | int]]:
    times = np.asarray([frame.time_us for frame in frames], dtype=float) / 1_000_000.0
    positions = np.asarray([[frame.x, frame.y] for frame in frames], dtype=float)
    if len(frames) < 2:
        empty = _summary([])
        return {"velocity_px_s": empty, "acceleration_px_s2": empty, "jerk_px_s3": empty}
    dt = np.diff(times)
    valid = dt > 0
    if not np.any(valid):
        empty = _summary([])
        return {"velocity_px_s": empty, "acceleration_px_s2": empty, "jerk_px_s3": empty}
    velocity = np.diff(positions, axis=0)[valid] / dt[valid, None]
    speed = np.linalg.norm(velocity, axis=1)
    if len(velocity) < 2:
        acceleration = np.empty((0, 2))
    else:
        velocity_dt = np.diff(times)[valid]
        acceleration = np.diff(velocity, axis=0) / np.maximum(velocity_dt[1:, None], 1e-9)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    if len(acceleration) < 2:
        jerk = np.empty((0, 2))
    else:
        jerk_dt = np.diff(times)[valid][2:]
        jerk = np.diff(acceleration, axis=0) / np.maximum(jerk_dt[:, None], 1e-9)
    return {
        "velocity_px_s": _summary(speed),
        "acceleration_px_s2": _summary(acceleration_norm),
        "jerk_px_s3": _summary(np.linalg.norm(jerk, axis=1)),
    }


def _normalized_entropy(angles: Sequence[float], bins: int = 16) -> float:
    if not angles:
        return 0.0
    values = (np.asarray(angles, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
    histogram, _ = np.histogram(values, bins=bins, range=(-math.pi, math.pi))
    probabilities = histogram[histogram > 0] / float(len(values))
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(entropy / math.log(bins))


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(right) < 3:
        return None
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if np.std(left_array) <= 1e-9 or np.std(right_array) <= 1e-9:
        return None
    value = float(np.corrcoef(left_array, right_array)[0, 1])
    return value if math.isfinite(value) else None


def _run_metrics(
    plan: MapPlan,
    frames: Sequence[TraceFrame],
    planner: HumanTracePlanner | None = None,
) -> dict[str, Any]:
    timeline_start = plan.objects[0].start_time_ms - 1500.0
    events = rising_press_events(frames, timeline_start_ms=timeline_start)
    assignments = match_press_events(plan.objects, events, minimum_window_ms=180.0)
    contexts = build_contexts(plan)
    radial: list[float] = []
    press_radial: list[float] = []
    angles: list[float] = []
    timing: list[float] = []
    successful = 0
    judged_objects = 0
    matched = 0
    rows: list[dict[str, Any]] = []
    for object_position, obj in enumerate(plan.objects):
        if obj.kind == "spinner":
            continue
        judged_objects += 1
        press_position = assignments.get(object_position)
        if press_position is None:
            rows.append({"object_position": object_position, "matched": False, "success": False})
            continue
        matched += 1
        event = events[press_position]
        frame = frames[event.frame_index]
        timing_error = float(event.time_ms - obj.start_time_ms)
        offset_x = float(frame.x - obj.position.x)
        offset_y = float(frame.y - obj.position.y)
        aim_error = math.hypot(offset_x, offset_y)
        press_normalized_radius = aim_error / max(float(obj.radius), 1e-9)
        planned_offset = planner.landing_offsets.get(object_position) if planner is not None else None
        if planned_offset is None:
            offset = (offset_x, offset_y)
        else:
            offset = planned_offset
        normalized_radius = math.hypot(offset[0], offset[1]) / max(float(obj.radius), 1e-9)
        success = abs(timing_error) <= obj.hit_windows.meh_ms and aim_error <= obj.radius
        radial.append(normalized_radius)
        press_radial.append(press_normalized_radius)
        timing.append(timing_error)
        if math.hypot(*offset) > 1e-9:
            angles.append(math.atan2(offset[1], offset[0]))
        successful += int(success)
        rows.append(
            {
                "object_position": object_position,
                "matched": True,
                "success": success,
                "radial": normalized_radius,
                "press_radial": press_normalized_radius,
                "planned_offset_x": float(offset[0]),
                "planned_offset_y": float(offset[1]),
                "timing_error_ms": timing_error,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "context": contexts[object_position].label,
            }
        )

    radial_array = np.asarray(radial, dtype=float)
    timing_array = np.asarray(timing, dtype=float)
    lag_correlations: dict[str, float] = {}
    for lag in range(2, 9):
        if len(radial_array) > lag:
            value = _correlation(radial_array[:-lag], radial_array[lag:])
            if value is not None:
                lag_correlations[str(lag)] = round(abs(value), 6)
    return {
        "objects": len(plan.objects),
        "judged_objects": judged_objects,
        "presses": len(events),
        "matched_objects": matched,
        "successful_landings": successful,
        "matching_one_to_one": len(set(assignments.values())) == len(assignments),
        "radial": {
            "mean": round(float(np.mean(radial_array)), 6) if len(radial_array) else 0.0,
            "p95": round(float(np.percentile(radial_array, 95)), 6) if len(radial_array) else 0.0,
            "rim_share_gt_0_8": round(float(np.mean(radial_array > 0.8)), 6) if len(radial_array) else 0.0,
            "samples": int(len(radial_array)),
        },
        "press_radial": {
            "mean": round(float(np.mean(press_radial)), 6) if press_radial else 0.0,
            "p95": round(float(np.percentile(press_radial, 95)), 6) if press_radial else 0.0,
            "rim_share_gt_0_8": round(float(np.mean(np.asarray(press_radial) > 0.8)), 6) if press_radial else 0.0,
            "samples": len(press_radial),
        },
        "angular_entropy": round(_normalized_entropy(angles), 6),
        "angular_samples": len(angles),
        "lag_abs_correlation": lag_correlations,
        "timing": {
            "mean_signed_ms": round(float(np.mean(timing_array)), 6) if len(timing_array) else 0.0,
            "std_ms": round(float(np.std(timing_array)), 6) if len(timing_array) else 0.0,
            "p95_abs_ms": round(float(np.percentile(np.abs(timing_array), 95)), 6) if len(timing_array) else 0.0,
            "early_share": round(float(np.mean(timing_array < 0.0)), 6) if len(timing_array) else 0.0,
            "samples": int(len(timing_array)),
        },
        "kinematics": _kinematic_summary(frames),
        "rows": rows,
    }


def _cross_seed_metrics(run_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    correlations: list[float] = []
    deltas: list[float] = []
    common_sample_count = 0
    pair_common_sample_counts: list[int] = []
    for left, right in itertools.combinations(run_records, 2):
        left_rows = {row["object_position"]: row for row in left["metrics"]["rows"] if row["matched"]}
        right_rows = {row["object_position"]: row for row in right["metrics"]["rows"] if row["matched"]}
        common = sorted(set(left_rows) & set(right_rows))
        if len(common) < 3:
            continue
        common_sample_count += len(common)
        pair_common_sample_counts.append(len(common))
        left_radial = [left_rows[index]["radial"] for index in common]
        right_radial = [right_rows[index]["radial"] for index in common]
        correlation = _correlation(left_radial, right_radial)
        if correlation is not None:
            correlations.append(abs(correlation))
        deltas.extend(
            math.hypot(
                left_rows[index].get("planned_offset_x", left_rows[index]["offset_x"])
                - right_rows[index].get("planned_offset_x", right_rows[index]["offset_x"]),
                left_rows[index].get("planned_offset_y", left_rows[index]["offset_y"])
                - right_rows[index].get("planned_offset_y", right_rows[index]["offset_y"]),
            )
            for index in common
        )
    return {
        "absolute_correlation": _summary(correlations),
        "landing_delta_px": _summary(deltas),
        "common_object_samples": common_sample_count,
        "minimum_pair_common_samples": min(pair_common_sample_counts, default=0),
        "checked": bool(pair_common_sample_counts) and min(pair_common_sample_counts) >= 32,
    }


def _aggregate_run_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    radial = [record["metrics"]["radial"]["mean"] for record in records]
    timing = [record["metrics"]["timing"]["p95_abs_ms"] for record in records]
    rim = [record["metrics"]["radial"]["rim_share_gt_0_8"] for record in records]
    entropy = [record["metrics"]["angular_entropy"] for record in records]
    success = [
        record["metrics"]["successful_landings"] / max(1, record["metrics"]["judged_objects"])
        for record in records
    ]
    lag_values = [
        value
        for record in records
        for value in record["metrics"]["lag_abs_correlation"].values()
    ]
    return {
        "runs": len(records),
        "radial_mean": _summary(radial),
        "timing_p95_abs_ms": _summary(timing),
        "rim_share_gt_0_8": _summary(rim),
        "angular_entropy": _summary(entropy),
        "angular_sample_count": int(sum(record["metrics"]["angular_samples"] for record in records)),
        "success_rate": _summary(success),
        "lag_abs_correlation_max": round(float(max(lag_values)), 6) if lag_values else 0.0,
        "successful_landings_total": int(sum(record["metrics"]["successful_landings"] for record in records)),
    }


def _profile_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = record["metrics"]
        rows.append(
            {
                "map": record["map"],
                "skill": record["skill"],
                "effort": record["effort"],
                "success_rate": metrics["successful_landings"] / max(1, metrics["judged_objects"]),
                "radial_mean": metrics["radial"]["mean"],
                "timing_p95_abs_ms": metrics["timing"]["p95_abs_ms"],
            }
        )
    return rows


def _monotonicity(rows: Sequence[dict[str, Any]], axis: str) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    groups: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["map"], row["effort"]) if axis == "skill" else (row["map"], row["skill"])
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        by_axis_value: dict[float, dict[str, Any]] = {}
        for row in sorted(group, key=lambda row: row[axis]):
            by_axis_value.setdefault(float(row[axis]), row)
        ordered = [by_axis_value[value] for value in sorted(by_axis_value)]
        for left, right in zip(ordered, ordered[1:]):
            success_delta = right["success_rate"] - left["success_rate"]
            radial_ratio = right["radial_mean"] / max(left["radial_mean"], 1e-9)
            timing_ratio = right["timing_p95_abs_ms"] / max(left["timing_p95_abs_ms"], 1e-9)
            expected_improvement = success_delta >= -0.12 and radial_ratio <= 1.35 and timing_ratio <= 1.35
            comparisons.append(
                {
                    "from": left[axis],
                    "to": right[axis],
                    "success_delta": round(success_delta, 6),
                    "radial_ratio": round(radial_ratio, 6),
                    "timing_ratio": round(timing_ratio, 6),
                    "pass": expected_improvement,
                }
            )
    failures = sum(not comparison["pass"] for comparison in comparisons)
    return {"axis": axis, "comparisons": comparisons, "failures": failures, "pass": failures <= max(1, len(comparisons) // 3)}


def _gate_report(
    *,
    same_seed: bool,
    map_reports: Sequence[dict[str, Any]],
    profile_report: dict[str, Any],
    gates: BenchmarkGates,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, value: Any, limit: Any, note: str) -> None:
        checks.append({"name": name, "pass": bool(passed), "value": value, "limit": limit, "note": note})

    add("same_seed_exact", same_seed, same_seed, True, "same map/profile/seed must reproduce exact trace frames")
    matching_ok = all(
        run["metrics"]["matching_one_to_one"]
        for report in map_reports
        for run in report["runs"]
    )
    add("one_to_one_matching", matching_ok, matching_ok, True, "a press may match at most one object")
    cross_values = [
        report["cross_seed"]["absolute_correlation"]["max"]
        for report in map_reports
        if report["cross_seed"]["checked"]
    ]
    cross_max = max(cross_values, default=0.0)
    add("cross_seed_correlation", cross_max <= gates.max_cross_seed_abs_correlation, round(cross_max, 6), gates.max_cross_seed_abs_correlation, "broad ceiling; checked only when every seed pair has at least 32 common object samples")
    lag_max = max((report["aggregate"]["lag_abs_correlation_max"] for report in map_reports), default=0.0)
    add("lag_periodicity", lag_max <= gates.max_lag_abs_correlation, round(lag_max, 6), gates.max_lag_abs_correlation, "lag-2..8 absolute correlation ceiling")
    rim_max = max((report["aggregate"]["rim_share_gt_0_8"]["max"] for report in map_reports), default=0.0)
    add("rim_share", rim_max <= gates.max_rim_share, round(rim_max, 6), gates.max_rim_share, "normalized landing radius > 0.8 must remain a minority")
    entropy_values = [
        report["aggregate"]["angular_entropy"]["min"]
        for report in map_reports
        if report["aggregate"]["angular_sample_count"] >= gates.entropy_min_samples
    ]
    entropy_min = min(entropy_values, default=None)
    add(
        "angular_entropy",
        entropy_min is None or entropy_min >= gates.min_angular_entropy,
        None if entropy_min is None else round(entropy_min, 6),
        gates.min_angular_entropy,
        f"checked only when there are at least {gates.entropy_min_samples} angular samples",
    )
    total_success = sum(report["aggregate"]["successful_landings_total"] for report in map_reports)
    add("successful_landings", total_success >= gates.min_successful_landings, total_success, gates.min_successful_landings, "benchmark must produce at least one judged landing")
    add("skill_monotonicity", profile_report["skill"]["pass"], profile_report["skill"]["failures"], "<= one-third failures", "higher skill should not broadly worsen error or success")
    add("effort_monotonicity", profile_report["effort"]["pass"], profile_report["effort"]["failures"], "<= one-third failures", "higher effort should not broadly worsen error or success")
    return {"pass": all(check["pass"] for check in checks), "checks": checks}


def run_benchmark(
    *,
    scope: str = "compact",
    map_paths: Iterable[str | Path] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    skill: float = DEFAULT_PROFILE[0],
    effort: float = DEFAULT_PROFILE[1],
    sample_rate_hz: int = 500,
    monotonicity_profiles: Sequence[tuple[float, float]] = MONOTONICITY_PROFILES,
    gates: BenchmarkGates | None = None,
    classification: str = "planner-only/not-runtime-validated",
    runtime_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a deterministic multi-map/multi-seed benchmark in memory."""
    selected_paths = [Path(path) for path in map_paths] if map_paths is not None else default_map_paths(scope)
    selected_paths = [path.resolve() for path in selected_paths]
    if not selected_paths or any(not path.is_file() for path in selected_paths):
        missing = [str(path) for path in selected_paths if not path.is_file()]
        raise FileNotFoundError(f"Benchmark map plan missing: {', '.join(missing)}")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one benchmark seed is required")
    gates = gates or BenchmarkGates()

    map_reports: list[dict[str, Any]] = []
    same_seed_results: list[dict[str, Any]] = []
    for map_path in selected_paths:
        plan = load_map_plan(map_path)
        map_name = map_path.stem.removesuffix(".map.ndjson")
        records: list[dict[str, Any]] = []
        for seed in seed_values:
            profile = HumanProfile(
                percentile=99.5,
                seed=seed,
                sample_rate_hz=sample_rate_hz,
                skill_level=float(skill),
                effort_level=float(effort),
            )
            planner = HumanTracePlanner(plan, profile)
            frames = planner.generate()
            metrics = _run_metrics(plan, frames, planner)
            record = {"map": map_name, "path": str(map_path), "seed": seed, "skill": float(skill), "effort": float(effort), "metrics": metrics}
            records.append(record)

        repeat_profile = HumanProfile(99.5, seed_values[0], sample_rate_hz, skill_level=float(skill), effort_level=float(effort))
        first_frames = HumanTracePlanner(plan, repeat_profile).generate()
        second_frames = HumanTracePlanner(plan, repeat_profile).generate()
        first_signature = [(frame.time_us, frame.x, frame.y, frame.k1, frame.k2) for frame in first_frames]
        second_signature = [(frame.time_us, frame.x, frame.y, frame.k1, frame.k2) for frame in second_frames]
        same_seed_results.append({"map": map_name, "seed": seed_values[0], "exact": first_signature == second_signature, "frames": len(first_frames)})
        map_reports.append(
            {
                "map": map_name,
                "path": str(map_path),
                "aggregate": _aggregate_run_metrics(records),
                "cross_seed": _cross_seed_metrics(records),
                "runs": records,
            }
        )

    monotonicity_records: list[dict[str, Any]] = []
    monotonicity_seed = seed_values[0]
    for map_path in selected_paths[:1]:
        plan = load_map_plan(map_path)
        map_name = map_path.stem.removesuffix(".map.ndjson")
        for profile_skill, profile_effort in monotonicity_profiles:
            profile = HumanProfile(99.5, monotonicity_seed, sample_rate_hz, skill_level=float(profile_skill), effort_level=float(profile_effort))
            planner = HumanTracePlanner(plan, profile)
            frames = planner.generate()
            monotonicity_records.append(
                {
                    "map": map_name,
                    "skill": float(profile_skill),
                    "effort": float(profile_effort),
                    "metrics": _run_metrics(plan, frames, planner),
                }
            )
    profile_rows = _profile_rows(monotonicity_records)
    profile_report = {"skill": _monotonicity(profile_rows[:], "skill"), "effort": _monotonicity(profile_rows[:], "effort"), "rows": profile_rows}
    same_seed = all(result["exact"] for result in same_seed_results)
    gate_report = _gate_report(same_seed=same_seed, map_reports=map_reports, profile_report=profile_report, gates=gates)
    return {
        "schema_version": 1,
        "benchmark": "human-sim-statistical-regression",
        "planner_version": PLANNER_VERSION,
        "scope": scope,
        "classification": classification,
        "runtime_quality": runtime_quality,
        "configuration": {
            "maps": [str(path) for path in selected_paths],
            "seeds": list(seed_values),
            "skill": float(skill),
            "effort": float(effort),
            "sample_rate_hz": sample_rate_hz,
            "gates": gates.__dict__,
        },
        "same_seed": {"all_exact": same_seed, "runs": same_seed_results},
        "maps": map_reports,
        "monotonicity": profile_report,
        "gates": gate_report,
    }


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Human-sim benchmark: {report['scope']} / {report['classification']}",
        f"planner={report['planner_version']} maps={len(report['maps'])} seeds={report['configuration']['seeds']}",
        f"same-seed-exact={'PASS' if report['same_seed']['all_exact'] else 'FAIL'}",
    ]
    for map_report in report["maps"]:
        aggregate = map_report["aggregate"]
        cross = map_report["cross_seed"]["absolute_correlation"]
        cross_value = f"{cross['max']:.3f}" if map_report["cross_seed"]["checked"] else "n/a"
        lines.append(
            f"{map_report['map']}: success={aggregate['success_rate']['mean']:.3f} "
            f"radial_mean={aggregate['radial_mean']['mean']:.3f} "
            f"rim={aggregate['rim_share_gt_0_8']['mean']:.3f} "
            f"entropy={aggregate['angular_entropy']['mean']:.3f} "
            f"timing_p95={aggregate['timing_p95_abs_ms']['mean']:.2f}ms "
            f"cross_seed_abs_corr_max={cross_value}"
        )
    lines.append(f"skill-monotonicity={'PASS' if report['monotonicity']['skill']['pass'] else 'FAIL'} effort-monotonicity={'PASS' if report['monotonicity']['effort']['pass'] else 'FAIL'}")
    lines.append(f"gates={'PASS' if report['gates']['pass'] else 'FAIL'}")
    for check in report["gates"]["checks"]:
        if not check["pass"]:
            lines.append(f"  FAIL {check['name']}: {check['value']} ({check['note']})")
    if report.get("runtime_quality"):
        quality = report["runtime_quality"]
        lines.append(f"runtime-quality={quality.get('status')} ({quality.get('classification')})")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: str | Path, summary_output: str | Path | None = None) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary = format_summary(report) + "\n"
    if summary_output:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary, encoding="utf-8")
