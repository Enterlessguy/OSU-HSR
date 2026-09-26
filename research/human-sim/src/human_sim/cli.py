from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from .benchmark import format_summary, run_benchmark, write_report
from .boundary_aware_execution import boundary_aware_warp_math_trace
from .collector import collect_replays
from .coherent_execution import (
    COHERENT_EXECUTION_VERSION,
    apply_coherent_trajectory_model,
    load_coherent_model,
)
from .dataset import extract_features
from .execution import warp_math_trace
from .execution_acquisition import (
    audit_candidate_pilot,
    audit_decoded_candidates,
    summarize_index,
    write_json,
)
from .execution_benchmark import run_paired_benchmark_for_map, write_paired_report
from .execution_exploratory import build_admission_table, train_exploratory_candidate
from .execution_training import (
    audit_corpus,
    evaluate_candidate,
    inspect_candidate,
    retrain_candidate,
    train_candidate,
    load_phase_model,
    load_execution_corpus,
)
from .io import load_map_plan, repository_git_commit, write_trace
from .library_audit import audit_library
from .modeling import evaluate, fit_models
from .planner import PLANNER_VERSION, HumanTracePlanner
from .runtime_quality import RuntimeQualityConfig, assess_runtime_quality
from .schemas import HumanProfile, RunManifest, SKILL_PRESETS
from .validation import validate_trace


def _plan(args: argparse.Namespace) -> int:
    map_plan = load_map_plan(args.map_plan)
    execution_blend = float(args.execution_blend)
    if not math.isfinite(execution_blend) or not 0.0 <= execution_blend <= 1.0:
        raise ValueError("execution blend must be between 0 and 1")
    motion_mode = args.motion_mode or ("perfect" if args.perfect_baseline else "profile")
    perfect_baseline = bool(args.perfect_baseline or motion_mode == "perfect")
    profile = HumanProfile(
        float(args.percentile),
        int(args.seed),
        int(args.sample_rate),
        perfect_baseline=perfect_baseline,
        skill_level=float(args.skill if args.skill is not None else args.percentile),
        effort_level=float(args.effort),
    )
    planner = HumanTracePlanner(map_plan, profile, args.model_bundle)
    frames = planner.generate()
    execution_result = None
    execution_model = None
    execution_model_load_error = None
    requested_execution_mode = args.execution_mode or (
        "hybrid" if args.execution_model or execution_blend > 0.0 else "math-only"
    )
    execution_adapter = str(getattr(args, "execution_adapter", "legacy"))
    boundary_segments = _parse_segment_indices(getattr(args, "execution_boundary_segments", None))
    if requested_execution_mode not in {"math-only", "hybrid", "coherent"}:
        raise ValueError("execution mode must be math-only, hybrid, or coherent")
    if requested_execution_mode == "math-only" and (args.execution_model or execution_blend > 0.0):
        raise ValueError("math-only execution cannot receive an execution model or non-zero blend")
    if requested_execution_mode == "coherent" and execution_blend <= 0.0:
        raise ValueError("coherent execution requires a positive blend; use math-only for the control arm")
    if requested_execution_mode == "math-only" and (execution_adapter != "legacy" or boundary_segments):
        raise ValueError("math-only execution cannot select an execution adapter or boundary segments")
    if execution_adapter == "boundary-aware-v2" and not boundary_segments:
        raise ValueError("boundary-aware-v2 requires --execution-boundary-segments")
    if execution_adapter != "boundary-aware-v2" and boundary_segments:
        raise ValueError("--execution-boundary-segments requires --execution-adapter boundary-aware-v2")
    if requested_execution_mode == "coherent":
        if not args.execution_model:
            raise ValueError("coherent execution requires --execution-model")
        try:
            coherent_model = load_coherent_model(args.execution_model)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"coherent model rejected: {type(error).__name__}: {error}") from error
        execution_model = coherent_model
        frames, execution_result = apply_coherent_trajectory_model(
            frames,
            map_plan,
            coherent_model,
            blend=execution_blend,
        )
        execution_result["motion_mode"] = motion_mode
        if execution_result.get("learned_segment_count", 0) <= 0 or execution_result.get("changed_sample_count", 0) <= 0:
            raise ValueError("coherent model produced no delivered learned movement")
    elif requested_execution_mode == "hybrid":
        if args.execution_model:
            try:
                execution_model = load_phase_model(args.execution_model)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                execution_model_load_error = f"model_rejected:{type(error).__name__}"
        if execution_model is not None and execution_blend > 0.0:
            timeline_start = map_plan.objects[0].start_time_ms - 1500.0
            segment_boundaries = [obj.start_time_ms - timeline_start for obj in map_plan.objects]
            segment_modes = ["break_free_roam"]
            for left, right in zip(map_plan.objects, map_plan.objects[1:]):
                if left.kind == "circle" and right.kind == "circle":
                    segment_modes.append("circle")
                elif left.kind == "slider" or right.kind == "slider":
                    segment_modes.append("slider")
                elif left.kind == "spinner" or right.kind == "spinner":
                    segment_modes.append("spinner")
                else:
                    segment_modes.append("break_free_roam")
            segment_modes.append("break_free_roam")
            if execution_adapter == "boundary-aware-v2":
                _, frames, execution_result = boundary_aware_warp_math_trace(
                    frames,
                    model=execution_model,
                    blend=execution_blend,
                    segment_boundaries_ms=segment_boundaries,
                    segment_modes=segment_modes,
                    selected_segment_indices=boundary_segments,
                )
                execution_result["blend"] = execution_blend
                execution_result["segment_count"] = len(boundary_segments)
            else:
                frames, execution_trace = warp_math_trace(
                    frames,
                    model=execution_model,
                    blend=execution_blend,
                    sample_rate_hz=profile.sample_rate_hz,
                    route_id=f"{map_plan.beatmap_sha256}:planner-trace",
                    mode="mixed",
                    segment_boundaries_ms=segment_boundaries,
                    segment_modes=segment_modes,
                    time_aware_phase=execution_adapter == "time-aware-v3",
                )
                execution_result = execution_trace.diagnostics.as_dict()
                execution_result["route_sha256"] = execution_trace.route_sha256
                execution_result["adapter"] = execution_adapter
            execution_result["requested_mode"] = requested_execution_mode
            execution_result["effective_mode"] = "hybrid" if execution_result.get("learned_segment_count") else "math-only"
            execution_result["motion_mode"] = motion_mode
            execution_result["segment_modes"] = segment_modes
        else:
            execution_result = {
                "requested_mode": requested_execution_mode,
                "effective_mode": "math-only",
                "motion_mode": motion_mode,
                "blend": execution_blend,
                "model_sha256": getattr(execution_model, "model_sha256", None),
                "fallback": True,
                "fallback_reason": execution_model_load_error or "missing_model_or_zero_blend",
                "segment_count": 0,
                "learned_segment_count": 0,
                "fallback_segment_count": 0,
                "adapter": execution_adapter,
            }
        if execution_model_load_error:
            execution_result["fallback"] = True
            execution_result["fallback_reason"] = execution_model_load_error
    else:
        execution_result = {
            "requested_mode": "math-only",
            "effective_mode": "math-only",
            "motion_mode": motion_mode,
            "blend": 0.0,
            "model_sha256": None,
            "fallback": False,
            "fallback_reason": None,
            "segment_count": 0,
            "learned_segment_count": 0,
            "fallback_segment_count": 0,
            "adapter": "legacy",
        }
    model_hash = hashlib.sha256(Path(args.model_bundle).read_bytes()).hexdigest() if args.model_bundle else None
    if execution_model is not None:
        model_hash = getattr(execution_model, "model_sha256", None) or getattr(execution_model, "canonical_sha256", None)
    model_version = model_hash or ("built_in_perfect_baseline_v1" if profile.perfect_baseline else "built_in_baseline_v1")
    git_commit = repository_git_commit()
    build_identity = f"human-sim-python:{PLANNER_VERSION}:{git_commit}"
    digest = write_trace(
        args.output,
        map_plan=map_plan,
        profile_percentile=profile.percentile,
        seed=profile.seed,
        sample_rate_hz=profile.sample_rate_hz,
        frames=frames,
        model_sha256=model_version,
        diagnostic_perfect=profile.perfect_baseline,
        planner_version=PLANNER_VERSION,
        git_commit=git_commit,
        build_identity=build_identity,
        motion_mode=motion_mode,
        execution_mode=requested_execution_mode,
        execution_blend=execution_blend,
        execution_adapter=COHERENT_EXECUTION_VERSION if requested_execution_mode == "coherent" else execution_adapter,
        execution_diagnostics=execution_result,
    )
    result = validate_trace(args.output)
    result["trace_sha256"] = digest
    result["strength"] = planner.strength_stats
    result["aim"] = planner.aim_stats
    result["timing"] = planner.timing_stats
    configuration = json.dumps(
        {
            "percentile": profile.percentile,
            "seed": profile.seed,
            "sample_rate_hz": profile.sample_rate_hz,
            "perfect_baseline": profile.perfect_baseline,
            "planner_version": PLANNER_VERSION,
            "skill_level": profile.skill_level,
            "effort_level": profile.effort_level,
            "execution_model": model_hash,
            "execution_blend": execution_blend,
            "execution_mode": requested_execution_mode,
            "execution_adapter": COHERENT_EXECUTION_VERSION if requested_execution_mode == "coherent" else execution_adapter,
            "execution_boundary_segments": boundary_segments,
            "motion_mode": motion_mode,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    manifest = RunManifest(
        schema_version=1,
        planner_version=PLANNER_VERSION,
        git_commit=git_commit,
        build_identity=build_identity,
        synthetic=True,
        beatmap_sha256=map_plan.beatmap_sha256,
        map_plan_sha256=hashlib.sha256(Path(args.map_plan).read_bytes()).hexdigest(),
        model_sha256=model_version,
        corpus_manifest_sha256=_optional_hash(args.corpus_manifest),
        trace_sha256=digest,
        client_build_sha256=_optional_hash(args.client_build),
        configuration_sha256=hashlib.sha256(configuration).hexdigest(),
        captured_replay_sha256=None,
        profile_percentile=profile.percentile,
        seed=profile.seed,
        motion_mode=motion_mode,
        execution_mode=requested_execution_mode,
        execution_blend=execution_blend,
    )
    manifest_path = Path(args.manifest or f"{args.output}.manifest.json")
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    result["manifest"] = str(manifest_path)
    if execution_result is not None:
        result["execution"] = execution_result
        print(
            "[execution] "
            f"requested={execution_result.get('requested_mode', requested_execution_mode)} "
            f"effective={execution_result.get('effective_mode', 'unknown')} "
            f"blend={execution_result.get('blend', execution_blend):.3f} "
            f"model_sha256={execution_result.get('model_sha256') or 'none'} "
            f"segments={execution_result.get('segment_count', 0)} "
            f"learned={execution_result.get('learned_segment_count', 0)} "
            f"fallback={execution_result.get('fallback_segment_count', 0)}"
            + (
                f" reason={execution_result.get('fallback_reason')}"
                if execution_result.get("fallback_reason")
                else ""
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _optional_hash(path: str | None) -> str | None:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def _validate(args: argparse.Namespace) -> int:
    print(json.dumps(validate_trace(args.trace), indent=2, sort_keys=True))
    return 0


def _collect(args: argparse.Namespace) -> int:
    print(json.dumps(collect_replays(args.score_ids, args.output, args.delay), indent=2, sort_keys=True))
    return 0


def _extract(args: argparse.Namespace) -> int:
    print(json.dumps(extract_features(args.replay, args.map_plan, args.output), indent=2, sort_keys=True))
    return 0


def _fit(args: argparse.Namespace) -> int:
    print(json.dumps(fit_models(args.features, args.output, args.seed), indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    print(json.dumps(evaluate(args.human, args.synthetic, args.output, args.seed), indent=2, sort_keys=True))
    return 0


def _parse_segment_indices(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    try:
        result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise ValueError("execution boundary segments must be comma-separated integers") from error
    if any(item < 0 for item in result):
        raise ValueError("execution boundary segments must be non-negative")
    return result


def _execution_audit(args: argparse.Namespace) -> int:
    report = audit_corpus(args.manifest)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _execution_source_audit(args: argparse.Namespace) -> int:
    """Summarize a public archive without admitting it to human training."""
    report = summarize_index(args.index)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _execution_candidate_audit(args: argparse.Namespace) -> int:
    candidate = json.loads(Path(args.candidate_report).read_text(encoding="utf-8"))
    evidence = (
        json.loads(Path(args.score_evidence).read_text(encoding="utf-8"))
        if args.score_evidence
        else None
    )
    result = {
        "schema_version": 1,
        "provenance_review": audit_candidate_pilot(candidate, evidence),
        "decoded_audit": audit_decoded_candidates(candidate, args.decoded_root, args.map_plan_root),
    }
    result["admission_table"] = build_admission_table(candidate, evidence or {}, result["decoded_audit"])
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        result["player_hash_salt_id"] = manifest.get("player_hash_salt_id")
        result["player_hash_salt_configured"] = bool(manifest.get("player_hash_salt_id"))
        result["salt_value_exposed_in_report"] = False
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _execution_exploratory_train(args: argparse.Namespace) -> int:
    candidate = json.loads(Path(args.candidate_report).read_text(encoding="utf-8"))
    evidence = json.loads(Path(args.score_evidence).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.candidate_audit).read_text(encoding="utf-8"))
    report = train_exploratory_candidate(
        candidate,
        evidence,
        audit["decoded_audit"],
        args.decoded_root,
        args.map_plan_root,
        args.output,
        seed=args.seed,
        max_windows_per_player=args.max_windows_per_player,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "exploratory_not_promoted" else 1


def _execution_train(args: argparse.Namespace) -> int:
    report = train_candidate(args.manifest, args.output, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") != "blocked" else 2


def _execution_extract(args: argparse.Namespace) -> int:
    rows, report = load_execution_corpus(args.manifest)
    encoded_rows = []
    for row in rows:
        encoded = dict(row)
        for key in ("x", "y", "phase_increments"):
            if key in encoded:
                encoded[key] = list(encoded[key])
        encoded_rows.append(encoded)
    result = {"schema_version": 1, "report": report, "windows": encoded_rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _execution_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_candidate(args.manifest, args.candidate, split=args.split)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _execution_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_candidate(args.candidate), indent=2, sort_keys=True))
    return 0


def _execution_retrain(args: argparse.Namespace) -> int:
    report = retrain_candidate(args.manifest, args.previous, args.output, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") != "blocked" else 2


def _execution_benchmark(args: argparse.Namespace) -> int:
    report = run_paired_benchmark_for_map(
        args.map_plan,
        model_path=args.model,
        blend=args.blend,
        sample_rate_hz=args.sample_rate,
        skill_group=args.skill_group,
        effort_level=args.effort,
        seed=args.seed,
        export_dir=args.export_dir,
    )
    write_paired_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "complete" and report["route_identity"]["all_pairs_shared_frozen_route"] else 1


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def _runtime_quality_config(args: argparse.Namespace) -> RuntimeQualityConfig:
    defaults = RuntimeQualityConfig()
    return RuntimeQualityConfig(
        max_dispatch_p95_us=float(args.max_dispatch_p95_us if args.max_dispatch_p95_us is not None else defaults.max_dispatch_p95_us),
        max_dispatch_p99_us=float(args.max_dispatch_p99_us if args.max_dispatch_p99_us is not None else defaults.max_dispatch_p99_us),
        max_dispatch_max_us=float(args.max_dispatch_max_us if args.max_dispatch_max_us is not None else defaults.max_dispatch_max_us),
        max_key_down_p95_us=float(args.max_key_down_p95_us if args.max_key_down_p95_us is not None else defaults.max_key_down_p95_us),
        max_send_input_p95_us=float(args.max_send_input_p95_us if args.max_send_input_p95_us is not None else defaults.max_send_input_p95_us),
        max_send_input_max_us=float(args.max_send_input_max_us if args.max_send_input_max_us is not None else defaults.max_send_input_max_us),
        max_deadline_coalesced_fraction=float(args.max_coalesced_fraction if args.max_coalesced_fraction is not None else defaults.max_deadline_coalesced_fraction),
        max_heartbeat_gap_ms=float(args.max_heartbeat_gap_ms if args.max_heartbeat_gap_ms is not None else defaults.max_heartbeat_gap_ms),
        minimum_heartbeat_count=int(args.minimum_heartbeat_count if args.minimum_heartbeat_count is not None else defaults.minimum_heartbeat_count),
    )


def _benchmark(args: argparse.Namespace) -> int:
    quality = None
    classification = "planner-only/not-runtime-validated"
    if args.runtime_telemetry:
        quality = assess_runtime_quality(args.runtime_telemetry, config=_runtime_quality_config(args))
        classification = quality["classification"]
    report = run_benchmark(
        scope=args.scope,
        map_paths=args.map_paths,
        seeds=_parse_seeds(args.seeds),
        skill=args.skill,
        effort=args.effort,
        sample_rate_hz=args.sample_rate,
        monotonicity_seeds=_parse_seeds(args.monotonicity_seeds) if args.monotonicity_seeds else None,
        classification=classification,
        runtime_quality=quality,
    )
    write_report(report, args.output, args.summary_output)
    print(format_summary(report))
    if not report["gates"]["pass"]:
        return 1
    if quality is not None and not quality["accepted"]:
        return 1
    return 0


def _runtime_quality(args: argparse.Namespace) -> int:
    assessment = assess_runtime_quality(args.source, config=_runtime_quality_config(args))
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if assessment["accepted"] else 1


def _audit_library(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[4]
    exporter = _dotnet_apphost(root, "HumanSim.MapExporter")
    report = audit_library(
        args.osu_storage,
        exporter,
        args.output,
        limit=args.limit,
        sample_rate_hz=args.sample_rate,
        seed=args.seed,
        workers=args.workers,
        environment=_runner_environment(),
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2, sort_keys=True))
    return 0 if report["failed"] == 0 else 1


def _runner_environment() -> dict[str, str]:
    # Some Windows hosts expose both `Path` and `PATH`. Deduplicate keys for
    # .NET's case-insensitive environment dictionary, but preserve the original
    # casing of every unrelated variable. Uppercasing the entire block can
    # break native/managed startup components which treat selected names as
    # case-sensitive even on Windows.
    environment: dict[str, str] = {}
    keys_by_casefold: dict[str, str] = {}
    for key, value in os.environ.items():
        folded = key.casefold()
        existing = keys_by_casefold.get(folded)
        if existing is not None:
            del environment[existing]
        selected = "Path" if folded == "path" else key
        environment[selected] = value
        keys_by_casefold[folded] = selected
    return environment


def _dotnet_apphost(root: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    debug = root / "research" / name / "bin" / "Debug" / "net8.0" / f"{name}{suffix}"
    release = root / "research" / name / "bin" / "Release" / "net8.0" / f"{name}{suffix}"
    return debug if debug.is_file() else release


def _default_osu_storage() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = _xdg_home("XDG_DATA_HOME", Path.home() / ".local" / "share")
    return base / "osu-development" / "files"


def _default_auto_output() -> Path:
    if os.name == "nt":
        return Path(__file__).resolve().parents[4] / "research" / "human-sim" / "output" / "auto"
    cache_home = _xdg_home("XDG_CACHE_HOME", Path.home() / ".cache")
    return cache_home / "intelligence-database-hsr" / "auto"


def _default_state_log() -> Path:
    state_home = _xdg_home("XDG_STATE_HOME", Path.home() / ".local" / "state")
    return state_home / "intelligence-database-hsr" / "logs"


def _xdg_home(variable: str, default: Path) -> Path:
    configured = os.environ.get(variable)
    if configured:
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path
    return default


def _run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[4]
    runner = _dotnet_apphost(root, "HumanSim.Runner")
    if not runner.is_file():
        raise RuntimeError("Build HumanSim.Runner and install a supported .NET 8 SDK before running a trace")
    completed = subprocess.run(
        [
            str(runner),
            "--client",
            str(Path(args.client).resolve()),
            "--trace",
            str(Path(args.trace).resolve()),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--left-key",
            args.left_key,
            "--right-key",
            args.right_key,
            "--input-lead-ms",
            str(args.input_lead_ms),
        ],
        check=False,
        env=_runner_environment(),
    )
    return int(completed.returncode)


def _run_auto(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[4]
    runner = _dotnet_apphost(root, "HumanSim.Runner")
    if not runner.exists():
        raise RuntimeError("Build HumanSim.Runner before launching automatic planning")
    execution_mode, execution_blend, execution_model = _resolve_auto_execution(
        root,
        args.execution_mode,
        args.execution_model,
        args.execution_blend,
    )
    command = [
        str(runner),
        "--client",
        str(Path(args.client).resolve()),
        "--auto-plan",
        args.mode,
        "--percentile",
        str(args.percentile),
        "--seed",
        str(args.seed),
        "--skill",
        str(args.skill if args.skill is not None else args.percentile),
        "--effort",
        str(args.effort),
        "--execution-mode",
        execution_mode,
        "--motion-mode",
        args.motion_mode or args.mode,
        "--execution-blend",
        str(execution_blend),
        "--sample-rate",
        str(args.sample_rate),
        "--cursor-rate",
        str(args.cursor_rate),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--planning-timeout-seconds",
        str(args.planning_timeout_seconds),
        "--left-key",
        args.left_key,
        "--right-key",
        args.right_key,
        "--input-lead-ms",
        str(args.input_lead_ms),
        "--workspace-root",
        str(root),
    ]
    if args.osu_storage:
        command.extend(["--osu-storage", str(Path(args.osu_storage).resolve())])
    if execution_model:
        command.extend(["--execution-model", str(Path(execution_model).resolve())])
    log_path = args.log_path or _default_state_log() / f"auto-run-{datetime.now():%Y%m%d-%H%M%S}.log"
    command.extend(["--log-path", str(Path(log_path).resolve())])
    output_directory = args.output_directory or _default_auto_output()
    command.extend(["--output-directory", str(Path(output_directory).resolve())])
    completed = subprocess.run(command, check=False, env=_runner_environment())
    return int(completed.returncode)


def _resolve_auto_execution(
    root: Path,
    execution_mode: str | None,
    execution_model: str | None,
    execution_blend: float | None,
) -> tuple[str, float, str | None]:
    """Resolve the HSR default without weakening the explicit math control."""
    mode = execution_mode or "coherent"
    if mode not in {"math-only", "hybrid", "coherent"}:
        raise ValueError("execution mode must be math-only, hybrid, or coherent")
    blend = float(execution_blend if execution_blend is not None else (0.0 if mode == "math-only" else 1.0))
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("execution blend must be between 0 and 1")
    if mode == "math-only":
        if execution_model or blend > 0.0:
            raise ValueError("math-only execution cannot receive an execution model or non-zero execution blend")
        return mode, 0.0, None
    if mode == "coherent" and blend <= 0.0:
        raise ValueError("coherent execution requires a positive blend; use math-only for the control arm")
    model = execution_model or str(
        root / "research" / "human-sim"
        / ("models/experimental/seed101-math-residual-g100-v4.json" if mode == "coherent"
           else "output/training/path-execution-ai-v2-final3/model.json")
    )
    if blend > 0.0 and not Path(model).is_file():
        raise RuntimeError(f"{mode} HSR requires its execution model artifact: {model}")
    return mode, blend, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="human-sim", description="Offline osu! human movement research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Generate a deterministic synthetic input trace")
    plan.add_argument("map_plan")
    plan.add_argument("output")
    plan.add_argument("--percentile", type=float, choices=SKILL_PRESETS, default=95.0)
    plan.add_argument("--seed", type=int, default=1)
    plan.add_argument("--sample-rate", type=int, default=500)
    plan.add_argument("--skill", type=float, help="Skill level 0-100 (default: percentile anchor)")
    plan.add_argument("--effort", type=float, default=100.0, help="Effort level 0-100")
    plan.add_argument(
        "--perfect-baseline",
        action="store_true",
        help="Disable stochastic aim/timing/error terms for infrastructure calibration",
    )
    plan.add_argument("--model-bundle", help="Optional fitted joblib bundle from the fit command")
    plan.add_argument(
        "--execution-model",
        help="Optional immutable execution-phase candidate; it can only warp an already generated mathematical trace",
    )
    plan.add_argument(
        "--execution-mode",
        choices=("math-only", "hybrid", "coherent"),
        help="Explicit execution dispatch mode; coherent uses the grouped trajectory test model",
    )
    plan.add_argument(
        "--execution-blend",
        type=float,
        default=0.0,
        help="Offline phase blend in [0,1]; zero is byte-identical math execution",
    )
    plan.add_argument(
        "--execution-adapter",
        choices=("legacy", "time-aware-v3", "boundary-aware-v2"),
        default="legacy",
        help="Explicit offline execution adapter; legacy remains the default",
    )
    plan.add_argument(
        "--execution-boundary-segments",
        help="Comma-separated zero-based segment indices; required only by boundary-aware-v2",
    )
    plan.add_argument(
        "--motion-mode",
        choices=("profile", "perfect"),
        help="Explicit mathematical motion mode; perfect is a machine baseline, profile is stochastic research motion",
    )
    plan.add_argument("--corpus-manifest", help="Optional corpus manifest to hash into the RunManifest")
    plan.add_argument("--client-build", help="Optional research client executable to hash into the RunManifest")
    plan.add_argument("--manifest", help="RunManifest output path; defaults beside the trace")
    plan.set_defaults(handler=_plan)
    validate = subparsers.add_parser("validate", help="Validate a generated trace")
    validate.add_argument("trace")
    validate.set_defaults(handler=_validate)
    collect = subparsers.add_parser("collect", help="Download public replays listed by score ID")
    collect.add_argument("score_ids")
    collect.add_argument("output")
    collect.add_argument("--delay", type=float, default=0.25)
    collect.set_defaults(handler=_collect)
    extract = subparsers.add_parser("extract", help="Match an officially decoded replay to a MapPlan")
    extract.add_argument("replay")
    extract.add_argument("map_plan")
    extract.add_argument("output")
    extract.set_defaults(handler=_extract)
    fit = subparsers.add_parser("fit", help="Fit context-conditioned quantile models")
    fit.add_argument("features")
    fit.add_argument("output")
    fit.add_argument("--seed", type=int, default=1)
    fit.set_defaults(handler=_fit)
    evaluate_parser = subparsers.add_parser("evaluate", help="Train defensive human-versus-synthetic baselines")
    evaluate_parser.add_argument("human")
    evaluate_parser.add_argument("synthetic")
    evaluate_parser.add_argument("output")
    evaluate_parser.add_argument("--seed", type=int, default=1)
    evaluate_parser.set_defaults(handler=_evaluate)
    execution_audit = subparsers.add_parser(
        "execution-audit",
        aliases=["audit-execution"],
        help="Audit the explicitly listed human-only execution corpus",
    )
    execution_audit.add_argument("manifest")
    execution_audit.add_argument("output")
    execution_audit.set_defaults(handler=_execution_audit)
    execution_source_audit = subparsers.add_parser(
        "execution-source-audit",
        aliases=["source-audit-execution"],
        help="Summarize public replay-archive metadata without assigning human skill tiers",
    )
    execution_source_audit.add_argument("index")
    execution_source_audit.add_argument("output")
    execution_source_audit.set_defaults(handler=_execution_source_audit)
    execution_candidate_audit = subparsers.add_parser(
        "execution-candidate-audit",
        aliases=["candidate-audit-execution"],
        help="Review and measure decoded public candidates without human-training admission",
    )
    execution_candidate_audit.add_argument("candidate_report")
    execution_candidate_audit.add_argument("decoded_root")
    execution_candidate_audit.add_argument("map_plan_root")
    execution_candidate_audit.add_argument("output")
    execution_candidate_audit.add_argument("--score-evidence")
    execution_candidate_audit.add_argument("--manifest")
    execution_candidate_audit.set_defaults(handler=_execution_candidate_audit)
    execution_exploratory_train = subparsers.add_parser(
        "execution-exploratory-train",
        aliases=["exploratory-train-execution"],
        help="Fit a separate skill-unknown exploratory phase candidate from admitted public candidates",
    )
    execution_exploratory_train.add_argument("candidate_report")
    execution_exploratory_train.add_argument("score_evidence")
    execution_exploratory_train.add_argument("candidate_audit")
    execution_exploratory_train.add_argument("decoded_root")
    execution_exploratory_train.add_argument("map_plan_root")
    execution_exploratory_train.add_argument("output")
    execution_exploratory_train.add_argument("--seed", type=int, default=42)
    execution_exploratory_train.add_argument("--max-windows-per-player", type=int, default=100)
    execution_exploratory_train.set_defaults(handler=_execution_exploratory_train)
    execution_train = subparsers.add_parser(
        "execution-train",
        aliases=["train-execution"],
        help="Train and register a CPU execution-phase candidate from verified human replays",
    )
    execution_train.add_argument("manifest")
    execution_train.add_argument("output")
    execution_train.add_argument("--seed", type=int, default=42)
    execution_train.set_defaults(handler=_execution_train)
    execution_extract = subparsers.add_parser(
        "execution-extract",
        aliases=["extract-execution"],
        help="Extract order-constrained phase labels from the explicitly listed human corpus",
    )
    execution_extract.add_argument("manifest")
    execution_extract.add_argument("output")
    execution_extract.set_defaults(handler=_execution_extract)
    execution_evaluate = subparsers.add_parser(
        "execution-evaluate",
        aliases=["evaluate-execution"],
        help="Evaluate an execution candidate on validation or the sealed test split",
    )
    execution_evaluate.add_argument("manifest")
    execution_evaluate.add_argument("candidate")
    execution_evaluate.add_argument("--split", choices=("train", "validation", "test"), default="test")
    execution_evaluate.add_argument("--output")
    execution_evaluate.set_defaults(handler=_execution_evaluate)
    execution_inspect = subparsers.add_parser(
        "execution-inspect",
        aliases=["inspect-execution", "model-inspect"],
        help="Inspect the immutable execution candidate contract and provenance",
    )
    execution_inspect.add_argument("candidate")
    execution_inspect.set_defaults(handler=_execution_inspect)
    execution_retrain = subparsers.add_parser(
        "execution-retrain",
        aliases=["retrain-execution"],
        help="Full-retrain a new execution candidate and compare validation with a prior candidate",
    )
    execution_retrain.add_argument("manifest")
    execution_retrain.add_argument("previous")
    execution_retrain.add_argument("output")
    execution_retrain.add_argument("--seed", type=int, default=42)
    execution_retrain.set_defaults(handler=_execution_retrain)
    execution_benchmark = subparsers.add_parser(
        "execution-benchmark",
        aliases=["benchmark-execution"],
        help="Run paired current-runtime-trace math-only versus execution-phase hybrid metrics",
    )
    execution_benchmark.add_argument("map_plan")
    execution_benchmark.add_argument("output")
    execution_benchmark.add_argument("--model")
    execution_benchmark.add_argument("--blend", type=float, default=1.0)
    execution_benchmark.add_argument("--sample-rate", type=int, default=500)
    execution_benchmark.add_argument("--skill-group", choices=("beginner", "intermediate", "expert", "competitive"), default="intermediate")
    execution_benchmark.add_argument("--effort", type=float, default=100.0)
    execution_benchmark.add_argument("--seed", type=int, default=42)
    execution_benchmark.add_argument("--export-dir")
    execution_benchmark.set_defaults(handler=_execution_benchmark)
    benchmark = subparsers.add_parser("benchmark", help="Run deterministic multi-map statistical planner regression checks")
    benchmark.add_argument("--scope", choices=("compact", "default", "full"), default="compact")
    benchmark.add_argument("--map", dest="map_paths", action="append", help="Explicit map plan; repeat for multiple maps")
    benchmark.add_argument("--seeds", default=",".join(str(seed) for seed in (42, 43, 44, 45, 46)))
    benchmark.add_argument("--monotonicity-seeds", help="Optional seed list for multi-map skill/effort checks; defaults to the first three benchmark seeds")
    benchmark.add_argument("--skill", type=float, default=50.0)
    benchmark.add_argument("--effort", type=float, default=80.0)
    benchmark.add_argument("--sample-rate", type=int, default=500)
    benchmark.add_argument("--output", default="output/analysis/statistical-benchmark.json")
    benchmark.add_argument("--summary-output", default="output/analysis/statistical-benchmark.txt")
    benchmark.add_argument("--runtime-telemetry", help="Runner log or telemetry JSON; omit for planner-only classification")
    for name, flag, default in (
        ("max_dispatch_p95_us", "--max-dispatch-p95-us", None),
        ("max_dispatch_p99_us", "--max-dispatch-p99-us", None),
        ("max_dispatch_max_us", "--max-dispatch-max-us", None),
        ("max_key_down_p95_us", "--max-key-down-p95-us", None),
        ("max_send_input_p95_us", "--max-send-input-p95-us", None),
        ("max_send_input_max_us", "--max-send-input-max-us", None),
        ("max_coalesced_fraction", "--max-coalesced-fraction", None),
        ("max_heartbeat_gap_ms", "--max-heartbeat-gap-ms", None),
        ("minimum_heartbeat_count", "--minimum-heartbeat-count", None),
    ):
        benchmark.add_argument(flag, dest=name, type=float if name != "minimum_heartbeat_count" else int, default=default)
    benchmark.set_defaults(handler=_benchmark)
    quality = subparsers.add_parser("runtime-quality", help="Assess runner latency and integrity telemetry")
    quality.add_argument("source", help="Runner log, telemetry JSON, or '-' for inline stdin is not supported")
    for name, flag, default in (
        ("max_dispatch_p95_us", "--max-dispatch-p95-us", None),
        ("max_dispatch_p99_us", "--max-dispatch-p99-us", None),
        ("max_dispatch_max_us", "--max-dispatch-max-us", None),
        ("max_key_down_p95_us", "--max-key-down-p95-us", None),
        ("max_send_input_p95_us", "--max-send-input-p95-us", None),
        ("max_send_input_max_us", "--max-send-input-max-us", None),
        ("max_coalesced_fraction", "--max-coalesced-fraction", None),
        ("max_heartbeat_gap_ms", "--max-heartbeat-gap-ms", None),
        ("minimum_heartbeat_count", "--minimum-heartbeat-count", None),
    ):
        quality.add_argument(flag, dest=name, type=float if name != "minimum_heartbeat_count" else int, default=default)
    quality.set_defaults(handler=_runtime_quality)
    audit = subparsers.add_parser(
        "audit-library",
        help="Decode and perfect-plan a deterministic sample of the local lazer osu!standard library",
    )
    audit.add_argument("output", help="JSON audit report path")
    audit.add_argument(
        "--osu-storage",
        default=str(_default_osu_storage()),
        help="lazer content-addressed files directory",
    )
    audit.add_argument("--limit", type=int, default=100, help="Number of evenly sampled difficulties; zero audits all")
    audit.add_argument("--sample-rate", type=int, default=500)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--workers", type=int, default=4)
    audit.set_defaults(handler=_audit_library)
    run = subparsers.add_parser("run", help="Launch the guarded research runner and client (Windows or X11 Linux)")
    run.add_argument("client")
    run.add_argument("trace")
    run.add_argument("--timeout-seconds", type=int, default=600)
    run.add_argument("--left-key", default="Z")
    run.add_argument("--right-key", default="X")
    run.add_argument(
        "--input-lead-ms",
        type=float,
        default=0.0,
        help="Bounded OS input-delivery compensation in milliseconds",
    )
    run.set_defaults(handler=_run)
    auto_run = subparsers.add_parser(
        "auto-run",
        help="Launch the guarded runner on Windows or X11 Linux and automatically plan the selected HSR map (Wayland is unsupported)",
    )
    auto_run.add_argument("client")
    auto_run.add_argument("--mode", choices=("perfect", "profile"), default="perfect")
    auto_run.add_argument(
        "--motion-mode",
        choices=("profile", "perfect"),
        help="Explicit motion mode forwarded to the planner; defaults to --mode",
    )
    auto_run.add_argument(
        "--execution-mode",
        choices=("math-only", "hybrid", "coherent"),
        help="Execution path; coherent selects the frozen grouped trajectory test model",
    )
    auto_run.add_argument(
        "--execution-model",
        help="Immutable execution-phase model; defaults to the final3 V2 artifact in hybrid mode",
    )
    auto_run.add_argument(
        "--execution-blend",
        type=float,
        default=None,
        help="Hybrid phase blend in [0,1]; defaults to 1 for V2 hybrid and 0 for math-only",
    )
    auto_run.add_argument("--percentile", type=float, choices=SKILL_PRESETS, default=99.5)
    auto_run.add_argument("--seed", type=int, default=42)
    auto_run.add_argument("--skill", type=float, help="Skill level 0-100 (default: percentile anchor)")
    auto_run.add_argument("--effort", type=float, default=100.0, help="Effort level 0-100")
    auto_run.add_argument("--sample-rate", type=int, default=500)
    auto_run.add_argument("--cursor-rate", type=int, default=1000)
    auto_run.add_argument("--timeout-seconds", type=int, default=600)
    auto_run.add_argument("--planning-timeout-seconds", type=int, default=180)
    auto_run.add_argument("--left-key", default="Z")
    auto_run.add_argument("--right-key", default="X")
    auto_run.add_argument(
        "--input-lead-ms",
        type=float,
        default=8.0 if os.name == "nt" else 0.0,
        help="Bounded input-delivery compensation in milliseconds (8 ms Windows default; no Linux calibration is assumed)",
    )
    auto_run.add_argument("--osu-storage", help="Override the lazer content-addressed files directory")
    auto_run.add_argument("--log-path", help="Write runner diagnostics to this file")
    auto_run.add_argument("--output-directory", help="Override the trace cache directory")
    auto_run.set_defaults(handler=_run_auto)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
