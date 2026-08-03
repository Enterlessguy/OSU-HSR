from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .collector import collect_replays
from .dataset import extract_features
from .io import load_map_plan, write_trace
from .library_audit import audit_library
from .modeling import evaluate, fit_models
from .planner import PLANNER_VERSION, HumanTracePlanner
from .schemas import HumanProfile, RunManifest, SKILL_PRESETS
from .validation import validate_trace


def _plan(args: argparse.Namespace) -> int:
    map_plan = load_map_plan(args.map_plan)
    profile = HumanProfile(
        float(args.percentile),
        int(args.seed),
        int(args.sample_rate),
        perfect_baseline=bool(args.perfect_baseline),
        skill_level=float(args.skill if args.skill is not None else args.percentile),
        effort_level=float(args.effort),
    )
    planner = HumanTracePlanner(map_plan, profile, args.model_bundle)
    frames = planner.generate()
    model_hash = hashlib.sha256(Path(args.model_bundle).read_bytes()).hexdigest() if args.model_bundle else None
    model_version = model_hash or ("built_in_perfect_baseline_v1" if profile.perfect_baseline else "built_in_baseline_v1")
    digest = write_trace(
        args.output,
        map_plan=map_plan,
        profile_percentile=profile.percentile,
        seed=profile.seed,
        sample_rate_hz=profile.sample_rate_hz,
        frames=frames,
        model_sha256=model_version,
        diagnostic_perfect=profile.perfect_baseline,
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
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    manifest = RunManifest(
        schema_version=1,
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
    )
    manifest_path = Path(args.manifest or f"{args.output}.manifest.json")
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    result["manifest"] = str(manifest_path)
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


def _audit_library(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[4]
    exporter = root / "research" / "HumanSim.MapExporter" / "bin" / "Debug" / "net8.0" / "HumanSim.MapExporter.exe"
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


def _run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[4]
    dotnet = root / ".dotnet" / "dotnet.exe"
    runner = root / "research" / "HumanSim.Runner" / "bin" / "Debug" / "net8.0-windows" / "HumanSim.Runner.dll"
    if not dotnet.exists() or not runner.exists():
        raise RuntimeError("Build HumanSim.Runner and install the workspace-local .NET SDK before running a trace")
    completed = subprocess.run(
        [
            str(dotnet),
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
    runner = root / "research" / "HumanSim.Runner" / "bin" / "Debug" / "net8.0-windows" / "HumanSim.Runner.exe"
    if not runner.exists():
        raise RuntimeError("Build HumanSim.Runner before launching automatic planning")
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
    log_path = args.log_path or root / "research" / "human-sim" / "output" / f"auto-run-{datetime.now():%Y%m%d-%H%M%S}.log"
    command.extend(["--log-path", str(Path(log_path).resolve())])
    completed = subprocess.run(command, check=False, env=_runner_environment())
    return int(completed.returncode)


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
    audit = subparsers.add_parser(
        "audit-library",
        help="Decode and perfect-plan a deterministic sample of the local lazer osu!standard library",
    )
    audit.add_argument("output", help="JSON audit report path")
    audit.add_argument(
        "--osu-storage",
        default=str(Path(os.environ.get("APPDATA", "")) / "osu-development" / "files"),
        help="lazer content-addressed files directory",
    )
    audit.add_argument("--limit", type=int, default=100, help="Number of evenly sampled difficulties; zero audits all")
    audit.add_argument("--sample-rate", type=int, default=500)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--workers", type=int, default=4)
    audit.set_defaults(handler=_audit_library)
    run = subparsers.add_parser("run", help="Launch the guarded Windows runner and research client")
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
        help="Launch the guarded runner and automatically plan the map/difficulty selected with HSR",
    )
    auto_run.add_argument("client")
    auto_run.add_argument("--mode", choices=("perfect", "profile"), default="perfect")
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
        default=8.0,
        help="OS input-delivery compensation; 8 ms is the research-build calibration default",
    )
    auto_run.add_argument("--osu-storage", help="Override the lazer content-addressed files directory")
    auto_run.add_argument("--log-path", help="Write runner diagnostics to this file")
    auto_run.set_defaults(handler=_run_auto)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
