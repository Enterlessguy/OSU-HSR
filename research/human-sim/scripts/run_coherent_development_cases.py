"""Generate paired math/coherent delivered traces on development maps only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

import human_sim.coherent_execution as coherent_execution
from human_sim.coherent_execution import apply_coherent_trajectory_model, load_coherent_model
from human_sim.io import load_map_plan
from human_sim.planner import HumanTracePlanner
from human_sim.schemas import HumanProfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"
MODEL = Path(os.environ.get("HSR_COHERENT_MODEL", str(ROOT / "output/grouped-coherent-learning-curve-v1-corrected/models/seed101-g200.json")))
SEED = 42


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "calibration", "validation"))
    parser.add_argument("--count", type=int)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--case-tag")
    args = parser.parse_args()
    if args.case_tag and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.case_tag):
        raise ValueError("case tag must be lowercase alphanumeric with hyphens")
    manifest_path = OUT / "development-batches" / args.split / "window-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"][:args.count] if args.count else manifest["records"]
    model_path = args.model.resolve()
    model = load_coherent_model(model_path)
    if model_path == MODEL.resolve() and model.canonical_sha256 != "fea5b14cc27aa9a5e72c8417faa2c5a6a1dfa639d07a229c865760c77543b979":
        raise ValueError("Deployed coherent model identity changed")
    dest = OUT / "development-cases" / args.split
    if args.case_tag:
        dest = dest / args.case_tag
    dest.mkdir(parents=True, exist_ok=True)
    adapter_sha = sha(Path(coherent_execution.__file__))
    summaries = []
    for number, row in enumerate(records, 1):
        map_md5 = row["map"]
        plan_path = OUT / "development-plans" / (map_md5 + ".map.ndjson.gz")
        case_path = dest / (map_md5 + f"-seed{SEED}.npz")
        summary_path = dest / (map_md5 + f"-seed{SEED}.json")
        if case_path.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["trace_sha256"] != sha(case_path):
                raise ValueError(f"Cached trace hash changed: {case_path}")
            if summary.get("model_canonical_sha256") != model.canonical_sha256 or summary.get("model_file_sha256") != sha(model_path):
                raise ValueError(f"Cached case used another model: {case_path}")
            if summary.get("map_plan_sha256") != sha(plan_path) or summary.get("time_basis") != "absolute_beatmap_ms":
                raise ValueError(f"Cached case used another map plan or time basis: {case_path}")
            if summary.get("adapter_source_sha256") and summary["adapter_source_sha256"] != adapter_sha:
                raise ValueError(f"Cached case used another adapter source: {case_path}")
            summaries.append(summary)
            print(f"[{number}/{len(records)}] cached {map_md5[:10]} learned={summary['learned_segment_count']}", flush=True)
            continue
        plan = load_map_plan(plan_path)
        profile = HumanProfile(99.5, SEED, 500, perfect_baseline=False, skill_level=99.5, effort_level=100.0)
        math_frames = HumanTracePlanner(plan, profile).generate()
        hybrid_frames, diagnostics = apply_coherent_trajectory_model(math_frames, plan, model, blend=1.0)
        if len(math_frames) != len(hybrid_frames):
            raise ValueError("Frame count changed")
        if any((a.time_us, a.k1, a.k2) != (b.time_us, b.k1, b.k2) for a, b in zip(math_frames, hybrid_frames)):
            raise ValueError("Timestamp or key schedule changed")
        timeline_start_ms = plan.objects[0].start_time_ms - 1500.0
        times_ms = np.asarray([timeline_start_ms + frame.time_us / 1000 for frame in math_frames], dtype=np.float64)
        math_positions = np.asarray([(frame.x, frame.y) for frame in math_frames], dtype=np.float64)
        hybrid_positions = np.asarray([(frame.x, frame.y) for frame in hybrid_frames], dtype=np.float64)
        keys = np.asarray([(frame.k1, frame.k2) for frame in math_frames], dtype=np.bool_)
        changed = np.linalg.norm(hybrid_positions - math_positions, axis=1)
        np.savez_compressed(case_path, times_ms=times_ms, math_positions=math_positions, hybrid_positions=hybrid_positions, keys=keys)
        summary = {
            "schema_version": "ordr-coherent-development-case-v1",
            "split": args.split, "map_md5": map_md5, "map_family": row["map_family"],
            "source_player_id": row["player"], "seed": SEED,
            "model_canonical_sha256": model.canonical_sha256,
            "model_file_sha256": sha(model_path), "map_plan_sha256": sha(plan_path),
            "adapter_source_sha256": adapter_sha,
            "trace_sha256": sha(case_path), "frame_count": len(math_frames),
            "time_basis": "absolute_beatmap_ms",
            "timeline_start_ms": timeline_start_ms,
            "timestamp_and_key_identity": True,
            "changed_sample_count": int(np.count_nonzero(changed > 1e-9)),
            "position_delta_rms_px": float(np.sqrt(np.mean(changed ** 2))),
            "learned_segment_count": diagnostics["learned_segment_count"],
            "fallback_segment_count": diagnostics["fallback_segment_count"],
            "effective_mode": diagnostics["effective_mode"],
            "diagnostics": diagnostics,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
        print(f"[{number}/{len(records)}] {map_md5[:10]} learned={summary['learned_segment_count']} fallback={summary['fallback_segment_count']} changed={summary['changed_sample_count']}", flush=True)
    report = {
        "schema_version": "ordr-coherent-development-cases-v1",
        "source_manifest_sha256": sha(manifest_path), "model_file_sha256": sha(model_path),
        "adapter_source_sha256": adapter_sha,
        "confirmation_access": False, "cases": len(summaries),
        "maps_with_learned_exposure": sum(s["learned_segment_count"] > 0 for s in summaries),
        "learned_segments": sum(s["learned_segment_count"] for s in summaries),
        "fallback_segments": sum(s["fallback_segment_count"] for s in summaries),
        "changed_samples": sum(s["changed_sample_count"] for s in summaries),
        "case_summaries": [s["map_md5"] + f"-seed{SEED}.json" for s in summaries],
    }
    report_path = dest / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "case_summaries"}, indent=2))


if __name__ == "__main__":
    main()
