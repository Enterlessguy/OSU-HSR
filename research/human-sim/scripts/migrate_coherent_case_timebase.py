"""Correct cached paired-case time arrays from planner-relative to map time."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from human_sim.io import load_map_plan


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"
CASES = OUT / "development-cases" / "validation"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    changes = []
    for summary_path in sorted(CASES.glob("*-seed42.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("time_basis") == "absolute_beatmap_ms":
            continue
        map_md5 = summary["map_md5"]
        trace_path = CASES / (map_md5 + "-seed42.npz")
        original_hash = sha(trace_path)
        if original_hash != summary["trace_sha256"]:
            raise ValueError(f"Cached case hash mismatch: {map_md5}")
        plan = load_map_plan(OUT / "development-plans" / (map_md5 + ".map.ndjson.gz"))
        timeline_start_ms = plan.objects[0].start_time_ms - 1500.0
        with np.load(trace_path) as archive:
            times = archive["times_ms"]
            math_positions = archive["math_positions"]
            hybrid_positions = archive["hybrid_positions"]
            keys = archive["keys"]
        if abs(float(times[0])) > 1e-6:
            raise ValueError(f"Expected planner-relative zero start for {map_md5}, got {times[0]}")
        new_times = times + timeline_start_ms
        temporary = trace_path.with_name(trace_path.stem + ".timebase-migration.npz")
        np.savez_compressed(temporary, times_ms=new_times, math_positions=math_positions, hybrid_positions=hybrid_positions, keys=keys)
        temporary.replace(trace_path)
        new_hash = sha(trace_path)
        summary.update({"trace_sha256": new_hash, "time_basis": "absolute_beatmap_ms", "timeline_start_ms": timeline_start_ms, "timebase_migration_original_trace_sha256": original_hash})
        atomic_json(summary_path, summary)
        changes.append({"map_md5": map_md5, "old_sha256": original_hash, "new_sha256": new_hash, "timeline_start_ms": timeline_start_ms})
        print(f"migrated {map_md5[:10]} offset={timeline_start_ms}", flush=True)
    stale = CASES / "circle-pilot-distance.json"
    if stale.exists():
        report = json.loads(stale.read_text(encoding="utf-8"))
        report.update({"status": "invalidated_wrong_generated_time_basis", "invalidation_reason": "Generated trace arrays used planner-relative time against absolute human beatmap time"})
        atomic_json(CASES / "circle-pilot-distance.invalid-timebase.json", report)
        stale.unlink()
    migration = {"schema_version": "coherent-development-timebase-correction-v1", "reason": "Map time offset was omitted from stored generated trace arrays; positions and keys unchanged", "confirmation_access": False, "cases": changes}
    target = CASES / "timebase-correction.json"
    atomic_json(target, migration)
    print(f"corrected={len(changes)} report={target} sha256={sha(target)}")


if __name__ == "__main__":
    main()
