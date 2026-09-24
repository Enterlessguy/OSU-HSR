"""Check paired delivered cursor positions at every circle hit time."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from human_sim.io import load_map_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/external-ordr-v155"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-tag", required=True)
    args = parser.parse_args()
    case_root = OUT / "development-cases/validation" / args.case_tag
    maps = []
    for summary_path in sorted(case_root.glob("*-seed42.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        map_md5 = summary["map_md5"]
        trace_path = case_root / f"{map_md5}-seed42.npz"
        if summary["trace_sha256"] != sha(trace_path) or not summary["timestamp_and_key_identity"]:
            raise ValueError(f"Trace or schedule identity changed: {map_md5}")
        plan = load_map_plan(OUT / "development-plans" / f"{map_md5}.map.ndjson.gz")
        with np.load(trace_path) as arrays:
            times = arrays["times_ms"]
            math = arrays["math_positions"]
            hybrid = arrays["hybrid_positions"]
        objects = [obj for obj in plan.objects if obj.kind == "circle" and times[0] <= obj.start_time_ms <= times[-1]]
        if not objects:
            continue
        event_times = np.asarray([obj.start_time_ms for obj in objects])
        targets = np.asarray([[obj.position.x, obj.position.y] for obj in objects])
        radius = np.asarray([obj.radius for obj in objects])
        positions = {
            arm: np.column_stack([np.interp(event_times, times, frames[:, axis]) for axis in range(2)])
            for arm, frames in (("math", math), ("hybrid", hybrid))
        }
        errors = {arm: np.linalg.norm(points - targets, axis=1) / radius for arm, points in positions.items()}
        math_legal = errors["math"] <= 1.0
        hybrid_legal = errors["hybrid"] <= 1.0
        maps.append({
            "map_md5": map_md5, "circles": len(objects),
            "math_legal": int(np.count_nonzero(math_legal)),
            "hybrid_legal": int(np.count_nonzero(hybrid_legal)),
            "math_legal_hybrid_illegal": int(np.count_nonzero(math_legal & ~hybrid_legal)),
            "math_illegal_hybrid_legal": int(np.count_nonzero(~math_legal & hybrid_legal)),
            "math_p95_error_radius": float(np.quantile(errors["math"], .95)),
            "hybrid_p95_error_radius": float(np.quantile(errors["hybrid"], .95)),
        })
    report = {
        "schema_version": "coherent-paired-circle-contact-audit-v1",
        "case_tag": args.case_tag, "confirmation_access": False,
        "maps": len(maps), "circles": sum(row["circles"] for row in maps),
        "math_legal_hybrid_illegal": sum(row["math_legal_hybrid_illegal"] for row in maps),
        "math_illegal_hybrid_legal": sum(row["math_illegal_hybrid_legal"] for row in maps),
        "map_rows": maps,
    }
    path = case_root / "circle-contact-audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "map_rows"}, indent=2))


if __name__ == "__main__":
    main()
