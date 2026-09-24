"""Check delivered paired traces for edits inside unsupported map modes."""

from __future__ import annotations

import argparse
from collections import Counter
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-tag")
    args = parser.parse_args()
    cases = CASES / args.case_tag if args.case_tag else CASES
    counts: Counter[str] = Counter()
    rows = []
    for summary_path in sorted(cases.glob("*-seed42.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        map_md5 = summary["map_md5"]
        trace_path = cases / (map_md5 + "-seed42.npz")
        if summary["trace_sha256"] != sha(trace_path) or summary.get("time_basis") != "absolute_beatmap_ms":
            raise ValueError(f"Unverified paired trace time basis or hash: {map_md5}")
        plan = load_map_plan(OUT / "development-plans" / (map_md5 + ".map.ndjson.gz"))
        with np.load(trace_path) as trace:
            times = trace["times_ms"]
            changed = np.linalg.norm(trace["hybrid_positions"] - trace["math_positions"], axis=1) > 1e-9
        for mode in ("slider", "spinner"):
            for obj in plan.objects:
                if obj.kind != mode:
                    continue
                mask = (times >= obj.start_time_ms) & (times <= obj.end_time_ms)
                counts[mode + "_windows"] += 1
                counts[mode + "_samples"] += int(np.count_nonzero(mask))
                n = int(np.count_nonzero(changed & mask))
                counts[mode + "_changed_samples"] += n
                if n:
                    rows.append({"map_md5": map_md5, "mode": mode, "object_index": obj.index, "changed_samples": n})
        for previous, current in zip(plan.objects, plan.objects[1:]):
            if current.start_time_ms - previous.end_time_ms < 1000:
                continue
            mask = (times >= previous.end_time_ms) & (times <= current.start_time_ms)
            counts["break_windows"] += 1
            counts["break_samples"] += int(np.count_nonzero(mask))
            n = int(np.count_nonzero(changed & mask))
            counts["break_changed_samples"] += n
            if n:
                rows.append({"map_md5": map_md5, "mode": "break_free_roam", "previous_index": previous.index, "next_index": current.index, "changed_samples": n})
    report = {"schema_version": "coherent-unsupported-mode-boundary-audit-v1", "confirmation_access": False, "case_count": len(list(cases.glob("*-seed42.json"))), "counts": counts, "changed_windows": rows, "unmodified_unsupported_modes": not rows}
    path = cases / "mode-boundary-audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "changed_windows"}, indent=2))
    print(f"changed_windows={len(rows)} report={path}")


if __name__ == "__main__":
    main()
