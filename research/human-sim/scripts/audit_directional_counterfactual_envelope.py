"""Audit sampled kinematic safety of the opened-validation directional trial."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from human_sim.boundary_aware_execution import exact_kinematic_summary
from human_sim.execution import ExecutionConstraints
from human_sim.io import load_map_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/external-ordr-v155"
BASE = OUT / "development-cases/validation/math-residual-trained-v1"
TRIAL = OUT / "development-cases/validation/math-residual-lateral-1p50-counterfactual"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    if source["cases"] != 43 or source["confirmation_access"]:
        raise ValueError("Expected complete opened validation only")
    failures = Counter()
    checked = 0
    worst_ratio = 0.0
    worst = None
    constraints = ExecutionConstraints()
    for name in source["case_summaries"]:
        original = json.loads((BASE / name).read_text(encoding="utf-8"))
        trial = json.loads((TRIAL / name).read_text(encoding="utf-8"))
        map_md5 = original["map_md5"]
        plan = load_map_plan(OUT / "development-plans" / f"{map_md5}.map.ndjson.gz")
        trace_path = TRIAL / name.replace(".json", ".npz")
        if trial["trace_sha256"] != sha(trace_path) or trial["runtime_delivered"]:
            raise ValueError(f"Counterfactual provenance failed: {map_md5}")
        with np.load(trace_path) as arrays:
            times = arrays["times_ms"]
            math = arrays["math_positions"]
            hybrid = arrays["hybrid_positions"]
        for record in original["diagnostics"]["windows"]:
            if not record.get("accepted"):
                continue
            objects = [plan.objects[index] for index in record["object_indices"]]
            mask = (times >= objects[0].start_time_ms - 1e-9) & (times <= objects[-1].start_time_ms + 1e-9)
            local_times = times[mask]
            baseline = exact_kinematic_summary(math[mask], local_times, constraints)
            candidate = exact_kinematic_summary(hybrid[mask], local_times, constraints)
            checked += 1
            for metric, limit in candidate["limits"].items():
                allowance = max(limit, 1.01 * baseline[metric]) + 1e-6
                ratio = candidate[metric] / allowance if allowance > 0 else 0.0
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst = {"map_md5": map_md5, "object_indices": record["object_indices"],
                             "metric": metric, "value": candidate[metric], "allowance": allowance}
                if candidate[metric] > allowance:
                    failures[metric] += 1
    report = {
        "schema_version": "directional-counterfactual-sampled-envelope-audit-v1",
        "runtime_delivered": False,
        "confirmation_access": False,
        "trial_report_sha256": sha(TRIAL / "counterfactual-report.json"),
        "auditor_sha256": sha(Path(__file__)),
        "accepted_windows_checked": checked,
        "failure_counts": dict(failures),
        "worst_ratio_to_allowance": worst_ratio,
        "worst": worst,
    }
    target = TRIAL / "sampled-envelope-audit.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
