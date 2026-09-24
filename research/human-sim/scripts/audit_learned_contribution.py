"""Separate learned displacement from the adapter's total math replacement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/external-ordr-v155/development-cases/validation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-tag", required=True)
    args = parser.parse_args()
    records = []
    for path in sorted((BASE / args.case_tag).glob("*-seed42.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        windows = [row for row in case["diagnostics"]["windows"] if row.get("accepted")]
        samples = sum(row["samples"] for row in windows)
        learned_sq = sum(row["samples"] * row["learned_contribution_rms_px"] ** 2 for row in windows)
        total_sq = sum(row["samples"] * row["total_replacement_rms_px"] ** 2 for row in windows)
        records.append({
            "map_md5": case["map_md5"], "windows": len(windows), "samples": samples,
            "learned_rms_px": float(np.sqrt(learned_sq / samples)) if samples else 0.0,
            "replacement_rms_px": float(np.sqrt(total_sq / samples)) if samples else 0.0,
            "learned_to_replacement_rms_ratio": float(np.sqrt(learned_sq / total_sq)) if total_sq else 0.0,
            "median_alpha": float(np.median([row["alpha"] for row in windows])) if windows else 0.0,
        })
    report = {
        "schema_version": "coherent-learned-contribution-audit-v1",
        "case_tag": args.case_tag, "map_count": len(records),
        "accepted_windows": sum(row["windows"] for row in records),
        "learned_to_replacement_ratio_median_map": float(np.median([row["learned_to_replacement_rms_ratio"] for row in records])),
        "learned_to_replacement_ratio_p10_map": float(np.quantile([row["learned_to_replacement_rms_ratio"] for row in records], .1)),
        "maps": records,
    }
    path = BASE / args.case_tag / "learned-contribution-audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "maps"}, indent=2))
    for row in sorted(records, key=lambda value: value["learned_to_replacement_rms_ratio"])[:5]:
        print("lowest", row)


if __name__ == "__main__":
    main()
