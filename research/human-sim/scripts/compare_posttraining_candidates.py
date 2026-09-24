"""Paired opened-validation comparison of two delivered circle-component reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "output/external-ordr-v155/development-cases/validation"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--candidate-tag", required=True)
    args = parser.parse_args()
    paths = {name: CASES / tag / "osi-v2-circle-component.json" for name, tag in
             (("base", args.base_tag), ("candidate", args.candidate_tag))}
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    base, candidate = reports["base"], reports["candidate"]
    for key in ("calibration_sha256", "feature_source_sha256", "bandwidth_source_sha256", "scorer_source_sha256"):
        if base[key] != candidate[key]:
            raise ValueError(f"Mismatched scoring contract: {key}")
    if not all(report["all_generated_maps_present"] and not report["confirmation_access"] for report in reports.values()):
        raise ValueError("Both reports must cover all opened validation maps and no confirmation")
    rows = {name: {row["map_md5"]: row for row in report["maps"]} for name, report in reports.items()}
    if rows["base"].keys() != rows["candidate"].keys():
        raise ValueError("Case lists differ")
    differences = []
    families = set()
    for map_md5 in sorted(rows["base"]):
        left, right = rows["base"][map_md5], rows["candidate"][map_md5]
        if left["status"] != right["status"] or left["map_family"] != right["map_family"]:
            raise ValueError(f"Support or family changed: {map_md5}")
        if left["status"] != "supported":
            continue
        if abs(left["math"]["circle_component_score"] - right["math"]["circle_component_score"]) > 1e-9:
            raise ValueError(f"Math arm differs: {map_md5}")
        if left["map_family"] in families:
            raise ValueError("Map families are not independent")
        families.add(left["map_family"])
        differences.append(right["hybrid"]["circle_component_score"] - left["hybrid"]["circle_component_score"])
    if not differences:
        raise ValueError("No supported maps")
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(20260925)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    bootstrap = np.mean(values[indices], axis=1)
    result = {
        "schema_version": "posttraining-delivered-circle-comparison-v1",
        "status": "opened_validation_circle_component_only",
        "confirmation_access": False,
        "base_tag": args.base_tag,
        "candidate_tag": args.candidate_tag,
        "base_report_sha256": sha(paths["base"]),
        "candidate_report_sha256": sha(paths["candidate"]),
        "supported_maps": len(values),
        "candidate_minus_base_mean": float(np.mean(values)),
        "candidate_minus_base_bootstrap_95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "maps_better": int(np.count_nonzero(values > 0)),
        "maps_worse": int(np.count_nonzero(values < 0)),
    }
    destination = CASES / args.candidate_tag / "posttraining-comparison.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
