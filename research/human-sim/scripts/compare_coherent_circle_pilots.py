"""Paired map-family uncertainty for exploratory circle distances only.

This deliberately does not label a candidate superior: OSI V2 and independent
perceptual confirmation are separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/external-ordr-v155/development-cases/validation"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(tag: str) -> tuple[Path, dict[str, dict]]:
    path = BASE / tag / "circle-pilot-distance.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["schema_version"] != "ordr-circle-pilot-distance-v1":
        raise ValueError(f"Wrong exploratory report: {path}")
    rows = {row["map_md5"]: row for row in report["cases"]}
    if len(rows) != len(report["cases"]):
        raise ValueError(f"Duplicate map in {path}")
    return path, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-tag", default="gapguard-v1")
    parser.add_argument("--candidate-tag", default="mixed-v1")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    old_path, old = read(args.baseline_tag)
    new_path, new = read(args.candidate_tag)
    maps = sorted(set(old) & set(new))
    if not args.allow_partial and (len(old) != 43 or len(new) != 43 or len(maps) != 43):
        raise ValueError(f"Need full 43 paired maps: old={len(old)} new={len(new)} common={len(maps)}")
    if not maps:
        raise ValueError("No common maps")
    for key in maps:
        if old[key]["map_family"] != new[key]["map_family"]:
            raise ValueError(f"Map-family mismatch: {key}")
        if abs(old[key]["math"]["composite_distance"] - new[key]["math"]["composite_distance"]) > 1e-10:
            raise ValueError(f"Math baseline differs between tags: {key}")
    if len({old[key]["map_family"] for key in maps}) != len(maps):
        raise ValueError("Bootstrap requires unique map families here")
    deltas = np.asarray([new[key]["hybrid_minus_math_distance"] for key in maps], dtype=float)
    old_deltas = np.asarray([old[key]["hybrid_minus_math_distance"] for key in maps], dtype=float)
    rng = np.random.default_rng(20260924)
    draw = rng.integers(0, len(maps), size=(20000, len(maps)))
    mean_boot = np.mean(deltas[draw], axis=1)
    difference_boot = np.mean((deltas - old_deltas)[draw], axis=1)
    report = {
        "schema_version": "ordr-exploratory-paired-circle-comparison-v1",
        "status": "exploratory_not_osi_v2_no_superiority_claim",
        "confirmation_access": False,
        "partial": len(maps) != 43,
        "baseline_report_sha256": sha(old_path),
        "candidate_report_sha256": sha(new_path),
        "maps": len(maps), "families": len(maps),
        "candidate_minus_math_mean_distance": float(np.mean(deltas)),
        "candidate_minus_math_bootstrap_95": [float(x) for x in np.quantile(mean_boot, [0.025, 0.975])],
        "old_minus_math_mean_distance": float(np.mean(old_deltas)),
        "candidate_minus_old_mean_distance": float(np.mean(deltas - old_deltas)),
        "candidate_minus_old_bootstrap_95": [float(x) for x in np.quantile(difference_boot, [0.025, 0.975])],
        "candidate_better_than_math_maps": int(np.sum(deltas < -1e-9)),
        "candidate_worse_than_math_maps": int(np.sum(deltas > 1e-9)),
        "candidate_tied_math_maps": int(np.sum(np.abs(deltas) <= 1e-9)),
        "map_deltas": {key: float(delta) for key, delta in zip(maps, deltas)},
    }
    path = BASE / args.candidate_tag / "circle-pilot-comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "map_deltas"}, indent=2))


if __name__ == "__main__":
    main()
