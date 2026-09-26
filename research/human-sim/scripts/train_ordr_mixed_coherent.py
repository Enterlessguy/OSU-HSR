"""Fit a new coherent model on disjoint old/new TRAIN groups only.

This preserves the deployed architecture and controller while adding newly
verified public replay groups.  Validation movement is never read by this fit.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import argparse
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"
WORKTREE = Path(os.environ.get("HSR_RESEARCH_ROOT", str(ROOT))).resolve()
OLD = WORKTREE / "output" / "grouped-coherent-learning-curve-v1-corrected"
DEST = OUT / "mixed-coherent-train-v1"
sys.path.insert(0, str(WORKTREE))
from human_sim import coherent_training as grouped
from human_sim.execution import canonical_sha256
from train_ordr_math_residual import atomic, jsonable  # noqa: E402
from human_sim.coherent_execution import load_coherent_model  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered(source: str, keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda key: hashlib.sha256(f"ordr-mixed-20260924:{source}:{key}".encode()).hexdigest())


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Stage reviewed TRAIN inputs for the public residual trainer without legacy data")
    args = parser.parse_args()
    audit_path = OUT / "development-identity-audit-v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    eligible = {(row["split"], row["replay_md5"]) for row in audit["records"] if row["metadata_eligible"]}
    train_manifest_path = OUT / "development-batches" / "train" / "window-manifest.json"
    manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    train_rows = [row for row in manifest["records"] if ("train", row["source_replay_md5"]) in eligible]
    if len(train_rows) < 100:
        raise ValueError(f"Only {len(train_rows)} identity-reviewed new TRAIN replays; require at least 100")
    recipients = []
    for split in ("calibration", "validation"):
        path = OUT / "development-batches" / split / "window-manifest.json"
        recipients.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    recipient_tokens = {(key, row[key]) for row in recipients for key in ("player", "replay_sha256", "map_family")}
    if any((key, row[key]) in recipient_tokens for row in train_rows for key in ("player", "replay_sha256", "map_family")):
        raise ValueError("TRAIN and development recipient identity overlap")

    source = DEST / "source"
    (source / "decoded").mkdir(parents=True, exist_ok=True)
    (source / "map-plans").mkdir(parents=True, exist_ok=True)
    for row in train_rows:
        decoded_source = OUT / "development-decoded" / (row["replay_sha256"] + ".ndjson")
        plan_source = OUT / "development-plans" / (row["map"] + ".map.ndjson.gz")
        if not decoded_source.exists() or not plan_source.exists():
            raise FileNotFoundError(f"Missing prepared TRAIN source for {row['source_replay_md5']}")
        for original, linked in ((decoded_source, source / "decoded" / decoded_source.name), (plan_source, source / "map-plans" / plan_source.name)):
            if not linked.exists():
                os.link(original, linked)
            if linked.stat().st_size != original.stat().st_size:
                raise ValueError(f"TRAIN hardlink size mismatch: {linked}")
    if args.prepare_only:
        print(f"Prepared {len(train_rows)} reviewed TRAIN inputs; confirmation untouched")
        return
    admitted, admission_report = grouped.admission({"records": train_rows}, source, recipients)
    admission_path = DEST / "new-admission-report.json"
    atomic(admission_path, admission_report)
    if admission_report["admitted_components"] < 100:
        raise ValueError(f"Only {admission_report['admitted_components']} new independent components; need 100")
    new_groups: dict[str, list[dict]] = defaultdict(list)
    for row in admitted:
        context = grouped.make_context(row["identity"], row["times"], row["positions"], row["events"], row["centers"])
        new_groups[row["identity"]["component_id"]].append(context)

    old_report_path = OLD / "admission-report.json"
    old_arrays_path = OLD / "admitted-contexts.npz"
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    old_groups: dict[str, list[dict]] = defaultdict(list)
    with np.load(old_arrays_path) as arrays:
        for i, row in enumerate(old_report["rows"]):
            prefix = f"c{i:03d}"
            context = grouped.make_context(row, arrays[prefix + "_times"], arrays[prefix + "_positions"], arrays[prefix + "_events"], arrays[prefix + "_centers"])
            old_groups[row["component_id"]].append(context)
    new_ids = ordered("new", list(new_groups))[:100]
    old_ids = ordered("old", list(old_groups))[:100]
    if len(new_ids) != 100 or len(old_ids) != 100:
        raise ValueError("Mixed fit requires 100 independent groups from each source")
    selected = [context for key in new_ids for context in new_groups[key]] + [context for key in old_ids for context in old_groups[key]]
    spec = {
        "schema_version": "ordr-mixed-coherent-fit-freeze-v1",
        "selection": "first 100 SHA256-ordered new and first 100 SHA256-ordered old independent TRAIN components; all admitted contexts per group",
        "new_component_ids": new_ids, "old_component_ids": old_ids,
        "new_contexts": sum(len(new_groups[key]) for key in new_ids),
        "old_contexts": sum(len(old_groups[key]) for key in old_ids),
        "seed": 101, "group_size": 200,
        "training_only": True, "confirmation_access": False,
        "source_hashes": {
            "development_identity_audit": sha(audit_path),
            "new_train_manifest": sha(train_manifest_path),
            "new_admission_report": sha(admission_path),
            "old_admission_report": sha(old_report_path),
            "old_admitted_arrays": sha(old_arrays_path),
            "trainer_script": sha(Path(__file__)),
            "grouped_trainer": sha(Path(grouped.__file__)),
            "coherent_trainer": sha(Path(grouped.__file__)),
        },
    }
    spec["sha256"] = canonical_sha256(spec)
    spec_path = DEST / "frozen-fit-spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError("Existing frozen fit specification differs")
    atomic(spec_path, spec)
    print(f"frozen groups=200 contexts={len(selected)} new={spec['new_contexts']} old={spec['old_contexts']}", flush=True)
    started = time.perf_counter()
    joint, fit_rows = grouped.train_joint(selected, grouped.bank_amplitudes(selected))
    wall = time.perf_counter() - started
    bundle = {
        "joint": joint, "seed": 101, "group_size": 200,
        "component_ids": ["new:" + key for key in new_ids] + ["old:" + key for key in old_ids],
        "fit_context_count": len(selected), "fit_wall_s": wall,
        "frozen_fit_spec_sha256": spec["sha256"],
        "source": "new public-score-matched o!rdr TRAIN and prior disjoint osu3k TRAIN",
        "human_provenance_limit": "public score metadata is corroboration, not proof of manual execution",
    }
    bundle["sha256"] = canonical_sha256(jsonable(bundle))
    model_path = DEST / "seed101-mixed-g200.json"
    atomic(model_path, bundle)
    verified = load_coherent_model(model_path)
    report = {"schema_version": "ordr-mixed-coherent-fit-v1", "fit_wall_s": wall, "fit_rows": len(fit_rows), "new_contexts": spec["new_contexts"], "old_contexts": spec["old_contexts"], "model_canonical_sha256": verified.canonical_sha256, "model_file_sha256": sha(model_path), "model_path": str(model_path), "confirmation_access": False, "validation_movement_access": False, "deployment_status": "candidate_not_promoted"}
    atomic(DEST / "fit-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
