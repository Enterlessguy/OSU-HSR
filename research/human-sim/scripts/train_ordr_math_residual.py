"""Fit a task-basis model to human-minus-math residuals on TRAIN only.

The existing joint architecture learns a correction relative to its task
curve. We replace each training observation with task + (human - math) so its
runtime correction has the meaning required by the isolated math-residual
adapter. Validation and confirmation movement are never read by this fit.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/external-ordr-v155"
WORKTREE = Path(os.environ.get("HSR_RESEARCH_ROOT", str(ROOT))).resolve()
VARIANT = ROOT / "output/adapter-experiments/math-residual-v1/src"
SOURCE = OUT / "mixed-coherent-train-v1/source"
DEST = OUT / "math-residual-train-v1"
sys.path.insert(0, str(WORKTREE))
import scripts.run_grouped_coherent_learning_curve_v1 as grouped  # noqa: E402
from human_sim.io import load_map_plan  # noqa: E402
from human_sim.planner import HumanTracePlanner  # noqa: E402
from human_sim.schemas import HumanProfile  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order(keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda key: hashlib.sha256(f"ordr-mixed-20260924:new:{key}".encode()).hexdigest())


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    grouped.coherent.pilot.detail.install(4)
    audit_path = OUT / "development-identity-audit-v1.json"
    manifest_path = OUT / "development-batches/train/window-manifest.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    eligible = {row["replay_md5"] for row in audit["records"] if row["split"] == "train" and row["metadata_eligible"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows = [row for row in manifest["records"] if row["source_replay_md5"] in eligible]
    recipients = []
    for split in ("calibration", "validation"):
        path = OUT / "development-batches" / split / "window-manifest.json"
        recipients.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    admitted, admission_report = grouped.admission({"records": train_rows}, SOURCE, recipients)
    grouped.atomic(DEST / "admission-report.json", admission_report)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in admitted:
        groups[row["identity"]["component_id"]].append(row)
    selected_ids = order(list(groups))[:100]
    if len(selected_ids) != 100:
        raise ValueError(f"Only {len(selected_ids)} independent new TRAIN groups")
    selected = [row for key in selected_ids for row in groups[key]]
    profile = HumanProfile(99.5, 42, 500, perfect_baseline=False, skill_level=99.5, effort_level=100.0)
    spec = {
        "schema_version": "ordr-math-residual-fit-freeze-v1",
        "selection": "same first 100 SHA256-ordered new TRAIN components as mixed fit",
        "component_ids": selected_ids, "contexts": len(selected), "seed": 101, "group_size": 100,
        "math_profile": {"skill": 99.5, "effort": 100.0, "seed": 42, "timing_level": 500, "perfect_baseline": False},
        "target": "task_baseline + observed_human - generated_math at identical absolute beatmap timestamps",
        "confirmation_access": False, "validation_movement_access": False,
        "source_hashes": {
            "identity_audit": sha(audit_path), "train_manifest": sha(manifest_path),
            "admission_report": sha(DEST / "admission-report.json"),
            "fit_script": sha(Path(__file__)),
            "trainer": sha(Path(grouped.coherent.__file__)),
            "variant_adapter": sha(VARIANT / "human_sim/coherent_execution.py"),
        },
    }
    spec["sha256"] = grouped.canonical_sha256(spec)
    spec_path = DEST / "frozen-fit-spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text(encoding="utf-8")) != spec:
        raise ValueError("Existing frozen fit spec differs")
    grouped.atomic(spec_path, spec)
    math_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    transformed = []
    residual_rms = []
    for number, row in enumerate(selected, 1):
        identity = row["identity"]
        map_md5 = identity["map"]
        if map_md5 not in math_cache:
            plan = load_map_plan(OUT / "development-plans" / f"{map_md5}.map.ndjson.gz")
            frames = HumanTracePlanner(plan, profile).generate()
            timeline_start = plan.objects[0].start_time_ms - 1500.0
            math_cache[map_md5] = (
                np.asarray([timeline_start + frame.time_us / 1000.0 for frame in frames], dtype=float),
                np.asarray([(frame.x, frame.y) for frame in frames], dtype=float),
            )
        context = grouped.make_context(identity, row["times"], row["positions"], row["events"], row["centers"])
        times = np.asarray(context["times"], dtype=float)
        math_times, math_positions = math_cache[map_md5]
        if times[0] < math_times[0] or times[-1] > math_times[-1]:
            raise ValueError(f"Math trace does not cover TRAIN context: {map_md5}")
        math_at_human_times = np.column_stack([np.interp(times, math_times, math_positions[:, axis]) for axis in range(2)])
        task_at_human_times = grouped.coherent.pilot.detail.design(times, context["nodes"], 4) @ context["baseline"]
        residual = context["observed"] - math_at_human_times
        context["observed"] = task_at_human_times + residual
        residual_rms.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
        transformed.append(context)
        if number % 20 == 0:
            print(f"MATH TARGET {number}/{len(selected)} maps={len(math_cache)}", flush=True)
    started = time.perf_counter()
    model, fit_rows = grouped.coherent.train_joint(transformed, grouped.bank_amplitudes(transformed))
    fit_wall = time.perf_counter() - started
    bundle = {
        "joint": model, "seed": 101, "group_size": 100,
        "component_ids": selected_ids, "fit_context_count": len(transformed),
        "fit_wall_s": fit_wall, "frozen_fit_spec_sha256": spec["sha256"],
        "target": spec["target"],
        "human_provenance_limit": "public score metadata corroborates identity, not manual execution",
    }
    bundle["sha256"] = grouped.canonical_sha256(grouped.coherent.pilot.backbone._jsonable(bundle))
    model_path = DEST / "seed101-math-residual-g100.json"
    grouped.atomic(model_path, bundle)
    # Verify in a fresh interpreter so the already imported TRAIN package does
    # not shadow the isolated group-100 research adapter. The desktop loader
    # remains unchanged.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(VARIANT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    validated = subprocess.run(
        [sys.executable, "-c", "import sys; from human_sim.coherent_execution import load_coherent_model; print(load_coherent_model(sys.argv[1]).canonical_sha256)", str(model_path)],
        capture_output=True, text=True, check=True, env=environment,
    )
    verified_hash = validated.stdout.strip()
    if verified_hash != bundle["sha256"]:
        raise ValueError("Fresh research runtime loader returned another model hash")
    report = {
        "schema_version": "ordr-math-residual-fit-v1",
        "contexts": len(transformed), "groups": len(selected_ids), "maps": len(math_cache),
        "fit_wall_s": fit_wall, "fit_rows": len(fit_rows),
        "training_residual_rms_median_px": float(np.median(residual_rms)),
        "model_canonical_sha256": verified_hash,
        "model_file_sha256": sha(model_path), "frozen_fit_spec_sha256": spec["sha256"],
        "confirmation_access": False, "validation_movement_access": False,
        "deployment_status": "research_candidate_not_promoted",
    }
    grouped.atomic(DEST / "fit-report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
