"""Diagnostic: isolate the coherent adapter's fixed task prior on open maps."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from human_sim.coherent_execution import apply_coherent_trajectory_model, load_coherent_model
from human_sim.io import load_map_plan
from human_sim.schemas import TraceFrame


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"
MODEL = Path(os.environ.get("HSR_COHERENT_MODEL", str(ROOT / "output/grouped-coherent-learning-curve-v1-corrected/models/seed101-g200.json")))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    case_dir = OUT / "development-cases" / "validation"
    manifest = json.loads((OUT / "development-batches" / "validation" / "window-manifest.json").read_text(encoding="utf-8"))
    summaries = [case_dir / (row["map"] + "-seed42.json") for row in manifest["records"][:3]]
    model = load_coherent_model(MODEL)
    results = []
    for path in summaries:
        source = json.loads(path.read_text(encoding="utf-8"))
        map_md5 = source["map_md5"]
        pair = np.load(case_dir / (map_md5 + "-seed42.npz"))
        times, points, keys = pair["times_ms"], pair["math_positions"], pair["keys"]
        frames = [TraceFrame(int(round(time * 1000)), float(point[0]), float(point[1]), bool(key[0]), bool(key[1])) for time, point, key in zip(times, points, keys)]
        plan = load_map_plan(OUT / "development-plans" / (map_md5 + ".map.ndjson.gz"))
        with patch("human_sim.coherent_execution._predict", side_effect=lambda context, _model: context["baseline"]):
            task_frames, diagnostic = apply_coherent_trajectory_model(frames, plan, model)
        task_positions = np.asarray([(frame.x, frame.y) for frame in task_frames])
        if any((a.time_us, a.k1, a.k2) != (b.time_us, b.k1, b.k2) for a, b in zip(frames, task_frames)):
            raise ValueError("Task diagnostic changed the schedule")
        output = case_dir / (map_md5 + "-seed42-task-prior.npz")
        np.savez_compressed(output, task_positions=task_positions)
        results.append({"map_md5": map_md5, "source_case_sha256": sha(path), "task_trace_sha256": sha(output), "accepted_task_windows": diagnostic["learned_segment_count"], "changed_samples": diagnostic["changed_sample_count"]})
        print(f"{map_md5[:10]} task_windows={diagnostic['learned_segment_count']} changed={diagnostic['changed_sample_count']}", flush=True)
    report = {"schema_version": "coherent-fixed-task-prior-diagnostic-v1", "purpose": "Separate fixed adapter prior from learned residual; not a deployable model", "confirmation_access": False, "cases": results}
    dest = case_dir / "task-prior-diagnostic.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={dest}")


if __name__ == "__main__":
    main()
