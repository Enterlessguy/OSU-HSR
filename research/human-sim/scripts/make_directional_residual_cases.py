"""Development-only directional post-training residual counterfactual.

The learned correction is decomposed along each four-circle window's longest
route chord. Only its lateral component is changed. This does not assert that
the runtime delivered the result or that its continuous safety gate passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from human_sim.io import load_map_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/external-ordr-v155"
SOURCE = OUT / "development-cases/validation/math-residual-trained-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lateral-gain", type=float, required=True)
    args = parser.parse_args()
    gain = args.lateral_gain
    if not np.isfinite(gain) or not 0.0 < gain <= 2.0 or gain == 1.0:
        raise ValueError("lateral gain must be in (0, 2] and differ from one")
    label = f"{gain:.2f}".replace(".", "p")
    if float(label.replace("p", ".")) != gain:
        raise ValueError("gain must have at most two decimal places")
    dest = OUT / "development-cases/validation" / f"math-residual-lateral-{label}-counterfactual"
    dest.mkdir(parents=True, exist_ok=True)
    original_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if original_report["cases"] != 43 or original_report["confirmation_access"]:
        raise ValueError("Expected complete opened-validation source only")
    summaries = []
    for name in original_report["case_summaries"]:
        old_summary_path = SOURCE / name
        old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
        map_md5 = old_summary["map_md5"]
        old_trace_path = SOURCE / name.replace(".json", ".npz")
        plan_path = OUT / "development-plans" / f"{map_md5}.map.ndjson.gz"
        if old_summary["trace_sha256"] != sha(old_trace_path) or old_summary["map_plan_sha256"] != sha(plan_path):
            raise ValueError(f"Source integrity mismatch: {name}")
        plan = load_map_plan(plan_path)
        with np.load(old_trace_path) as arrays:
            times = arrays["times_ms"]
            math = arrays["math_positions"]
            original_hybrid = arrays["hybrid_positions"]
            keys = arrays["keys"]
        hybrid = original_hybrid.copy()
        covered = np.zeros(len(times), dtype=np.int16)
        for window in old_summary["diagnostics"]["windows"]:
            if not window.get("accepted"):
                continue
            indices = window["object_indices"]
            objects = [plan.objects[index] for index in indices]
            centers = np.asarray([(obj.position.x, obj.position.y) for obj in objects], dtype=float)
            chord = centers[-1] - centers[0]
            if np.linalg.norm(chord) < 1e-8:
                chords = np.diff(centers, axis=0)
                chord = chords[np.argmax(np.linalg.norm(chords, axis=1))]
            norm = float(np.linalg.norm(chord))
            if norm < 1e-8:
                # Match the runtime's deterministic orientation for stacks.
                normal = np.asarray((0.0, 1.0))
            else:
                normal = np.asarray((-chord[1], chord[0])) / norm
            mask = (times >= objects[0].start_time_ms - 1e-9) & (times <= objects[-1].start_time_ms + 1e-9)
            delta = original_hybrid[mask] - math[mask]
            hybrid[mask] = original_hybrid[mask] + (gain - 1.0) * np.outer(delta @ normal, normal)
            covered[mask] += 1
        if np.any(covered > 2):
            raise ValueError(f"Overlapping accepted windows beyond shared endpoints: {map_md5}")
        change = np.linalg.norm(hybrid - math, axis=1)
        new_trace_path = dest / old_trace_path.name
        np.savez_compressed(new_trace_path, times_ms=times, math_positions=math,
                            hybrid_positions=hybrid, keys=keys)
        summary = dict(old_summary)
        summary["schema_version"] = "ordr-directional-residual-counterfactual-v1"
        summary["trace_sha256"] = sha(new_trace_path)
        summary["adapter_source_sha256"] = sha(Path(__file__))
        summary["source_adapter_sha256"] = old_summary["adapter_source_sha256"]
        summary["source_case_sha256"] = sha(old_summary_path)
        summary["source_trace_sha256"] = old_summary["trace_sha256"]
        summary["lateral_gain"] = gain
        summary["runtime_delivered"] = False
        summary["confirmation_access"] = False
        summary["changed_sample_count"] = int(np.count_nonzero(change > 1e-9))
        summary["position_delta_rms_px"] = float(np.sqrt(np.mean(change ** 2)))
        summary["diagnostics"] = {"counterfactual_only": True,
                                  "accepted_windows_from_source": old_summary["learned_segment_count"]}
        (dest / name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
    report = {
        "schema_version": "ordr-directional-residual-counterfactual-cases-v1",
        "source_case_tag": SOURCE.name,
        "source_report_sha256": sha(SOURCE / "report.json"),
        "generator_sha256": sha(Path(__file__)),
        "case_tag": dest.name,
        "lateral_gain": gain,
        "runtime_delivered": False,
        "confirmation_access": False,
        "cases": len(summaries),
        "learned_segments_from_source": sum(s["learned_segment_count"] for s in summaries),
        "fallback_segments_from_source": sum(s["fallback_segment_count"] for s in summaries),
        "changed_samples": sum(s["changed_sample_count"] for s in summaries),
    }
    (dest / "counterfactual-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
