"""Create development-only residual scale counterfactuals from paired traces.

This never reads confirmation data and does not claim runtime delivery. A chosen
scale must be implemented in the adapter and independently rerun before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "output/external-ordr-v155/development-cases/validation"
SOURCE_TAG = "math-residual-trained-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    scale = args.scale
    if not np.isfinite(scale) or not 0.0 < scale < 1.0:
        raise ValueError("counterfactual scale must be strictly between zero and one")
    scale_label = f"{scale:.2f}".replace(".", "p")
    if float(scale_label.replace("p", ".")) != scale:
        raise ValueError("scale must have at most two decimal places")
    source = CASE_ROOT / SOURCE_TAG
    destination = CASE_ROOT / f"math-residual-scale-{scale_label}-counterfactual"
    destination.mkdir(parents=True, exist_ok=True)
    source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if source_report["cases"] != 43 or source_report["confirmation_access"]:
        raise ValueError("Expected complete opened-validation source only")
    summaries = []
    for name in source_report["case_summaries"]:
        source_summary_path = source / name
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        source_trace_path = source / name.replace(".json", ".npz")
        if source_summary["trace_sha256"] != sha(source_trace_path):
            raise ValueError(f"Source trace hash mismatch: {name}")
        with np.load(source_trace_path) as arrays:
            times = arrays["times_ms"]
            math = arrays["math_positions"]
            original_hybrid = arrays["hybrid_positions"]
            keys = arrays["keys"]
        hybrid = math + scale * (original_hybrid - math)
        trace_path = destination / source_trace_path.name
        np.savez_compressed(trace_path, times_ms=times, math_positions=math,
                            hybrid_positions=hybrid, keys=keys)
        changed = np.linalg.norm(hybrid - math, axis=1)
        summary = dict(source_summary)
        summary["schema_version"] = "ordr-coherent-scale-counterfactual-v1"
        summary["trace_sha256"] = sha(trace_path)
        summary["adapter_source_sha256"] = sha(Path(__file__))
        summary["source_adapter_sha256"] = source_summary["adapter_source_sha256"]
        summary["source_case_sha256"] = sha(source_summary_path)
        summary["source_trace_sha256"] = source_summary["trace_sha256"]
        summary["posttraining_scale"] = scale
        summary["runtime_delivered"] = False
        summary["confirmation_access"] = False
        summary["changed_sample_count"] = int(np.count_nonzero(changed > 1e-9))
        summary["position_delta_rms_px"] = float(np.sqrt(np.mean(changed ** 2)))
        summary["diagnostics"] = {
            "counterfactual_only": True,
            "source_diagnostics_sha256": hashlib.sha256(
                json.dumps(source_summary["diagnostics"], sort_keys=True).encode()
            ).hexdigest(),
        }
        (destination / name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
    report = {
        "schema_version": "ordr-coherent-scale-counterfactual-cases-v1",
        "source_case_tag": SOURCE_TAG,
        "source_report_sha256": sha(source / "report.json"),
        "generator_sha256": sha(Path(__file__)),
        "case_tag": destination.name,
        "posttraining_scale": scale,
        "runtime_delivered": False,
        "confirmation_access": False,
        "cases": len(summaries),
        "maps_with_learned_exposure": sum(s["learned_segment_count"] > 0 for s in summaries),
        "learned_segments_from_source": sum(s["learned_segment_count"] for s in summaries),
        "fallback_segments_from_source": sum(s["fallback_segment_count"] for s in summaries),
        "changed_samples": sum(s["changed_sample_count"] for s in summaries),
        "case_summaries": source_report["case_summaries"],
    }
    (destination / "counterfactual-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "case_summaries"}, indent=2))


if __name__ == "__main__":
    main()
