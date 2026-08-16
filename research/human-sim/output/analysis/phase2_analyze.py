from __future__ import annotations

"""Phase-2 error-mix analyzer.

Runs the planner over a benchmark map set at several skill/effort profiles and
reports how misses and errors are distributed across the aim and timing
systems, plus context-split timing behaviour, early/late bias, unstable rate,
and aim-timing correlation.

Usage:
    python phase2_analyze.py [--version LABEL]
"""

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # research/human-sim
sys.path.insert(0, str(ROOT / "src"))

from human_sim.context import build_contexts  # noqa: E402
from human_sim.io import load_map_plan  # noqa: E402
from human_sim.matching import match_press_events, rising_press_events  # noqa: E402
from human_sim.planner import PLANNER_VERSION, HumanTracePlanner  # noqa: E402
from human_sim.schemas import HumanProfile  # noqa: E402

MAPS = {
    "kingborn-nm-od10": ROOT.parents[1] / "timing-tests" / "benchmark" / "kingborn-nm.map.ndjson.gz",
    "crystalia-lum-od10": ROOT / "output" / "auto" / "6147f69819fb-NM-r1-p99.5-s42-sk10-ef40-1000hz-profile.map.ndjson.gz",
    "heavenly-hrdt-od10": ROOT.parents[1] / "timing-tests" / "benchmark" / "heavenly-hrdt.map.ndjson.gz",
    "haiboku-hard-od6": ROOT / "output" / "auto" / "5418e5b6e63d-NM-r1-p99.5-s42-sk50-ef80-1000hz-profile.map.ndjson.gz",
    "centipede-od8": ROOT.parents[1] / "timing-tests" / "benchmark" / "centipede-generated.map.ndjson.gz",
}

PROFILES = [
    (10, 40),
    (20, 70),
    (50, 68),
    (50, 80),
    (80, 68),
    (99, 40),
]


def analyze_run(map_plan, frames, skill, effort, ghost_flags):
    frame_times = np.array([f.time_us / 1000.0 for f in frames])
    frame_xs = np.array([f.x for f in frames])
    frame_ys = np.array([f.y for f in frames])

    def position_at_time(time_ms):
        index = int(np.searchsorted(frame_times, time_ms))
        index = min(max(index, 0), len(frame_times) - 1)
        return float(frame_xs[index]), float(frame_ys[index])

    timeline_start = map_plan.objects[0].start_time_ms - 1500.0
    events = rising_press_events(frames, timeline_start_ms=timeline_start)
    press_times_absolute = np.array([event.time_ms for event in events])

    # Global greedy press matching: assign each press to the object it fits
    # best (smallest |timing error|), each press used once and each object
    # once. The old per-object nearest search cascaded - one missing press in
    # a dense section made neighbours consume each other's presses and
    # inflated the miss count.
    assigned_press = match_press_events(map_plan.objects, events, minimum_window_ms=180.0)
    contexts = build_contexts(map_plan)

    rows = []
    for object_position, obj in enumerate(map_plan.objects):
        ghost = bool(ghost_flags[object_position]) if object_position < len(ghost_flags) else False
        if obj.kind == "spinner":
            continue
        start = obj.start_time_ms
        radius = obj.radius
        windows = obj.hit_windows
        context = contexts[object_position].label

        if object_position not in assigned_press:
            rows.append(
                {
                    "context": context, "kind": obj.kind, "hit": False,
                    "timing_error": None, "aim_error": None,
                    "aim_miss": True, "timing_miss": True, "judgement": "miss",
                    "ghost": ghost, "attribution": "timing",
                }
            )
            continue
        best = assigned_press[object_position]
        press_relative_ms = float(press_times_absolute[best] - timeline_start)
        timing_error = float(press_times_absolute[best] - start)
        px, py = position_at_time(press_relative_ms)
        aim_error = math.hypot(px - obj.position.x, py - obj.position.y)

        aim_miss = aim_error > radius
        timing_miss = abs(timing_error) > windows.meh_ms
        if aim_miss or timing_miss:
            judgement = "miss"
        elif abs(timing_error) <= windows.great_ms:
            judgement = "300"
        elif abs(timing_error) <= windows.ok_ms:
            judgement = "100"
        else:
            judgement = "50"
        # A ghost press (tap fired before the cursor was free to move) is only
        # timing-induced when it actually missed; if the cursor was already
        # inside the circle (nearby objects), it is a real, early hit.
        if ghost and aim_miss:
            attribution = "timing"
        elif aim_miss and timing_miss:
            attribution = "both"
        elif aim_miss:
            attribution = "aim"
        elif timing_miss:
            attribution = "timing"
        else:
            attribution = "hit"
        rows.append(
            {
                "context": context, "kind": obj.kind, "hit": not (aim_miss or timing_miss),
                "timing_error": timing_error, "aim_error": aim_error,
                "aim_miss": aim_miss, "timing_miss": timing_miss, "judgement": judgement,
                "ghost": ghost, "attribution": attribution,
            }
        )

    judgements = [r["judgement"] for r in rows]
    counts = {j: judgements.count(j) for j in ("300", "100", "50", "miss")}
    weights = {"300": 1.0, "100": 1.0 / 3.0, "50": 1.0 / 6.0, "miss": 0.0}
    accuracy = 100.0 * sum(weights[j] for j in judgements) / len(judgements)

    timing = np.array([r["timing_error"] for r in rows if r["timing_error"] is not None], dtype=float)
    aim = np.array([r["aim_error"] for r in rows if r["aim_error"] is not None], dtype=float)
    hits = [r for r in rows if r["hit"]]

    def pct(values, p):
        return round(float(np.percentile(values, p)), 2) if len(values) else 0.0

    aim_only = sum(1 for r in rows if r["attribution"] == "aim")
    timing_only = sum(1 for r in rows if r["attribution"] == "timing")
    both = sum(1 for r in rows if r["attribution"] == "both")
    ghost = sum(1 for r in rows if r["attribution"] == "timing" and r["ghost"])
    total_misses = aim_only + timing_only + both + ghost
    timing_attributable = timing_only + both + ghost

    # Context split over pressed objects (no press -> miss is not a timing sample).
    contexts = {}
    for ctx in ("stream", "burst", "jump", "transition", "slider"):
        values = [r["timing_error"] for r in rows if r["context"] == ctx and r["timing_error"] is not None]
        ctx_rows = [r for r in rows if r["context"] == ctx]
        contexts[ctx] = {
            "n": len(ctx_rows),
            "timing_std_ms": round(float(np.std(values)), 2) if values else 0.0,
            "timing_abs_mean_ms": round(float(np.mean(np.abs(values))), 2) if values else 0.0,
            "timing_miss_pct": round(100.0 * sum(1 for r in ctx_rows if r["timing_miss"]) / max(1, len(ctx_rows)), 1),
            "aim_miss_pct": round(100.0 * sum(1 for r in ctx_rows if r["aim_miss"]) / max(1, len(ctx_rows)), 1),
        }

    # Aim/timing correlation among judged hits (aim error vs |timing error|).
    paired_aim = np.array([r["aim_error"] for r in hits], dtype=float)
    paired_timing = np.array([abs(r["timing_error"]) for r in hits], dtype=float)
    if len(paired_aim) > 2 and float(np.std(paired_aim)) > 0 and float(np.std(paired_timing)) > 0:
        correlation = float(np.corrcoef(paired_aim, paired_timing)[0, 1])
    else:
        correlation = 0.0

    # Section-level correlation: split the map into short windows and correlate
    # mean |timing error| with mean aim error per window. This is the "hard
    # sections degrade both systems together" signal that per-object noise
    # hides, and is the behaviour the shared pressure state is meant to create.
    window = 20
    section_aim = []
    section_timing = []
    for offset in range(0, len(rows), window):
        chunk = rows[offset : offset + window]
        pressed = [r for r in chunk if r["aim_error"] is not None and r["timing_error"] is not None]
        if len(pressed) >= 8:
            section_aim.append(float(np.mean([r["aim_error"] for r in pressed])))
            section_timing.append(float(np.mean([abs(r["timing_error"]) for r in pressed])))
    if len(section_aim) > 2 and float(np.std(section_aim)) > 0 and float(np.std(section_timing)) > 0:
        section_correlation = float(np.corrcoef(section_aim, section_timing)[0, 1])
    else:
        section_correlation = 0.0

    # Early/late bias: mean signed error and split averages (mimics the in-game
    # hit-error meter which averages early and late hits separately).
    early = timing[timing < 0]
    late = timing[timing > 0]

    return {
        "accuracy_pct": round(accuracy, 2),
        "judgements": counts,
        "miss_split": {
            "aim_only": aim_only,
            "timing_only": timing_only,
            "both": both,
            "ghost": ghost,
            "total": total_misses,
            "aim_share_pct": round(100.0 * (aim_only + both) / max(1, total_misses), 1),
            "timing_share_pct": round(100.0 * timing_attributable / max(1, total_misses), 1),
            "timing_attributable": timing_attributable,
        },
        "timing": {
            "ur": round(float(np.std(timing)) * 10.0, 1) if len(timing) else 0.0,
            "mean_signed_ms": round(float(np.mean(timing)), 2) if len(timing) else 0.0,
            "mean_abs_ms": round(float(np.mean(np.abs(timing))), 2) if len(timing) else 0.0,
            "p95_abs_ms": pct(np.abs(timing), 95),
            "early_mean_ms": round(float(np.mean(early)), 2) if len(early) else 0.0,
            "late_mean_ms": round(float(np.mean(late)), 2) if len(late) else 0.0,
            "early_share_pct": round(100.0 * len(early) / max(1, len(timing)), 1),
        },
        "aim": {
            "mean_error_px": round(float(np.mean(aim)), 2) if len(aim) else 0.0,
            "p95_error_px": pct(aim, 95),
            "miss_pct": round(100.0 * (aim_only + both) / max(1, len(rows)), 2),
        },
        "contexts": contexts,
        "aim_timing_corr": round(correlation, 3),
        "section_aim_timing_corr": round(section_correlation, 3),
        "n_objects": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=PLANNER_VERSION)
    args = parser.parse_args()

    results = []
    for map_name, map_path in MAPS.items():
        if not map_path.exists():
            print(f"skip {map_name}: {map_path} missing", file=sys.stderr)
            continue
        plan = load_map_plan(map_path)
        for skill, effort in PROFILES:
            profile = HumanProfile(
                percentile=99.5,
                seed=42,
                sample_rate_hz=500,
                perfect_baseline=False,
                skill_level=float(skill),
                effort_level=float(effort),
            )
            planner = HumanTracePlanner(plan, profile)
            frames = planner.generate()
            estimate = analyze_run(plan, frames, skill, effort, planner.ghost_flags)
            estimate["map"] = map_name
            estimate["skill"] = skill
            estimate["effort"] = effort
            estimate["strength_mean"] = planner.strength_stats.get("mean", 0.0)
            # Planner-internal timing attribution: pure mistaps (error beyond
            # the MEH window) plus ghost presses (press fired before the cursor
            # could physically be on the target - a timing-induced miss that
            # position-based attribution would otherwise count as an aim miss).
            estimate["planner_timing_attributable"] = (
                planner.timing_stats.get("mistaps", 0) + planner.timing_stats.get("ghost_presses", 0)
            )
            results.append(estimate)

    out = ROOT / "output" / "analysis" / f"phase2-benchmark-{args.version}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    header = (
        f"# Phase 2 error-mix benchmark ({args.version})\n\n"
        f"| map | sk/ef | acc% | UR | bias ms | aim-only | tmg-only | both | ghost | t-attrib | corr | sec-corr |\n"
        f"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for r in results:
        lines.append(
            f"| {r['map']} | {r['skill']}/{r['effort']} | {r['accuracy_pct']} | {r['timing']['ur']} | "
            f"{r['timing']['mean_signed_ms']} | {r['miss_split']['aim_only']} | {r['miss_split']['timing_only']} | "
            f"{r['miss_split']['both']} | {r['miss_split']['ghost']} | {r['miss_split']['timing_attributable']} | "
            f"{r['aim_timing_corr']} | {r['section_aim_timing_corr']} |"
        )
    lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
