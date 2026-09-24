"""Honest coverage and abstention reporting for Osu Realism Index V1."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


ORI_SPEC_VERSION = "osu-realism-index-v1:1.0"
ORI_MACHINE_ID = "osu-realism-index-v1"
TIERS = ("beginner", "intermediate", "expert", "competitive")
MODES = ("isolated_circle", "stream", "slider", "spinner", "break_free_roam", "join")
DURATION_BINS = ("<50", "50-100", "100-200", "200-500", ">500")


def _empty_cells() -> dict[str, dict[str, int]]:
    return {tier: {mode: 0 for mode in MODES} for tier in TIERS}


def build_realism_coverage_report(
    *,
    source_audit: Mapping[str, Any] | None = None,
    admission_table: Mapping[str, Any] | None = None,
    corpus_report: Mapping[str, Any] | None = None,
    model_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ORI coverage without inventing a score from insufficient data."""
    source_audit = source_audit or {}
    admission_table = admission_table or {}
    corpus_report = corpus_report or {}
    model_report = model_report or {}
    coverage = _empty_cells()
    for key, count in (corpus_report.get("coverage") or {}).items():
        try:
            split, tier = str(key).split("/", 1)
        except ValueError:
            continue
        if tier in coverage:
            # The corpus coverage is human-window coverage; ORI still remains
            # uncalibrated until independent human-vs-human reference pairs exist.
            coverage[tier]["join"] += int(count) if split == "validation" else 0
    blockers = [
        "independent human calibration pairs are absent",
        "all required four-tier human labels are absent from the admitted pilot",
        "required mode and player/map cell minima are not met",
        "flicker equivalence margins and perceptual validation are not calibrated",
    ]
    labelled_windows = admission_table.get("labelled_windows") or {tier: 0 for tier in TIERS}
    labelled_players = admission_table.get("labelled_players") or {tier: 0 for tier in TIERS}
    if any(int(labelled_windows.get(tier, 0)) > 0 for tier in TIERS):
        blockers.remove("all required four-tier human labels are absent from the admitted pilot")
    return {
        "schema_version": 1,
        "spec_version": ORI_SPEC_VERSION,
        "machine_id": ORI_MACHINE_ID,
        "status": "N/A",
        "score": None,
        "headline_claim": "abstain_until_calibrated",
        "blockers": blockers,
        "calibration": {
            "status": "not_calibrated",
            "human_vs_human_pairs": 0,
            "corruption_ladder": "not_frozen",
            "bootstrap_replicates": 0,
        },
        "coverage": {
            "tier_mode_windows": coverage,
            "labelled_windows": {tier: int(labelled_windows.get(tier, 0)) for tier in TIERS},
            "labelled_players": {tier: int(labelled_players.get(tier, 0)) for tier in TIERS},
            "source_rows": int(source_audit.get("rows", 0) or source_audit.get("row_count", 0) or 0),
            "admitted_exploratory_records": int(admission_table.get("exploratory_eligible", 0)),
            "quarantined_records": int(admission_table.get("quarantined", 0)),
            "model_training_status": model_report.get("model_training_status", "unknown"),
        },
        "metric_definitions": {
            "execution": "Matched distributional distances for timing along an already supplied frozen route.",
            "whole_movement": "Separate spatial/route-shape benchmark; not repaired by timing-only peAI.",
            "normalized_phase": "Cumulative arc-length phase in [0,1], compared after identical declared resampling; no completion reward.",
            "completion_hit_context": "Target completion and hit/miss are contextual diagnostics only, never realism rewards.",
            "flicker": "Raw displacement spikes, reversals, stop/restart bursts and bandwidth-controlled tail metrics; equivalence margins require calibration.",
            "missing_data": "Unsupported or uncovered cells are N/A, not zero and not extrapolated.",
        },
        "source": {
            "url": source_audit.get("source"),
            "license": source_audit.get("source_license"),
            "human_provenance": admission_table.get("policy", {}).get("not_claimed", []),
        },
    }


def summarize_candidate_modes(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count observed candidate modes without treating them as human coverage."""
    result: defaultdict[str, int] = defaultdict(int)
    for record in records:
        result[str(record.get("mode", "unknown"))] += 1
    return dict(sorted(result.items()))
