"""Small, explicitly unconditioned execution pilot for reviewed public candidates.

This module is intentionally separate from the four-tier human corpus.  It can
measure and fit a provisional phase law only for candidates with exact map
identity, no automation bits, a decoded replay, and a matching public score
identity.  The source license and score metadata do not certify manual input;
the resulting candidate is exploratory, skill-unknown, and never promotable
through the tiered training path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .execution import FrozenRoute
from .execution_acquisition import write_json
from .execution_benchmark import (
    RuntimeTraceCase,
    run_runtime_trace_paired_benchmark,
    runtime_trace_cases_from_map_plan,
)
from .execution_training import (
    FEATURE_NAMES,
    _fit_linear,
    _extract_phase_label_arrays,
    _phase_metrics,
    feature_vector,
    save_model,
    transition_route,
)
from .io import load_map_plan


MOD_BITS = {
    1: "NF",
    2: "EZ",
    4: "TD",
    8: "HD",
    16: "HR",
    32: "SD",
    64: "DT",
    128: "RX",
    256: "HT",
    512: "NC",
    1024: "FL",
    2048: "AT",
    4096: "SO",
    8192: "AP",
    16384: "PF",
    536870912: "CL",
}
ASSISTANCE_MODS = frozenset({"AT", "RX", "AP", "CL"})
EXPLORATORY_DECISION = "exploratory_candidate_unconditioned_skill_unknown"


def _mod_names(mask: int) -> list[str]:
    return [name for bit, name in MOD_BITS.items() if int(mask) & bit]


def _decoded_by_hash(decoded_audit: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item.get("replay_sha256")): item for item in decoded_audit.get("records", [])}


def _map_family_identity(admission: Mapping[str, Any]) -> tuple[str | None, str]:
    """Return beatmap-set identity and how it was established.

    An exact beatmap hash is not a substitute for a beatmap-set identity: a set
    can contain multiple difficulties. Unknown set identity therefore remains
    explicit and cannot be reported as strict family-separated coverage.
    """
    evidence = admission.get("source_identity_evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    resolved = evidence.get("resolved_beatmapset_id") or admission.get("map_family")
    if resolved:
        kind = "resolved_beatmapset_id" if evidence.get("resolved_beatmapset_id") else "trusted_manifest_map_family"
        return str(resolved), kind
    return None, "unknown_no_beatmapset_identity"


def _map_family(admission: Mapping[str, Any]) -> str | None:
    """Compatibility helper returning only a known set identity."""
    return _map_family_identity(admission)[0]


def _player_identity_key(admission: Mapping[str, Any]) -> str | None:
    """Return a stable grouping key without treating a blank display name as an identity."""
    value = admission.get("player_identity_key")
    if value:
        return str(value)
    evidence = admission.get("source_identity_evidence")
    if isinstance(evidence, Mapping) and evidence.get("player_id") is not None:
        return f"source-player-id:{evidence['player_id']}"
    display = str(admission.get("player_name_archive") or "").strip()
    if display and display.lower() != "guest":
        return display
    return None


def _assign_disjoint_components(
    eligible: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Assign connected player/map-family components to train/validation/test.

    A connected component is indivisible: if a player appears on a map family,
    both entities stay in the same split. This prevents both player leakage and
    beatmap-set leakage, including when the same set has several difficulties.
    """
    nodes: dict[str, set[str]] = defaultdict(set)
    component_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        player = _player_identity_key(row)
        family = _map_family(row)
        if player is None:
            continue
        if family is None:
            continue
        player_node = f"player:{player}"
        family_node = f"map_family:{family}"
        nodes[player_node].add(family_node)
        nodes[family_node].add(player_node)
        component_rows[player_node].append(row)

    unknown_map_family_records = sum(_map_family(row) is None for row in eligible)
    unknown_player_identity_records = sum(_player_identity_key(row) is None for row in eligible)
    if unknown_map_family_records or unknown_player_identity_records:
        return {}, {}, {
            "status": "blocked_unknown_grouping_identity" if unknown_player_identity_records else "blocked_unknown_map_family_identity",
            "component_count": 0,
            "required_components": 3,
            "deficit": 3,
            "unknown_map_family_records": unknown_map_family_records,
            "unknown_player_identity_records": unknown_player_identity_records,
            "strict_family_coverage": "unverified_unknown_records_excluded",
        }

    seen: set[str] = set()
    components: list[dict[str, Any]] = []
    for start in sorted(nodes):
        if start in seen:
            continue
        stack = [start]
        members: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            members.add(node)
            stack.extend(sorted(nodes[node] - seen, reverse=True))
        players = sorted(node.split(":", 1)[1] for node in members if node.startswith("player:"))
        families = sorted(node.split(":", 1)[1] for node in members if node.startswith("map_family:"))
        rows = [row for player in players for row in component_rows[f"player:{player}"]]
        components.append({
            "component_id": hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()[:16],
            "players": players,
            "map_families": families,
            "records": len(rows),
            "usable_movement_windows": int(sum(int(row.get("usable_movement_windows") or 0) for row in rows)),
        })

    if len(components) < 3:
        return {}, {}, {
            "status": "blocked_component_deficit",
            "component_count": len(components),
            "required_components": 3,
            "deficit": max(0, 3 - len(components)),
            "components": components,
            "unknown_map_family_records": unknown_map_family_records,
        }

    ordered = sorted(
        components,
        key=lambda item: (-int(item["usable_movement_windows"]), str(item["component_id"])),
    )
    split_by_player: dict[str, str] = {}
    split_by_family: dict[str, str] = {}
    for index, component in enumerate(ordered):
        split = "test" if index == 0 else "validation" if index == 1 else "train"
        for player in component["players"]:
            split_by_player[player] = split
        for family in component["map_families"]:
            split_by_family[family] = split
        component["split"] = split
    return split_by_player, split_by_family, {
        "status": "complete",
        "component_count": len(components),
        "components": ordered,
    }


def build_admission_table(
    candidate_report: Mapping[str, Any],
    score_evidence: Mapping[str, Any],
    decoded_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the exploratory-only evidence policy row by row."""
    evidence_by_score = {
        str(item.get("score_id")): item
        for item in score_evidence.get("records", [])
        if item.get("score_id") is not None
    }
    decoded_by_hash = _decoded_by_hash(decoded_audit)
    rows: list[dict[str, Any]] = []
    for record in candidate_report.get("records", []):
        header = dict(record.get("replay_header", {}))
        replay_hash = str(header.get("sha256", ""))
        score_id = str(header.get("online_score_id") or "")
        evidence = evidence_by_score.get(score_id) or record.get("source_identity_evidence")
        decoded = decoded_by_hash.get(replay_hash, {})
        raw_mods = int(header.get("mods", 0))
        mods = _mod_names(raw_mods)
        exact_map = bool(record.get("map_md5_matches_header")) and bool(
            decoded.get("decoded_map_md5_matches_candidate")
        )
        official_score_match = "not_available_no_online_score_id"
        stable_identity = "archive_player_name_and_private_hash_only"
        concerns = [
            "public_submission_archive_does_not_certify_manual_input_or_replay_session_provenance",
            "skill_group_unknown_rank_is_recorded_as_evidence_not_a_tier_label",
        ]
        if evidence is not None:
            if evidence.get("error"):
                official_score_match = "score_page_unavailable"
                concerns.append("official_score_page_unavailable")
            elif evidence.get("player_name_matches_archive") is False:
                official_score_match = "score_page_username_mismatch"
                concerns.append("archive_username_differs_from_current_public_username")
            elif evidence.get("beatmap_md5") == header.get("beatmap_md5"):
                official_score_match = "score_page_player_and_beatmap_match"
                stable_identity = "public_player_id_and_username_match_archive_record"
                concerns.append("public_score_metadata_is_supporting_identity_evidence_not_human_certification")
            else:
                official_score_match = "score_page_beatmap_mismatch"
                concerns.append("official_score_page_beatmap_mismatch")
        if evidence is not None and evidence.get("source_dataset_identity_match"):
            if evidence.get("beatmap_md5") == header.get("beatmap_md5"):
                official_score_match = "source_dataset_player_and_beatmap_match"
                stable_identity = "source_dataset_player_id_and_dated_metadata"
                concerns.append("source_dataset_metadata_is_identity_support_not_independent_score_page")
            else:
                official_score_match = "source_dataset_beatmap_mismatch"
                concerns.append("source_dataset_beatmap_mismatch")
        eligible = bool(
            exact_map
            and decoded.get("decoded_frame_count")
            and decoded.get("player_hash_present")
            and not bool(header.get("automation_mods"))
            and official_score_match in {
                "score_page_player_and_beatmap_match",
                "source_dataset_player_and_beatmap_match",
            }
            and evidence is not None
            and (
                evidence.get("player_is_bot") is False
                or evidence.get("source_dataset_identity_match") is True
            )
        )
        decision = EXPLORATORY_DECISION if eligible else "quarantined_unresolved_identity_or_provenance"
        map_family, map_family_identity_kind = _map_family_identity({
            "source_identity_evidence": evidence if isinstance(evidence, Mapping) else {},
            "exact_map_hash": header.get("beatmap_md5"),
        })
        player_identity_key = _player_identity_key({
            "source_identity_evidence": evidence if isinstance(evidence, Mapping) else {},
            "player_name_archive": header.get("player_name"),
        })
        if player_identity_key is None:
            concerns.append("stable_player_grouping_identity_unavailable")
            eligible = False
            decision = "quarantined_unresolved_identity_or_provenance"
        rows.append(
            {
                "replay_sha256": replay_hash,
                "player_name_archive": header.get("player_name"),
                "player_identity_key": player_identity_key,
                "source_url": candidate_report.get("source"),
                "source_license": candidate_report.get("source_license", "unrecorded"),
                "raw_mods": raw_mods,
                "mods": mods,
                "automation_mods": bool(header.get("automation_mods")),
                "exact_map_hash": header.get("beatmap_md5"),
                "exact_map_hash_match": exact_map,
                "decoded_frame_count": decoded.get("decoded_frame_count", 0),
                "usable_movement_windows": decoded.get("usable_movement_windows", 0),
                "player_hash_present": bool(decoded.get("player_hash_present")),
                "online_score_id": header.get("online_score_id"),
                "stable_player_identity_evidence": stable_identity,
                "official_score_match": official_score_match,
                "official_player_id": evidence.get("player_id") if evidence else None,
                "official_player_name": evidence.get("player_name") if evidence else None,
                "official_rank_global": evidence.get("rank_global") if evidence else None,
                "dated_rank_evidence_status": evidence.get("dated_rank_evidence_status") if evidence else None,
                "identity_evidence_kind": evidence.get("identity_evidence_kind") if evidence else None,
                "source_identity_evidence": dict(evidence) if isinstance(evidence, Mapping) else {},
                "map_family": map_family,
                "map_family_identity_kind": map_family_identity_kind,
                "skill_group": None,
                "unresolved_provenance_concerns": concerns,
                "admission_decision": decision,
            }
        )
    return {
        "schema_version": 1,
        "policy": {
            "source_license": "must_be_recorded; CC0 permits intended reuse for this source",
            "required_for_exploratory": [
                "exact replay/map MD5 match",
                "decoded finite replay with stable salted player hash",
                "no raw automation/assistance bits",
                "public score player and beatmap identity match",
                "public score does not mark the account as a bot",
            ],
            "not_claimed": [
                "manual-input certification",
                "four-tier skill label",
                "promotion to the tier-conditioned human model",
            ],
        },
        "source": candidate_report.get("source"),
        "source_license": candidate_report.get("source_license", "unrecorded"),
        "rows": rows,
        "exploratory_eligible": sum(row["admission_decision"] == EXPLORATORY_DECISION for row in rows),
        "quarantined": sum(row["admission_decision"] != EXPLORATORY_DECISION for row in rows),
        "map_family_identity": {
            "by_kind": dict(Counter(str(row["map_family_identity_kind"]) for row in rows)),
            "known_records": sum(row.get("map_family") is not None for row in rows),
            "unknown_records": sum(row.get("map_family") is None for row in rows),
            "strict_family_coverage": "verified_for_known_records" if all(row.get("map_family") is not None for row in rows) else "unverified_unknown_records",
        },
        "labelled_players": {group: 0 for group in ("beginner", "intermediate", "expert", "competitive")},
        "labelled_windows": {group: 0 for group in ("beginner", "intermediate", "expert", "competitive")},
        "promotion_status": "not_promotable_four_tier_gates_unmet",
    }


def _read_decoded(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
        frames = [json.loads(line) for line in stream if line.strip()]
    return header, frames


def _manifest_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _phase_diagnostics(rows: Sequence[Mapping[str, Any]], model: Any) -> dict[str, Any]:
    """Report phase errors in normalized units as well as log-increment units."""
    if not rows:
        return {
            "windows": 0,
            "log_increment_rmse": None,
            "normalized_increment_rmse": None,
            "normalized_cumulative_phase_rmse": None,
            "minimum_jerk_cumulative_phase_rmse": None,
            "completion_mean": None,
            "full_population_baseline": {"windows": 0},
            "delivered_fallback": {"windows": 0, "fallback_share": 0.0, "reason_counts": {}},
        }
    log_errors: list[np.ndarray] = []
    increment_errors: list[np.ndarray] = []
    cumulative_errors: list[np.ndarray] = []
    minimum_jerk_errors: list[np.ndarray] = []
    baseline_log_errors: list[np.ndarray] = []
    baseline_increment_errors: list[np.ndarray] = []
    baseline_cumulative_errors: list[np.ndarray] = []
    completion_errors: list[float] = []
    out_of_distribution_rows = 0
    phase = np.linspace(0.0, 1.0, 32)
    minimum_jerk = np.diff(10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    minimum_jerk = minimum_jerk / np.sum(minimum_jerk)
    for row in rows:
        truth_log = np.asarray(row["y"], dtype=float)
        truth = np.exp(truth_log)
        truth = truth / max(float(np.sum(truth)), 1e-12)
        truth_log = truth_log - np.logaddexp.reduce(truth_log)
        baseline_log = np.log(np.maximum(minimum_jerk, 1e-12))
        baseline_log = baseline_log - np.logaddexp.reduce(baseline_log)
        baseline_log_errors.append(baseline_log - truth_log)
        baseline_increment_errors.append(minimum_jerk - truth)
        baseline_cumulative_errors.append(np.cumsum(minimum_jerk) - np.cumsum(truth))
        try:
            predicted = np.asarray(model.predict_phase_weights_from_features(row["x"], 32), dtype=float)
        except ValueError as error:
            if str(error) != "out_of_distribution_features":
                raise
            out_of_distribution_rows += 1
            continue
        predicted = predicted / max(float(np.sum(predicted)), 1e-12)
        predicted_log = np.log(np.maximum(predicted, 1e-12))
        log_errors.append(predicted_log - truth_log)
        increment_errors.append(predicted - truth)
        cumulative_errors.append(np.cumsum(predicted) - np.cumsum(truth))
        minimum_jerk_errors.append(np.cumsum(minimum_jerk) - np.cumsum(truth))
        completion_errors.append(float(row.get("completion_fraction", 0.0)))
    baseline = {
        "windows": len(rows),
        "log_increment_rmse": float(np.sqrt(np.mean(np.asarray(baseline_log_errors) ** 2))),
        "normalized_increment_rmse": float(np.sqrt(np.mean(np.asarray(baseline_increment_errors) ** 2))),
        "normalized_cumulative_phase_rmse": float(np.sqrt(np.mean(np.asarray(baseline_cumulative_errors) ** 2))),
    }
    return {
        "windows": len(rows),
        "model_evaluated_windows": len(log_errors),
        "out_of_distribution_rows": out_of_distribution_rows,
        "supported_subset": {
            "windows": len(log_errors),
            "metrics_are_not_full_population": True,
        },
        "log_increment_rmse": float(np.sqrt(np.mean(np.asarray(log_errors) ** 2))) if log_errors else None,
        "normalized_increment_rmse": float(np.sqrt(np.mean(np.asarray(increment_errors) ** 2))) if increment_errors else None,
        "normalized_cumulative_phase_rmse": float(np.sqrt(np.mean(np.asarray(cumulative_errors) ** 2))) if cumulative_errors else None,
        "minimum_jerk_cumulative_phase_rmse": float(np.sqrt(np.mean(np.asarray(minimum_jerk_errors) ** 2))) if minimum_jerk_errors else None,
        "completion_mean": float(np.mean(completion_errors)) if completion_errors else None,
        "full_population_baseline": baseline,
        "delivered_fallback": {
            "windows": out_of_distribution_rows,
            "fallback_share": float(out_of_distribution_rows / len(rows)),
            "reason_counts": {"out_of_distribution_features": out_of_distribution_rows} if out_of_distribution_rows else {},
        },
    }


def _benchmark_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    hybrid = report["aggregates"]["hybrid"]
    math = report["aggregates"]["math_only"]
    return {
        "blend": report["blend"],
        "pairs": len(report["pairs"]),
        "fallback_share": hybrid.get("fallback_share"),
        "fallback_reason_counts": hybrid.get("fallback_reason_counts", {}),
        "projection_fraction": hybrid.get("projection_fraction"),
        "math_completion_share": math.get("completion_share"),
        "hybrid_completion_share": hybrid.get("completion_share"),
        "math_speed_p95": math.get("speed_p95"),
        "hybrid_speed_p95": hybrid.get("speed_p95"),
        "math_jerk_p95": math.get("jerk_p95"),
        "hybrid_jerk_p95": hybrid.get("jerk_p95"),
        "all_pairs_shared_frozen_route": report["route_identity"]["all_pairs_shared_frozen_route"],
        "reproducible": report["reproducible"],
        "route_geometry_sources": report.get("route_geometry_sources", []),
        "scope_note": report.get("scope_note"),
    }


def _mark_exploratory_distribution(report: dict[str, Any]) -> dict[str, Any]:
    report["human_execution_distribution"] = {
        "status": "not_available_skill_unknown_exploratory_pilot",
        "reason": "This comparison uses quarantined public candidates and is not a tier distribution or human-certification claim.",
    }
    return report


def train_exploratory_candidate(
    candidate_report: Mapping[str, Any],
    score_evidence: Mapping[str, Any],
    decoded_audit: Mapping[str, Any],
    decoded_root: str | Path,
    map_plan_root: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    max_windows_per_player: int = 100,
    evaluate_sealed_test: bool = False,
    model_version: str = "execution-phase-ridge-v1",
    ridge_alpha: float = 1e-3,
    candidate_machine_id: str | None = None,
    training_source_stage: str | None = None,
    fit_profile: str = "exploratory_v1_default",
    fit_changes: Sequence[str] = (),
) -> dict[str, Any]:
    """Fit a skill-unknown candidate with disjoint player/map-family holdouts.

    The default routine uses train and validation only. The final test partition
    remains sealed and is not read for selection or benchmarking unless an
    explicit release-milestone caller opts in with ``evaluate_sealed_test``.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("exploratory candidate directories are immutable")
    output.mkdir(parents=True, exist_ok=True)
    table = build_admission_table(candidate_report, score_evidence, decoded_audit)
    eligible = [row for row in table["rows"] if row["admission_decision"] == EXPLORATORY_DECISION]
    if len(eligible) < 3:
        raise ValueError("exploratory candidate needs at least three eligible player groups")
    split_by_player, split_by_family, component_ledger = _assign_disjoint_components(eligible)
    if component_ledger["status"] != "complete":
        raise ValueError(json.dumps({"reason": "player_map_family_component_deficit", **component_ledger}, sort_keys=True))
    split_ledger = {
        "players": split_by_player,
        "map_families": split_by_family,
        "components": component_ledger["components"],
    }
    decoded_root = Path(decoded_root)
    map_plan_root = Path(map_plan_root)
    rows: list[dict[str, Any]] = []
    route_by_id: dict[str, FrozenRoute] = {}
    rejected: Counter[str] = Counter()
    used_by_player: Counter[str] = Counter()
    seen_route_keys: set[tuple[str, str, str]] = set()
    candidate_by_hash = {str(record["replay_sha256"]): record for record in eligible}
    for replay_hash, admission in candidate_by_hash.items():
        player = str(admission["player_identity_key"])
        map_family = str(admission["map_family"])
        decoded_path = decoded_root / f"{replay_hash}.ndjson"
        map_plan = load_map_plan(map_plan_root / f"{admission['exact_map_hash']}.map.ndjson.gz")
        _, frames = _read_decoded(decoded_path)
        times = np.asarray([float(frame["time_ms"]) for frame in frames], dtype=float)
        positions = np.asarray([[float(frame["x"]), float(frame["y"])] for frame in frames], dtype=float)
        for index in range(1, len(map_plan.objects)):
            if used_by_player[player] >= max_windows_per_player:
                break
            route_key = (player, map_family, f"{replay_hash}:{index}")
            if route_key in seen_route_keys:
                rejected["duplicate_route_window"] += 1
                continue
            try:
                route = transition_route(
                    map_plan,
                    index,
                    skill_group="intermediate",
                    record_id=f"exploratory:{replay_hash}:{index}",
                )
                label = _extract_phase_label_arrays(route, times, positions)
            except ValueError as error:
                rejected[str(error)] += 1
                continue
            seen_route_keys.add(route_key)
            split = split_by_player[player]
            row = {
                "x": feature_vector(route, "intermediate"),
                "y": np.asarray(label["log_phase_increments"], dtype=float),
                "completion_fraction": float(label["completion_fraction"]),
                "cross_track_rmse_px": float(label["cross_track_rmse_px"]),
                "source_gap_fraction": float(label["source_gap_fraction"]),
                "source_cadence_median_ms": float(label["source_cadence_median_ms"]),
                "split": split,
                "skill_group": None,
                "player": player,
                "player_identity_key": player,
                "player_display_name": admission.get("player_name_archive"),
                "map": admission["exact_map_hash"],
                "map_family": map_family,
                "route_id": route.route_id,
                "duration_ms": route.duration_ms,
                "distance_px": route.length_px,
                "mode": route.mode,
                "replay_sha256": replay_hash,
                "source_window_index": index,
                "route": route,
            }
            rows.append(row)
            route_by_id[route.route_id] = route
            used_by_player[player] += 1
    if not rows:
        raise ValueError("no exploratory windows survived candidate review")
    manifest = {
        "schema_version": 1,
        "kind": "execution_exploratory_unconditioned_candidate",
        "source": candidate_report.get("source"),
        "source_license": candidate_report.get("source_license", "unrecorded"),
        "skill_group": None,
        "player_map_split_ledger": split_ledger,
        "max_windows_per_player": max_windows_per_player,
        "candidate_machine_id": candidate_machine_id,
        "model_version": model_version,
        "training_source_stage": training_source_stage,
        "fit_profile": fit_profile,
        "fit_changes": list(fit_changes),
        "records": [
            {
                "replay_sha256": row["replay_sha256"],
                "player": row["player_identity_key"],
                "player_display_name": row.get("player_display_name"),
                "map": row["exact_map_hash"],
                "map_family": row["map_family"],
                "split": split_by_player[row["player_identity_key"]],
                "admission": EXPLORATORY_DECISION,
            }
            for row in eligible
        ],
        "promotion_status": "not_promotable_four_tier_gates_unmet",
    }
    manifest_hash = _manifest_sha256(manifest)
    (output / "exploratory-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model = _fit_linear(
        [row for row in rows if row["split"] == "train"],
        seed,
        manifest_hash,
        model_version=model_version,
        ridge_alpha=ridge_alpha,
    )
    model_hashes = save_model(model, output / "model.json")
    print(
        f"[training] fresh_fit_complete model={model.model_version} model_sha256={model.model_sha256} "
        f"manifest_sha256={manifest_hash} fitted_records={len(rows)}",
        flush=True,
    )
    legacy_log_metrics = {
        split: _phase_metrics([row for row in rows if row["split"] == split], model)
        for split in ("train", "validation")
    }
    metrics = {
        split: _phase_diagnostics([row for row in rows if row["split"] == split], model)
        for split in ("train", "validation")
    }
    print("[validation] phase_diagnostics_complete", flush=True)
    skill_coefficient_max = max(
        (abs(float(coefficient)) for row in model.coefficients for coefficient in row[1 + 8 :]),
        default=0.0,
    )

    def _cases_for_split(split: str, limit: int | None = None) -> tuple[RuntimeTraceCase, ...]:
        """Build benchmark cases from current runtime planner samples.

        The phase labels above intentionally use a strict two-circle route. The
        runtime benchmark is separate and freezes one complete map trace emitted
        by the current planner, including timestamps, key states, and explicit
        object-mode boundaries. Unsupported segments remain fallbacks.
        """
        map_hashes = []
        seen_maps: set[str] = set()
        for row in rows:
            if row["split"] != split or row["map"] in seen_maps:
                continue
            seen_maps.add(row["map"])
            map_hashes.append(str(row["map"]))
        cases: list[PairedRouteCase] = []
        for map_hash in map_hashes:
            plan = load_map_plan(map_plan_root / f"{map_hash}.map.ndjson.gz")
            cases.extend(
                runtime_trace_cases_from_map_plan(
                    plan,
                    skill_group="intermediate",
                    effort_level=100.0,
                    seed=seed,
                    sample_rate_hz=500,
                )
            )
            if limit is not None and len(cases) >= limit:
                break
        if limit is not None:
            cases = cases[:limit]
        return tuple(cases)

    validation_cases = _cases_for_split("validation", limit=None)
    test_cases = _cases_for_split("test", limit=None) if evaluate_sealed_test else ()
    if not validation_cases:
        raise ValueError("exploratory candidate needs a non-empty validation route group")
    candidate_blends = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
    validation_ablation: list[dict[str, Any]] = []
    math_trace_cache: dict[str, dict[str, Any]] = {}
    ablation_progress_path = output / "validation-ablation-progress.json"
    write_json(ablation_progress_path, {
        "schema_version": 1,
        "status": "in_progress",
        "candidate_blends": list(candidate_blends),
        "completed_blends": [],
        "validation_cases": len(validation_cases),
        "coverage_policy": "all validation cases retained; only non-selection tail diagnostics omitted",
    })
    for blend in candidate_blends:
        ablation_report = _mark_exploratory_distribution(
            run_runtime_trace_paired_benchmark(
                validation_cases,
                model=model,
                blend=blend,
                sample_rate_hz=500,
                math_trace_cache=math_trace_cache,
                verify_reproducibility=False,
                include_tail_diagnostics=False,
                progress_label=f"validation-ablation blend={blend}",
                progress_path=output / f"validation-ablation-blend-{str(blend).replace('.', '_')}.json",
            )
        )
        summary = _benchmark_summary(ablation_report)
        validation_ablation.append(summary)
        write_json(ablation_progress_path, {
            "schema_version": 1,
            "status": "in_progress",
            "candidate_blends": list(candidate_blends),
            "completed_blends": [item["blend"] for item in validation_ablation],
            "latest_summary": summary,
            "validation_cases": len(validation_cases),
            "coverage_policy": "all validation cases retained; only non-selection tail diagnostics omitted",
        })
    write_json(ablation_progress_path, {
        "schema_version": 1,
        "status": "complete",
        "candidate_blends": list(candidate_blends),
        "completed_blends": [item["blend"] for item in validation_ablation],
        "summaries": validation_ablation,
        "validation_cases": len(validation_cases),
        "coverage_policy": "all validation cases retained; only non-selection tail diagnostics omitted",
    })
    print(f"[validation] blend_ablation_complete cases={len(validation_cases)} blends={len(validation_ablation)}", flush=True)

    def _selection_key(summary: Mapping[str, Any]) -> tuple[float, float, float]:
        jerk = summary.get("hybrid_jerk_p95") or {}
        speed = summary.get("hybrid_speed_p95") or {}
        return (
            float(summary.get("fallback_share") if summary.get("fallback_share") is not None else 1.0),
            float(jerk.get("mean") if jerk.get("mean") is not None else float("inf")),
            float(speed.get("mean") if speed.get("mean") is not None else float("inf")),
        )

    selected_summary = min(validation_ablation, key=_selection_key)
    selected_blend = float(selected_summary["blend"])
    selection_reason = (
        "Validation-only selection lexicographically minimizes fallback share, then hybrid jerk p95 mean, "
        "then hybrid speed p95 mean. Completion is reported but is not an optimization objective; "
        "no verified tiered human distribution or calibrated flicker gate is available for this exploratory candidate."
    )
    validation_selection = {
        "candidate_blends": list(candidate_blends),
        "ablation_summaries": validation_ablation,
        "selected_blend": selected_blend,
        "selected_summary": selected_summary,
        "selection_objective": "fallback_share, hybrid_jerk_p95_mean, hybrid_speed_p95_mean",
        "selection_reason": selection_reason,
        "validation_cases": len(validation_cases),
        "human_distribution_status": "insufficient_evidence_no_verified_tiered_human_distribution",
        "flicker_gate": "not_available_no_calibrated_human_reference",
    }
    validation_map_inventory = sorted({str(row["map"]) for row in rows if row["split"] == "validation"})
    validation_case_maps = sorted({str(case.map_id) for case in validation_cases})
    validation_family_inventory = sorted({str(row["map_family"]) for row in rows if row["split"] == "validation"})
    validation_case_map_families = sorted({
        str(row["map_family"])
        for row in rows
        if row["split"] == "validation" and str(row["map"]) in set(validation_case_maps)
    })
    runtime_validation_coverage = {
        "validation_case_count": len(validation_cases),
        "validation_map_count": len(validation_map_inventory),
        "validation_case_map_count": len(validation_case_maps),
        "validation_map_hashes_omitted": sorted(set(validation_map_inventory) - set(validation_case_maps)),
        "validation_map_family_count": len(validation_family_inventory),
        "validation_case_map_family_count": len(validation_case_map_families),
        "validation_map_families_omitted": sorted(set(validation_family_inventory) - set(validation_case_map_families)),
        "coverage_policy": "all fitted validation map hashes are converted to runtime planner cases; no case-count cap",
    }
    print(
        f"[validation] runtime_case_inventory maps={len(validation_map_inventory)} cases={len(validation_cases)} "
        f"families={len(validation_family_inventory)} omitted_maps={len(runtime_validation_coverage['validation_map_hashes_omitted'])}",
        flush=True,
    )

    benchmark_scope = "sealed_test_release_milestone" if evaluate_sealed_test else "validation_only_exploratory"
    benchmark = _mark_exploratory_distribution(
        run_runtime_trace_paired_benchmark(
            test_cases if evaluate_sealed_test else validation_cases,
            model=model,
            blend=selected_blend,
            sample_rate_hz=500,
            math_trace_cache=math_trace_cache,
            verify_reproducibility=True,
            include_tail_diagnostics=True,
            progress_label="validation-selected",
            progress_path=output / "validation-selected-progress.json",
            export_dir=output / ("paired-test-routes-selected" if evaluate_sealed_test else "paired-validation-routes-selected"),
        )
    )
    unregularized_benchmark = _mark_exploratory_distribution(
        run_runtime_trace_paired_benchmark(
            test_cases if evaluate_sealed_test else validation_cases,
            model=model,
            blend=1.0,
            sample_rate_hz=500,
            math_trace_cache=math_trace_cache,
            verify_reproducibility=True,
            include_tail_diagnostics=True,
            progress_label="validation-unregularized",
            progress_path=output / "validation-unregularized-progress.json",
            export_dir=output / ("paired-test-routes-unregularized" if evaluate_sealed_test else "paired-validation-routes-unregularized"),
        )
    )
    print("[validation] selected_and_unregularized_benchmarks_complete", flush=True)
    benchmark_filename = "paired-test-benchmark.json" if evaluate_sealed_test else "paired-validation-benchmark.json"
    unregularized_filename = "paired-test-benchmark-unregularized.json" if evaluate_sealed_test else "paired-validation-benchmark-unregularized.json"
    write_json(output / benchmark_filename, benchmark)
    write_json(output / unregularized_filename, unregularized_benchmark)
    report = {
        "schema_version": 1,
        "status": "exploratory_not_promoted",
        "model_training_status": "exploratory_unconditioned_skill_unknown",
        "promotion_status": "not_promotable_four_tier_gates_unmet",
        "model_version": model.model_version,
        "candidate_machine_id": candidate_machine_id,
        "fit_profile": fit_profile,
        "ridge_alpha": ridge_alpha,
        "training_source_stage": training_source_stage,
        "fit_changes": list(fit_changes),
        "fresh_fit": True,
        "historical_model_reused": False,
        "model_sha256": model.model_sha256,
        "model_file_sha256": model_hashes["file_sha256"],
        "training_manifest_sha256": manifest_hash,
        "training_manifest_file_sha256": hashlib.sha256(
            (output / "exploratory-manifest.json").read_bytes()
        ).hexdigest(),
        "training_manifest_hash_semantics": "training_manifest_sha256 is the canonical compact manifest-object hash; training_manifest_file_sha256 hashes the pretty-printed artifact bytes",
        "feature_names": list(FEATURE_NAMES),
        "skill_feature_policy": "neutral_placeholder_only; no observed skill labels; skill coefficients disabled by constant-feature fit",
        "skill_feature_coefficient_max_abs": skill_coefficient_max,
        "seed": seed,
        "source_license": candidate_report.get("source_license", "unrecorded"),
        "player_grouping_policy": "stable_source_player_id_when_available; display name retained separately; blank or Guest display names are not grouping keys",
        "eligible_identity_coverage": {
            "records": len(eligible),
            "unique_player_identity_keys": len({str(row["player_identity_key"]) for row in eligible}),
            "blank_display_name_records": sum(not str(row.get("player_name_archive") or "").strip() for row in eligible),
            "guest_display_name_records": sum(str(row.get("player_name_archive") or "").strip().lower() == "guest" for row in eligible),
            "unknown_player_identity_records": sum(row.get("player_identity_key") is None for row in eligible),
            "unknown_map_family_records": sum(row.get("map_family") is None for row in eligible),
        },
        "candidate_admission_table": table,
        "split_ledger": split_ledger,
        "rows": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
        "players": {split: sorted({row["player"] for row in rows if row["split"] == split}) for split in ("train", "validation", "test")},
        "maps": {split: sorted({row["map"] for row in rows if row["split"] == split}) for split in ("train", "validation", "test")},
        "map_families": {split: sorted({row["map_family"] for row in rows if row["split"] == split}) for split in ("train", "validation", "test")},
        "split_integrity": {
            "player_disjoint": all(
                not (
                    {row["player"] for row in rows if row["split"] == left}
                    & {row["player"] for row in rows if row["split"] == right}
                )
                for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
            ),
            "map_family_disjoint": all(
                not (
                    {row["map_family"] for row in rows if row["split"] == left}
                    & {row["map_family"] for row in rows if row["split"] == right}
                )
                for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
            ),
            "component_assignment": component_ledger["status"],
        },
        "metrics": metrics,
        "legacy_log_metrics": legacy_log_metrics,
        "rejected_during_extraction": dict(rejected),
        "validation_blend_selection": validation_selection,
        "runtime_validation_coverage": runtime_validation_coverage,
        "benchmark_scope": benchmark_scope,
        "sealed_test": {
            "status": "evaluated_at_explicit_release_milestone" if evaluate_sealed_test else "sealed_not_evaluated",
            "rows_available": sum(row["split"] == "test" for row in rows),
            "routes_available": len(test_cases),
        },
        "paired_benchmark": {
            "path": str(output / benchmark_filename),
            **_benchmark_summary(benchmark),
        },
        "paired_benchmark_unregularized": {
            "path": str(output / unregularized_filename),
            **_benchmark_summary(unregularized_benchmark),
        },
        "limitations": [
            "Skill is unknown and deliberately excluded from the exploratory feature interpretation.",
            "Public score metadata and non-bot flags do not certify manual input.",
            "This candidate cannot satisfy the four-tier training or promotion gates.",
            "Player and map-family connected components are kept wholly within one split.",
            "The final test partition is sealed by default and was not read for selection or benchmarking.",
            "No verified tiered human distribution or calibrated flicker gate was available, so realism improvement is unproven.",
            "Use the artifact only to test the execution-only phase mechanism while collecting reviewed cohorts.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    (output / "candidate.lock").write_text(
        json.dumps({"immutable": True, "model_sha256": model.model_sha256}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")
