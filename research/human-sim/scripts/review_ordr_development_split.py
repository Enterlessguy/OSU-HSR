"""Audit stable player/map identities across acquired development partitions."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"
OLD = Path(os.environ.get("HSR_PRIOR_OUTPUT", str(ROOT / "output/prior")))
AUDIT = Path(os.environ.get("HSR_PRIOR_CONFIRMATION_AUDIT", str(ROOT / "output/prior/confirmation-cohort-audit-v1/audit-report.json")))


def old_identity_sets() -> tuple[set[str], set[str], set[str]]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    players = {str(player).removeprefix("source-player-id:") for c in audit["components"] for player in c.get("players", [])}
    families = {str(family) for c in audit["components"] for family in c.get("map_families", [])}
    maps = {str(md5) for c in audit["components"] for md5 in c.get("maps", [])}
    for source in (OLD / "external-osu3k" / "data.csv", OLD / "osu3k-chunk-02" / "dataset" / "data.csv"):
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("userid"):
                    players.add(row["userid"].split(".")[0])
    return players, families, maps


def main() -> None:
    proposal_path = OUT / "split-proposal-v1.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    old_players, old_families, old_maps = old_identity_sets()
    latest = []
    journal_hashes = {}
    for split in ("train", "calibration", "validation"):
        path = OUT / "development-batches" / split / "acquisition.jsonl"
        if not path.exists():
            continue
        rows = {row["replay_md5"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        allowed = {row["replay_md5"] for row in proposal["partitions"][split]}
        if not set(rows).issubset(allowed):
            raise ValueError(f"Journal contains a replay outside the {split} proposal")
        latest.extend(rows.values())
        journal_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()

    by_player: dict[str, list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in latest:
        if row.get("stable_identity_match"):
            by_player[str(row["public_score_evidence"]["player_id"])].append(row)
            by_family[str(row["map_family"])].append(row)
    reviewed = []
    for row in latest:
        reasons = []
        page = row.get("public_score_evidence") or {}
        player_id = str(page.get("player_id") or "")
        if not row.get("file_integrity_pass"):
            reasons.append("file_integrity_failed")
        if not row.get("stable_identity_match"):
            reasons.append("stable_public_score_identity_unverified")
        if row["map_family"] in old_families or row["map_md5"] in old_maps:
            reasons.append("prior_map_exposure")
        if player_id and player_id in old_players:
            reasons.append("prior_player_exposure")
        if player_id and len(by_player[player_id]) > 1:
            reasons.append("stable_player_id_collision")
        if len(by_family[row["map_family"]]) > 1:
            reasons.append("map_family_collision")
        reviewed.append({
            "split": row["split"], "replay_md5": row["replay_md5"],
            "map_md5": row["map_md5"], "map_family": row["map_family"],
            "source_player_id": player_id or None,
            "source_score_id": row.get("score_id"),
            "hosted_replay_flag": page.get("has_replay"),
            "metadata_eligible": not reasons,
            "quarantine_reasons": reasons,
        })
    counts = Counter(row["split"] for row in reviewed if row["metadata_eligible"])
    hosted_counts = Counter(row["split"] for row in reviewed if row["metadata_eligible"] and row["hosted_replay_flag"])
    report = {
        "schema_version": "ordr-development-identity-audit-v1",
        "status": "metadata_review_only_manual_execution_not_certified",
        "source_proposal_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        "journal_sha256": journal_hashes,
        "confirmation_access": False,
        "old_player_ids": len(old_players), "old_map_families": len(old_families),
        "reviewed": len(reviewed),
        "metadata_eligible_by_split": dict(counts),
        "hosted_replay_subset_by_split": dict(hosted_counts),
        "quarantine_reasons": dict(Counter(reason for row in reviewed for reason in row["quarantine_reasons"])),
        "records": reviewed,
    }
    path = OUT / "development-identity-audit-v1.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(f"report={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
