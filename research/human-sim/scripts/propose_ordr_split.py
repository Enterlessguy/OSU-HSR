"""Propose disjoint research cohorts from metadata; confirmation remains unopened."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"
SEED = "ordr-v155-metadata-split-20260924-v1"
SPLITS = {"train": 400, "calibration": 100, "validation": 100, "confirmation": 100, "reserve": 300}


def key(row: dict) -> str:
    return hashlib.sha256(f"{SEED}:{row['replay_md5']}".encode()).hexdigest()


def main() -> None:
    source = OUT / "metadata-audit-v1.json"
    audit = json.loads(source.read_text(encoding="utf-8"))
    selected = []
    players: set[str] = set()
    families: set[str] = set()
    replays: set[str] = set()
    for row in sorted(audit["candidate_pool"], key=key):
        player = row["player_name"].casefold()
        family = row["map_family"]
        replay = row["replay_md5"]
        if player in players or family in families or replay in replays:
            continue
        selected.append(row)
        players.add(player)
        families.add(family)
        replays.add(replay)
        if len(selected) == sum(SPLITS.values()):
            break
    if len(selected) != sum(SPLITS.values()):
        raise ValueError("Not enough disjoint metadata rows for the fixed split proposal")

    partitions: dict[str, list[dict]] = {}
    cursor = 0
    for name, size in SPLITS.items():
        partitions[name] = selected[cursor:cursor + size]
        cursor += size
    assert len(players) == len(families) == len(replays) == len(selected)
    report = {
        "schema_version": "ordr-split-proposal-v1",
        "status": "provisional_metadata_only",
        "source_audit_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection_seed": SEED,
        "selection_rule": "ascending SHA256(seed:replay_md5), greedily one casefolded player name and map set per row",
        "attrition_rule": "quarantine identity/provenance/MD5 failures; take next disjoint reserve in written order without inspecting movement outcomes",
        "stable_player_id_verification": "pending",
        "human_provenance_review": "pending",
        "confirmation_movement_access": False,
        "confirmation_usage": "sealed until model, benchmark code, scales, exclusions and cases are frozen",
        "partition_sizes": {name: len(rows) for name, rows in partitions.items()},
        "partitions": partitions,
    }
    path = OUT / "split-proposal-v1.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "partitions"}, indent=2))
    print(f"proposal={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
