"""Metadata-only eligibility inventory for the pinned o!rdr index snapshot."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "output" / "external-ordr-v155" / "index-snapshot"
OLD = Path(os.environ.get("HSR_PRIOR_OUTPUT", str(ROOT / "output/prior")))
AUDIT = Path(os.environ.get("HSR_PRIOR_CONFIRMATION_AUDIT", str(ROOT / "output/prior/confirmation-cohort-audit-v1/audit-report.json")))
AUTOMATION_MASK = 2048 | 128 | 8192
MD5 = re.compile(r"^[0-9a-f]{32}$")


def read_old_names() -> tuple[set[str], set[str]]:
    names: set[str] = set()
    user_ids: set[str] = set()
    for path in (OLD / "external-osu3k" / "data.csv", OLD / "osu3k-chunk-02" / "dataset" / "data.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("player"):
                    names.add(row["player"].strip().casefold())
                user_ids.add(str(row.get("userid", "")).split(".")[0])
    return names, user_ids


def main() -> None:
    snapshot = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    old_names, old_ids = read_old_names()
    old_families = {str(family) for component in audit["components"] for family in component.get("map_families", [])}
    old_maps = {str(md5) for component in audit["components"] for md5 in component.get("maps", [])}
    old_ids.update(str(player).removeprefix("source-player-id:") for component in audit["components"] for player in component.get("players", []))

    counts: Counter[str] = Counter()
    eligible: list[dict[str, str | int | float]] = []
    with (SNAPSHOT / "index.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            counts["rows"] += 1
            family = row.get("beatmap-SetId", "").removesuffix(".0")
            map_md5 = row.get("beatmapHash", "").lower()
            replay_md5 = row.get("replayHash", "").lower()
            name = row.get("playerName", "").strip()
            try:
                mods = int(row.get("mods", ""))
                accuracy = float(row.get("performance-Accuracy", ""))
                circles = int(row.get("beatmap-Circles", ""))
            except ValueError:
                counts["bad_numeric_metadata"] += 1
                continue
            if not MD5.fullmatch(map_md5) or not MD5.fullmatch(replay_md5) or not family.isdecimal():
                counts["bad_identity_metadata"] += 1
                continue
            if family in old_families or map_md5 in old_maps:
                counts["old_map_or_family"] += 1
                continue
            if not name or name.casefold() in old_names:
                counts["old_or_blank_player_name"] += 1
                continue
            if mods & AUTOMATION_MASK:
                counts["automation_mod"] += 1
                continue
            # Start with an NM, successful, sufficiently populated source pool.
            # This is a candidate filter, not a human-provenance certification.
            if mods != 0:
                counts["other_mod"] += 1
                continue
            if row.get("performance-IsFail", "").lower() == "true" or not 0.85 <= accuracy <= 1 or circles < 100:
                counts["performance_or_map_filter"] += 1
                continue
            counts["candidate_rows"] += 1
            eligible.append({
                "replay_md5": replay_md5,
                "map_md5": map_md5,
                "map_id": row.get("beatmap-Id", ""),
                "map_family": family,
                "player_name": name,
                "date": row.get("date", ""),
                "accuracy": accuracy,
                "circles": circles,
                "mods": mods,
            })

    by_player = Counter(str(row["player_name"]).casefold() for row in eligible)
    by_family = Counter(str(row["map_family"]) for row in eligible)
    report = {
        "schema_version": "ordr-metadata-audit-v1",
        "source_manifest_sha256": hashlib.sha256((SNAPSHOT / "manifest.json").read_bytes()).hexdigest(),
        "old_audit_canonical_sha256": audit["canonical_sha256"],
        "old_name_count": len(old_names),
        "old_player_id_count": len(old_ids),
        "old_map_family_count": len(old_families),
        "old_exact_map_count": len(old_maps),
        "counts": counts,
        "candidate_players": len(by_player),
        "candidate_families": len(by_family),
        "player_name_is_stable_id": False,
        "human_provenance": "unverified_public_archive; review score-page identity and replay headers before admission",
        "movement_opened": False,
        "candidate_pool": eligible,
    }
    dest = SNAPSHOT.parent / "metadata-audit-v1.json"
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "candidate_pool"}, indent=2, default=dict))
    print(f"candidate_pool={dest} sha256={hashlib.sha256(dest.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
