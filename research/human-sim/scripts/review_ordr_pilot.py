"""Review pilot file integrity and public score metadata without decoding frames."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from human_sim.execution_acquisition import fetch_public_score_evidence_batch, read_osr_header


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "output" / "external-ordr-v155" / "development-pilot-v1"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    acquisition_path = PILOT / "acquisition-report.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    records = []
    for item in acquisition["records"]:
        replay_path = PILOT / "files" / "replays" / "osr" / (item["replay_md5"] + ".osr")
        map_path = PILOT / "files" / "beatmaps" / (item["map_md5"] + ".osu")
        header = read_osr_header(replay_path)
        checks = {
            "replay_file_md5": md5(replay_path) == item["replay_md5"],
            "map_file_md5": md5(map_path) == item["map_md5"],
            "replay_map_md5": header["beatmap_md5"].lower() == item["map_md5"],
            "player_name": header["player_name"].casefold() == item["player_name"].casefold(),
            "standard_mode": header["mode"] == 0,
            "no_mods": header["mods"] == 0,
        }
        records.append({"split": item["split"], "replay_md5": item["replay_md5"], "map_md5": item["map_md5"], "file_integrity_pass": all(checks.values()), "checks": checks, "score_id": header["online_score_id"], "player_name": header["player_name"]})
    score_ids = [row["score_id"] for row in records if row["file_integrity_pass"] and row["score_id"]]
    evidence = {str(item["score_id"]): item for item in fetch_public_score_evidence_batch(score_ids, max_workers=4)}
    for row in records:
        page = evidence.get(str(row["score_id"]))
        row["public_score_evidence"] = page
        row["public_identity_status"] = (
            "matched_metadata_pending_human_review"
            if page and page.get("public_score_page_found") and not page.get("player_is_bot")
            and page.get("beatmap_md5") == row["map_md5"]
            and page.get("player_name", "").casefold() == row["player_name"].casefold()
            else "unverified_or_mismatch"
        )
    report = {
        "schema_version": "ordr-pilot-identity-review-v1",
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "source_acquisition_sha256": hashlib.sha256(acquisition_path.read_bytes()).hexdigest(),
        "confirmation_access": False,
        "movement_decoded": False,
        "file_integrity_pass": sum(row["file_integrity_pass"] for row in records),
        "public_identity_match": sum(row["public_identity_status"] == "matched_metadata_pending_human_review" for row in records),
        "score_ids_present": len(score_ids),
        "records": records,
    }
    path = PILOT / "identity-review.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(f"status={[row['public_identity_status'] for row in records]}")


if __name__ == "__main__":
    main()
