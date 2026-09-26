"""Acquire a small, preselected development pilot; never touch confirmation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from human_sim.execution_acquisition import (
    ArchiveLayout,
    BoundedZip64Archive,
    _archive_size,
    iter_central_entries,
    kaggle_signed_url,
    read_osr_header,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"
PILOT_PER_SPLIT = 5


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def main() -> None:
    proposal_path = OUT / "split-proposal-v1.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    snapshot = OUT / "index-snapshot"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    central = (snapshot / "central-directory.bin").read_bytes()
    if hashlib.sha256(central).hexdigest() != manifest["central_directory"]["sha256"]:
        raise ValueError("Cached ZIP central directory hash changed")
    rows = [row | {"split": split} for split in ("train", "calibration", "validation") for row in proposal["partitions"][split][:PILOT_PER_SPLIT]]
    wanted = {f"replays/osr/{row['replay_md5']}.osr" for row in rows} | {f"beatmaps/{row['map_md5']}.osu" for row in rows}
    entries = {entry.name: entry for entry in iter_central_entries(central) if entry.name in wanted}
    if len(entries) != len(wanted):
        raise ValueError(f"Missing selected archive entries: {sorted(wanted - entries.keys())}")
    signed_url = kaggle_signed_url("skihikingkevin", "ordr-replay-dump", 155)
    if _archive_size(signed_url) != manifest["archive_layout"]["archive_size"]:
        raise ValueError("Remote archive size differs from cached metadata")
    archive = BoundedZip64Archive.__new__(BoundedZip64Archive)
    archive.url = signed_url
    archive.archive_size = manifest["archive_layout"]["archive_size"]
    archive.layout = ArchiveLayout(**manifest["archive_layout"])
    archive._entries = central
    files = OUT / "development-pilot-v1" / "files"

    def acquire(row: dict) -> dict:
        replay_name = f"replays/osr/{row['replay_md5']}.osr"
        map_name = f"beatmaps/{row['map_md5']}.osu"
        replay_path = files / replay_name
        map_path = files / map_name
        try:
            replay = archive.extract(entries[replay_name], replay_path, max_uncompressed_bytes=8_000_000)
            beatmap = archive.extract(entries[map_name], map_path, max_uncompressed_bytes=8_000_000)
            header = read_osr_header(replay_path)
            checks = {
                "map_file_md5": md5(map_path) == row["map_md5"],
                "replay_file_md5": md5(replay_path) == row["replay_md5"],
                "replay_map_md5": header["beatmap_md5"].lower() == row["map_md5"],
                "player_name": header["player_name"].casefold() == row["player_name"].casefold(),
                "standard_mode": header["mode"] == 0,
                "no_mods": header["mods"] == 0,
                "score_id_present": bool(header["online_score_id"]),
            }
            return {"split": row["split"], "replay_md5": row["replay_md5"], "map_md5": row["map_md5"], "player_name": row["player_name"], "checks": checks, "passed": all(checks.values()), "replay": replay, "beatmap": beatmap, "header": header}
        except Exception as error:
            return {"split": row["split"], "replay_md5": row["replay_md5"], "map_md5": row["map_md5"], "passed": False, "error": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(acquire, rows))
    report = {
        "schema_version": "ordr-development-pilot-v1",
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "source_proposal_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        "confirmation_or_reserve_access": False,
        "movement_decoded": False,
        "requested": len(rows),
        "passed": sum(row["passed"] for row in records),
        "records": records,
    }
    report_path = OUT / "development-pilot-v1" / "acquisition-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(f"failed={[(r['split'], r['replay_md5'], r.get('error'), r.get('checks')) for r in records if not r['passed']]}")


if __name__ == "__main__":
    main()
