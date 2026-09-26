"""Resumable, bounded acquisition of preselected development replay/map pairs.

Confirmation and reserve partitions are refused.  A public score match is
recorded as identity evidence, not proof that the play was manual.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.error

from human_sim.execution_acquisition import (
    ArchiveLayout,
    BoundedZip64Archive,
    _archive_size,
    fetch_public_score_evidence,
    iter_central_entries,
    kaggle_signed_url,
    read_osr_header,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "calibration", "validation"))
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be 1â€“4")
    proposal_path = OUT / "split-proposal-v1.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    partition = proposal["partitions"][args.split]
    if not 1 <= args.count <= len(partition):
        raise ValueError("count exceeds the preselected partition")
    rows = partition[:args.count]
    snapshot = OUT / "index-snapshot"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    central = (snapshot / "central-directory.bin").read_bytes()
    if hashlib.sha256(central).hexdigest() != manifest["central_directory"]["sha256"]:
        raise ValueError("Cached ZIP central directory hash changed")
    wanted = {f"replays/osr/{row['replay_md5']}.osr" for row in rows} | {f"beatmaps/{row['map_md5']}.osu" for row in rows}
    entries = {entry.name: entry for entry in iter_central_entries(central) if entry.name in wanted}
    if len(entries) != len(wanted):
        raise ValueError(f"Missing selected entries: {len(wanted) - len(entries)}")

    batch = OUT / "development-batches" / args.split
    batch.mkdir(parents=True, exist_ok=True)
    journal = batch / "acquisition.jsonl"
    completed: dict[str, dict] = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["replay_md5"]] = item
    # A completed mismatch in replay/header identity is a terminal quarantine,
    # whereas a network exception is retryable.  Never keep refetching a file
    # whose verified bytes disagree with the source metadata.
    pending = [
        row for row in rows
        if row["replay_md5"] not in completed
        or (not completed[row["replay_md5"]].get("file_integrity_pass") and "error" in completed[row["replay_md5"]])
    ]
    print(f"split={args.split} selected={len(rows)} cached={len(rows)-len(pending)} pending={len(pending)}", flush=True)
    if not pending:
        return

    for attempt in range(4):
        try:
            signed_url = kaggle_signed_url("skihikingkevin", "ordr-replay-dump", 155)
            remote_size = _archive_size(signed_url)
            break
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if remote_size != manifest["archive_layout"]["archive_size"]:
        raise ValueError("Remote archive size differs from cached metadata")
    archive = BoundedZip64Archive.__new__(BoundedZip64Archive)
    archive.url = signed_url
    archive.archive_size = manifest["archive_layout"]["archive_size"]
    archive.layout = ArchiveLayout(**manifest["archive_layout"])
    archive._entries = central
    files = OUT / "development-files"
    map_locks: dict[str, threading.Lock] = {row["map_md5"]: threading.Lock() for row in pending}

    def acquire(row: dict) -> dict:
        replay_name = f"replays/osr/{row['replay_md5']}.osr"
        map_name = f"beatmaps/{row['map_md5']}.osu"
        replay_path = files / replay_name
        map_path = files / map_name
        result = {"split": args.split, "replay_md5": row["replay_md5"], "map_md5": row["map_md5"], "map_family": row["map_family"], "player_name": row["player_name"], "acquired_utc": datetime.now(timezone.utc).isoformat(), "confirmation_access": False}
        try:
            if not replay_path.exists() or md5(replay_path) != row["replay_md5"]:
                result["replay"] = archive.extract(entries[replay_name], replay_path, max_uncompressed_bytes=8_000_000)
            with map_locks[row["map_md5"]]:
                if not map_path.exists() or md5(map_path) != row["map_md5"]:
                    result["beatmap"] = archive.extract(entries[map_name], map_path, max_uncompressed_bytes=8_000_000)
            header = read_osr_header(replay_path)
            checks = {
                "replay_file_md5": md5(replay_path) == row["replay_md5"],
                "map_file_md5": md5(map_path) == row["map_md5"],
                "replay_map_md5": header["beatmap_md5"].lower() == row["map_md5"],
                "player_name": header["player_name"].casefold() == row["player_name"].casefold(),
                "standard_mode": header["mode"] == 0,
                "no_mods": header["mods"] == 0,
            }
            result["file_integrity_pass"] = all(checks.values())
            result["checks"] = checks
            result["score_id"] = header["online_score_id"]
            result["replay_sha256"] = header["sha256"]
            if result["file_integrity_pass"] and header["online_score_id"]:
                try:
                    page = fetch_public_score_evidence(header["online_score_id"])
                except Exception as error:
                    page = {"score_id": str(header["online_score_id"]), "error": f"{type(error).__name__}: {error}"}
                result["public_score_evidence"] = page
                result["stable_identity_match"] = bool(
                    page.get("public_score_page_found") and page.get("player_id")
                    and not page.get("player_is_bot")
                    and page.get("beatmap_md5") == row["map_md5"]
                    and str(page.get("player_name", "")).casefold() == row["player_name"].casefold()
                )
            else:
                result["stable_identity_match"] = False
        except Exception as error:
            result["file_integrity_pass"] = False
            result["stable_identity_match"] = False
            result["error"] = f"{type(error).__name__}: {error}"
        return result

    with journal.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(acquire, row): row for row in pending}
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{number}/{len(pending)}] {result['replay_md5'][:10]} integrity={result['file_integrity_pass']} identity={result['stable_identity_match']}", flush=True)
    print(f"journal={journal} sha256={hashlib.sha256(journal.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
