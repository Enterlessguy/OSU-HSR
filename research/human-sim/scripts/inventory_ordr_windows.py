"""Build a development-only circle-transition inventory for scorer design."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

from human_sim.scoped_realism import load_human_rows


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "calibration", "validation"))
    args = parser.parse_args()
    batch = OUT / "development-batches" / args.split
    acquisition_path = batch / "acquisition.jsonl"
    preparation_path = batch / "preparation.jsonl"
    acquisition = {row["replay_md5"]: row for row in map(json.loads, acquisition_path.read_text(encoding="utf-8").splitlines())}
    preparation = {row["replay_md5"]: row for row in map(json.loads, preparation_path.read_text(encoding="utf-8").splitlines())}
    records = []
    decoded_root = OUT / "development-decoded"
    for replay_md5, row in acquisition.items():
        prepared = preparation.get(replay_md5)
        if not row.get("stable_identity_match") or not prepared or not prepared.get("prepared"):
            continue
        replay_sha = row["replay_sha256"]
        source = decoded_root / (replay_md5 + ".ndjson")
        alias = decoded_root / (replay_sha + ".ndjson")
        if not alias.exists():
            os.link(source, alias)
        if sha(alias) != prepared["decoded_sha256"]:
            raise ValueError(f"Decoded replay changed for {replay_md5}")
        records.append({
            "split": args.split, "player": "source-player-id:" + str(row["public_score_evidence"]["player_id"]),
            "map": row["map_md5"], "map_family": row["map_family"],
            "replay_sha256": replay_sha, "source_replay_md5": replay_md5,
        })
    manifest = {"schema_version": "ordr-development-manifest-v1", "max_windows_per_player": 100, "records": records, "confirmation_access": False, "source_acquisition_sha256": sha(acquisition_path), "source_preparation_sha256": sha(preparation_path)}
    manifest_path = batch / "window-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"loading {len(records)} {args.split} records", flush=True)
    rows, inventory = load_human_rows(manifest_path, decoded_root, OUT / "development-plans", split=args.split)
    inventory.update({"schema_version": "ordr-development-window-inventory-v1", "source_manifest_sha256": sha(manifest_path), "confirmation_access": False})
    report_path = batch / "window-inventory.json"
    report_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    rows_path = batch / "human-windows.json.gz"
    with gzip.open(rows_path, "wt", encoding="utf-8") as stream:
        json.dump(rows, stream, separators=(",", ":"))
    print(json.dumps(inventory, indent=2))
    print(f"human_windows={rows_path} sha256={sha(rows_path)}")


if __name__ == "__main__":
    main()
