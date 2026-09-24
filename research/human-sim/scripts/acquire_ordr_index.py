"""Cache the metadata index of a public o!rdr archive without replay payloads.

This is discovery material only.  No row becomes an admissible human trace
without provenance and identity review, and this script never opens .osr data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from human_sim.execution_acquisition import BoundedZip64Archive, kaggle_signed_url


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, default=155)
    args = parser.parse_args()
    if args.version != 155:
        raise ValueError("Review the pinned source version before changing it")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing snapshot: {manifest_path}")

    url = kaggle_signed_url("skihikingkevin", "ordr-replay-dump", args.version)
    archive = BoundedZip64Archive(url, max_central_bytes=256 * 1024 * 1024)
    print(f"archive bytes={archive.archive_size} central bytes={archive.layout.central_size}", flush=True)
    if archive.archive_size != 28_988_527_483 or archive.layout.entry_count != 821_359:
        raise ValueError("Archive changed since the preliminary metadata probe")
    central_path = output / "central-directory.bin"
    central_path.write_bytes(archive._entries)
    central_sha = digest(central_path)
    print(f"saved central SHA256={central_sha}", flush=True)

    counts: Counter[str] = Counter()
    index_entry = None
    for entry in archive.iter_entries():
        counts[Path(entry.name).suffix.lower()] += 1
        if Path(entry.name).name.lower() == "index.csv":
            if index_entry is not None:
                raise ValueError("Multiple index.csv entries")
            index_entry = entry
    if index_entry is None:
        raise FileNotFoundError("index.csv missing from archive")
    print(f"index compressed bytes={index_entry.compressed_size}", flush=True)
    index_path = output / "index.csv"
    extracted = archive.extract(index_entry, index_path, max_uncompressed_bytes=200_000_000)
    print(f"saved index SHA256={extracted['sha256']}", flush=True)
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        preview = [next(reader) for _ in range(3)]

    manifest = {
        "source": "https://www.kaggle.com/datasets/skihikingkevin/ordr-replay-dump/data",
        "owner": "skihikingkevin",
        "dataset": "ordr-replay-dump",
        "version": args.version,
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "archive_layout": vars(archive.layout),
        "central_directory": {"path": central_path.name, "sha256": central_sha},
        "entry_suffix_counts": dict(counts),
        "index": {**extracted, "path": index_path.name, "header": header, "preview": preview},
        "admissibility": "Discovery metadata only; replay execution and provenance unreviewed",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
