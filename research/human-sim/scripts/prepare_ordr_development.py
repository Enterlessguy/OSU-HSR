"""Decode identity-matched development rows and export matching map plans."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "external-ordr-v155"
BIN = Path(os.environ.get("HSR_RESEARCH_BIN", str(ROOT.parent)))
REPLAY_DLL = BIN / "HumanSim.ReplayExtractor" / "bin" / "Debug" / "net8.0" / "HumanSim.ReplayExtractor.dll"
MAP_DLL = BIN / "HumanSim.MapExporter" / "bin" / "Debug" / "net8.0" / "HumanSim.MapExporter.dll"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "calibration", "validation"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be 1–4")
    journal = OUT / "development-batches" / args.split / "acquisition.jsonl"
    latest = {row["replay_md5"]: row for row in map(json.loads, journal.read_text(encoding="utf-8").splitlines())}
    audit_path = OUT / "development-identity-audit-v1.json"
    identity_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    eligible = {row["replay_md5"] for row in identity_audit["records"] if row["split"] == args.split and row["metadata_eligible"]}
    rows = [row for row in latest.values() if row["replay_md5"] in eligible and row.get("file_integrity_pass") and row.get("stable_identity_match")]
    files = OUT / "development-files"
    decoded_dir = OUT / "development-decoded"
    plans_dir = OUT / "development-plans"
    results_dir = OUT / "development-batches" / args.split
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_path = results_dir / "preparation.jsonl"
    done = {}
    if progress_path.exists():
        done = {row["replay_md5"]: row for row in map(json.loads, progress_path.read_text(encoding="utf-8").splitlines())}
    pending = [row for row in rows if row["replay_md5"] not in done or not done[row["replay_md5"]].get("prepared")]
    print(f"split={args.split} identity_matched={len(rows)} pending={len(pending)}", flush=True)
    env = os.environ.copy()
    env["HUMAN_SIM_PLAYER_SALT"] = "ordr-v155-development-local-20260924"

    def prepare(row: dict) -> dict:
        map_md5 = row["map_md5"]
        replay_md5 = row["replay_md5"]
        beatmap = files / "beatmaps" / (map_md5 + ".osu")
        replay = files / "replays" / "osr" / (replay_md5 + ".osr")
        plan = plans_dir / (map_md5 + ".map.ndjson.gz")
        decoded = decoded_dir / (replay_md5 + ".ndjson")
        result = {"split": args.split, "replay_md5": replay_md5, "map_md5": map_md5, "map_family": row["map_family"], "source_player_id": row["public_score_evidence"]["player_id"], "prepared_utc": datetime.now(timezone.utc).isoformat(), "confirmation_access": False}
        try:
            plan.parent.mkdir(parents=True, exist_ok=True)
            decoded.parent.mkdir(parents=True, exist_ok=True)
            if not plan.exists():
                subprocess.run(["dotnet", str(MAP_DLL), "--map", str(beatmap), "--output", str(plan)], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if not decoded.exists():
                subprocess.run(["dotnet", str(REPLAY_DLL), "--replay", str(replay), "--map", str(beatmap), "--output", str(decoded)], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            with decoded.open("r", encoding="utf-8") as stream:
                header = json.loads(stream.readline())
                count = sum(1 for line in stream if line.strip())
            if header.get("beatmap_md5") != map_md5 or header.get("frame_count") != count or count < 4:
                raise ValueError("Decoded replay header or frame count mismatch")
            result.update({"prepared": True, "decoded_frames": count, "decoded_sha256": sha(decoded), "plan_sha256": sha(plan)})
        except Exception as error:
            result.update({"prepared": False, "error": f"{type(error).__name__}: {error}"})
        return result

    with progress_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(prepare, row) for row in pending]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{number}/{len(pending)}] {result['replay_md5'][:10]} prepared={result['prepared']}", flush=True)
    print(f"journal={progress_path} sha256={sha(progress_path)}")


if __name__ == "__main__":
    main()
