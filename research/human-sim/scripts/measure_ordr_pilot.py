"""Decode only public-score-matched development pilot replays and audit cadence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "output" / "external-ordr-v155" / "development-pilot-v1"
DLL = Path(os.environ.get("HSR_REPLAY_EXTRACTOR_DLL", str(ROOT.parent / "HumanSim.ReplayExtractor/bin/Debug/net8.0/HumanSim.ReplayExtractor.dll")))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def main() -> None:
    identity_path = PILOT / "identity-review.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    rows = [row for row in identity["records"] if row["public_identity_status"] == "matched_metadata_pending_human_review"]
    results = []
    for row in rows:
        replay = PILOT / "files" / "replays" / "osr" / (row["replay_md5"] + ".osr")
        beatmap = PILOT / "files" / "beatmaps" / (row["map_md5"] + ".osu")
        decoded = PILOT / "decoded" / (row["replay_md5"] + ".ndjson")
        decoded.parent.mkdir(parents=True, exist_ok=True)
        if not decoded.exists():
            env = os.environ.copy()
            env["HUMAN_SIM_PLAYER_SALT"] = "ordr-v155-pilot-local-20260924"
            subprocess.run(["dotnet", str(DLL), "--replay", str(replay), "--map", str(beatmap), "--output", str(decoded)], check=True, env=env, stdout=subprocess.DEVNULL)
        with decoded.open("r", encoding="utf-8") as stream:
            header = json.loads(stream.readline())
            frames = [json.loads(line) for line in stream if line.strip()]
        times = [float(frame["time_ms"]) for frame in frames]
        gaps = [b - a for a, b in zip(times, times[1:])]
        finite = all(math.isfinite(value) for frame in frames for value in (float(frame["x"]), float(frame["y"]), float(frame["time_ms"])))
        result = {
            "split": row["split"], "replay_md5": row["replay_md5"],
            "source_player_id": row["public_score_evidence"]["player_id"],
            "map_md5_matches": header["beatmap_md5"] == row["map_md5"],
            "frame_count": len(frames), "header_count_matches": header["frame_count"] == len(frames),
            "finite": finite, "median_gap_ms": percentile(gaps, 0.5),
            "p95_gap_ms": percentile(gaps, 0.95), "max_gap_ms": max(gaps) if gaps else None,
            "gaps_over_40_ms": sum(gap > 40 for gap in gaps),
            "nonpositive_gaps": sum(gap <= 0 for gap in gaps),
            "out_of_playfield_frames": sum(not (0 <= frame["x"] <= 512 and 0 <= frame["y"] <= 384) for frame in frames),
            "decoded_sha256": hashlib.sha256(decoded.read_bytes()).hexdigest(),
        }
        results.append(result)
        print(f"{row['split']} {row['replay_md5'][:10]} frames={len(frames)} median_gap={result['median_gap_ms']} p95={result['p95_gap_ms']} finite={finite}", flush=True)
    report = {
        "schema_version": "ordr-development-cadence-pilot-v1",
        "measured_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        "confirmation_access": False,
        "source_replay_cadence_is_native": True,
        "records": results,
    }
    path = PILOT / "cadence-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={path}")


if __name__ == "__main__":
    main()
