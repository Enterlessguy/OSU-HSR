from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_map_plan


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _read_replay(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with _open(path) as stream:
        header = json.loads(stream.readline())
        frames = [json.loads(line) for line in stream if line.strip()]
    if header.get("kind") != "replay_header" or int(header.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported replay export: {path}")
    return header, frames


def _context(kind: str, distance: float, interval_ms: float) -> str:
    if kind in {"slider", "spinner"}:
        return kind
    if interval_ms < 130:
        return "stream"
    if interval_ms < 220:
        return "burst"
    if distance > 120:
        return "jump"
    return "transition"


def extract_features(replay_path: str, map_plan_path: str, output_path: str) -> dict[str, Any]:
    """Match decoded replay key-downs to lazer-transformed hit objects and write Parquet."""
    import polars as pl

    plan = load_map_plan(map_plan_path)
    header, raw_frames = _read_replay(Path(replay_path))
    rate = plan.clock_rate
    frames = sorted(raw_frames, key=lambda value: float(value["time_ms"]))
    times = np.asarray([float(f["time_ms"]) / rate for f in frames])
    xs = np.asarray([float(f["x"]) for f in frames])
    ys = np.asarray([float(f["y"]) for f in frames])

    downs: list[tuple[float, int, int]] = []
    active_since: dict[int, float] = {}
    holds: dict[tuple[int, int], float] = {}
    previous = [False, False]
    for index, frame in enumerate(frames):
        now = float(frame["time_ms"]) / rate
        state = [bool(frame.get("k1")), bool(frame.get("k2"))]
        for key in range(2):
            if state[key] and not previous[key]:
                downs.append((now, key, index))
                active_since[key] = now
            elif previous[key] and not state[key] and key in active_since:
                holds[(key, index)] = now - active_since.pop(key)
        previous = state

    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    previous_object = None
    for obj in plan.objects:
        candidates = [(abs(t - obj.start_time_ms), i, t, key, frame_i) for i, (t, key, frame_i) in enumerate(downs) if i not in used]
        nearest = min(candidates, default=None)
        hit = nearest is not None and nearest[0] <= max(180.0, obj.hit_windows.miss_ms)
        if hit:
            _, down_index, down_time, key, frame_index = nearest
            used.add(down_index)
            aim_x = float(np.interp(down_time, times, xs)) - obj.position.x
            aim_y = float(np.interp(down_time, times, ys)) - obj.position.y
            release_time = next(
                (float(frames[j]["time_ms"]) / rate for j in range(frame_index + 1, len(frames)) if not bool(frames[j].get("k1" if key == 0 else "k2"))),
                down_time,
            )
            hold = max(0.0, release_time - down_time)
            hit_error = down_time - obj.start_time_ms
        else:
            key, aim_x, aim_y, hold, hit_error = -1, np.nan, np.nan, np.nan, np.nan

        if previous_object is None:
            distance, interval = 0.0, 1000.0
        else:
            distance = float(np.hypot(obj.position.x - previous_object.end_position.x, obj.position.y - previous_object.end_position.y))
            interval = obj.start_time_ms - previous_object.end_time_ms
        rows.append(
            {
                "player_hash": header["player_hash"],
                "beatmap_sha256": plan.beatmap_sha256,
                "object_index": obj.index,
                "context": _context(obj.kind, distance, interval),
                "object_kind": obj.kind,
                "distance": distance,
                "interval_ms": interval,
                "strain": min(1.0, distance / max(16.0, interval) / 2.2),
                "clock_rate": rate,
                "hidden": "HD" in plan.mods,
                "hit": hit,
                "key": key,
                "hit_error_ms": hit_error,
                "aim_offset_x": aim_x,
                "aim_offset_y": aim_y,
                "hold_duration_ms": hold,
            }
        )
        previous_object = obj

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(destination)
    return {"rows": len(rows), "hits": sum(bool(row["hit"]) for row in rows), "output": str(destination)}
