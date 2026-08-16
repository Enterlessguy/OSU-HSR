from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from .context import build_contexts
from .io import load_map_plan
from .matching import PressEvent, match_press_events


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _read_replay(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with _open(path) as stream:
        header = json.loads(stream.readline())
        frames = [json.loads(line) for line in stream if line.strip()]
    if header.get("kind") != "replay_header" or int(header.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported replay export: {path}")
    return header, frames


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

    downs: list[PressEvent] = []
    previous = [False, False]
    for index, frame in enumerate(frames):
        now = float(frame["time_ms"]) / rate
        state = [bool(frame.get("k1")), bool(frame.get("k2"))]
        for key in range(2):
            if state[key] and not previous[key]:
                downs.append(PressEvent(time_ms=now, key=key, frame_index=index))
        previous = state

    rows: list[dict[str, Any]] = []
    contexts = build_contexts(plan)
    assignments = match_press_events(plan.objects, downs, minimum_window_ms=180.0)
    for object_position, obj in enumerate(plan.objects):
        down_index = assignments.get(object_position)
        hit = down_index is not None
        if hit:
            press = downs[down_index]
            down_time, key, frame_index = press.time_ms, press.key, press.frame_index
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

        feature_row = contexts[object_position].as_feature_dict()
        feature_row.update(
            {
                "player_hash": header["player_hash"],
                "beatmap_sha256": plan.beatmap_sha256,
                "object_index": obj.index,
                "hit": hit,
                "key": key,
                "hit_error_ms": hit_error,
                "aim_offset_x": aim_x,
                "aim_offset_y": aim_y,
                "hold_duration_ms": hold,
            }
        )
        rows.append(feature_row)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(destination)
    return {"rows": len(rows), "hits": sum(bool(row["hit"]) for row in rows), "output": str(destination)}
