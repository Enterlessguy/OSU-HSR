from __future__ import annotations

import math
from pathlib import Path

from .io import iter_trace


def validate_trace(path: str | Path) -> dict[str, float | int | str]:
    iterator = iter_trace(path)
    header = next(iterator)
    last_time = -1
    count = 0
    max_speed = 0.0
    previous = None
    k1_down = False
    k2_down = False
    for value in iterator:
        time_us = int(value["time_us"])
        x = float(value["x"])
        y = float(value["y"])
        if time_us <= last_time:
            raise ValueError(f"Non-monotonic timestamp at line {value['_line']}")
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Non-finite coordinate at line {value['_line']}")
        if previous is not None:
            delta_seconds = (time_us - previous[0]) / 1_000_000.0
            max_speed = max(max_speed, math.hypot(x - previous[1], y - previous[2]) / delta_seconds)
        previous = (time_us, x, y)
        last_time = time_us
        k1_down = bool(value["k1"])
        k2_down = bool(value["k2"])
        count += 1
    if count == 0:
        raise ValueError("Trace contains no frames")
    if k1_down or k2_down:
        raise ValueError("Trace ends with a held key")
    # Machine-perfect infrastructure traces must follow the beatmap's authentic
    # geometry, including novelty maps whose bursts teleport across the field.
    # Keep the strict human-like guard for learned profiles, and reject only
    # absurd discontinuities in diagnostic mode.
    speed_limit = 1_000_000 if bool(header.get("diagnostic_perfect", False)) else 100_000
    if max_speed > speed_limit:
        raise ValueError(
            f"Implausible cursor discontinuity: {max_speed:.1f} osu! pixels/s "
            f"(limit {speed_limit:.0f})"
        )
    return {
        "status": "valid",
        "frames": count,
        "duration_us": last_time,
        "max_speed_osu_px_s": round(max_speed, 3),
        "profile_percentile": float(header["profile_percentile"]),
        "planner_version": str(header.get("planner_version", "unknown")),
        "git_commit": str(header.get("git_commit", "unknown")),
        "build_identity": str(header.get("build_identity", "unknown")),
        "identity_complete": all(
            bool(header.get(key))
            for key in ("planner_version", "git_commit", "build_identity")
        ),
    }
