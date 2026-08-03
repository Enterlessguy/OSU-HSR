from __future__ import annotations

import gzip
import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .schemas import MapPlan, TraceFrame


@contextmanager
def _open_text(path: Path, mode: str):
    if path.suffix.lower() == ".gz":
        if mode == "w":
            with path.open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                        yield text
            return
        with gzip.open(path, mode + "t", encoding="utf-8", newline="\n") as text:
            yield text
        return
    with path.open(mode, encoding="utf-8", newline="\n") as text:
        yield text


def load_map_plan(path: str | Path) -> MapPlan:
    source = Path(path)
    with _open_text(source, "r") as stream:
        line = stream.readline()
    if not line:
        raise ValueError(f"Map plan is empty: {source}")
    return MapPlan.from_dict(json.loads(line))


def write_trace(
    path: str | Path,
    *,
    map_plan: MapPlan,
    profile_percentile: float,
    seed: int,
    sample_rate_hz: int,
    frames: Iterable[TraceFrame],
    model_sha256: str | None = None,
    diagnostic_perfect: bool = False,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(destination, "w") as stream:
        header = {
            "kind": "trace_header",
            "schema_version": 1,
            "beatmap_sha256": map_plan.beatmap_sha256,
            "beatmap_md5": map_plan.beatmap_md5,
            "mods": list(map_plan.mods),
            "profile_percentile": profile_percentile,
            "seed": seed,
            "sample_rate_hz": sample_rate_hz,
            "clock_rate": map_plan.clock_rate,
            "timeline_start_effective_ms": map_plan.objects[0].start_time_ms - 1500.0
            if map_plan.objects
            else 0.0,
            "coordinate_space": "osu_playfield_512x384",
            "synthetic": True,
            "diagnostic_perfect": diagnostic_perfect,
            "model_sha256": model_sha256 or "built_in_baseline_v1",
        }
        encoded = json.dumps(header, separators=(",", ":"), sort_keys=True) + "\n"
        stream.write(encoded)
        for frame in frames:
            encoded = json.dumps(frame.as_dict(), separators=(",", ":"), sort_keys=True) + "\n"
            stream.write(encoded)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def iter_trace(path: str | Path):
    source = Path(path)
    with _open_text(source, "r") as stream:
        header = json.loads(stream.readline())
        if header.get("kind") != "trace_header" or header.get("schema_version") != 1:
            raise ValueError("Unsupported or missing trace header")
        yield header
        for line_number, line in enumerate(stream, start=2):
            if line.strip():
                value = json.loads(line)
                value["_line"] = line_number
                yield value
