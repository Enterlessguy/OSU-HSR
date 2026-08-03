from __future__ import annotations

from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from .io import load_map_plan
from .planner import HumanTracePlanner
from .schemas import HumanProfile, MapPlan, TraceFrame


def discover_standard_beatmaps(storage_root: str | Path) -> list[Path]:
    root = Path(storage_root)
    if not root.is_dir():
        raise FileNotFoundError(f"lazer content store was not found: {root}")

    beatmaps: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as stream:
                prefix = stream.read(65_536)
        except OSError:
            continue
        if not prefix.startswith(b"osu file format v"):
            continue
        mode = 0
        for raw_line in prefix.splitlines():
            line = raw_line.strip()
            if line.startswith(b"Mode:"):
                try:
                    mode = int(line.split(b":", 1)[1].strip())
                except ValueError:
                    mode = -1
                break
        if mode == 0:
            beatmaps.append(path)
    return sorted(beatmaps, key=lambda value: value.name)


def _even_sample(values: list[Path], limit: int | None) -> list[Path]:
    if limit is None or limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indices]


def _rising_edges(frames: list[TraceFrame]) -> list[tuple[TraceFrame, int]]:
    previous_k1 = previous_k2 = False
    edges: list[tuple[TraceFrame, int]] = []
    for frame in frames:
        if frame.k1 and not previous_k1:
            edges.append((frame, 0))
        if frame.k2 and not previous_k2:
            edges.append((frame, 1))
        previous_k1, previous_k2 = frame.k1, frame.k2
    return edges


def verify_perfect_plan(plan: MapPlan, frames: list[TraceFrame], sample_rate_hz: int) -> dict[str, Any]:
    step_ms = 1000.0 / sample_rate_hz
    timeline_start = plan.objects[0].start_time_ms - 1500.0
    edges = _rising_edges(frames)
    if len(edges) != len(plan.objects):
        raise ValueError(f"key-down coverage mismatch: {len(edges)} transitions for {len(plan.objects)} objects")

    max_hit_time_error_ms = 0.0
    max_head_position_error = 0.0
    slider_tail_holds = 0
    frame_times_us = [frame.time_us for frame in frames]
    for obj, (edge, key) in zip(plan.objects, edges):
        edge_time_ms = timeline_start + edge.time_us / 1000.0
        time_error = abs(edge_time_ms - obj.start_time_ms)
        max_hit_time_error_ms = max(max_hit_time_error_ms, time_error)
        if time_error > step_ms + 0.001:
            raise ValueError(f"object {obj.index} key-down is {time_error:.3f} ms from its start")

        if obj.kind != "spinner":
            position_error = math.hypot(edge.x - obj.position.x, edge.y - obj.position.y)
            max_head_position_error = max(max_head_position_error, position_error)
            if position_error > 1.0:
                raise ValueError(f"object {obj.index} head position is off by {position_error:.3f} osu! pixels")

        if obj.kind == "slider":
            tail_time_us = round((obj.end_time_ms - timeline_start) * 1000.0)
            # Event frames are intentionally inserted between regular 500 Hz
            # samples, so elapsed time can no longer be converted directly to
            # a list index. Inspect the first real frame at/after the tail.
            tail_index = min(bisect_left(frame_times_us, tail_time_us), len(frames) - 1)
            tail = frames[tail_index]
            held = tail.k1 if key == 0 else tail.k2
            if not held:
                raise ValueError(f"slider {obj.index} is not held at its tail")
            slider_tail_holds += 1

    max_speed = 0.0
    for previous, current in zip(frames, frames[1:]):
        seconds = (current.time_us - previous.time_us) / 1_000_000.0
        max_speed = max(max_speed, math.hypot(current.x - previous.x, current.y - previous.y) / seconds)

    return {
        "frames": len(frames),
        "key_downs": len(edges),
        "max_hit_time_error_ms": round(max_hit_time_error_ms, 6),
        "max_head_position_error_osu_px": round(max_head_position_error, 6),
        "max_speed_osu_px_s": round(max_speed, 3),
        "slider_tail_holds": slider_tail_holds,
    }


def audit_library(
    storage_root: str | Path,
    exporter: str | Path,
    output: str | Path,
    *,
    limit: int | None = 100,
    sample_rate_hz: int = 500,
    seed: int = 42,
    workers: int = 4,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    maps = discover_standard_beatmaps(storage_root)
    selected = _even_sample(maps, limit)
    exporter_path = Path(exporter)
    if not exporter_path.is_file():
        raise FileNotFoundError(f"MapExporter has not been built: {exporter_path}")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    def build_report(*, complete: bool) -> dict[str, Any]:
        ordered = sorted(results, key=lambda result: result["index"])
        passed = sum(result["status"] == "passed" for result in ordered)
        return {
            "schema_version": 1,
            "complete": complete,
            "storage_root": str(Path(storage_root).resolve()),
            "discovered_standard_beatmaps": len(maps),
            "requested": len(selected),
            "audited": len(ordered),
            "passed": passed,
            "failed": len(ordered) - passed,
            "pass_rate": passed / len(ordered) if ordered else 0.0,
            "sample_rate_hz": sample_rate_hz,
            "workers": workers,
            "results": ordered,
        }

    with tempfile.TemporaryDirectory(prefix="human-sim-library-audit-") as temporary:
        temp_root = Path(temporary)

        def audit_one(index: int, source: Path) -> dict[str, Any]:
            map_plan_path = temp_root / f"{index:04d}.map.ndjson.gz"
            record: dict[str, Any] = {
                "index": index,
                "content_hash": source.name,
                "status": "failed",
                "stage": "export",
            }
            try:
                started = time.perf_counter()
                completed = subprocess.run(
                    [str(exporter_path), "--map", str(source), "--output", str(map_plan_path), "--mods", ""],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=environment,
                )
                record["export_duration_s"] = round(time.perf_counter() - started, 3)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"exporter code {completed.returncode}")
                plan = load_map_plan(map_plan_path)
                counts = {kind: sum(obj.kind == kind for obj in plan.objects) for kind in ("circle", "slider", "spinner")}
                record.update(
                    {
                        "stage": "plan",
                        "beatmap_sha256": plan.beatmap_sha256,
                        "title": str(plan.metadata.get("title", "")),
                        "artist": str(plan.metadata.get("artist", "")),
                        "difficulty": str(plan.metadata.get("difficulty_name", "")),
                        "object_count": len(plan.objects),
                        "counts": counts,
                    }
                )
                profile = HumanProfile(99.5, seed, sample_rate_hz, perfect_baseline=True)
                started = time.perf_counter()
                frames = HumanTracePlanner(plan, profile).generate()
                record["planning_duration_s"] = round(time.perf_counter() - started, 3)
                record["stage"] = "verify"
                verification = verify_perfect_plan(plan, frames, sample_rate_hz)
                record.update(
                    {
                        "status": "passed",
                        "stage": "complete",
                        "verification": verification,
                    }
                )
            except Exception as exception:
                record["error"] = f"{type(exception).__name__}: {exception}"
            return record

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(audit_one, index, source): (index, source)
                for index, source in enumerate(selected, start=1)
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                index, source = futures[future]
                try:
                    record = future.result()
                except Exception as exception:
                    record = {
                        "index": index,
                        "content_hash": source.name,
                        "status": "failed",
                        "stage": "worker",
                        "error": f"{type(exception).__name__}: {exception}",
                    }
                results.append(record)
                checkpoint = build_report(complete=False)
                destination.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")
                print(f"[{completed_count}/{len(selected)}] {record['status']}: {record.get('title') or source.name}", flush=True)

    report = build_report(complete=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
