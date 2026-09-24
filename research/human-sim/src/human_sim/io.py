from __future__ import annotations

import gzip
import hashlib
import io
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .schemas import MapPlan, TraceFrame


def repository_git_commit(repository_root: str | Path | None = None) -> str:
    """Return the checked-out commit used to produce a trace, if available."""
    root = Path(repository_root) if repository_root else Path(__file__).resolve().parents[4]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if commit else "unknown"


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
    planner_version: str | None = None,
    git_commit: str | None = None,
    build_identity: str | None = None,
    motion_mode: str = "profile",
    execution_mode: str = "math-only",
    execution_blend: float = 0.0,
    execution_adapter: str = "legacy",
    execution_diagnostics: dict[str, Any] | None = None,
) -> str:
    if planner_version is None:
        from .planner import PLANNER_VERSION

        planner_version = PLANNER_VERSION
    git_commit = git_commit or repository_git_commit()
    build_identity = build_identity or f"human-sim-python:{planner_version}:{git_commit}"
    execution_diagnostics = execution_diagnostics or {}
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
            "motion_mode": motion_mode,
            "execution_mode": execution_mode,
            "execution_blend": execution_blend,
            "execution_adapter": execution_adapter,
            "model_sha256": model_sha256 or "built_in_baseline_v1",
            "execution_diagnostics_version": 1,
            "execution_model_sha256": str(execution_diagnostics.get("model_sha256") or model_sha256 or "none"),
            "execution_effective_mode": str(execution_diagnostics.get("effective_mode") or execution_mode),
            "execution_fallback": bool(execution_diagnostics.get("fallback", False)),
            "execution_fallback_reason": execution_diagnostics.get("fallback_reason"),
            "execution_segment_count": int(execution_diagnostics.get("segment_count", 0)),
            "execution_learned_segment_count": int(execution_diagnostics.get("learned_segment_count", 0)),
            "execution_changed_sample_count": int(execution_diagnostics.get("changed_sample_count", 0)),
            "execution_fallback_segment_count": int(execution_diagnostics.get("fallback_segment_count", 0)),
            "planner_version": planner_version,
            "git_commit": git_commit,
            "build_identity": build_identity,
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
