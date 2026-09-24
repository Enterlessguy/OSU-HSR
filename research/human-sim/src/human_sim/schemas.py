from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_MODS = frozenset({"HD", "HR", "DT", "HT", "FL"})
SKILL_PRESETS = (50.0, 75.0, 90.0, 95.0, 99.0, 99.5)


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def from_value(cls, value: dict[str, Any] | list[float]) -> "Point":
        if isinstance(value, dict):
            return cls(float(value["x"]), float(value["y"]))
        return cls(float(value[0]), float(value[1]))


@dataclass(frozen=True)
class HitWindows:
    great_ms: float
    ok_ms: float
    meh_ms: float
    miss_ms: float

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "HitWindows":
        value = value or {}
        return cls(
            float(value.get("great_ms", 40.0)),
            float(value.get("ok_ms", 80.0)),
            float(value.get("meh_ms", 120.0)),
            float(value.get("miss_ms", 150.0)),
        )


@dataclass(frozen=True)
class MapObject:
    index: int
    kind: str
    start_time_ms: float
    end_time_ms: float
    position: Point
    end_position: Point
    radius: float
    repeat_count: int = 0
    path_samples: tuple[Point, ...] = ()
    hit_windows: HitWindows = field(default_factory=lambda: HitWindows(40, 80, 120, 150))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MapObject":
        kind = str(value["kind"]).lower()
        if kind not in {"circle", "slider", "spinner"}:
            raise ValueError(f"Unsupported object kind: {kind}")
        return cls(
            index=int(value["index"]),
            kind=kind,
            start_time_ms=float(value["effective_start_time_ms"]),
            end_time_ms=float(value["effective_end_time_ms"]),
            position=Point.from_value(value["position"]),
            end_position=Point.from_value(value.get("end_position", value["position"])),
            radius=float(value.get("radius", 32.0)),
            repeat_count=int(value.get("repeat_count", 0)),
            path_samples=tuple(Point.from_value(p) for p in value.get("path_samples", [])),
            hit_windows=HitWindows.from_dict(value.get("hit_windows")),
        )


@dataclass(frozen=True)
class MapPlan:
    schema_version: int
    beatmap_sha256: str
    beatmap_md5: str
    clock_rate: float
    mods: tuple[str, ...]
    objects: tuple[MapObject, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MapPlan":
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("Only MapPlan schema version 1 is supported")
        mods = tuple(sorted(str(m).upper() for m in value.get("mods", [])))
        unknown = set(mods) - SUPPORTED_MODS
        if unknown:
            raise ValueError(f"Unsupported mods: {sorted(unknown)}")
        if "DT" in mods and "HT" in mods:
            raise ValueError("Double Time and Half Time cannot be combined")
        objects = tuple(MapObject.from_dict(o) for o in value.get("objects", []))
        if not objects:
            raise ValueError("Map plan contains no hit objects")
        if any(b.start_time_ms < a.start_time_ms for a, b in zip(objects, objects[1:])):
            raise ValueError("Map objects must be ordered by effective start time")
        return cls(
            schema_version=1,
            beatmap_sha256=str(value["beatmap_sha256"]).lower(),
            beatmap_md5=str(value.get("beatmap_md5", "")).lower(),
            clock_rate=float(value.get("clock_rate", 1.0)),
            mods=mods,
            objects=objects,
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class HumanProfile:
    percentile: float
    seed: int
    sample_rate_hz: int = 500
    perfect_baseline: bool = False
    # Skill/effort axes (0-100). Skill sets the execution ceiling for every
    # criterion; effort sets how consistently that ceiling is reached.
    # Defaults keep legacy runs identical: skill anchors to the percentile,
    # effort is fully committed.
    skill_level: float = 99.5
    effort_level: float = 100.0
    model_version: str = "baseline-v1"
    tapping_style: str = "alternating"
    motion_device: dict[str, float] = field(default_factory=lambda: {"sample_rate_hz": 500.0, "gain": 1.0})
    fatigue_initial: float = 0.0

    def validate(self) -> None:
        if self.percentile not in SKILL_PRESETS:
            raise ValueError(f"Percentile must be one of {SKILL_PRESETS}")
        if not 0 <= self.skill_level <= 100:
            raise ValueError("Skill level must be between 0 and 100")
        if not 0 <= self.effort_level <= 100:
            raise ValueError("Effort level must be between 0 and 100")
        if not 60 <= self.sample_rate_hz <= 1000:
            raise ValueError("Sample rate must be between 60 and 1000 Hz")


@dataclass(frozen=True)
class TraceFrame:
    time_us: int
    x: float
    y: float
    k1: bool
    k2: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_us": self.time_us,
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "k1": self.k1,
            "k2": self.k2,
        }


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    planner_version: str
    git_commit: str
    build_identity: str
    synthetic: bool
    beatmap_sha256: str
    map_plan_sha256: str
    model_sha256: str
    corpus_manifest_sha256: str | None
    trace_sha256: str
    client_build_sha256: str | None
    configuration_sha256: str
    captured_replay_sha256: str | None
    profile_percentile: float
    seed: int
    motion_mode: str = "profile"
    execution_mode: str = "math-only"
    execution_blend: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)
