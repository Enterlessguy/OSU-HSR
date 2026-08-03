from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .schemas import HumanProfile, MapObject, MapPlan, TraceFrame


# Bump when trace-generation behaviour changes so the runner's content-addressed
# trace cache is invalidated (the configuration hash includes this string).
PLANNER_VERSION = "settle-chord-v3"


@dataclass(frozen=True)
class SkillParameters:
    aim_sigma: float
    timing_sigma_ms: float
    timing_rho: float
    curvature_ratio: float
    correction_probability: float
    hold_mean_ms: float
    hold_sigma_ms: float
    fatigue_gain: float


PRESETS: dict[float, SkillParameters] = {
    50.0: SkillParameters(5.0, 17.0, 0.58, 0.085, 0.52, 78, 18, 0.16),
    75.0: SkillParameters(3.6, 12.0, 0.60, 0.072, 0.46, 75, 15, 0.13),
    90.0: SkillParameters(2.5, 8.0, 0.62, 0.060, 0.40, 72, 12, 0.10),
    95.0: SkillParameters(1.9, 6.0, 0.64, 0.052, 0.36, 70, 10, 0.08),
    99.0: SkillParameters(1.25, 4.2, 0.66, 0.044, 0.31, 68, 8, 0.06),
    99.5: SkillParameters(0.95, 3.4, 0.68, 0.040, 0.28, 67, 7, 0.05),
}

# Aim-sigma anchors by skill percentile (px). Skill 100 is machine-perfect.
AIM_SIGMA_ANCHORS = [
    (0.0, 7.5),
    (50.0, 5.0),
    (75.0, 3.6),
    (90.0, 2.5),
    (95.0, 1.9),
    (99.0, 1.25),
    (99.5, 0.95),
    (100.0, 0.0),
]

# Per-criterion importance weights for the overall strength function
# str(t) = 1 - weakness(t), where weakness is the weighted sum of the
# normalised per-criterion weaknesses. Unimplemented criteria contribute 0
# until their phase lands.
STRENGTH_WEIGHTS = {
    "aim": 0.30,
    "timing": 0.25,
    "randomness": 0.15,
    "correlation": 0.10,
    "curvature": 0.10,
    "fatigue": 0.10,
}

# Parameter interpolation across the skill axis. Skill now drives every
# criterion (not just aim): timing, curvature, corrections, holds, fatigue.
# Anchors below the 50th percentile extrapolate to a beginner profile.
SKILL_ANCHORS: dict[str, list[tuple[float, float]]] = {
    "aim_sigma": [(0.0, 7.5), (50.0, 5.0), (75.0, 3.6), (90.0, 2.5), (95.0, 1.9), (99.0, 1.25), (99.5, 0.95), (100.0, 0.0)],
    "timing_sigma_ms": [(0.0, 22.0), (50.0, 17.0), (75.0, 12.0), (90.0, 8.0), (95.0, 6.0), (99.0, 4.2), (99.5, 3.4), (100.0, 0.0)],
    "timing_rho": [(0.0, 0.55), (50.0, 0.58), (75.0, 0.60), (90.0, 0.62), (95.0, 0.64), (99.0, 0.66), (99.5, 0.68), (100.0, 0.68)],
    "curvature_ratio": [(0.0, 0.100), (50.0, 0.085), (75.0, 0.072), (90.0, 0.060), (95.0, 0.052), (99.0, 0.044), (99.5, 0.040), (100.0, 0.040)],
    "correction_probability": [(0.0, 0.60), (50.0, 0.52), (75.0, 0.46), (90.0, 0.40), (95.0, 0.36), (99.0, 0.31), (99.5, 0.28), (100.0, 0.28)],
    "hold_mean_ms": [(0.0, 80.0), (50.0, 78.0), (75.0, 75.0), (90.0, 72.0), (95.0, 70.0), (99.0, 68.0), (99.5, 67.0), (100.0, 67.0)],
    "hold_sigma_ms": [(0.0, 20.0), (50.0, 18.0), (75.0, 15.0), (90.0, 12.0), (95.0, 10.0), (99.0, 8.0), (99.5, 7.0), (100.0, 7.0)],
    "fatigue_gain": [(0.0, 0.18), (50.0, 0.16), (75.0, 0.13), (90.0, 0.10), (95.0, 0.08), (99.0, 0.06), (99.5, 0.05), (100.0, 0.05)],
}


def interpolate_skill_parameters(skill_level: float) -> SkillParameters:
    skill_level = float(np.clip(skill_level, 0.0, 100.0))

    def interpolate(name: str) -> float:
        anchors = SKILL_ANCHORS[name]
        return float(np.interp(skill_level, [a[0] for a in anchors], [a[1] for a in anchors]))

    return SkillParameters(
        aim_sigma=interpolate("aim_sigma"),
        timing_sigma_ms=interpolate("timing_sigma_ms"),
        timing_rho=interpolate("timing_rho"),
        curvature_ratio=interpolate("curvature_ratio"),
        correction_probability=interpolate("correction_probability"),
        hold_mean_ms=interpolate("hold_mean_ms"),
        hold_sigma_ms=interpolate("hold_sigma_ms"),
        fatigue_gain=interpolate("fatigue_gain"),
    )


def _minimum_jerk(u: np.ndarray) -> np.ndarray:
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def _normalised(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-9 else np.zeros(2)


def _ou_bridge(
    rng: np.random.Generator,
    count: int,
    sigma: float,
    correlation_ms: float,
    step_ms: float,
) -> np.ndarray:
    """Generate correlated residuals pinned to zero at both boundaries."""
    if count <= 1 or sigma <= 0:
        return np.zeros(count)
    rho = math.exp(-step_ms / max(correlation_ms, step_ms))
    values = np.zeros(count)
    innovation = sigma * math.sqrt(max(1e-9, 1 - rho * rho))
    for index in range(1, count):
        values[index] = rho * values[index - 1] + rng.normal(0, innovation)
    values -= np.linspace(values[0], values[-1], count)
    return values


class HumanTracePlanner:
    def __init__(self, map_plan: MapPlan, profile: HumanProfile, model_bundle_path: str | None = None):
        profile.validate()
        self.map = map_plan
        self.profile = profile
        self.params = interpolate_skill_parameters(profile.skill_level)
        self.rng = np.random.default_rng(profile.seed)
        self.step_ms = 1000.0 / profile.sample_rate_hz
        self.timing_state = 0.0
        self.aim_state = np.zeros(2)
        self.aim_consistency = 0.0
        self.fatigue = profile.fatigue_initial

        # Skill/effort axes (0..1). Skill sets the aim ceiling; effort sets how
        # consistently that ceiling is reached (consistency + lapse rate).
        self.skill = float(np.clip(profile.skill_level, 0.0, 100.0)) / 100.0
        self.effort = float(np.clip(profile.effort_level, 0.0, 100.0)) / 100.0
        anchor_skills = [anchor[0] for anchor in AIM_SIGMA_ANCHORS]
        anchor_sigmas = [anchor[1] for anchor in AIM_SIGMA_ANCHORS]
        self.aim_sigma_base = float(np.interp(profile.skill_level, anchor_skills, anchor_sigmas))

        # Per-player aim "habit": a fixed offset plus a directional octant bias
        # that persists across the whole run (the player's own fingerprint).
        self.aim_habit = self.rng.normal(0.0, 0.35 * self.aim_sigma_base, 2)
        self.octant_bias = self.rng.normal(0.0, 0.25 * self.aim_sigma_base, (8, 2))

        # Phase-1 diagnostics: aim error distribution and strength samples.
        self.aim_errors: list[float] = []
        self.timing_errors: list[float] = []
        self.aim_lapses = 0
        self.aim_misses = 0
        self.last_aim_error = 0.0
        self.strength_samples: list[tuple[float, float]] = []

        self.model_bundle = None
        if model_bundle_path:
            import joblib

            self.model_bundle = joblib.load(model_bundle_path)

    def generate(self) -> list[TraceFrame]:
        first_time = self.map.objects[0].start_time_ms
        # Gameplay itself starts before beatmap time zero (osu!standard uses at
        # least a two-second pre-roll). Preserve 1.5 seconds of that negative
        # clock range so maps whose first object is near 0 ms do not require an
        # impossible centre-to-object jump after the audio clock reaches zero.
        timeline_start = first_time - 1500.0
        cursor: list[tuple[float, np.ndarray]] = [(timeline_start, np.array([256.0, 192.0]))]
        key_intervals: list[tuple[float, float, int]] = []
        last_time = timeline_start
        last_position = cursor[0][1]
        key_index = 0
        previous_hit_time = float("-inf")

        for index, obj in enumerate(self.map.objects):
            strain = self._strain(index)
            self.fatigue = min(1.0, self.fatigue * 0.985 + strain * self.params.fatigue_gain * 0.015)
            hit_time = self._sample_hit_time(index, obj, strain)
            previous_hit = previous_hit_time
            previous_hit_time = hit_time

            if obj.kind == "spinner":
                self._append_spinner(cursor, obj, hit_time, last_time, last_position, strain)
                key_intervals.append((hit_time, max(hit_time + 50, obj.end_time_ms), key_index))
                key_index ^= 1
                last_time = obj.end_time_ms
                last_position = cursor[-1][1]
                self._record_strength(hit_time, obj, 0.0)
                continue

            # Visualisation-style maps stack many circles at (nearly) the same
            # instant. Re-arming a key between objects 1-2 ms apart is
            # impossible at any sane sample rate, so merge such circles into a
            # single chord press: one key-down covers every member, and the
            # cursor stays on the first member's position.
            chorded = (
                obj.kind == "circle"
                and index > 0
                and hit_time - previous_hit < self.step_ms
                and key_intervals
            )
            if chorded:
                self._record_strength(hit_time, obj, self.last_aim_error)
                begin, end, key = key_intervals[-1]
                hold = 64.0 if self.profile.perfect_baseline else max(
                    24.0,
                    self._model_sample(
                        "hold_duration_ms",
                        index,
                        self.rng.normal(self.params.hold_mean_ms, self.params.hold_sigma_ms),
                    ),
                )
                key_intervals[-1] = (begin, max(end, hit_time + hold), key)
                last_time = hit_time
                continue

            target = self._sample_target(index, obj, strain)
            self._record_strength(hit_time, obj, self.last_aim_error)
            hold = 64.0 if self.profile.perfect_baseline else max(
                24.0,
                self._model_sample(
                    "hold_duration_ms",
                    index,
                    self.rng.normal(self.params.hold_mean_ms, self.params.hold_sigma_ms),
                ),
            )

            if hit_time <= last_time:
                # The sampled press lands before the cursor is free to move
                # (large early timing error on a dense section). Keep the
                # cursor on its current path so the press misses naturally
                # instead of emitting a backwards-in-time teleport.
                key_intervals.append((hit_time, hit_time + hold, key_index))
                key_index ^= 1
                continue

            self._append_transition(cursor, last_time, hit_time, last_position, target, strain, long_settle=index == 0)
            key_intervals.append((hit_time, hit_time + hold, key_index))
            key_index ^= 1

            if obj.kind == "slider":
                self._append_slider(cursor, obj, hit_time, strain)
                last_time = max(hit_time, obj.end_time_ms)
                last_position = cursor[-1][1]
                # A normal sampled key hold must cover the complete slider. A
                # negative release offset is a slider break, which should only
                # be introduced by an explicit skill-conditioned error model.
                # Keeping a small post-tail guard also avoids quantisation at
                # the 500 Hz macro boundary releasing one frame too early.
                release_guard = 20.0 if self.profile.perfect_baseline else max(
                    self.step_ms * 2.0,
                    self.rng.normal(18.0, 6.0),
                )
                key_intervals[-1] = (
                    hit_time,
                    max(hit_time + hold, obj.end_time_ms + release_guard),
                    key_intervals[-1][2],
                )
            else:
                last_time = hit_time
                last_position = target

        key_intervals = self._ensure_key_rearm(key_intervals)
        end_time = max(last_time, max(end for _, end, _ in key_intervals)) + 100.0
        return self._resample(cursor, key_intervals, timeline_start, end_time)

    def _ensure_key_rearm(
        self, intervals: list[tuple[float, float, int]]
    ) -> list[tuple[float, float, int]]:
        """Guarantee a sampled key-up before either physical key is reused."""
        adjusted = list(intervals)
        for key in (0, 1):
            indices = [index for index, interval in enumerate(adjusted) if interval[2] == key]
            for current_index, next_index in zip(indices, indices[1:]):
                begin, end, _ = adjusted[current_index]
                next_begin = adjusted[next_index][0]
                latest_release = next_begin - self.step_ms
                if end <= latest_release:
                    continue
                if latest_release - begin < self.step_ms * 2.0:
                    raise ValueError(
                        "Trace sample rate is too low to release and re-arm a tapping key "
                        f"between objects at {begin:.3f} ms and {next_begin:.3f} ms"
                    )
                adjusted[current_index] = (begin, latest_release, key)
        return adjusted

    def _strain(self, index: int) -> float:
        if index == 0:
            return 0.0
        current = self.map.objects[index]
        previous = self.map.objects[index - 1]
        delta_ms = max(16.0, current.start_time_ms - previous.end_time_ms)
        distance = math.hypot(
            current.position.x - previous.end_position.x,
            current.position.y - previous.end_position.y,
        )
        return float(np.clip((distance / delta_ms) / 2.2, 0.0, 1.0))

    def _sample_hit_time(self, index: int, obj: MapObject, strain: float) -> float:
        if self.profile.perfect_baseline:
            self.timing_errors.append(0.0)
            return obj.start_time_ms
        sigma = self.params.timing_sigma_ms * (1.0 + 0.7 * strain + 0.35 * self.fatigue)
        rho = self.params.timing_rho
        innovation = self._model_sample("hit_error_ms", index, self.rng.normal(0.0, sigma))
        self.timing_state = rho * self.timing_state + innovation * math.sqrt(1.0 - rho**2)
        limit = max(2.0, obj.hit_windows.meh_ms * 0.82)
        error = float(np.clip(self.timing_state, -limit, limit))
        self.timing_errors.append(error)
        return obj.start_time_ms + error

    def _sample_target(self, index: int, obj: MapObject, strain: float) -> np.ndarray:
        if self.profile.perfect_baseline:
            return np.array([obj.position.x, obj.position.y])
        offset = self._aim_offset(index, obj, strain)
        result = np.array([obj.position.x, obj.position.y]) + offset
        margin = min(obj.radius * 0.12, 8.0)
        return np.array([np.clip(result[0], margin, 512 - margin), np.clip(result[1], margin, 384 - margin)])

    def _aim_offset(self, index: int, obj: MapObject, strain: float) -> np.ndarray:
        """Phase-1 aim structure: skill ceiling x effort consistency, plus the
        player's persistent aim habit and rare large aim lapses."""
        # Effort-modulated consistency: the aim quality wanders slowly during a
        # run (good moments, bad moments). Low effort means wider wandering.
        consistency_rho = math.exp(-1.0 / max(40.0, 400.0 * (1.0 - self.effort) + 40.0))
        consistency_innovation = 0.35 * (1.0 - self.effort) * math.sqrt(max(1e-9, 1.0 - consistency_rho**2))
        self.aim_consistency = consistency_rho * self.aim_consistency + self.rng.normal(0.0, consistency_innovation)
        self.aim_consistency = float(np.clip(self.aim_consistency, -1.0, 1.0))
        # Rebalanced strain scaling: hard sections degrade aim gradually
        # instead of blowing past the circle in bursts. Capped at ~1.8x base.
        sigma = self.aim_sigma_base * (1.0 + 0.5 * self.aim_consistency) * (1.0 + min(0.5 * strain + 0.3 * self.fatigue, 0.8))

        # Direction octant for the habit bias (relative to the previous object).
        if index:
            previous = self.map.objects[index - 1]
            dx = obj.position.x - previous.end_position.x
            dy = obj.position.y - previous.end_position.y
        else:
            dx, dy = obj.position.x - 256.0, obj.position.y - 192.0
        octant = int((math.atan2(dy, dx) + math.tau) % math.tau / (math.tau / 8.0)) % 8
        habit = self.aim_habit + self.octant_bias[octant]

        innovation = np.array(
            [
                self._model_sample("aim_offset_x", index, self.rng.normal(0.0, sigma)),
                self._model_sample("aim_offset_y", index, self.rng.normal(0.0, sigma)),
            ]
        )
        # Slightly less persistent aim state: individual errors, fewer streaks.
        self.aim_state = 0.45 * self.aim_state + innovation * math.sqrt(1 - 0.45**2)
        offset = self.aim_state + habit

        # Aim lapse: a rare large over/undershoot that lands outside the circle.
        lapse_probability = (1.0 - self.skill) ** 2 * (1.0 - self.effort) * (0.01 + 0.05 * strain)
        if self.rng.random() < lapse_probability:
            direction = self.rng.uniform(0.0, math.tau)
            lapse_distance = obj.radius * self.rng.uniform(1.05, 1.5)
            offset = offset + lapse_distance * np.array([math.cos(direction), math.sin(direction)])
            self.aim_lapses += 1

        error = float(np.linalg.norm(offset))
        self.last_aim_error = error
        self.aim_errors.append(error)
        if error > obj.radius:
            self.aim_misses += 1
        return offset

    def _record_strength(self, t: float, obj: MapObject, aim_error: float) -> None:
        """Sample the overall strength str(t) = 1 - weakness(t).

        Weakness is the importance-weighted sum of the six per-criterion
        weaknesses. Phase 1 implements aim and fatigue; the remaining criteria
        contribute 0 until their phases land.
        """
        radius = max(obj.radius, 1e-6)
        aim_weakness = min(1.0, aim_error / radius) if not self.profile.perfect_baseline else 0.0
        fatigue_weakness = self.fatigue * (0.5 + 0.5 * (1.0 - self.effort))
        weakness = (
            STRENGTH_WEIGHTS["aim"] * aim_weakness
            + STRENGTH_WEIGHTS["timing"] * 0.0
            + STRENGTH_WEIGHTS["randomness"] * 0.0
            + STRENGTH_WEIGHTS["correlation"] * 0.0
            + STRENGTH_WEIGHTS["curvature"] * 0.0
            + STRENGTH_WEIGHTS["fatigue"] * fatigue_weakness
        )
        self.strength_samples.append((t, 1.0 - weakness))

    @property
    def strength_stats(self) -> dict[str, float]:
        if not self.strength_samples:
            return {}
        values = [strength for _, strength in self.strength_samples]
        return {
            "initial": round(values[0], 4),
            "mean": round(float(np.mean(values)), 4),
            "p50": round(float(np.percentile(values, 50)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "min": round(min(values), 4),
            "end": round(values[-1], 4),
            "samples": len(values),
        }

    @property
    def aim_stats(self) -> dict[str, float]:
        return {
            "mean_error_px": round(float(np.mean(self.aim_errors)), 3) if self.aim_errors else 0.0,
            "p95_error_px": round(float(np.percentile(self.aim_errors, 95)), 3) if self.aim_errors else 0.0,
            "max_error_px": round(float(np.max(self.aim_errors)), 3) if self.aim_errors else 0.0,
            "lapses": self.aim_lapses,
            "misses_from_aim": self.aim_misses,
        }

    @property
    def timing_stats(self) -> dict[str, float]:
        return {
            "mean_error_ms": round(float(np.mean(self.timing_errors)), 3) if self.timing_errors else 0.0,
            "std_error_ms": round(float(np.std(self.timing_errors)), 3) if self.timing_errors else 0.0,
            "p95_abs_error_ms": round(float(np.percentile(np.abs(self.timing_errors), 95)), 3) if self.timing_errors else 0.0,
            "max_abs_error_ms": round(float(np.max(np.abs(self.timing_errors))), 3) if self.timing_errors else 0.0,
        }

    def _model_sample(self, target: str, index: int, fallback: float) -> float:
        if self.model_bundle is None:
            return float(fallback)
        models = self.model_bundle.get("models", {}).get(target)
        if not models:
            return float(fallback)
        import pandas as pd

        obj = self.map.objects[index]
        if index:
            previous = self.map.objects[index - 1]
            distance = math.hypot(obj.position.x - previous.end_position.x, obj.position.y - previous.end_position.y)
            interval = obj.start_time_ms - previous.end_time_ms
        else:
            distance, interval = 0.0, 1000.0
        strain = min(1.0, distance / max(16.0, interval) / 2.2)
        if obj.kind in {"slider", "spinner"}:
            context = obj.kind
        elif interval < 130:
            context = "stream"
        elif interval < 220:
            context = "burst"
        elif distance > 120:
            context = "jump"
        else:
            context = "transition"
        features = pd.DataFrame(
            [{"distance": distance, "interval_ms": interval, "strain": strain, "clock_rate": self.map.clock_rate,
              "hidden": "HD" in self.map.mods,
              "context": context, "object_kind": obj.kind}]
        )
        q10, q50, q90 = (float(models[str(q)].predict(features)[0]) for q in (0.1, 0.5, 0.9))
        u = self.rng.random()
        if u < 0.5:
            return q10 + (q50 - q10) * (u / 0.5)
        return q50 + (q90 - q50) * ((u - 0.5) / 0.5)

    def _append_transition(
        self,
        points: list[tuple[float, np.ndarray]],
        available_from: float,
        hit_time: float,
        start: np.ndarray,
        target: np.ndarray,
        strain: float,
        long_settle: bool = False,
    ) -> None:
        distance = float(np.linalg.norm(target - start))
        fitts_ms = (70.0 + 92.0 * math.log2(distance / 64.0 + 1.0)) * (1.08 - self.profile.percentile / 600.0)
        available = max(self.step_ms, hit_time - available_from)

        # OS/game cursor delivery can lag the dispatched position by one or two
        # game frames (~12-20 ms observed on the benchmark host). A key press
        # processed while the cursor is still mid-transition misses fast jumps
        # ("too slow to catch up"). Reserve a settle window so the cursor is
        # already on the target when the press is processed.
        # The game's input pipeline can lag one or two frames at map start
        # (first seconds: shaders, high-performance session). The first target
        # gets a much longer settle so the cursor parks early and the laggy
        # startup sampling catches up before the first press.
        settle = min(120.0 if long_settle else 24.0, max(2.0, available * 0.5))
        settle = min(settle, available - self.step_ms)
        move_budget = max(self.step_ms, available - settle)
        duration = min(move_budget, max(45.0, fitts_ms * (1.0 + 0.25 * strain)))

        # Never collapse a movement into a sub-millisecond teleport. If the
        # available time cannot fit the distance at a plausible speed, the
        # cursor simply arrives late (and the press naturally misses) instead
        # of teleporting.
        max_transition_speed_px_s = 50_000.0
        min_duration = max(self.step_ms, distance / max_transition_speed_px_s)
        if duration < min_duration:
            duration = min_duration
        start_time = max(available_from, hit_time - settle - duration)
        if start_time > points[-1][0] + self.step_ms:
            points.append((start_time, start.copy()))

        arrival_time = hit_time - settle
        if start_time >= available_from and arrival_time - start_time < min_duration:
            arrival_time = start_time + min_duration
        count = max(2, int(math.ceil((arrival_time - start_time) / self.step_ms)) + 1)
        times = np.linspace(start_time, arrival_time, count)
        u = np.linspace(0.0, 1.0, count)
        progress = _minimum_jerk(u)
        direction = _normalised(target - start)
        normal = np.array([-direction[1], direction[0]])
        curve = 0.0 if self.profile.perfect_baseline else self.rng.normal(
            0.0, max(0.5, distance * self.params.curvature_ratio)
        )
        residual = np.zeros(count) if self.profile.perfect_baseline else _ou_bridge(
            self.rng,
            count,
            self.params.aim_sigma * (0.25 + 0.55 * strain),
            self.rng.uniform(35.0, 110.0),
            self.step_ms,
        )
        correction = np.zeros(count)
        if not self.profile.perfect_baseline and self.rng.random() < self.params.correction_probability:
            centre = self.rng.uniform(0.68, 0.88)
            width = self.rng.uniform(0.06, 0.14)
            correction = self.rng.normal(0.0, self.params.aim_sigma * 0.9) * np.exp(-0.5 * ((u - centre) / width) ** 2)

        for time_ms, amount, phase, noise, correction_value in zip(times, progress, 4 * u * (1 - u), residual, correction):
            position = start + (target - start) * amount + normal * (curve * phase + noise + correction_value)
            points.append((float(time_ms), position))

        if arrival_time + self.step_ms < hit_time:
            points.append((hit_time, target.copy()))

    def _append_slider(self, points: list[tuple[float, np.ndarray]], obj: MapObject, hit_time: float, strain: float) -> None:
        if len(obj.path_samples) < 2 or obj.end_time_ms <= obj.start_time_ms:
            return
        duration = obj.end_time_ms - obj.start_time_ms
        # Slider motion follows the beatmap clock, not the sampled head-hit
        # clock. An early hit waits at the head until the slider starts; a late
        # hit catches up from the head to the curve's current position. Shifting
        # the entire curve by hit error made the tail late/early by the same
        # amount and could lose repeats or tails despite a valid head hit.
        tracking_start = max(hit_time, obj.start_time_ms)
        tracking_duration = max(self.step_ms, obj.end_time_ms - tracking_start)
        count = max(2, int(math.ceil(tracking_duration / self.step_ms)) + 1)
        times = np.linspace(tracking_start, obj.end_time_ms, count)
        path = np.array([[point.x, point.y] for point in obj.path_samples])
        noise = np.zeros(count) if self.profile.perfect_baseline else _ou_bridge(
            self.rng, count, self.params.aim_sigma * (0.25 + 0.45 * strain), 80.0, self.step_ms
        )
        lag = 0.0 if self.profile.perfect_baseline else float(
            np.clip(self.rng.normal(0.0, 0.012 + strain * 0.01), -0.035, 0.035)
        )
        entry_position = points[-1][1].copy()
        first_base: np.ndarray | None = None
        for index, time_ms in enumerate(times):
            # MapPlan path samples come from lazer's StackedPositionAt() over
            # the complete [start, end] slider duration. They already include
            # every repeat and reversal. Applying repeat_count here a second
            # time made repeat sliders traverse too many spans and immediately
            # broke tracking at the first repeat slider in real maps.
            progress = np.clip((time_ms - obj.start_time_ms) / duration + lag, 0.0, 1.0)
            path_index = progress * (len(path) - 1)
            lower = int(math.floor(path_index))
            upper = min(len(path) - 1, lower + 1)
            blend = path_index - lower
            base = path[lower] * (1 - blend) + path[upper] * blend
            if first_base is None:
                first_base = base.copy()
            entry_correction = (entry_position - first_base) * math.exp(-(time_ms - times[0]) / 80.0)
            tangent = _normalised(path[upper] - path[max(0, lower - 1)])
            normal = np.array([-tangent[1], tangent[0]])
            points.append((float(time_ms), base + entry_correction + normal * noise[index]))

    def _append_spinner(
        self,
        points: list[tuple[float, np.ndarray]],
        obj: MapObject,
        hit_time: float,
        available_from: float,
        start: np.ndarray,
        strain: float,
    ) -> None:
        duration = max(self.step_ms, obj.end_time_ms - obj.start_time_ms)
        count = max(2, int(math.ceil(duration / self.step_ms)) + 1)
        if self.profile.perfect_baseline:
            centre = np.array([256.0, 192.0])
            radius_x = radius_y = 90.0
            rpm = 480.0
            direction = 1.0
            phase = 0.0
        else:
            centre = np.array([256.0, 192.0]) + self.rng.normal(0.0, self.params.aim_sigma * 0.7, 2)
            radius_x = self.rng.uniform(70.0, 105.0)
            radius_y = radius_x * self.rng.uniform(0.82, 1.08)
            rpm = self.rng.uniform(300.0, 420.0) * (0.92 + self.profile.percentile / 1250.0)
            direction = -1.0 if self.rng.random() < 0.5 else 1.0
            phase = self.rng.uniform(0.0, math.tau)
        entry = centre + np.array([radius_x * math.cos(phase), radius_y * math.sin(phase)])
        self._append_transition(points, available_from, hit_time, start, entry, strain, long_settle=True)
        times = np.linspace(hit_time, hit_time + duration, count)
        angles = phase + direction * ((times - times[0]) / 1000.0) * rpm / 60.0 * math.tau
        radial_noise = np.zeros(count) if self.profile.perfect_baseline else _ou_bridge(
            self.rng, count, self.params.aim_sigma, 80.0, self.step_ms
        )
        for time_ms, angle, noise in zip(times, angles, radial_noise):
            position = centre + np.array([(radius_x + noise) * math.cos(angle), (radius_y + noise) * math.sin(angle)])
            points.append((float(time_ms), position))

    def _resample(
        self,
        cursor: list[tuple[float, np.ndarray]],
        intervals: list[tuple[float, float, int]],
        start_ms: float,
        end_ms: float,
    ) -> list[TraceFrame]:
        # Keep only strictly increasing path times. Overlapping presses can
        # leave stale/backwards points that corrupt np.interp into teleports.
        ordered: list[tuple[float, np.ndarray]] = []
        last_kept_time = float("-inf")
        for time_ms, position in cursor:
            time_ms = round(float(time_ms), 6)
            if time_ms <= last_kept_time:
                continue
            ordered.append((time_ms, position))
            last_kept_time = time_ms
        source_times = np.array([item[0] for item in ordered])
        xs = np.array([item[1][0] for item in ordered])
        ys = np.array([item[1][1] for item in ordered])
        regular_times = np.arange(start_ms, end_ms + self.step_ms * 0.5, self.step_ms)
        # Preserve the nominal sampling cadence, but also emit frames at every
        # input transition.  A grid anchored at ``start_ms`` will usually not
        # contain later object timestamps exactly.  Without these event frames
        # a key-down is delayed to the next sample and the cursor is still
        # interpolating towards the hit object when the key is pressed.  The
        # runner deliberately dispatches cursor position with every key change,
        # so merging transition timestamps here makes that atomic dispatch land
        # on the planned hit point while retaining 500 Hz path sampling.
        event_times = np.array(
            [
                time_ms
                for begin, end, _key in intervals
                for time_ms in (begin, end)
                if start_ms <= time_ms <= end_ms
            ],
            dtype=float,
        )
        if event_times.size:
            frame_times = np.unique(np.round(np.concatenate((regular_times, event_times)), 6))
        else:
            frame_times = np.unique(np.round(regular_times, 6))
        out_x = np.interp(frame_times, source_times, xs)
        out_y = np.interp(frame_times, source_times, ys)

        # Overlapping presses can briefly make the source path go backward,
        # which np.interp would turn into a teleport. In human profile mode,
        # clamp every resampled step to a plausible speed cap (perfect mode
        # keeps its own much higher machine ceiling).
        if not self.profile.perfect_baseline:
            max_profile_speed_px_s = 90_000.0
            for index in range(1, len(out_x)):
                delta_seconds = (frame_times[index] - frame_times[index - 1]) / 1000.0
                if delta_seconds <= 0:
                    continue
                delta_x = out_x[index] - out_x[index - 1]
                delta_y = out_y[index] - out_y[index - 1]
                distance = math.hypot(delta_x, delta_y)
                allowed = max_profile_speed_px_s * delta_seconds
                if distance > allowed:
                    scale = allowed / distance
                    out_x[index] = out_x[index - 1] + delta_x * scale
                    out_y[index] = out_y[index - 1] + delta_y * scale

        key_states: list[np.ndarray] = []
        for key in (0, 1):
            begins = np.sort(np.array([begin for begin, _end, interval_key in intervals if interval_key == key]))
            ends = np.sort(np.array([end for _begin, end, interval_key in intervals if interval_key == key]))
            # Intervals are begin-inclusive and end-exclusive.  Counting all
            # starts/ends at each timestamp preserves that definition even
            # when holds overlap, while avoiding an O(frames * objects) scan.
            active_count = np.searchsorted(begins, frame_times, side="right") - np.searchsorted(
                ends, frame_times, side="right"
            )
            key_states.append(active_count > 0)
        frames: list[TraceFrame] = []
        previous_us = -1
        for time_ms, x, y, k1, k2 in zip(frame_times, out_x, out_y, key_states[0], key_states[1]):
            relative_us = max(previous_us + 1, int(round((time_ms - start_ms) * 1000.0)))
            previous_us = relative_us
            frames.append(
                TraceFrame(
                    relative_us,
                    float(np.clip(x, -64, 576)),
                    float(np.clip(y, -64, 448)),
                    bool(k1),
                    bool(k2),
                )
            )
        return frames
