from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .schemas import HumanProfile, MapObject, MapPlan, TraceFrame


# Bump when trace-generation behaviour changes so the runner's content-addressed
# trace cache is invalidated (the configuration hash includes this string).
PLANNER_VERSION = "timing-sync-v2.11"


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
# v4 nerfs the low-skill end hard: a 10% player now routinely lands outside a
# radius-36 circle instead of pinning the centre.
AIM_SIGMA_ANCHORS = [
    (0.0, 16.0),
    (20.0, 12.5),
    (50.0, 8.5),
    (75.0, 5.5),
    (90.0, 3.5),
    (95.0, 2.6),
    (99.0, 1.6),
    (99.5, 1.15),
    (100.0, 0.0),
]

# Per-skill cursor speed ceiling in osu! playfield px/s (512x384 space).
# The old model's 50k transition cap and 90k resample guard allowed inhuman
# flicks; real humans on the benchmark host move at ~500-4000 px/s. Short
# moves are additionally capped lower than long ones (see _speed_ceiling).
SPEED_CEILING_ANCHORS = [
    (0.0, 2_400.0),
    (20.0, 2_800.0),
    (50.0, 4_200.0),
    (75.0, 6_000.0),
    (90.0, 8_000.0),
    (95.0, 9_500.0),
    (99.0, 12_000.0),
    (99.5, 13_000.0),
    (100.0, 1_000_000.0),
]

# Spinner rotation speed by skill (RPM). A 10% player spins at ~120 RPM, not
# the 300-420 RPM the old model always used; orbit speed would otherwise
# exceed the movement ceiling for low-skill profiles.
SPINNER_RPM_ANCHORS = [
    (0.0, 120.0),
    (50.0, 250.0),
    (75.0, 330.0),
    (90.0, 380.0),
    (95.0, 400.0),
    (99.5, 430.0),
    (100.0, 480.0),
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
    "aim_sigma": [(0.0, 16.0), (20.0, 12.5), (50.0, 8.5), (75.0, 5.5), (90.0, 3.5), (95.0, 2.6), (99.0, 1.6), (99.5, 1.15), (100.0, 0.0)],
    # v2.1: base timing sigma reduced ~10% because the new pressure/context
    # system adds pattern-dependent inflation on top; a clean isolated note is
    # now tighter than before while streams/bursts are looser (see
    # TIMING_CONTEXT_SCALES). v2.4: another ~8% cut so 90%+ runs produce a
    # realistic OK count (~25-30) instead of drifting toward MEHs/misses.
    "timing_sigma_ms": [(0.0, 28.0), (50.0, 18.2), (75.0, 12.4), (90.0, 8.3), (95.0, 6.2), (99.0, 4.1), (99.5, 3.3), (100.0, 0.0)],
    "timing_rho": [(0.0, 0.55), (50.0, 0.58), (75.0, 0.60), (90.0, 0.62), (95.0, 0.64), (99.0, 0.66), (99.5, 0.68), (100.0, 0.68)],
    "curvature_ratio": [(0.0, 0.050), (50.0, 0.042), (75.0, 0.035), (90.0, 0.028), (95.0, 0.024), (99.0, 0.020), (99.5, 0.018), (100.0, 0.018)],
    "correction_probability": [(0.0, 0.35), (50.0, 0.30), (75.0, 0.26), (90.0, 0.22), (95.0, 0.20), (99.0, 0.17), (99.5, 0.15), (100.0, 0.15)],
    "hold_mean_ms": [(0.0, 80.0), (50.0, 78.0), (75.0, 75.0), (90.0, 72.0), (95.0, 70.0), (99.0, 68.0), (99.5, 67.0), (100.0, 67.0)],
    "hold_sigma_ms": [(0.0, 20.0), (50.0, 18.0), (75.0, 15.0), (90.0, 12.0), (95.0, 10.0), (99.0, 8.0), (99.5, 7.0), (100.0, 7.0)],
    "fatigue_gain": [(0.0, 0.18), (50.0, 0.16), (75.0, 0.13), (90.0, 0.10), (95.0, 0.08), (99.0, 0.06), (99.5, 0.05), (100.0, 0.05)],
}

# Phase-2 (v2.1) timing-context scales. Real players time streams and bursts
# much less consistently than isolated notes (cognitive/motor load), while
# slider heads are deliberately lenient (ScoreV1 only needs the MEH window for
# the head) and jumps are usually tapped to the beat, so their timing stays
# steadier than their aim.
TIMING_CONTEXT_SCALES = {
    "stream": 1.28,
    "burst": 1.05,
    "transition": 1.00,
    "jump": 0.94,
    "slider": 0.70,
    "spinner": 1.00,
}

# Rare "panic tap" probability multiplier per context: streams/bursts produce
# most mistaps, isolated notes almost none.
MISTAP_CONTEXT_SCALES = {
    "stream": 1.9,
    "burst": 1.45,
    "transition": 0.35,
    "jump": 0.55,
    "slider": 0.22,
    "spinner": 0.0,
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
        self.aim_bias_state = 0.0
        # v1.5 continuous-motion states: a slow absolute drift and a persistent
        # curve direction. Unlike per-transition random draws, these evolve
        # smoothly across the whole run so adjacent moves keep one continuous
        # path instead of resetting into independent squiggles.
        self.wander = np.zeros(2)
        self.curve_state = 0.0
        self.fatigue = profile.fatigue_initial
        # Phase-2 (v2.1) shared aim/timing state: one pressure signal and one
        # short stress-episode signal drive both systems, so a hard pattern
        # degrades aim and timing together instead of only one of them.
        self.pressure = 0.0
        self.stress_episode = 0.0
        self.rush_draw = 0.0

        # Skill/effort axes (0..1). Skill sets the aim ceiling; effort sets how
        # consistently that ceiling is reached (consistency + lapse rate).
        self.skill = float(np.clip(profile.skill_level, 0.0, 100.0)) / 100.0
        self.effort = float(np.clip(profile.effort_level, 0.0, 100.0)) / 100.0
        # Per-run early/late leaning plus a slow session drift (effort scales
        # the drift). Real hit-error meters show each player consistently
        # leaning early or late by a few ms, with the lean wandering over the
        # course of a play.
        if self.profile.perfect_baseline:
            self.timing_bias_ms = 0.0
            self.timing_drift_per_ms = 0.0
        else:
            bias_sigma = 1.5 + 12.0 * (1.0 - self.skill) ** 2
            self.timing_bias_ms = float(self.rng.normal(0.0, bias_sigma))
            drift_sigma = 0.8 + 4.5 * (1.0 - self.effort) ** 2
            self.timing_drift_per_ms = float(self.rng.normal(0.0, drift_sigma / 60_000.0))
        anchor_skills = [anchor[0] for anchor in AIM_SIGMA_ANCHORS]
        anchor_sigmas = [anchor[1] for anchor in AIM_SIGMA_ANCHORS]
        self.aim_sigma_base = float(np.interp(profile.skill_level, anchor_skills, anchor_sigmas))

        # Phase-1 diagnostics: aim error distribution and strength samples.
        self.aim_errors: list[float] = []
        self.timing_errors: list[float] = []
        self.aim_lapses = 0
        self.aim_misses = 0
        self.timing_mistaps = 0
        self.ghost_presses = 0
        self.ghost_flags: list[bool] = []
        self.idle_wanders = 0
        self.last_aim_error = 0.0
        self.strength_samples: list[tuple[float, float]] = []

        self.model_bundle = None
        if model_bundle_path:
            import joblib

            self.model_bundle = joblib.load(model_bundle_path)

    def _speed_ceiling(self, distance: float) -> float:
        """Per-move speed ceiling: skill sets the base, distance scales it so
        short corrections stay slower/more controlled while long jumps may use
        more of the ceiling. A gentle distance bias (v1.6) gives short moves a
        little more speed allowance and long moves a little less, so very short
        moves are slightly easier to land and long jumps are slightly harder
        without changing the human speed envelope."""
        anchors = SPEED_CEILING_ANCHORS
        base = float(np.interp(self.skill * 100.0, [a[0] for a in anchors], [a[1] for a in anchors]))
        # Distance bias: 1.03 on very short moves, 1.0 around the 160 px
        # crossover, down to 0.93 on long jumps (with a soft floor so real
        # players keep a plausible long-jump speed).
        if distance <= 80.0:
            scale = 1.03
        elif distance <= 160.0:
            scale = 1.03 - 0.03 * (distance - 80.0) / 80.0
        elif distance <= 320.0:
            scale = 1.0 - 0.07 * (distance - 160.0) / 160.0
        else:
            scale = 0.93
        scale = float(np.clip(0.55 + 0.45 * (distance / 260.0), 0.55, 1.15) * scale)
        return base * scale

    def _advance_motion_states(self, dt_ms: float) -> None:
        """Advance the continuous wander/curve states over a time step so the
        path stays smooth and correlated across object transitions."""
        if self.profile.perfect_baseline:
            return
        dt = max(1.0, dt_ms)
        wander_rho = math.exp(-dt / 650.0)
        wander_sigma = self.aim_sigma_base * (0.10 + 0.08 * (1.0 - self.effort))
        innovation_scale = math.sqrt(max(1e-9, 1.0 - wander_rho * wander_rho))
        self.wander = wander_rho * self.wander + self.rng.normal(0.0, wander_sigma * innovation_scale, 2)
        curve_rho = math.exp(-dt / 800.0)
        self.curve_state = curve_rho * self.curve_state + self.rng.normal(
            0.0, math.sqrt(max(1e-9, 1.0 - curve_rho * curve_rho))
        )

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
            self.ghost_flags.append(False)
            strain = self._strain(index)
            context = self._context(index, obj, strain)
            self._update_pressure(index, obj, strain, context)
            self.fatigue = min(1.0, self.fatigue * 0.985 + strain * self.params.fatigue_gain * 0.015)
            hit_time = self._sample_hit_time(index, obj, strain, context)
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
                and 0.0 <= hit_time - previous_hit < self.step_ms
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

            target = self._sample_target(index, obj, strain, context)
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
                self.ghost_presses += 1
                miss_nerf = self._miss_nerf(strain, self.pressure)
                if self.rng.random() >= miss_nerf and not self.profile.perfect_baseline:
                    # v2.9 catch-up tap: the player rushes, then taps late on
                    # arrival instead of combo-breaking - the press lands in
                    # the OK/MEH band as a late hit. The cursor has the extra
                    # time to arrive, so this converts a dense-section miss
                    # into an inaccuracy, which is what high-accuracy players
                    # actually do.
                    if self.rng.random() < 0.65:
                        late_error = obj.hit_windows.great_ms * self.rng.uniform(1.15, 1.9)
                    else:
                        late_error = obj.hit_windows.ok_ms * self.rng.uniform(1.05, 1.5)
                    hit_time = obj.start_time_ms + late_error
                    self.timing_errors[-1] = late_error
                    self.ghost_flags[-1] = False
                    previous_hit_time = hit_time
                else:
                    # The sampled press lands before the cursor is free to move
                    # (large early timing error on a dense section). Keep the
                    # cursor on its current path so the press misses naturally
                    # instead of emitting a backwards-in-time teleport.
                    self.ghost_flags[-1] = True
                    key_intervals.append((hit_time, hit_time + hold, key_index))
                    key_index ^= 1
                    continue

            # When there is genuinely spare time (long gaps, map breaks, the
            # pre-roll), let the cursor doodle instead of parking on the next
            # note. The wander ends early enough for the real approach.
            wander_end_time, wander_end_position = self._append_idle_wander(
                cursor, last_time, hit_time, last_position, target, strain, long_settle=index == 0
            )
            if wander_end_time is not None:
                last_time = wander_end_time
                last_position = wander_end_position
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
                # The transition path may have been truncated at the press time
                # when the speed ceiling left the cursor mid-flight (late
                # arrival -> natural miss). Continue the next move from where
                # the cursor actually is, not from the planned target.
                last_time = max(hit_time, cursor[-1][0])
                last_position = cursor[-1][1]

        key_intervals = self._ensure_key_rearm(key_intervals)
        end_time = max(last_time, max(end for _, end, _ in key_intervals)) + 100.0
        return self._resample(cursor, key_intervals, timeline_start, end_time)

    def _ensure_key_rearm(
        self, intervals: list[tuple[float, float, int]]
    ) -> list[tuple[float, float, int]]:
        """Guarantee a sampled key-up before either physical key is reused."""
        adjusted: list[tuple[float, float, int]] = []
        for key in (0, 1):
            chain = sorted(
                (interval for interval in intervals if interval[2] == key),
                key=lambda interval: interval[0],
            )
            for begin, end, interval_key in chain:
                if not adjusted or adjusted[-1][2] != key:
                    adjusted.append((begin, end, interval_key))
                    continue
                previous_begin, previous_end, previous_key = adjusted[-1]
                latest_release = begin - self.step_ms
                if previous_end <= latest_release:
                    adjusted.append((begin, end, interval_key))
                    continue
                if latest_release - previous_begin >= self.step_ms * 2.0:
                    # Enough room to release and re-arm: clip the previous
                    # hold so both presses stay distinct.
                    adjusted[-1] = (previous_begin, latest_release, previous_key)
                    adjusted.append((begin, end, interval_key))
                else:
                    # Two same-key presses closer than any release/re-arm
                    # cycle (dense sections with large timing errors). A human
                    # would cover both with one press: merge the holds.
                    adjusted[-1] = (previous_begin, max(previous_end, end), previous_key)
        return sorted(adjusted, key=lambda interval: interval[0])

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

    def _context(self, index: int, obj: MapObject, strain: float) -> str:
        """Classify the pattern the object sits in (stream/burst/jump/...).

        Mirrors research/human-sim/src/human_sim/dataset.py so the simulator
        and the replay-analysis pipeline agree on what "stream" means.
        """
        if index == 0:
            return "transition"
        previous = self.map.objects[index - 1]
        interval = obj.start_time_ms - previous.end_time_ms
        distance = math.hypot(
            obj.position.x - previous.end_position.x,
            obj.position.y - previous.end_position.y,
        )
        if obj.kind in {"slider", "spinner"}:
            return obj.kind
        if interval < 130:
            return "stream"
        if interval < 220:
            return "burst"
        if distance > 120:
            return "jump"
        return "transition"

    def _miss_nerf(self, strain: float, pressure: float) -> float:
        """v2.9: above 50% skill, miss probability collapses.

        Real players at 90%+ accuracy simply do not combo-break like the old
        model did. The nerf is 1.0 at skill 50 and falls roughly quadratically
        to a ~8% floor at skill 100, but very fast / very complex sections
        (high strain or pressure) partially escape it, so a genuinely overfaced
        moment can still miss. It never reaches zero.
        """
        if self.skill <= 0.5:
            return 1.0
        drop = (self.skill - 0.5) / 0.5
        base = 0.08 + 0.92 * (1.0 - drop) ** 2
        escape = min(1.0, 0.55 * strain + 0.35 * pressure)
        return base + (1.0 - base) * escape

    def _update_pressure(self, index: int, obj: MapObject, strain: float, context: str) -> None:
        """Advance the shared aim/timing pressure state and stress episodes.

        One AR(1)-style "pressure" signal drives both aim and timing, so dense
        or fast patterns degrade both systems together (the aim/timing
        correlation real players show under load). A shorter stress-episode
        state occasionally spikes on top for a note or two - the shared
        "freak-out" moments where both the tap and the aim fall apart.
        """
        if self.profile.perfect_baseline:
            return
        if index == 0:
            delta_ms = 1000.0
        else:
            previous = self.map.objects[index - 1]
            delta_ms = max(16.0, obj.start_time_ms - previous.end_time_ms)
        context_factor = {
            "stream": 1.00,
            "burst": 0.65,
            "jump": 0.45,
            "transition": 0.18,
            "slider": 0.30,
            "spinner": 0.12,
        }[context]
        skill_damp = 1.25 - 0.55 * self.skill
        target = min(
            1.0,
            (strain * (0.55 + 0.45 * context_factor) + 0.28 * context_factor + 0.30 * self.fatigue)
            * skill_damp,
        )
        rho = math.exp(-delta_ms / 500.0)
        self.pressure = rho * self.pressure + (1.0 - rho) * target
        episode_probability = 0.035 * self.pressure * (0.6 + 0.6 * (1.0 - self.effort))
        if self.rng.random() < episode_probability:
            self.stress_episode = min(1.6, self.stress_episode + self.rng.uniform(0.5, 1.0))
        else:
            self.stress_episode *= math.exp(-delta_ms / 220.0)
        # One shared signed "rush" draw per object: positive means the player
        # is rushing (tapping early and cutting the aim short), negative means
        # late + overshooting. Same draw feeds both systems, which is what
        # creates the real-world aim/timing correlation under pressure.
        self.rush_draw = float(self.rng.normal(0.0, 1.0))

    def _sample_hit_time(self, index: int, obj: MapObject, strain: float, context: str) -> float:
        if self.profile.perfect_baseline:
            self.timing_errors.append(0.0)
            return obj.start_time_ms
        skill_gain = 0.15 + 0.85 * (1.0 - self.skill)
        context_scale = 1.0 + (TIMING_CONTEXT_SCALES[context] - 1.0) * (0.25 + 0.75 * (1.0 - self.skill))
        sigma = (
            self.params.timing_sigma_ms
            * (1.0 + 0.45 * (1.0 - self.effort))
            * context_scale
            * (1.0 + (0.45 * self.pressure + 0.30 * self.stress_episode) * skill_gain)
            * (1.0 + 0.35 * self.fatigue)
        )
        # v2.4: tighter error cap (74% of the MEH window instead of 82%) so
        # the ordinary AR(1) tap can never drift far enough to ghost-press or
        # become a 50/miss; those classes now come almost exclusively from the
        # explicit (OK-biased) mistap component.
        limit = max(2.0, obj.hit_windows.meh_ms * 0.74)
        # Persistent per-run early/late leaning + slow session drift. The core
        # AR(1) stays zero-mean; the bias shifts the whole run (like the hit
        # error meter showing "-8.4 - 3.1 avg").
        elapsed_ms = obj.start_time_ms - self.map.objects[0].start_time_ms
        bias_now = self.timing_bias_ms + self.timing_drift_per_ms * elapsed_ms

        # Rare timing inaccuracy ("mistap"): the error is shaped so that at
        # high accuracy the player produces far more OKs than MEHs or misses
        # (real-player bias). The magnitude is a 3-band mixture - ~65% OK-band,
        # ~25% MEH-band, ~10% beyond the MEH window into a timing miss. The
        # probability falls quadratically with skill and additionally with
        # effort, so high-skill/high-effort runs have almost no timing misses.
        mistap_probability = (
            (1.0 - self.skill) ** 2
            * (0.07 + 0.10 * strain + 0.06 * self.pressure)
            # v2.6: effort suppresses mistaps much harder at high effort
            # (ef 95/100 nearly eliminates them; ef 50 still meaningful).
            * (0.15 + 0.85 * (1.0 - self.effort) ** 2)
            * MISTAP_CONTEXT_SCALES[context]
        )
        if self.rng.random() < mistap_probability:
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            miss_nerf = self._miss_nerf(strain, self.pressure)
            band = self.rng.random()
            if band < 0.65:
                # OK band: mostly 100s with a few 50s at the window edge.
                magnitude = obj.hit_windows.ok_ms * self.rng.uniform(0.78, 1.02)
            elif band < 0.95:
                # MEH band: 50s.
                magnitude = obj.hit_windows.meh_ms * self.rng.uniform(0.72, 0.97)
            elif self.rng.random() < miss_nerf:
                # Very rare panic miss beyond the MEH window - v2.9: further
                # suppressed above 50% skill so 90%+ runs keep misses near
                # zero; downgrades to a MEH otherwise.
                magnitude = obj.hit_windows.meh_ms * self.rng.uniform(1.02, 1.28)
            else:
                magnitude = obj.hit_windows.meh_ms * self.rng.uniform(0.72, 0.97)
            error = sign * magnitude
            self.timing_mistaps += 1
            # A mistap is an isolated event; do not drag the AR state into it.
            self.timing_state = 0.55 * self.timing_state
        else:
            rho = self.params.timing_rho
            innovation = self._model_sample("hit_error_ms", index, self.rng.normal(0.0, sigma))
            self.timing_state = rho * self.timing_state + innovation * math.sqrt(1.0 - rho**2)
            rush_timing = self.rush_draw * 0.70 * self.pressure * sigma * skill_gain
            error = float(np.clip(self.timing_state + bias_now - rush_timing, -limit, limit))
        self.timing_errors.append(error)
        return obj.start_time_ms + error

    def _sample_target(self, index: int, obj: MapObject, strain: float, context: str) -> np.ndarray:
        if self.profile.perfect_baseline:
            return np.array([obj.position.x, obj.position.y])
        offset = self._aim_offset(index, obj, strain, context)
        result = np.array([obj.position.x, obj.position.y]) + offset
        margin = min(obj.radius * 0.12, 8.0)
        return np.array([np.clip(result[0], margin, 512 - margin), np.clip(result[1], margin, 384 - margin)])

    def _aim_offset(self, index: int, obj: MapObject, strain: float, context: str) -> np.ndarray:
        """v2.5 aim landing: centre-anchored, undershoot-biased, speed-coupled.

        There is no persistent habit/octant pattern. Real players aim for the
        middle of the circle and slightly favour undershooting (landing short
        of the target) over overshooting. The landing is therefore an
        elliptical Gaussian anchored just short of the centre along the
        approach axis: overwhelming probability in a rough neighbourhood of
        the centre, the entry-side "beginning" and the middle most likely, the
        far side and the edges unlikely. The spread still grows with speed,
        strain, effort and fatigue, so hard maps at low skill keep their
        natural misses while a 90%+ run has huge margin inside the circle.
        """
        if index:
            previous = self.map.objects[index - 1]
            dx = obj.position.x - previous.end_position.x
            dy = obj.position.y - previous.end_position.y
        else:
            dx, dy = obj.position.x - 256.0, obj.position.y - 192.0
        distance = math.hypot(dx, dy)
        approach_angle = math.atan2(dy, dx)

        # How hard this move has to be: Fitts natural pace vs the skill speed
        # ceiling. Fast moves (long jumps at low skill) inflate the spread.
        fitts_ms = (70.0 + 92.0 * math.log2(distance / 64.0 + 1.0)) * (1.08 - self.profile.percentile / 600.0)
        fitts_speed = distance / max(1.0, fitts_ms) * 1000.0
        speed_ratio = float(np.clip(fitts_speed / max(1.0, self._speed_ceiling(distance)), 0.0, 1.0))

        # Distance accuracy factor (v1.6): tightens short moves (~0.94x aim
        # spread at very short range) and loosens long moves (~1.06x), so
        # close targets are more reliably hit and far targets miss a little
        # more often. Smooth, small, and speed-independent.
        if distance <= 80.0:
            distance_factor = 0.92
        elif distance <= 160.0:
            distance_factor = 0.92 + 0.08 * (distance - 80.0) / 80.0
        elif distance <= 320.0:
            distance_factor = 1.0 + 0.10 * (distance - 160.0) / 160.0
        else:
            distance_factor = 1.10

        skill_gain = 0.15 + 0.85 * (1.0 - self.skill)
        sigma = (
            self.aim_sigma_base
            # Skill-dependent aim compensation: low-skill profiles gain timing
            # misses (mistaps/ghost presses) so their aim must soften a little
            # to keep the overall accuracy ladder unchanged; top profiles keep
            # the v1.6 aim tightness.
            * (0.74 + 0.26 * self.skill)
            # v2.4: final ~8% landing-tightening so 90%+ runs keep near-misses
            # inside the circle (1-2 misses instead of 4-5); hard-map misses
            # stay driven by strain/speed, not this constant.
            * 0.92
            * (1.0 + 0.6 * (1.0 - self.effort))
            # v2.4: slightly gentler speed inflation so a 90%+ run keeps its
            # hard-jump misses down to the rare case. v2.6: gentler again -
            # fast/dense sections spread the landing less, so overfaced runs
            # produce OK/MEH misses-of-timing instead of aim combo breaks.
            * (1.0 + speed_ratio * (0.24 + 0.48 * (1.0 - self.skill)))
            * (1.0 + min(0.45 * strain + 0.25 * self.fatigue, 0.7))
            * distance_factor
            * (1.0 + (0.12 * self.pressure + 0.06 * self.stress_episode) * skill_gain)
        )

        # Landing angle is tight around the approach axis (sides unlikely);
        # a small uniform-direction chance remains at low effort / high speed
        # (the "effort = consistency" clause), kept small so edges stay rare.
        # v2.8: tighter angular cone so the aggregate cloud stays visibly
        # elongated along the travel direction instead of rounding out.
        # v2.11: irregular per-object landing shapes. Every object draws its
        # own ellipse parameters (undershoot depth, elongation, angular
        # scatter, and a two-cluster mixture), so the aggregate cloud is a
        # ragged scatter biased toward undershoot instead of one clean
        # synthetic "arrow". The radial boundary is soft and per-object, so
        # the heatmap never shows a hard ring or a consistent outline.
        angular_sigma_rad = math.radians(
            self.rng.uniform(8.0, 26.0) + 20.0 * (1.0 - self.skill) + 12.0 * speed_ratio
        )
        angle = approach_angle + self.rng.normal(0.0, angular_sigma_rad)
        # Small scatter-direction chance (effort = consistency), biased toward
        # the approach side so it never draws a full 360-degree ring.
        uniform_probability = float(np.clip(0.04 + 0.10 * speed_ratio + 0.06 * (1.0 - self.effort), 0.0, 0.30))
        if self.rng.random() < uniform_probability:
            angle = approach_angle + self.rng.uniform(-math.pi * 0.75, math.pi * 0.75)

        # Per-object undershoot depth and elongation: the aggregate is a
        # mixture of differently-shaped ellipses, so consecutive runs never
        # trace the same outline. A persistent signed bias state still makes
        # landings drift in runs (temporal correlation) without pinning one
        # geometric shape.
        undershoot = (
            obj.radius
            * self.rng.uniform(0.10, 0.28)
            * (1.35 - 0.60 * self.skill)
            * (1.0 + 0.35 * speed_ratio)
        )
        self.aim_bias_state = 0.92 * self.aim_bias_state + self.rng.normal(0.0, 0.18 * sigma)
        perpendicular_factor = self.rng.uniform(0.25, 0.65)
        # Two-cluster mixture: a tight undershoot cluster (most landings) plus
        # a looser scattered cluster, with per-object mixture probability. This
        # creates lumps and asymmetry instead of a smooth analytic cloud.
        cluster_probability = self.rng.uniform(0.55, 0.85)
        if self.rng.random() < cluster_probability:
            radial = self.rng.normal(-undershoot - self.aim_bias_state, sigma * self.rng.uniform(0.70, 1.00))
        else:
            radial = self.rng.normal(
                -undershoot * self.rng.uniform(0.1, 0.6) - self.aim_bias_state,
                sigma * self.rng.uniform(1.15, 1.60),
            )
        perpendicular = self.rng.normal(0.0, sigma * perpendicular_factor)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        offset = radial * np.array([cos_a, sin_a]) + perpendicular * np.array([-sin_a, cos_a])

        # Weak correlated 2D wander keeps consecutive landings related without
        # a fixed pattern; kept light so the undershoot bias stays visible.
        self.aim_state = 0.55 * self.aim_state + self.rng.normal(0.0, 0.30 * sigma, 2)
        offset = offset + 0.15 * self.aim_state
        # Shared rush/panic component: the same signed draw used by timing,
        # projected onto the approach axis. Rushing cuts the aim short of the
        # circle; falling behind overshoots past it - and both correlate with
        # early/late taps.
        rush_aim = self.rush_draw * 0.70 * self.pressure * sigma * skill_gain
        if distance > 1e-6:
            offset = offset - rush_aim * np.array([dx / distance, dy / distance])

        # Soft radial boundary (v2.11): a per-object limit with a gentle
        # pull-back instead of a hard ring. Far landings are squashed toward
        # the limit at 30%, so a sparse outer scatter exists and the boundary
        # is ragged while rim landings stay rare.
        soft_limit = obj.radius * (0.80 - 0.12 * self.skill) * self.rng.uniform(0.92, 1.08)
        landing_error = float(np.linalg.norm(offset))
        if landing_error > soft_limit and landing_error > 1e-9:
            offset = offset / landing_error * (soft_limit + (landing_error - soft_limit) * 0.30)

        # Aim lapse: a rare large over/undershoot that lands outside the circle.
        lapse_probability = (
            (1.0 - self.skill) ** 2
            * (1.0 - self.effort)
            # v2.4/v2.5: lapse base cut so a 90%+ run keeps near-zero misses;
            # strain still keeps lapses present on hard/dense maps.
            * (0.002 + 0.05 * strain + 0.02 * self.pressure)
            # v2.9: above 50% skill, lapses collapse too (context-exempt on
            # very fast/complex sections).
            * self._miss_nerf(strain, self.pressure)
        )
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
        weaknesses. Phase 1 implemented aim and fatigue; Phase 2 (v2.1) also
        wires timing (error normalised by the MEH window). The remaining
        criteria contribute 0 until their phases land.
        """
        radius = max(obj.radius, 1e-6)
        aim_weakness = min(1.0, aim_error / radius) if not self.profile.perfect_baseline else 0.0
        timing_error = self.timing_errors[-1] if self.timing_errors else 0.0
        timing_weakness = (
            min(1.0, abs(timing_error) / max(obj.hit_windows.meh_ms, 1e-6))
            if not self.profile.perfect_baseline
            else 0.0
        )
        fatigue_weakness = self.fatigue * (0.5 + 0.5 * (1.0 - self.effort))
        weakness = (
            STRENGTH_WEIGHTS["aim"] * aim_weakness
            + STRENGTH_WEIGHTS["timing"] * timing_weakness
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
            "mistaps": self.timing_mistaps,
            "ghost_presses": self.ghost_presses,
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

    def _append_idle_wander(
        self,
        points: list[tuple[float, np.ndarray]],
        available_from: float,
        hit_time: float,
        start: np.ndarray,
        target: np.ndarray,
        strain: float,
        long_settle: bool = False,
    ) -> tuple[float | None, np.ndarray | None]:
        """Insert a smooth human-like idle wander when there is enough spare
        time before the next object (long gaps, map breaks, pre-roll).

        Real players rarely park their cursor on the next note during slow
        sections; they doodle - figure-8s, circles/ellipses, or loose random
        sine paths - then re-approach just before the object. The wander only
        runs when the time left after reserving the approach is meaningful
        (>= 350 ms), stays inside the playfield, and is sampled at the trace
        rate so the resampled path is continuous. A minimum-jerk ease blends
        the parked cursor into the pattern so the doodle never starts with a
        jerk. Perfect-baseline mode never wanders (calibration traces stay
        exact).
        """
        if self.profile.perfect_baseline:
            return None, None

        # The wander envelope (below) guarantees the cursor converges onto the
        # target by the time the approach starts, so the reserve only needs to
        # cover a short (~60 px) final approach plus settle and rearm slack.
        # This is the mathematical fix for wander-induced late-arrival misses:
        # the return distance at wander end is ~0, so the reserved approach
        # time can never be insufficient.
        approach_distance = 60.0
        fitts_ms = (
            (70.0 + 92.0 * math.log2(approach_distance / 64.0 + 1.0))
            * (1.08 - self.profile.percentile / 600.0)
        )
        approach = max(45.0, fitts_ms * (1.0 + 0.25 * strain))
        # Mirror the transition's speed-ceiling clamp (with its per-move random
        # multiplier, worst case) plus settle and rearm slack.
        min_duration = max(
            self.step_ms,
            approach_distance / max(1.0, self._speed_ceiling(approach_distance)) * 1000.0 * 1.9 * 1.25,
        )
        reserve = max(approach, min_duration) + 32.0 + self.step_ms * 4.0 + (120.0 if long_settle else 0.0)
        wander_start_time = max(available_from + self.step_ms, points[-1][0] + self.step_ms)
        budget_ms = hit_time - reserve - wander_start_time
        if budget_ms < 350.0:
            return None, None

        # v2.7: the doodle roams a big neighbourhood around the next target
        # (80-200 px away, playfield-safe) instead of hovering right next to
        # it; the convergence envelope still brings it home in time.
        center_distance = self.rng.uniform(80.0, 200.0)
        center_angle = self.rng.uniform(0.0, math.tau)
        center = np.array(
            [
                target[0] + center_distance * math.cos(center_angle),
                target[1] + center_distance * math.sin(center_angle),
            ]
        )
        center = np.clip(center, [70.0, 70.0], [442.0, 314.0])

        # Pattern parameters are drawn once per idle window so the shape stays
        # stable for its whole duration (a pattern that redraws itself every
        # sample would just be noise).
        # v2.7: pattern pace scales with how much time is available - a long
        # idle window gets slow, grand figure-8s; a short one gets a tighter,
        # quicker doodle.
        period_ms = self.rng.uniform(1400.0, 3200.0) * float(np.clip(budget_ms / 2500.0, 0.7, 1.6))
        phase = self.rng.uniform(0.0, math.tau)
        rotation = self.rng.uniform(0.0, math.tau)
        axis_a = self.rng.uniform(60.0, 110.0)
        axis_b = self.rng.uniform(45.0, 85.0)
        speed_mult = self.rng.uniform(0.6, 1.4)
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        wobble_freq = self.rng.uniform(0.2, 0.6)
        wobble_phase = self.rng.uniform(0.0, math.tau)
        wobble_amp = self.rng.uniform(2.0, 6.0)
        drift_amp = self.rng.uniform(20.0, 45.0)
        drift_freq = self.rng.uniform(0.03, 0.10)
        drift_phase = self.rng.uniform(0.0, math.tau)
        shrink_power = self.rng.uniform(1.0, 1.8)
        kind_roll = self.rng.random()
        # v2.7: smooth figure-8s dominate (~65%), circles are a smaller
        # minority (~20%), loose random paths are now rare (~15%) so the idle
        # motion reads as a confident doodle, not noise.
        kind = "figure8" if kind_roll < 0.65 else "circle" if kind_roll < 0.85 else "random"
        rand_freqs = [self.rng.uniform(0.15, 1.0) for _ in range(3)]
        rand_amps = [self.rng.uniform(25.0, 55.0) for _ in range(3)]
        rand_phases = [self.rng.uniform(0.0, math.tau) for _ in range(3)]

        def pattern(clock_ms: float) -> np.ndarray:
            u = math.tau * (clock_ms / period_ms) + phase
            if kind == "figure8":
                # Lissajous 1:2 - the classic infinity doodle.
                px = axis_a * math.sin(u)
                py = axis_b * math.sin(2.0 * u + phase * 0.5)
            elif kind == "circle":
                px = axis_a * math.cos(u)
                py = axis_b * math.sin(u)
            else:
                px = sum(
                    amp * math.sin(math.tau * freq * clock_ms / 1000.0 + p)
                    for amp, freq, p in zip(rand_amps, rand_freqs, rand_phases)
                )
                py = sum(
                    amp * math.sin(math.tau * freq * clock_ms / 1000.0 + p + (index + 1) * 1.9)
                    for index, (amp, freq, p) in enumerate(zip(rand_amps, rand_freqs, rand_phases))
                )
            # A slow wobble keeps the pattern from looking mathematically
            # perfect, and a very slow centre drift makes the doodle roam
            # instead of orbiting one fixed spot.
            wobble = wobble_amp * math.sin(math.tau * wobble_freq * clock_ms / 1000.0 + wobble_phase)
            local = np.array([(px + wobble) * speed_mult, py * speed_mult])
            drift = np.array(
                [
                    drift_amp * math.sin(math.tau * drift_freq * clock_ms / 1000.0 + drift_phase),
                    drift_amp * math.cos(math.tau * drift_freq * clock_ms / 1000.0 + drift_phase + 1.3),
                ]
            )
            return center + drift + np.array(
                [local[0] * cos_r - local[1] * sin_r, local[0] * sin_r + local[1] * cos_r]
            )

        entry_ms = min(300.0, budget_ms * 0.35)
        # v2.7: larger doodles need a bit more headroom; still well inside the
        # human envelope.
        max_step = 1600.0 * self.step_ms / 1000.0
        wander_end_time = hit_time - reserve
        duration_ms = max(1.0, wander_end_time - wander_start_time)
        noise_state = np.zeros(2)
        # Time-varying speed: a smooth OU multiplier on the pattern clock, so
        # the doodle accelerates and slows organically instead of one robotic
        # orbital speed.
        speed_env = 1.0
        speed_rho = math.exp(-self.step_ms / 420.0)
        speed_innovation = 0.22 * math.sqrt(max(1e-9, 1.0 - speed_rho * speed_rho))
        pattern_clock = 0.0
        previous_position = start
        emitted = 0
        t = wander_start_time
        while t <= wander_end_time - self.step_ms * 0.5:
            progress = min(1.0, (t - wander_start_time) / duration_ms)
            # Envelope: the doodle is anchored to the target and shrinks onto
            # it as the wander ends, so the return distance is always feasible.
            shrink = (1.0 - progress) ** shrink_power
            entry_fraction = min(1.0, (t - wander_start_time) / entry_ms)
            ease = _minimum_jerk(np.array([entry_fraction]))[0]
            # Small, slow jitter: ~0.18 px per 2 ms step (~90 px/s), so the
            # doodle has natural hand-tremor texture without squiggling.
            noise_state = 0.96 * noise_state + self.rng.normal(0.0, 0.18, 2)
            speed_env = float(np.clip(speed_rho * speed_env + self.rng.normal(0.0, speed_innovation), 0.45, 1.4))
            pattern_clock += self.step_ms * speed_env
            blended = start + (pattern(pattern_clock) - start) * ease
            position = target + (blended - target) * shrink + noise_state * shrink
            position = np.clip(position, [8.0, 8.0], [504.0, 376.0])
            delta = position - previous_position
            delta_length = float(np.linalg.norm(delta))
            if delta_length > max_step:
                position = previous_position + delta / delta_length * max_step
            points.append((t, position))
            previous_position = position
            emitted += 1
            t += self.step_ms

        if emitted == 0:
            return None, None
        self.idle_wanders += 1
        return points[-1][0], points[-1][1]

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

        # v2.7 flow motion: the cursor leaves the previous target right after
        # the press and spends the whole gap gliding to the next landing, so
        # movements connect into one continuous curve instead of "park then
        # shoot". Only a short park remains before the press - just enough for
        # the game's 1-2 frame delivery lag - and the first target keeps a
        # longer startup park.
        # v2.10: density-adaptive pre-press park. With the window-origin fix,
        # the game-visible cursor matches the dispatch (verified delta ~0), so
        # a long park is no longer needed for delivery. Dense sections (short
        # intervals) use a ~1-frame park so fast jumpstreams have enough
        # movement time; normal gaps keep a comfortable 32 ms.
        if available < 140.0:
            park = min(16.0, max(2.0, available * 0.25))
        else:
            park = min(120.0 if long_settle else 32.0, max(2.0, available * 0.5))
        park = min(park, available - self.step_ms)
        duration = max(self.step_ms, available - park)

        # Never collapse a movement into a sub-millisecond teleport. If the
        # available time cannot fit the distance at a plausible speed, the
        # cursor simply arrives late (and the press naturally misses) instead
        # of teleporting.
        if self.profile.perfect_baseline:
            max_transition_speed_px_s = 50_000.0
        else:
            # Skill ceiling x distance scaling x per-move randomness. The
            # random multiplier gives each move its own slightly different
            # pace so speeds are never deterministic.
            move_speed_multiplier = self.rng.uniform(0.82, 1.08)
            max_transition_speed_px_s = self._speed_ceiling(distance) * move_speed_multiplier
        # distance (px) / speed (px/s) yields seconds; min_duration is in
        # milliseconds. The missing *1000 made the speed ceiling never bind
        # (it always collapsed to one sample step), which let the cursor
        # flick across long jumps at 10-50x the intended speed.
        # The minimum-jerk profile peaks at 15/8 = 1.875x its average speed
        # mid-move, and the curvature/noise jitter terms stack on top of that.
        # v2.10: the movement floor uses ~1.9x (min-jerk peaks at 1.875x
        # average) instead of 2.4x, so 94-ms-interval jumpstreams at the
        # player's ceiling are feasible instead of being declared impossible.
        # The hard per-step resample clamp remains the flick guard.
        min_duration = max(self.step_ms, distance / max_transition_speed_px_s * 1000.0 * 1.9)
        if duration < min_duration:
            duration = min_duration
        start_time = max(available_from, hit_time - park - duration)
        if start_time > points[-1][0] + self.step_ms:
            points.append((start_time, start.copy()))

        arrival_time = hit_time - park
        if start_time >= available_from and arrival_time - start_time < min_duration:
            arrival_time = start_time + min_duration
        self._advance_motion_states(max(1.0, arrival_time - start_time))
        count = max(2, int(math.ceil((arrival_time - start_time) / self.step_ms)) + 1)
        times = np.linspace(start_time, arrival_time, count)
        u = np.linspace(0.0, 1.0, count)
        progress = _minimum_jerk(u)
        direction = _normalised(target - start)
        normal = np.array([-direction[1], direction[0]])
        # Continuous signed bow: the persistent curve state keeps consecutive
        # moves bending the same way instead of darting left/right at random.
        # v2.7: per-move curvature variation - sometimes flowing and curvy,
        # sometimes straighter - on top of the persistent signed bow, so
        # consecutive moves read as one connected curve, not straight bullets.
        curve = 0.0 if self.profile.perfect_baseline else self.curve_state * max(
            0.5, distance * self.params.curvature_ratio
        ) * self.rng.uniform(0.8, 2.2)
        residual = np.zeros(count) if self.profile.perfect_baseline else _ou_bridge(
            self.rng,
            count,
            self.params.aim_sigma * (0.07 + 0.14 * strain),
            self.rng.uniform(160.0, 340.0),
            self.step_ms,
        )
        correction = np.zeros(count)
        # Occasional small mid-path correction jerk: rare, brief, subtle.
        if not self.profile.perfect_baseline and self.rng.random() < self.params.correction_probability * 0.5:
            centre = self.rng.uniform(0.68, 0.88)
            width = self.rng.uniform(0.06, 0.14)
            correction = self.rng.normal(0.0, self.params.aim_sigma * 0.45) * np.exp(-0.5 * ((u - centre) / width) ** 2)

        for time_ms, amount, phase, noise, correction_value in zip(times, progress, 4 * u * (1 - u), residual, correction):
            if time_ms > hit_time:
                # The speed ceiling left the cursor still en route when the
                # press fires (late arrival -> the press misses naturally).
                # Emit no points past the press: a later object's transition
                # starts from the cursor's actual position at this moment.
                break
            position = start + (target - start) * amount + normal * (curve * phase + noise + correction_value) + self.wander
            points.append((float(time_ms), position))

        if arrival_time + self.step_ms < hit_time:
            points.append((hit_time, target + self.wander))

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
            self.rng, count, self.params.aim_sigma * (0.06 + 0.12 * strain), 220.0, self.step_ms
        )
        # Slider-follow speed bound. Sliders legitimately move at map velocity,
        # but the entry correction (merging a cursor parked far from the head)
        # must not yank the cursor across the playfield faster than a human can
        # physically move. Floor keeps very-low-skill profiles from crawling.
        max_slider_speed_px_s = max(2_400.0, self._speed_ceiling(80.0) * 1.6)
        lag = 0.0 if self.profile.perfect_baseline else float(
            np.clip(self.rng.normal(0.0, 0.012 + strain * 0.01), -0.035, 0.035)
        )
        entry_position = points[-1][1].copy()
        first_base: np.ndarray | None = None
        noise_sign = 1.0
        previous_tangent: np.ndarray | None = None
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
            if previous_tangent is not None and float(np.dot(tangent, previous_tangent)) < 0.0:
                # The path turned around (repeat/reversal) or the sampled curve
                # wiggled by sub-pixel amounts. Keep the lateral noise offset on
                # the same physical side of the slider so the cursor never
                # snaps across the curve.
                noise_sign = -noise_sign
            previous_tangent = tangent
            normal = np.array([-tangent[1], tangent[0]])
            position = base + entry_correction + normal * (noise_sign * noise[index]) + self.wander
            if index > 0 and time_ms > points[-1][0]:
                max_step = max_slider_speed_px_s * (time_ms - points[-1][0]) / 1000.0
                delta = position - points[-1][1]
                step = float(np.linalg.norm(delta))
                if step > max_step:
                    position = points[-1][1] + delta * (max_step / step)
            points.append((float(time_ms), position))

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
            anchors = SPINNER_RPM_ANCHORS
            base_rpm = float(np.interp(self.skill * 100.0, [a[0] for a in anchors], [a[1] for a in anchors]))
            rpm = base_rpm * self.rng.uniform(0.88, 1.10)
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
        # clamp every resampled step to the skill speed ceiling (with a small
        # jitter allowance). This is the hard guarantee that the delivered
        # trace never contains a humanly-impossible flick, whatever residual
        # jitter stacks on the generated path. Perfect mode keeps its own much
        # higher machine ceiling.
        if not self.profile.perfect_baseline:
            max_profile_speed_px_s = max(3_200.0, self._speed_ceiling(260.0) * 1.35)
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
