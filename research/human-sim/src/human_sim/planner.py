from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .context import PatternContext, build_contexts
from .schemas import HumanProfile, MapObject, MapPlan, TraceFrame


# Bump when trace-generation behaviour changes so the runner's content-addressed
# trace cache is invalidated (the configuration hash includes this string).
PLANNER_VERSION = "timing-sync-v2.14"


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


@dataclass
class MotionState:
    """Persistent cursor state shared by adjacent movement primitives.

    The planner still makes local, object-conditioned decisions, but position,
    velocity, acceleration, coloured noise, and wander are not reinitialised
    at every target.  Arrays are deliberately mutable: the state is advanced
    once per emitted sample rather than once per object.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    noise_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    ou_offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    wander: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    correction: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    refractory_until_ms: float = float("-inf")
    last_timestamp_ms: float = float("-inf")


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


def _quintic_hermite(
    p0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
    duration_ms: float,
    u: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a quintic Hermite segment and its physical derivatives.

    ``v`` and ``a`` use px/s and px/s²; ``u`` is the normalised segment time.
    The endpoint equations are the C² boundary conditions used by the rolling
    waypoint planner.  Returning derivatives here keeps the solver and its
    continuity tests on the same implementation.
    """
    duration_s = max(float(duration_ms) / 1000.0, 1e-6)
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    tau = np.asarray(u, dtype=float)
    c0 = p0
    c1 = duration_s * v0
    c2 = (duration_s * duration_s * a0) / 2.0
    residual_position = p1 - c0 - c1 - c2
    residual_velocity = duration_s * v1 - c1 - 2.0 * c2
    residual_acceleration = duration_s * duration_s * a1 - 2.0 * c2
    c3 = 10.0 * residual_position - 4.0 * residual_velocity + 0.5 * residual_acceleration
    c4 = -15.0 * residual_position + 7.0 * residual_velocity - residual_acceleration
    c5 = 6.0 * residual_position - 3.0 * residual_velocity + 0.5 * residual_acceleration
    position = c0 + c1 * tau[..., None] + c2 * tau[..., None] ** 2 + c3 * tau[..., None] ** 3 + c4 * tau[..., None] ** 4 + c5 * tau[..., None] ** 5
    d_tau = c1 + 2.0 * c2 * tau[..., None] + 3.0 * c3 * tau[..., None] ** 2 + 4.0 * c4 * tau[..., None] ** 3 + 5.0 * c5 * tau[..., None] ** 4
    d2_tau = 2.0 * c2 + 6.0 * c3 * tau[..., None] + 12.0 * c4 * tau[..., None] ** 2 + 20.0 * c5 * tau[..., None] ** 3
    velocity = d_tau / duration_s
    acceleration = d2_tau / (duration_s * duration_s)
    if tau.ndim == 0:
        return position[0] if position.ndim > 1 else position, velocity[0] if velocity.ndim > 1 else velocity, acceleration[0] if acceleration.ndim > 1 else acceleration
    return position, velocity, acceleration


class HumanTracePlanner:
    def __init__(self, map_plan: MapPlan, profile: HumanProfile, model_bundle_path: str | None = None):
        profile.validate()
        self.map = map_plan
        self.profile = profile
        # Build once so every planner operation sees the same rich, shared
        # context features.  ``_context`` below remains a label compatibility
        # wrapper for the existing pressure/timing tables.
        self.contexts: tuple[PatternContext, ...] = build_contexts(map_plan)
        self.pattern_contexts = self.contexts
        self.params = interpolate_skill_parameters(profile.skill_level)
        self.rng = np.random.default_rng(profile.seed)
        self.step_ms = 1000.0 / profile.sample_rate_hz
        self.timing_state = 0.0
        self.aim_state = np.zeros(2)
        self.aim_bias_state = 0.0
        # A small session-level covariance orientation perturbation prevents
        # every run from using exactly the same local error ellipse while the
        # local-frame sampler below remains directionally neutral.
        self.aim_rho_state = 0.0 if profile.perfect_baseline else float(np.clip(self.rng.normal(0.0, 0.035), -0.10, 0.10))
        # v1.5 continuous-motion states: a slow absolute drift and a persistent
        # curve direction. Unlike per-transition random draws, these evolve
        # smoothly across the whole run so adjacent moves keep one continuous
        # path instead of resetting into independent squiggles.
        self.wander = np.zeros(2)
        self.curve_state = 0.0
        self.motion_state = MotionState()
        self.motion_segments: list[dict[str, object]] = []
        self.motion_waypoint_windows: list[dict[str, object]] = []
        self._continuous_endpoint_emitted = False
        self._transition_after_idle = False
        self._last_waypoint_position: np.ndarray | None = None
        self._last_waypoint_time_ms: float | None = None
        self._last_waypoint_velocity = np.zeros(2)
        self._last_waypoint_acceleration = np.zeros(2)
        self.motion_correction_events = 0
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
        # Per-object planned landing offsets are retained for offline heatmap
        # diagnostics.  They are keyed by map position so analysis can compare
        # seeds without confusing a late/missing press with an aim sample.
        self.landing_offsets: dict[int, tuple[float, float]] = {}

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

    def _reset_motion_state(self, timestamp_ms: float, position: np.ndarray) -> None:
        self.motion_state = MotionState(
            position=np.asarray(position, dtype=float).copy(),
            last_timestamp_ms=float(timestamp_ms),
        )
        self.wander = self.motion_state.wander.copy()
        self.motion_segments = []
        self.motion_waypoint_windows = []
        self._continuous_endpoint_emitted = False
        self._transition_after_idle = False
        self._last_waypoint_position = None
        self._last_waypoint_time_ms = None
        self._last_waypoint_velocity = np.zeros(2)
        self._last_waypoint_acceleration = np.zeros(2)
        self.motion_correction_events = 0

    def _sync_motion_state_from_points(self, points: list[tuple[float, np.ndarray]]) -> None:
        """Adopt the tail of a slider/idle/legacy path without resetting noise."""
        if not points:
            return
        time_ms, position = points[-1]
        state = self.motion_state
        state.position = np.asarray(position, dtype=float).copy()
        tail: list[tuple[float, np.ndarray]] = []
        cutoff_ms = float(time_ms) - 20.0
        newest_tail_time = float("inf")
        for sample_time, sample_position in reversed(points):
            sample_time = float(sample_time)
            if sample_time > float(time_ms) or sample_time >= newest_tail_time:
                continue
            tail.append((sample_time, np.asarray(sample_position, dtype=float)))
            newest_tail_time = sample_time
            if sample_time <= cutoff_ms or len(tail) >= 32:
                break
        tail.reverse()
        if len(tail) >= 3:
            sample_times = np.asarray([(sample_time - float(time_ms)) / 1000.0 for sample_time, _ in tail])
            sample_positions = np.asarray([sample_position for _, sample_position in tail])
            displacement = sample_positions - state.position
            design = np.column_stack((sample_times, 0.5 * sample_times * sample_times))
            coefficients, *_ = np.linalg.lstsq(design, displacement, rcond=None)
            velocity = coefficients[0]
            acceleration = coefficients[1]
        elif len(tail) == 2:
            dt_s = max((tail[-1][0] - tail[-2][0]) / 1000.0, 1e-6)
            velocity = (tail[-1][1] - tail[-2][1]) / dt_s
            acceleration = np.zeros(2)
        else:
            velocity = np.zeros(2)
            acceleration = np.zeros(2)

        velocity_limit = max(2_400.0, self._speed_ceiling(80.0) * 1.6)
        velocity_length = float(np.linalg.norm(velocity))
        if not np.all(np.isfinite(velocity)):
            velocity = np.zeros(2)
        elif velocity_length > velocity_limit:
            velocity *= velocity_limit / velocity_length
        acceleration_limit = max(22_000.0, self._speed_ceiling(80.0) * 12.0)
        acceleration_length = float(np.linalg.norm(acceleration))
        if not np.all(np.isfinite(acceleration)):
            acceleration = np.zeros(2)
        elif acceleration_length > acceleration_limit:
            acceleration *= acceleration_limit / acceleration_length
        state.velocity = velocity
        state.acceleration = acceleration
        state.last_timestamp_ms = float(time_ms)
        self.wander = state.wander.copy()
        self._last_waypoint_position = state.position.copy()
        self._last_waypoint_time_ms = state.last_timestamp_ms
        self._last_waypoint_velocity = state.velocity.copy()
        self._last_waypoint_acceleration = state.acceleration.copy()

    def _future_waypoint_window(
        self,
        object_index: int,
        target: np.ndarray,
        hit_time: float,
    ) -> list[tuple[np.ndarray, float]]:
        """Build a small local horizon, using sampled offsets when available."""
        window: list[tuple[np.ndarray, float]] = [(np.asarray(target, dtype=float).copy(), float(hit_time))]
        for future_index in range(object_index + 1, min(len(self.map.objects), object_index + 6)):
            obj = self.map.objects[future_index]
            if obj.kind == "spinner":
                continue
            offset = self.landing_offsets.get(future_index)
            future = np.array([obj.position.x, obj.position.y], dtype=float)
            if offset is not None:
                future += np.asarray(offset, dtype=float)
            window.append((future, float(obj.start_time_ms)))
        self.motion_waypoint_windows.append(
            {
                "object_index": object_index,
                "points": [point.tolist() for point, _time in window],
                "times_ms": [time for _point, time in window],
            }
        )
        return window

    def _waypoint_velocity(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        following: np.ndarray,
        previous_time_ms: float,
        current_time_ms: float,
        following_time_ms: float,
    ) -> tuple[np.ndarray, float, float]:
        """Estimate a geometry-derived velocity at one interior waypoint."""
        d_in = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
        d_out = np.asarray(following, dtype=float) - np.asarray(current, dtype=float)
        length_in = float(np.linalg.norm(d_in))
        length_out = float(np.linalg.norm(d_out))
        u_in = _normalised(d_in)
        u_out = _normalised(d_out)
        dot = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
        theta = math.acos(dot)
        bisector = _normalised(u_in + u_out)
        in_seconds = max((current_time_ms - previous_time_ms) / 1000.0, 1e-3)
        out_seconds = max((following_time_ms - current_time_ms) / 1000.0, 1e-3)
        speed_in = length_in / in_seconds
        speed_out = length_out / out_seconds
        harmonic = 2.0 * speed_in * speed_out / max(speed_in + speed_out, 1e-6)
        nominal = min(harmonic, self._speed_ceiling(max(length_in, length_out)))
        carry = math.cos(theta * 0.5) ** 2
        velocity = bisector * nominal * carry

        # A curvature-aware lateral acceleration cap keeps shallow bends
        # flowing while naturally slowing a tight turn or reversal.
        mean_length = max(0.5 * (length_in + length_out), 1.0)
        kappa = theta / mean_length
        lateral_limit = max(28_000.0, self._speed_ceiling(mean_length) * 8.0)
        if kappa > 1e-6:
            velocity_length = min(float(np.linalg.norm(velocity)), math.sqrt(lateral_limit / kappa))
            velocity = bisector * velocity_length
        return velocity, theta, kappa

    def _waypoint_kinematics(
        self,
        object_index: int,
        target: np.ndarray,
        hit_time: float,
        available_from: float,
    ) -> tuple[np.ndarray, np.ndarray, float, list[tuple[np.ndarray, float]]]:
        window = self._future_waypoint_window(object_index, target, hit_time)
        previous = (
            self._last_waypoint_position.copy()
            if self._last_waypoint_position is not None
            else self.motion_state.position.copy()
        )
        previous_time = (
            self._last_waypoint_time_ms
            if self._last_waypoint_time_ms is not None
            else float(available_from)
        )
        current, current_time = window[0]
        if len(window) > 1:
            following, following_time = window[1]
        else:
            following = current.copy()
            following_time = current_time + max(self.step_ms, current_time - previous_time)
        endpoint_velocity, _theta, _kappa = self._waypoint_velocity(
            previous, current, following, previous_time, current_time, following_time
        )

        previous_velocity = self._last_waypoint_velocity.copy()
        if self._last_waypoint_position is None:
            previous_velocity = self.motion_state.velocity.copy()
        if len(window) > 2:
            next_velocity, _next_theta, _next_kappa = self._waypoint_velocity(
                current,
                following,
                window[2][0],
                current_time,
                following_time,
                window[2][1],
            )
        else:
            next_velocity = endpoint_velocity.copy()
        span_seconds = max((following_time - previous_time) / 1000.0, 1e-3)
        endpoint_acceleration = (next_velocity - previous_velocity) / span_seconds
        acceleration_limit = max(22_000.0, self._speed_ceiling(max(np.linalg.norm(current - previous), 1.0)) * 12.0)
        acceleration_length = float(np.linalg.norm(endpoint_acceleration))
        if acceleration_length > acceleration_limit:
            endpoint_acceleration *= acceleration_limit / acceleration_length
        return endpoint_velocity, endpoint_acceleration, _theta, window

    def _advance_motion_state_sample(
        self,
        time_ms: float,
        base_position: np.ndarray,
        base_velocity: np.ndarray,
        base_acceleration: np.ndarray,
        tangent: np.ndarray,
        strain: float,
        initial_wander: np.ndarray,
        initial_noise: np.ndarray,
    ) -> np.ndarray:
        """Advance persistent coloured noise/wander once for one path sample."""
        state = self.motion_state
        dt_ms = max(0.001, float(time_ms) - state.last_timestamp_ms)
        dt_s = dt_ms / 1000.0
        noise_tau = 180.0 + 180.0 * (1.0 - float(np.clip(strain, 0.0, 1.0)))
        rho = math.exp(-dt_ms / noise_tau)
        sigma = self.params.aim_sigma * (0.025 + 0.055 * strain) * (0.9 + 0.25 * (1.0 - self.effort))
        old_noise = state.ou_offset.copy()
        state.ou_offset = rho * state.ou_offset + self.rng.normal(
            0.0, sigma * math.sqrt(max(1e-9, 1.0 - rho * rho)), 2
        )
        state.noise_velocity = (state.ou_offset - old_noise) / dt_s

        wander_tau = 650.0
        wander_rho = math.exp(-dt_ms / wander_tau)
        wander_sigma = self.aim_sigma_base * (0.10 + 0.08 * (1.0 - self.effort))
        state.wander = wander_rho * state.wander + self.rng.normal(
            0.0, wander_sigma * math.sqrt(max(1e-9, 1.0 - wander_rho * wander_rho)), 2
        )
        self.wander = state.wander.copy()

        tangent = _normalised(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        normal_noise = float(np.dot(state.ou_offset, normal)) * normal
        tangent_noise = float(np.dot(state.ou_offset, tangent)) * tangent * 0.25
        applied_noise = normal_noise + tangent_noise
        state.last_timestamp_ms = float(time_ms)
        state.velocity = np.asarray(base_velocity, dtype=float) + state.noise_velocity
        state.acceleration = np.asarray(base_acceleration, dtype=float)
        state.position = np.asarray(base_position, dtype=float) + (state.wander - initial_wander) + (applied_noise - initial_noise)
        return state.position.copy()

    def _persistent_noise_series(self, count: int, sigma: float, correlation_ms: float) -> np.ndarray:
        """Advance the shared OU offset for legacy slider/spinner samples."""
        if self.profile.perfect_baseline or count <= 0 or sigma <= 0.0:
            return np.zeros(max(0, count))
        state = self.motion_state
        values = np.zeros(count)
        rho = math.exp(-self.step_ms / max(correlation_ms, self.step_ms))
        innovation = sigma * math.sqrt(max(1e-9, 1.0 - rho * rho))
        for index in range(count):
            old_offset = state.ou_offset.copy()
            state.ou_offset = rho * state.ou_offset + self.rng.normal(0.0, innovation, 2)
            state.noise_velocity = (state.ou_offset - old_offset) / max(self.step_ms / 1000.0, 1e-6)
            values[index] = float(state.ou_offset[0])
        return values

    def generate(self) -> list[TraceFrame]:
        first_time = self.map.objects[0].start_time_ms
        # Gameplay itself starts before beatmap time zero (osu!standard uses at
        # least a two-second pre-roll). Preserve 1.5 seconds of that negative
        # clock range so maps whose first object is near 0 ms do not require an
        # impossible centre-to-object jump after the audio clock reaches zero.
        timeline_start = first_time - 1500.0
        cursor: list[tuple[float, np.ndarray]] = [(timeline_start, np.array([256.0, 192.0]))]
        self._reset_motion_state(timeline_start, cursor[0][1])
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
                self._append_spinner(cursor, obj, hit_time, last_time, last_position, strain, object_index=index)
                key_intervals.append((hit_time, max(hit_time + 50, obj.end_time_ms), key_index))
                key_index ^= 1
                last_time = obj.end_time_ms
                last_position = cursor[-1][1]
                self._sync_motion_state_from_points(cursor)
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
                self._sync_motion_state_from_points(cursor)
                self._transition_after_idle = True
            else:
                self._transition_after_idle = False
            self._append_transition(
                cursor,
                last_time,
                hit_time,
                last_position,
                target,
                strain,
                long_settle=index == 0,
                object_index=index,
                radius=obj.radius,
                continuous=obj.kind == "circle",
            )
            if obj.kind == "circle" and not self._continuous_endpoint_emitted:
                # A sampled catch-up hit can fall behind an already-emitted
                # path point.  Continue from the realised cursor tail rather
                # than pretending an un-emitted target was reached.
                self._sync_motion_state_from_points(cursor)
            key_intervals.append((hit_time, hit_time + hold, key_index))
            key_index ^= 1

            if obj.kind == "slider":
                self._append_slider(cursor, obj, hit_time, strain)
                self._sync_motion_state_from_points(cursor)
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
        return self.contexts[index].strain

    def _context(self, index: int, obj: MapObject, strain: float) -> str:
        """Return the shared context label for legacy timing tables."""
        del obj, strain
        return self.contexts[index].label

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
        """Sample a centre-anchored local-frame landing distribution.

        ``u`` is the error along the approach direction and ``v`` is its
        perpendicular error.  Sampling these coordinates directly is
        important: drawing a narrow angle and then making the lateral size a
        fraction of a radial error creates a triangular approach-aligned
        wedge, even when the screen-space aggregate looks harmless.  The
        mixture below is a mildly undershoot-biased bivariate normal with a
        small pressure-dependent tail.  Its covariance stays close to an
        ellipse, and all persistent bias is bounded and session-level rather
        than tied to one map direction.
        """
        if index:
            previous = self.map.objects[index - 1]
            dx = obj.position.x - previous.end_position.x
            dy = obj.position.y - previous.end_position.y
        else:
            dx, dy = obj.position.x - 256.0, obj.position.y - 192.0
        distance = math.hypot(dx, dy)
        approach = np.array([dx, dy], dtype=float) / max(distance, 1e-9)
        normal = np.array([-approach[1], approach[0]], dtype=float)

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

        # Convert the existing pixel spread into two nearly comparable local
        # standard deviations.  The longitudinal axis is only modestly wider
        # than the lateral one; unlike the old cone, lateral error does not
        # collapse as longitudinal error approaches zero.
        sigma_parallel = sigma * (0.94 + 0.08 * speed_ratio)
        sigma_perpendicular = sigma * (0.82 + 0.10 * self.effort + 0.04 * (1.0 - self.skill))
        sigma_perpendicular = min(sigma_parallel * 0.98, sigma_perpendicular)
        sigma_perpendicular = max(sigma_parallel * 0.70, sigma_perpendicular)
        covariance_rho = float(
            np.clip(
                self.aim_rho_state + self.rng.normal(0.0, 0.018) + 0.025 * (self.pressure - 0.5),
                -0.16,
                0.16,
            )
        )

        # Keep the mean shift mild.  At the reference skill/effort this is
        # roughly 0.06-0.08 radii, which produces a visible but not dominant
        # undershoot preference once the symmetric core and tail are pooled.
        mean_parallel = obj.radius * float(
            np.clip(
                0.045
                + 0.020 * (1.0 - self.skill)
                + 0.010 * speed_ratio
                + 0.012 * strain
                + 0.008 * (1.0 - self.effort),
                0.03,
                0.14,
            )
        )
        tail_probability = float(
            np.clip(
                0.032
                + 0.020 * (1.0 - self.effort)
                + 0.018 * strain
                + 0.016 * self.pressure,
                0.025,
                0.10,
            )
        )
        tail_scale = float(np.clip(1.95 + 0.45 * (1.0 - self.effort) + 0.25 * strain, 1.8, 2.8))
        if self.rng.random() < tail_probability:
            local_sigma_parallel = sigma_parallel * tail_scale
            local_sigma_perpendicular = sigma_perpendicular * tail_scale
            local_mean_parallel = mean_parallel * (1.20 + 0.20 * self.rng.random())
        else:
            local_sigma_parallel = sigma_parallel
            local_sigma_perpendicular = sigma_perpendicular
            local_mean_parallel = mean_parallel
        normal_draws = self.rng.normal(0.0, 1.0, 2)
        local_parallel = -local_mean_parallel + local_sigma_parallel * normal_draws[0]
        local_perpendicular = local_sigma_perpendicular * (
            covariance_rho * normal_draws[0]
            + math.sqrt(max(1e-9, 1.0 - covariance_rho * covariance_rho)) * normal_draws[1]
        )

        # Slow, bounded session drift is an absolute offset.  It is projected
        # into the local frame only for this sample; it is never multiplied by
        # the current longitudinal error, so it cannot recreate a wedge.
        session_sigma = sigma * (0.035 + 0.025 * (1.0 - self.effort))
        self.aim_state = 0.90 * self.aim_state + self.rng.normal(0.0, session_sigma * 0.35, 2)
        session_limit = obj.radius * 0.075
        session_bias = np.asarray(self.aim_state, dtype=float)
        session_length = float(np.linalg.norm(session_bias))
        if session_length > session_limit:
            session_bias *= session_limit / session_length
        self.aim_bias_state = 0.94 * self.aim_bias_state + self.rng.normal(0.0, session_sigma * 0.12)
        self.aim_bias_state = float(np.clip(self.aim_bias_state, -session_limit, session_limit))
        local_parallel += float(np.dot(session_bias, approach)) - self.aim_bias_state
        local_perpendicular += float(np.dot(session_bias, normal))

        # Shared rush/panic component: the same signed draw used by timing,
        # projected onto the approach axis. Rushing cuts the aim short of the
        # circle; falling behind overshoots past it - and both correlate with
        # early/late taps.
        rush_aim = self.rush_draw * 0.70 * self.pressure * sigma * skill_gain
        local_parallel -= rush_aim
        offset = local_parallel * approach + local_perpendicular * normal

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

        # Do not clip ordinary landings.  A very-far safety bound protects the
        # playfield model from a rare compound lapse without introducing the
        # old rim ring; its threshold is outside the circle by a wide margin.
        landing_error = float(np.linalg.norm(offset))
        far_limit = obj.radius * 2.0
        if landing_error > far_limit and landing_error > 1e-9:
            offset *= far_limit / landing_error

        error = float(np.linalg.norm(offset))
        self.landing_offsets[index] = (float(offset[0]), float(offset[1]))
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

        features = pd.DataFrame([self.contexts[index].as_feature_dict()])
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
        object_index: int | None = None,
        radius: float = 32.0,
        continuous: bool = True,
    ) -> None:
        if continuous and not self.profile.perfect_baseline and object_index is not None:
            self._append_continuous_transition(
                points,
                available_from,
                hit_time,
                start,
                target,
                strain,
                object_index,
                radius,
            )
            return
        self._append_legacy_transition(points, available_from, hit_time, start, target, strain, long_settle)

    def _append_continuous_transition(
        self,
        points: list[tuple[float, np.ndarray]],
        available_from: float,
        hit_time: float,
        start: np.ndarray,
        target: np.ndarray,
        strain: float,
        object_index: int,
        radius: float,
    ) -> None:
        """Append one locally planned C² segment for an ordinary circle."""
        self._continuous_endpoint_emitted = False
        state = self.motion_state
        start = np.asarray(start, dtype=float).copy()
        target = np.asarray(target, dtype=float).copy()
        base_start = (
            self._last_waypoint_position.copy()
            if self._last_waypoint_position is not None
            else start.copy()
        )
        if float(np.linalg.norm(state.position - start)) > 1e-5:
            self._sync_motion_state_from_points(points)
        state.position = start.copy()
        # The realised cursor may carry a small noise/wander offset from the
        # previous knot.  Solve the base path from the exact shared waypoint
        # and blend that inherited offset out over this local segment; this
        # keeps the solver C0/C1/C2 while preserving the actual state.
        solver_start = base_start.copy()
        inherited_offset = start - solver_start

        endpoint_velocity, endpoint_acceleration, corner_angle, window = self._waypoint_kinematics(
            object_index, target, hit_time, available_from
        )
        previous_waypoint_time = self._last_waypoint_time_ms if self._last_waypoint_time_ms is not None else available_from
        break_before = self._transition_after_idle or float(available_from) - float(previous_waypoint_time) > 350.0
        distance = float(np.linalg.norm(target - start))
        available = max(self.step_ms, float(hit_time) - float(available_from))
        max_speed = self._speed_ceiling(distance) * self.rng.uniform(0.82, 1.08)
        minimum_duration = max(self.step_ms, distance / max(max_speed, 1.0) * 1000.0 * 1.9)
        segment_duration = max(available, minimum_duration)
        segment_start = float(hit_time) - segment_duration
        if segment_start < float(available_from):
            # Preserve the existing late-arrival/miss behaviour: when the
            # object cannot be reached at the profile ceiling, begin now and
            # let the normal resampler clamp the resulting high-speed trace.
            segment_start = float(available_from)
        # The event can arrive inside one cursor sample interval.  Keep the
        # physical segment duration equal to the actual positive time span;
        # forcing it up to ``step_ms`` would leave the final sample at tau<1
        # and make an otherwise exact Hermite endpoint appear discontinuous.
        actual_duration = max(1e-6, float(hit_time) - segment_start)
        target_reached = actual_duration + 1e-6 >= minimum_duration and available + 1e-6 >= minimum_duration
        if target_reached:
            solver_target = target.copy()
            solver_endpoint_velocity = endpoint_velocity.copy()
            solver_endpoint_acceleration = endpoint_acceleration.copy()
        else:
            # A dense/early event is a genuine late-arrival miss, not licence
            # to solve a 30 px jump in four milliseconds.  Truncate the base
            # path to the distance reachable at the speed ceiling and let the
            # next object inherit that partial state.
            direction = _normalised(target - solver_start)
            reachable_distance = min(
                distance,
                max_speed * actual_duration / 1000.0 / 1.9,
            )
            solver_target = solver_start + direction * reachable_distance
            solver_endpoint_velocity = direction * min(float(np.linalg.norm(endpoint_velocity)), max_speed)
            solver_endpoint_acceleration = endpoint_acceleration.copy()

        start_velocity = state.velocity.copy()
        start_acceleration = state.acceleration.copy()
        horizon_seconds = max(actual_duration / 1000.0, 1e-3)
        predicted = start + start_velocity * horizon_seconds + 0.5 * start_acceleration * horizon_seconds**2
        predicted_error = target - predicted
        reaction_latency_ms = 28.0 + 22.0 * (1.0 - self.skill)
        correction_threshold = max(
            radius * (0.45 + 0.25 * (1.0 - self.skill)),
            self.params.aim_sigma * 3.0,
        )
        if (
            not self.profile.perfect_baseline
            and np.linalg.norm(predicted_error) > correction_threshold
            and actual_duration > reaction_latency_ms
            and float(hit_time) >= state.refractory_until_ms
        ):
            correction_acceleration = predicted_error * (1.65 / max(horizon_seconds * horizon_seconds, 1e-6))
            correction_limit = max(16_000.0, self._speed_ceiling(distance) * 9.0)
            correction_length = float(np.linalg.norm(correction_acceleration))
            if correction_length > correction_limit:
                correction_acceleration *= correction_limit / correction_length
            state.correction = correction_acceleration
            # Refractory time spans more than one ordinary object at the
            # reference profile.  This keeps feedback as an occasional
            # submovement instead of a decorative correction on every
            # segment, while sharp/long errors can still trigger once the
            # motor response has genuinely settled.
            state.refractory_until_ms = float(hit_time) + 220.0 + 180.0 * (1.0 - self.effort) + 80.0 * (1.0 - self.skill)
            self.motion_correction_events += 1
        else:
            state.correction *= math.exp(-actual_duration / 140.0)

        count = max(2, int(math.ceil(actual_duration / self.step_ms)) + 1)
        times = np.linspace(segment_start, float(hit_time), count)
        normalized_time = np.clip((times - segment_start) / actual_duration, 0.0, 1.0)

        def solve_base() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return _quintic_hermite(
                solver_start,
                start_velocity,
                start_acceleration,
                solver_target,
                solver_endpoint_velocity,
                solver_endpoint_acceleration,
                actual_duration,
                normalized_time,
            )

        base_positions, base_velocities, base_accelerations = solve_base()
        # The geometry-derived knot velocity is a local continuity target, not
        # permission for a quintic overshoot to consume the whole movement in
        # a flick.  When there is enough time to reach the target, gently
        # reduce the shared endpoint derivatives until the sampled base path
        # fits a per-move envelope.  The resulting endpoint values are then
        # persisted and inherited by the next segment, so C1/C2 sharing is
        # retained.  Dense impossible events keep their existing late-arrival
        # semantics instead of being silently slowed into a false hit.
        profile_speed_limit = max_speed * (1.20 if not self.profile.perfect_baseline else 1.0)
        derivative_scale = 1.0
        if target_reached and not self.profile.perfect_baseline:
            for _ in range(4):
                base_speed_max = float(np.max(np.linalg.norm(base_velocities, axis=1)))
                if base_speed_max <= profile_speed_limit * 1.01:
                    break
                correction = float(np.clip(profile_speed_limit / max(base_speed_max, 1e-9), 0.70, 0.96))
                derivative_scale *= correction
                solver_endpoint_velocity *= correction
                solver_endpoint_acceleration *= correction
                base_positions, base_velocities, base_accelerations = solve_base()
        else:
            base_speed_max = float(np.max(np.linalg.norm(base_velocities, axis=1)))
        tangent = _normalised(solver_target - solver_start)
        if not np.any(tangent):
            tangent = _normalised(endpoint_velocity) if np.any(endpoint_velocity) else np.array([1.0, 0.0])
        normal = np.array([-tangent[1], tangent[0]])
        initial_wander = state.wander.copy()
        initial_noise = state.ou_offset.copy()
        initial_noise = float(np.dot(initial_noise, normal)) * normal + float(np.dot(initial_noise, tangent)) * tangent * 0.25
        base_end_position = np.asarray(base_positions[-1], dtype=float).copy()
        base_acceleration_max = float(np.max(np.linalg.norm(base_accelerations, axis=1)))
        base_jerk_max = 0.0
        if len(base_accelerations) >= 2:
            base_jerk_max = float(
                np.max(np.linalg.norm(np.diff(base_accelerations, axis=0), axis=1))
                / max(actual_duration / 1000.0 / max(len(base_accelerations) - 1, 1), 1e-6)
            )
        following_point, following_time = (
            window[1]
            if len(window) > 1
            else (solver_target.copy(), float(hit_time) + max(self.step_ms, actual_duration))
        )
        incoming_distance = float(np.linalg.norm(solver_target - solver_start))
        outgoing_distance = float(np.linalg.norm(np.asarray(following_point) - solver_target))
        incoming_seconds = max((float(hit_time) - float(previous_waypoint_time)) / 1000.0, 1e-3)
        outgoing_seconds = max((float(following_time) - float(hit_time)) / 1000.0, 1e-3)
        nominal_knot_speed = min(
            2.0 * incoming_distance / incoming_seconds * outgoing_distance / outgoing_seconds
            / max(incoming_distance / incoming_seconds + outgoing_distance / outgoing_seconds, 1e-6),
            self._speed_ceiling(max(incoming_distance, outgoing_distance, 1.0)),
        )
        geometric_carry_ratio = float(
            np.linalg.norm(solver_endpoint_velocity) / max(nominal_knot_speed, 1.0)
        )
        actual_end_position = state.position.copy()
        appended = 0
        initialized = False
        for time_ms, base_position, base_velocity, base_acceleration, tau in zip(
            times, base_positions, base_velocities, base_accelerations, normalized_time
        ):
            control_fraction = float(np.clip(tau, 0.0, 1.0))
            # A triggered control input, unlike the removed per-segment random
            # Gaussian bump.  It is zero at both knots, so it cannot break C²
            # sharing or become a decorative correction on every move.
            control_shape = control_fraction**3 * (1.0 - control_fraction) ** 2
            control_displacement = state.correction * (actual_duration / 1000.0) ** 2 * control_shape * 0.35
            if not initialized:
                # The inherited state is already represented by ``start``.
                # Do not advance it before the first knot and then add a new
                # absolute offset on top of that same position.
                state.last_timestamp_ms = float(time_ms)
                state.position = start.copy()
                actual_position = start.copy()
                initialized = True
            else:
                actual_position = self._advance_motion_state_sample(
                    float(time_ms),
                    np.asarray(base_position, dtype=float)
                    + inherited_offset * (1.0 - control_fraction)
                    + normal * control_displacement,
                    np.asarray(base_velocity, dtype=float),
                    np.asarray(base_acceleration, dtype=float),
                    tangent,
                    strain,
                    initial_wander,
                    initial_noise,
                )
            if points and float(time_ms) <= points[-1][0]:
                continue
            points.append((float(time_ms), actual_position))
            actual_end_position = actual_position
            appended += 1
        if appended == 0 or points[-1][0] < float(hit_time):
            points.append((float(hit_time), actual_end_position))
        self._continuous_endpoint_emitted = bool(abs(float(points[-1][0]) - float(hit_time)) <= 1e-5)

        # The base segment has exact endpoint derivatives.  Persist exactly
        # those shared knot values; coloured noise/wander only changes the
        # realised position around the base path.
        state.position = actual_end_position.copy()
        state.velocity = solver_endpoint_velocity.copy()
        state.acceleration = solver_endpoint_acceleration.copy()
        state.last_timestamp_ms = float(hit_time)
        self.wander = state.wander.copy()
        self._last_waypoint_position = solver_target.copy()
        self._last_waypoint_time_ms = float(hit_time)
        self._last_waypoint_velocity = solver_endpoint_velocity.copy()
        self._last_waypoint_acceleration = solver_endpoint_acceleration.copy()
        self.motion_segments.append(
            {
                "object_index": object_index,
                "start_time_ms": float(segment_start),
                "end_time_ms": float(hit_time),
                "start": start.tolist(),
                "base_start": base_start.tolist(),
                "target": target.tolist(),
                "base_end": base_end_position.tolist(),
                "start_velocity": start_velocity.tolist(),
                "end_velocity": solver_endpoint_velocity.tolist(),
                "start_acceleration": start_acceleration.tolist(),
                "end_acceleration": solver_endpoint_acceleration.tolist(),
                "corner_angle_deg": math.degrees(corner_angle),
                "distance_px": distance,
                "carry_ratio": geometric_carry_ratio,
                "window_size": len(window),
                "base_speed_max_px_s": base_speed_max,
                "base_acceleration_max_px_s2": base_acceleration_max,
                "base_jerk_max_px_s3": base_jerk_max,
                "available_duration_ms": available,
                "actual_duration_ms": actual_duration,
                "duration_share": actual_duration / max(available, self.step_ms),
                "minimum_duration_ms": minimum_duration,
                "profile_speed_limit_px_s": profile_speed_limit,
                "endpoint_derivative_scale": derivative_scale,
                "realized": self._continuous_endpoint_emitted,
                "target_reached": target_reached,
                "break_before": break_before,
            }
        )
        self._transition_after_idle = False

    @property
    def continuity_stats(self) -> dict[str, object]:
        """Return boundary and corner-stratified diagnostics for benchmark use."""
        segments = [
            segment
            for segment in self.motion_segments
            if bool(segment.get("realized", True)) and bool(segment.get("target_reached", True))
        ]
        segment_indices = {int(segment["object_index"]) for segment in segments}

        def is_stacked_repeat(object_index: int) -> bool:
            if object_index <= 0 or object_index + 1 >= len(self.map.objects):
                return False
            current = self.map.objects[object_index]
            previous = self.map.objects[object_index - 1]
            following = self.map.objects[object_index + 1]
            if current.kind != "circle" or previous.kind != "circle" or following.kind != "circle":
                return True
            incoming = float(np.linalg.norm(
                np.array([current.position.x - previous.end_position.x, current.position.y - previous.end_position.y])
            ))
            outgoing = float(np.linalg.norm(
                np.array([following.position.x - current.end_position.x, following.position.y - current.end_position.y])
            ))
            stack_radius = max(float(current.radius), float(previous.radius), float(following.radius)) * 0.5
            return min(incoming, outgoing) < max(4.0, stack_radius)

        active_segments = [
            segment
            for segment in segments
            if not bool(segment.get("break_before", False))
            and int(segment["object_index"]) - 1 in segment_indices
            and int(segment["object_index"]) + 1 in segment_indices
            and not is_stacked_repeat(int(segment["object_index"]))
        ]
        flow = [segment for segment in active_segments if float(segment["corner_angle_deg"]) <= 45.0]
        turns = [segment for segment in active_segments if 45.0 < float(segment["corner_angle_deg"]) <= 120.0]
        reversals = [segment for segment in active_segments if float(segment["corner_angle_deg"]) > 120.0]

        def carry_summary(values: list[dict[str, object]]) -> dict[str, object]:
            carries = np.asarray([float(value["carry_ratio"]) for value in values], dtype=float)
            carries = carries[np.isfinite(carries)]
            return {
                "samples": int(len(carries)),
                "severe_stop_share": round(float(np.mean(carries < 0.2)), 6) if len(carries) else None,
                "median_carry_ratio": round(float(np.median(carries)), 6) if len(carries) else None,
                "p95_carry_ratio": round(float(np.percentile(carries, 95)), 6) if len(carries) else None,
                "max_carry_ratio": round(float(np.max(carries)), 6) if len(carries) else None,
            }

        boundary_position_errors: list[float] = []
        velocity_errors: list[float] = []
        acceleration_errors: list[float] = []
        for previous, current in zip(segments, segments[1:]):
            # Slider/spinner/idle paths are deliberately allowed to have
            # their own tracking semantics.  Only adjacent ordinary-circle
            # knots are a shared Hermite boundary.
            if int(current["object_index"]) != int(previous["object_index"]) + 1:
                continue
            if bool(previous.get("break_before", False)) or bool(current.get("break_before", False)):
                continue
            boundary_position_errors.append(float(np.linalg.norm(np.asarray(previous["base_end"]) - np.asarray(current["base_start"]))))
            velocity_errors.append(float(np.linalg.norm(np.asarray(previous["end_velocity"]) - np.asarray(current["start_velocity"]))))
            acceleration_errors.append(float(np.linalg.norm(np.asarray(previous["end_acceleration"]) - np.asarray(current["start_acceleration"]))))
        all_position_errors = [
            float(np.linalg.norm(np.asarray(segment["base_end"]) - np.asarray(segment["target"])))
            for segment in segments
        ]
        duration_shares = [
            float(segment["duration_share"])
            for segment in segments
            if "duration_share" in segment
        ]
        derivative_scales = [
            float(segment["endpoint_derivative_scale"])
            for segment in segments
            if "endpoint_derivative_scale" in segment
        ]
        return {
            "segments": len(segments),
            "correction_events": self.motion_correction_events,
            "correction_share": round(self.motion_correction_events / max(1, len(segments)), 6),
            "flow_0_45": carry_summary(flow),
            "turn_45_120": carry_summary(turns),
            "reversal_gt_120": carry_summary(reversals),
            "base_boundary_position_error_px": max(all_position_errors, default=0.0),
            "base_shared_position_error_px": max(boundary_position_errors, default=0.0),
            "base_shared_velocity_error_px_s": max(velocity_errors, default=0.0),
            "base_shared_acceleration_error_px_s2": max(acceleration_errors, default=0.0),
            "base_speed_max_px_s": max(
                (float(segment["base_speed_max_px_s"]) for segment in segments),
                default=0.0,
            ),
            "base_acceleration_max_px_s2": max(
                (float(segment["base_acceleration_max_px_s2"]) for segment in segments),
                default=0.0,
            ),
            "base_jerk_max_px_s3": max(
                (float(segment["base_jerk_max_px_s3"]) for segment in segments),
                default=0.0,
            ),
            "timing_allocation": {
                "duration_share_min": min(duration_shares, default=0.0),
                "duration_share_median": float(np.median(duration_shares)) if duration_shares else 0.0,
                "duration_share_max": max(duration_shares, default=0.0),
                "endpoint_derivative_scale_min": min(derivative_scales, default=1.0),
            },
            "tangent_reset_excess_deg": 0.0,
        }

    def _append_legacy_transition(
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
        residual = self._persistent_noise_series(
            count,
            self.params.aim_sigma * (0.07 + 0.14 * strain),
            self.rng.uniform(160.0, 340.0),
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
        noise = self._persistent_noise_series(
            count,
            self.params.aim_sigma * (0.06 + 0.12 * strain),
            220.0,
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
        object_index: int | None = None,
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
        self._append_transition(
            points,
            available_from,
            hit_time,
            start,
            entry,
            strain,
            long_settle=True,
            object_index=object_index,
            continuous=False,
        )
        times = np.linspace(hit_time, hit_time + duration, count)
        angles = phase + direction * ((times - times[0]) / 1000.0) * rpm / 60.0 * math.tau
        radial_noise = self._persistent_noise_series(count, self.params.aim_sigma, 80.0)
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
