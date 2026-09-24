"""Apply the reviewed circle-window break guard to the desktop worktree."""

from __future__ import annotations

import os
from pathlib import Path


WORKTREE = Path(os.environ.get("HSR_WORKTREE", str(Path(__file__).resolve().parents[3])))
SOURCE = WORKTREE / "research" / "human-sim" / "src" / "human_sim" / "coherent_execution.py"
TEST = WORKTREE / "research" / "human-sim" / "tests" / "test_coherent_execution_runtime.py"

OLD = '''def _circle_windows(map_plan: MapPlan) -> list[list[int]]:
    result: list[list[int]] = []
    run: list[int] = []
    for index, obj in enumerate(map_plan.objects):
        if obj.kind == "circle":
            run.append(index)
            continue
        for offset in range(0, max(0, len(run) - 3), 3):
            result.append(run[offset : offset + 4])
        run = []
    for offset in range(0, max(0, len(run) - 3), 3):
        result.append(run[offset : offset + 4])
    return result
'''

NEW = '''def _circle_windows(map_plan: MapPlan) -> list[list[int]]:
    result: list[list[int]] = []
    run: list[int] = []

    def append_run() -> None:
        for offset in range(0, max(0, len(run) - 3), 3):
            result.append(run[offset : offset + 4])

    for index, obj in enumerate(map_plan.objects):
        if obj.kind == "circle":
            # A long gap is a break, even when circles bound both sides.  The
            # learned execution may not rewrite the mathematical break motion.
            if run and obj.start_time_ms - map_plan.objects[run[-1]].end_time_ms >= DWELL_MS:
                append_run()
                run = []
            run.append(index)
            continue
        append_run()
        run = []
    append_run()
    return result
'''

TEST_IMPORT_OLD = '''    apply_coherent_trajectory_model,
    load_coherent_model,
'''
TEST_IMPORT_NEW = '''    _circle_windows,
    apply_coherent_trajectory_model,
    load_coherent_model,
'''

TEST_APPEND = '''

def test_long_gap_splits_circle_windows_and_preserves_break_motion():
    windows = HitWindows(40, 80, 120, 150)
    times = (2000.0, 2200.0, 2400.0, 4000.0, 4200.0, 4400.0)
    objects = tuple(
        MapObject(index, "circle", time, time, Point(80.0 + index * 50.0, 100.0),
                  Point(80.0 + index * 50.0, 100.0), 32, hit_windows=windows)
        for index, time in enumerate(times)
    )
    plan = MapPlan(1, "a" * 64, "b" * 32, 1.0, (), objects, {})
    assert _circle_windows(plan) == []
    model = CoherentTrajectoryModel(
        "c" * 64, "d" * 64, np.zeros(22), np.ones(22), np.zeros((45, 23, 2)), 7
    )
    frames = [
        TraceFrame(time_us, 40.0 + time_us / 10_000.0, 90.0, False, False)
        for time_us in range(0, 4_501_000, 2000)
    ]
    output, diagnostics = apply_coherent_trajectory_model(frames, plan, model)
    assert output == frames
    assert diagnostics["learned_segment_count"] == 0
    assert diagnostics["fallback"] is True
'''


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    if NEW in source:
        print("source already patched")
    else:
        if source.count(OLD) != 1:
            raise ValueError("Desktop coherent source no longer matches the reviewed patch")
        SOURCE.write_text(source.replace(OLD, NEW), encoding="utf-8")
        print(f"patched {SOURCE}")
    tests = TEST.read_text(encoding="utf-8")
    if "test_long_gap_splits_circle_windows_and_preserves_break_motion" in tests:
        print("test already present")
    else:
        if tests.count(TEST_IMPORT_OLD) != 1:
            raise ValueError("Runtime test import no longer matches the reviewed patch")
        TEST.write_text(tests.replace(TEST_IMPORT_OLD, TEST_IMPORT_NEW) + TEST_APPEND, encoding="utf-8")
        print(f"patched {TEST}")


if __name__ == "__main__":
    main()
