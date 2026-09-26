#!/usr/bin/env python3
"""Check the compiled runner, exporter and learned planner using synthetic input."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def verify(root: Path) -> None:
    root = root.resolve()
    suffix = ".exe" if os.name == "nt" else ""
    model = root / "research/human-sim/models/experimental/seed101-math-residual-g100-v4.json"
    expected_model = "3ba4d157124fa078b862062a38578561265bf03aaa2180a566440874b22bc7a2"
    if hashlib.sha256(model.read_bytes()).hexdigest() != expected_model:
        raise RuntimeError("Installed model does not match the reviewed checkpoint")

    with tempfile.TemporaryDirectory(prefix="hsr-installed-plan-") as directory:
        temporary = Path(directory)
        coordinates = [(128, 192), (256, 64), (384, 192), (256, 320)]
        objects = "\n".join(
            f"{coordinates[i % 4][0]},{coordinates[i % 4][1]},{2000 + 500 * i},1,0,0:0:0:0:"
            for i in range(80)
        )
        content = (
            "osu file format v14\n\n[General]\nAudioFilename: none.wav\nMode: 0\n\n"
            "[Metadata]\nTitle:HSR synthetic compiler check\nArtist:HSR\nCreator:HSR\nVersion:Research\n\n"
            "[Difficulty]\nHPDrainRate:5\nCircleSize:4\nOverallDifficulty:7\nApproachRate:8\n"
            "SliderMultiplier:1.4\nSliderTickRate:1\n\n[TimingPoints]\n0,500,4,2,1,100,1,0\n\n"
            f"[HitObjects]\n{objects}\n"
        ).encode()
        map_hash = hashlib.sha256(content).hexdigest()
        storage = temporary / "files"
        beatmap = storage / map_hash[0] / map_hash[:2] / map_hash
        beatmap.parent.mkdir(parents=True)
        beatmap.write_bytes(content)
        command = [
            str(root / f"research/HumanSim.Runner/bin/Release/net8.0/HumanSim.Runner{suffix}"),
            "--client", str(root / f"osu.Desktop/bin/Release/net8.0/osu!{suffix}"),
            "--auto-plan", "profile", "--execution-mode", "coherent",
            "--execution-model", str(model), "--execution-blend", "1",
            "--skill", "99.5", "--effort", "100", "--sample-rate", "1000",
            "--workspace-root", str(root), "--osu-storage", str(storage),
            "--output-directory", str(temporary / "traces"), "--verify-plan", map_hash,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=240)
        if completed.returncode:
            raise RuntimeError(f"Installed compiler failed:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")
        verification = next(
            json.loads(line) for line in completed.stdout.splitlines()
            if line.startswith('{"verification":"compiled_runner_planning_only"')
        )
        header = verification["header"]
        if (
            not header["synthetic"] or header["beatmap_sha256"] != map_hash
            or header["execution_effective_mode"] != "coherent" or header["execution_fallback"]
            or header["execution_learned_segment_count"] <= 0
            or header["execution_changed_sample_count"] <= 0
            or header["execution_fallback_segment_count"] != 0
        ):
            raise RuntimeError("Installed compiler did not produce the required learned trace")
        print(json.dumps({
            "installed_compiler": "passed", "frames": verification["frames"],
            "learned_segments": header["execution_learned_segment_count"],
            "changed_samples": header["execution_changed_sample_count"],
            "fallback_segments": header["execution_fallback_segment_count"],
            "model_sha256": expected_model, "input_dispatched": False,
        }))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: verify-installed-planner.py /path/to/HSR/root")
    verify(Path(sys.argv[1]))
