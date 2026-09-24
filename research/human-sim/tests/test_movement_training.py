from __future__ import annotations

import json

import numpy as np
import pytest

from human_sim import movement_training as movement


def test_empty_corpus_reports_missing_data_and_cannot_train(tmp_path):
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"schema_version": 1, "replays": []}))
    rows, report = movement.load_corpus(str(manifest))
    assert rows == []
    assert report["status"] == "awaiting_human_replays"
    with pytest.raises(ValueError, match="Insufficient train/beginner"):
        movement.train(str(manifest), str(tmp_path / "run"))


def test_hsr_is_rejected_even_when_manifest_claims_human(tmp_path):
    replay = tmp_path / "replay.ndjson"
    replay.write_text(json.dumps({"kind": "replay_header", "schema_version": 1, "mods": ["HSR"]}) + "\n")
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"schema_version": 1, "replays": [{
        "replay": replay.name, "map_plan": "not-read.json", "human_verified": True,
        "source": "test fixture", "skill_group": "beginner", "skill_evidence": "fixture", "split": "train"}]}))
    with pytest.raises(ValueError, match="Synthetic/automated"):
        movement.load_corpus(str(manifest))


def test_training_reports_every_tier_and_preserves_previous_candidate(tmp_path, monkeypatch):
    # Artificial fixture data validates the pipeline only. Never added to corpus.
    rows = []
    rng = np.random.default_rng(7)
    for split in movement.SPLITS:
        for group in movement.SKILL_GROUPS:
            for index in range(104 if split == "train" else 26):
                x = rng.normal(size=8)
                rows.append({"split": split, "skill_group": group, "player": f"{split}-{group}-{index % 2}",
                             "x": x, "y": x * 0.1, "map": split})
    monkeypatch.setattr(movement, "load_corpus", lambda _: (rows, {"manifest_sha256": "fixture-only"}))
    output = tmp_path / "candidate"
    report = movement.train("fixture", str(output))
    assert len(report["metrics"]) == 8
    assert report["status"] == "experimental_not_promoted"
    assert (output / "movement.joblib").exists()
    with pytest.raises(ValueError, match="never overwrite"):
        movement.train("fixture", str(output))


def _human_fixture(tmp_path, name, player="player-a", split="train"):
    map_path = tmp_path / f"{name}.map.json"
    map_path.write_text(json.dumps({"schema_version": 1, "beatmap_sha256": "a" * 64,
        "beatmap_md5": "b" * 32, "clock_rate": 1, "mods": [], "objects": [
            {"index": index, "kind": "circle", "effective_start_time_ms": 100 + 100 * index,
             "effective_end_time_ms": 100 + 100 * index,
             "position": {"x": 100 + 50 * index, "y": 192}, "radius": 32} for index in range(4)]}))
    replay = tmp_path / f"{name}.ndjson"
    header = {"kind": "replay_header", "schema_version": 1, "player_hash": player,
              "beatmap_md5": "b" * 32, "mods": []}
    frames = [{"time_ms": t, "x": 100 + 0.5 * (t - 100), "y": 192 + np.sin(t / 50),
               "k1": False, "k2": False} for t in range(80, 421, 10)]
    replay.write_text("\n".join(json.dumps(value) for value in [header, *frames]) + "\n")
    return {"human_verified": True, "source": "test fixture only", "skill_evidence": "fixture",
            "skill_group": "beginner", "split": split, "map_plan": map_path.name, "replay": replay.name}


def test_extracts_continuous_windows_without_requiring_successful_hits(tmp_path):
    record = _human_fixture(tmp_path, "one")
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"schema_version": 1, "replays": [record]}))
    rows, report = movement.load_corpus(str(manifest))
    assert len(rows) == 3
    assert report["coverage"]["train/beginner"] == 3
    assert all(len(row["y"]) == 8 and np.all(np.isfinite(row["y"])) for row in rows)


def test_map_leakage_rejected_for_different_players(tmp_path):
    first = _human_fixture(tmp_path, "one")
    second = _human_fixture(tmp_path, "two", player="player-b", split="test")
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"schema_version": 1, "replays": [first, second]}))
    with pytest.raises(ValueError, match="Player/map leakage"):
        movement.load_corpus(str(manifest))
