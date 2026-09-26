from __future__ import annotations

import os
from pathlib import Path

import pytest

from human_sim.cli import _default_auto_output, _default_osu_storage, _default_state_log, _dotnet_apphost


@pytest.mark.skipif(os.name != "posix", reason="XDG defaults apply to the Linux/macOS CLI")
def test_linux_cli_uses_xdg_data_cache_state_and_release_apphost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert _default_osu_storage() == tmp_path / "data" / "osu-development" / "files"
    assert _default_auto_output() == tmp_path / "cache" / "intelligence-database-hsr" / "auto"
    assert _default_state_log() == tmp_path / "state" / "intelligence-database-hsr" / "logs"

    project_root = tmp_path / "workspace"
    release_host = project_root / "research" / "HumanSim.Runner" / "bin" / "Release" / "net8.0" / "HumanSim.Runner"
    release_host.parent.mkdir(parents=True)
    release_host.touch()
    assert _dotnet_apphost(project_root, "HumanSim.Runner") == release_host


@pytest.mark.skipif(os.name != "posix", reason="XDG defaults apply to the Linux/macOS CLI")
def test_relative_xdg_home_values_fall_back_to_absolute_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("XDG_DATA_HOME", "relative-data")
    monkeypatch.setenv("XDG_CACHE_HOME", "relative-cache")
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")

    assert _default_osu_storage() == tmp_path / "home" / ".local" / "share" / "osu-development" / "files"
    assert _default_auto_output() == tmp_path / "home" / ".cache" / "intelligence-database-hsr" / "auto"
    assert _default_state_log() == tmp_path / "home" / ".local" / "state" / "intelligence-database-hsr" / "logs"
