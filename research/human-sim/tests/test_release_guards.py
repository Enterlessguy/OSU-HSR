from pathlib import Path
from urllib.request import Request

import numpy as np
import pytest

from human_sim import collector
from human_sim.coherent_execution import _context, _continuous_envelope, _design, _predict, _scaled_envelope, load_coherent_model
from human_sim.network import _HTTPSRedirect, require_https


@pytest.mark.parametrize("url", ["file:///private", "http://example.org/a", "ftp://example.org", "https://user:secret@example.org/a"])
def test_acquisition_rejects_non_https_and_embedded_credentials(url):
    with pytest.raises(ValueError):
        require_https(url)


def test_redirect_rejects_downgrade_and_removes_cross_host_token():
    handler = _HTTPSRedirect()
    request = Request("https://osu.ppy.sh/a", headers={"Authorization": "Bearer test"})
    with pytest.raises(ValueError):
        handler.redirect_request(request, None, 302, "redirect", {}, "http://example.org/b")
    other = handler.redirect_request(request, None, 302, "redirect", {}, "https://example.org/b")
    assert other.get_header("Authorization") is None


@pytest.mark.parametrize("score_id", ["../escape", "1/2", "１２３", "-1"])
def test_score_id_cannot_escape_download_directory(tmp_path, monkeypatch, score_id):
    ids = tmp_path / "ids.txt"
    ids.write_text(score_id, encoding="utf-8")
    monkeypatch.setattr(collector, "_oauth_token", lambda: "test")
    with pytest.raises(ValueError, match="ASCII decimal"):
        collector.collect_replays(ids, tmp_path / "replays")
    assert not list(tmp_path.rglob("*.osr"))


@pytest.mark.parametrize("alpha", [1.0, .32, .08, .003, 0.0])
def test_scaled_polynomial_envelope_matches_direct_extrema(alpha):
    nodes = np.array([0., 80., 220., 300.])
    coefficients = np.random.default_rng(101).normal(size=(3 * len(nodes) + 4 * (len(nodes) - 1), 2))
    full = _continuous_envelope(coefficients, nodes)
    scaled = _scaled_envelope(full, alpha)
    direct = _continuous_envelope(coefficients * alpha, nodes)
    for metric in ("speed_px_s", "acceleration_px_s2", "jerk_px_s3"):
        assert scaled[metric] == pytest.approx(direct[metric], rel=1e-7, abs=1e-6)
    assert scaled["valid"] == direct["valid"]
    assert scaled["failure_families"] == direct["failure_families"]


@pytest.mark.parametrize("contacts", [
    [[100, 100]] * 4,
    [[100, 100], [300, 100], [100, 100], [300, 100]],
    [[100, 100], [300, 100], [300, 300], [100, 100]],
])
def test_pinned_model_keeps_contact_and_c2_outer_joins_on_edge_geometry(contacts):
    root = Path(__file__).resolve().parents[1]
    model = load_coherent_model(root / "models/experimental/seed101-math-residual-g100-v4.json")
    events = np.array([0., 100., 240., 410.])
    context = _context(events, np.array(contacts, dtype=float), np.array([[20., 10.], [-10., 30.]]), np.zeros((2, 2)))
    residual = _predict(context, model) - context["baseline"]
    assert np.all(np.isfinite(residual))
    assert np.max(np.abs(_design(events, context["nodes"]) @ residual)) < 1e-8
    for order in (0, 1, 2):
        assert np.max(np.abs(_design(events[[0, -1]], context["nodes"], order=order) @ residual)) < 1e-6
