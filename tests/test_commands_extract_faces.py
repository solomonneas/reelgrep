"""CLI tests for `reelgrep extract-faces`."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from reelgrep import config
from reelgrep.cli import main as cli
from reelgrep.db import connect, migrate


def _fake_insightface_singleface(monkeypatch):
    """Inject a fake insightface that returns one detection per frame."""
    rng = np.random.RandomState(0)
    e = rng.randn(512).astype(np.float32)
    e /= np.linalg.norm(e)
    fake_face = types.SimpleNamespace(
        bbox=np.array([10, 20, 110, 140], dtype=np.float32),
        normed_embedding=e,
    )
    fake_app_module = types.SimpleNamespace(
        FaceAnalysis=lambda *a, **kw: types.SimpleNamespace(
            prepare=lambda *a, **kw: None,
            get=lambda img: [fake_face],
        ),
    )
    monkeypatch.setitem(sys.modules, "insightface",
                        types.SimpleNamespace(app=fake_app_module))
    monkeypatch.setitem(sys.modules, "insightface.app", fake_app_module)
    monkeypatch.setitem(
        sys.modules, "cv2",
        types.SimpleNamespace(imread=lambda p: np.zeros((480, 640, 3), dtype=np.uint8)),
    )


def _seed(tmp_path: Path, *, video_path: str = "/v.mp4") -> Path:
    db = tmp_path / "idx.sqlite"
    config.set_db_override(db)
    conn = connect(db)
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + ("a" * 64), video_path, 5000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid = conn.execute("SELECT id FROM videos WHERE path = ?", (video_path,)).fetchone()[0]
    fp = tmp_path / f"f_{vid}.jpg"
    fp.write_bytes(b"x")
    conn.execute(
        "INSERT INTO frames(video_id, timestamp_ms, path, sampling_strategy) VALUES (?,?,?,?)",
        (vid, 0, str(fp), "every_n"),
    )
    conn.commit()
    conn.close()
    return db


@pytest.fixture(autouse=True)
def _reset_config():
    yield
    config.reset_settings()


def test_extract_faces_command_reports_detections(tmp_path, monkeypatch):
    _seed(tmp_path)
    _fake_insightface_singleface(monkeypatch)

    result = CliRunner().invoke(cli, ["extract-faces", "/v.mp4"])
    assert result.exit_code == 0, result.output
    assert "detections: 1" in result.output


def test_extract_faces_command_handles_missing_extra(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.setitem(sys.modules, "insightface", None)
    monkeypatch.setitem(sys.modules, "insightface.app", None)
    result = CliRunner().invoke(cli, ["extract-faces", "/v.mp4"])
    assert result.exit_code == 1
    # InsightFaceMissingError surfaces via stderr (or combined output).
    combined = (result.output or "") + (result.stderr or "")
    assert "insightface" in combined.lower()


def test_extract_faces_all_backfills_every_video(tmp_path, monkeypatch):
    db = _seed(tmp_path, video_path="/v1.mp4")
    # Add a second video + frame.
    conn = connect(db)
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + ("b" * 64), "/v2.mp4", 5000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid2 = conn.execute("SELECT id FROM videos WHERE path='/v2.mp4'").fetchone()[0]
    fp = tmp_path / "f_v2.jpg"
    fp.write_bytes(b"x")
    conn.execute(
        "INSERT INTO frames(video_id, timestamp_ms, path, sampling_strategy) VALUES (?,?,?,?)",
        (vid2, 0, str(fp), "every_n"),
    )
    conn.commit()
    conn.close()

    _fake_insightface_singleface(monkeypatch)
    result = CliRunner().invoke(cli, ["extract-faces", "--all"])
    assert result.exit_code == 0, result.output
    assert "videos processed: 2" in result.output


def test_extract_faces_requires_video_or_all():
    result = CliRunner().invoke(cli, ["extract-faces"])
    assert result.exit_code != 0
    assert "VIDEO" in result.output or "--all" in result.output
