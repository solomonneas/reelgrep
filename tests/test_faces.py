"""Unit tests for reelgrep.faces.extract_faces and helpers."""
from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from reelgrep import config
from reelgrep.db import connect, migrate


def _fake_insightface(monkeypatch, *, detections_per_frame):
    """Inject a fake insightface.app.FaceAnalysis returning a scripted sequence."""
    calls = {"n": 0}

    class FakeFace:
        def __init__(self, bbox, embedding):
            self.bbox = np.array(bbox, dtype=np.float32)  # xyxy
            self.normed_embedding = embedding.astype(np.float32)

    class FakeApp:
        def __init__(self, *a, **kw): pass
        def prepare(self, *a, **kw): pass
        def get(self, img):
            i = calls["n"]
            calls["n"] += 1
            return [FakeFace((x, y, x + w, y + h), e)
                    for (x, y, w, h), e in detections_per_frame[i % len(detections_per_frame)]]

    fake_app_module = types.SimpleNamespace(FaceAnalysis=FakeApp)
    fake_insightface = types.SimpleNamespace(app=fake_app_module)
    monkeypatch.setitem(sys.modules, "insightface", fake_insightface)
    monkeypatch.setitem(sys.modules, "insightface.app", fake_app_module)
    fake_cv2 = types.SimpleNamespace(imread=lambda p: np.zeros((480, 640, 3), dtype=np.uint8))
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)


def _seed_video_with_frames(conn: sqlite3.Connection, *, n_frames: int, frame_dir: Path) -> int:
    """Insert a videos row + n frames. Returns the video_id."""
    cur = conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "a"*64, "/some/video.mp4", n_frames * 5000,
         "2026-05-18T00:00:00Z", "{}"),
    )
    video_id = cur.lastrowid
    for i in range(n_frames):
        p = frame_dir / f"f{i}.jpg"
        p.write_bytes(b"fake")
        conn.execute(
            "INSERT INTO frames(video_id, timestamp_ms, path, sampling_strategy) "
            "VALUES (?,?,?,?)",
            (video_id, i * 5000, str(p), "every_n"),
        )
    conn.commit()
    return video_id


@pytest.fixture(autouse=True)
def _reset_config():
    yield
    config.reset_settings()


def test_extract_faces_writes_one_detection_per_face(tmp_path, monkeypatch):
    """One frame with two faces produces two rows; embeddings round-trip as float32[512]."""
    config.set_db_override(tmp_path / "idx.sqlite")

    emb_a = np.random.RandomState(0).randn(512).astype(np.float32)
    emb_b = np.random.RandomState(1).randn(512).astype(np.float32)
    emb_a /= np.linalg.norm(emb_a)
    emb_b /= np.linalg.norm(emb_b)
    _fake_insightface(monkeypatch, detections_per_frame=[
        [((10, 20, 100, 120), emb_a), ((300, 50, 80, 100), emb_b)],
    ])

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    _seed_video_with_frames(conn, n_frames=1, frame_dir=tmp_path)
    conn.close()

    from reelgrep.faces import extract_faces
    result = extract_faces("/some/video.mp4", db_path=tmp_path / "idx.sqlite")

    assert result.detections_added == 2
    assert result.frames_scanned == 1
    assert result.embedding_model == "insightface_buffalo_l"

    conn = sqlite3.connect(tmp_path / "idx.sqlite")
    rows = conn.execute(
        "SELECT bbox_x, bbox_y, bbox_w, bbox_h, embedding FROM face_detections ORDER BY id"
    ).fetchall()
    assert [(r[0], r[1], r[2], r[3]) for r in rows] == [(10, 20, 100, 120), (300, 50, 80, 100)]
    decoded = np.frombuffer(rows[0][4], dtype=np.float32)
    assert decoded.shape == (512,)
    np.testing.assert_allclose(decoded, emb_a, atol=1e-6)


def test_extract_faces_idempotent_for_same_model(tmp_path, monkeypatch):
    """Running extract_faces twice on a video without --force is a no-op the second time."""
    config.set_db_override(tmp_path / "idx.sqlite")
    emb = np.random.RandomState(2).randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    _fake_insightface(monkeypatch, detections_per_frame=[[((0, 0, 50, 50), emb)]])

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    _seed_video_with_frames(conn, n_frames=1, frame_dir=tmp_path)
    conn.close()

    from reelgrep.faces import extract_faces
    extract_faces("/some/video.mp4", db_path=tmp_path / "idx.sqlite")
    second = extract_faces("/some/video.mp4", db_path=tmp_path / "idx.sqlite")
    assert second.detections_added == 0
    assert second.skipped_existing is True


def test_extract_faces_raises_when_extra_missing(tmp_path, monkeypatch):
    """If insightface is not importable, raise InsightFaceMissingError."""
    monkeypatch.setitem(sys.modules, "insightface", None)
    monkeypatch.setitem(sys.modules, "insightface.app", None)
    config.set_db_override(tmp_path / "idx.sqlite")

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    _seed_video_with_frames(conn, n_frames=1, frame_dir=tmp_path)
    conn.close()

    from reelgrep.faces import InsightFaceMissingError, extract_faces
    with pytest.raises(InsightFaceMissingError):
        extract_faces("/some/video.mp4", db_path=tmp_path / "idx.sqlite")
