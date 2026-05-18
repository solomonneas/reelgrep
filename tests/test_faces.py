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


def test_extract_faces_force_replaces_existing_detections(tmp_path, monkeypatch):
    """`extract_faces(..., force=True)` replaces detections instead of appending."""
    config.set_db_override(tmp_path / "idx.sqlite")
    emb = np.random.RandomState(11).randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    _fake_insightface(monkeypatch, detections_per_frame=[[((0, 0, 50, 50), emb)]])

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    _seed_video_with_frames(conn, n_frames=1, frame_dir=tmp_path)
    conn.close()

    from reelgrep.faces import extract_faces
    extract_faces("/some/video.mp4", db_path=tmp_path / "idx.sqlite")
    extract_faces("/some/video.mp4", db_path=tmp_path / "idx.sqlite", force=True)

    conn = sqlite3.connect(tmp_path / "idx.sqlite")
    rows = conn.execute("SELECT COUNT(*) FROM face_detections").fetchone()[0]
    assert rows == 1, f"force=True should replace, not append; got {rows} rows"


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


def _seed_synthetic_detections(conn, *, video_id, frame_dir, embeddings_per_frame):
    """Insert frames + face_detections from a list[list[np.ndarray]]."""
    for i, embs in enumerate(embeddings_per_frame):
        fpath = frame_dir / f"sf{i}.jpg"
        fpath.write_bytes(b"x")
        cur = conn.execute(
            "INSERT INTO frames(video_id, timestamp_ms, path, sampling_strategy) "
            "VALUES (?,?,?,?)",
            (video_id, i * 5000, str(fpath), "every_n"),
        )
        fid = cur.lastrowid
        for j, e in enumerate(embs):
            e = e.astype(np.float32)
            conn.execute(
                "INSERT INTO face_detections(frame_id, bbox_x, bbox_y, bbox_w, bbox_h, "
                "embedding, embedding_model, detected_at) VALUES (?,?,?,?,?,?,?,?)",
                (fid, 10 * j, 20 * j, 80, 100, e.tobytes(),
                 "insightface_buffalo_l", "2026-05-18T00:00:00Z"),
            )
    conn.commit()


def test_cluster_faces_finds_three_well_separated_clusters(tmp_path):
    """Three Gaussian-separated identity clusters with 10 detections each become 3 clusters."""
    config.set_db_override(tmp_path / "idx.sqlite")
    rng = np.random.RandomState(42)
    centers = [rng.randn(512).astype(np.float32) * 5 for _ in range(3)]
    centers = [c / np.linalg.norm(c) for c in centers]
    embs_per_frame: list[list[np.ndarray]] = []
    for c in centers:
        for _ in range(10):
            e = c + 0.05 * rng.randn(512).astype(np.float32)
            e /= np.linalg.norm(e)
            embs_per_frame.append([e])

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "f"*64, "/v.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid = conn.execute("SELECT id FROM videos").fetchone()[0]
    _seed_synthetic_detections(
        conn, video_id=vid, frame_dir=tmp_path, embeddings_per_frame=embs_per_frame,
    )
    conn.close()

    from reelgrep.faces import cluster_faces
    report = cluster_faces(min_cluster_size=5, db_path=tmp_path / "idx.sqlite")
    assert report.clusters_found == 3
    assert report.detections_clustered == 30
    assert report.noise_detections == 0


def test_cluster_faces_preserves_label_across_rerun(tmp_path):
    """A labeled cluster keeps its label after recluster when its centroid stays close."""
    config.set_db_override(tmp_path / "idx.sqlite")
    rng = np.random.RandomState(7)
    c = rng.randn(512).astype(np.float32)
    c /= np.linalg.norm(c)
    embs = []
    for _ in range(8):
        e = c + 0.03 * rng.randn(512).astype(np.float32)
        e /= np.linalg.norm(e)
        embs.append([e])

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "e"*64, "/v.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid = conn.execute("SELECT id FROM videos").fetchone()[0]
    _seed_synthetic_detections(conn, video_id=vid, frame_dir=tmp_path, embeddings_per_frame=embs)
    conn.close()

    from reelgrep.faces import cluster_faces
    cluster_faces(min_cluster_size=5, db_path=tmp_path / "idx.sqlite")
    conn = sqlite3.connect(tmp_path / "idx.sqlite")
    conn.execute("UPDATE face_clusters SET label = ? WHERE id = ?", ("Speaker A", 1))
    conn.commit()
    conn.close()

    # Re-run; the same detections should re-cluster the same way and re-attach the label.
    cluster_faces(min_cluster_size=5, db_path=tmp_path / "idx.sqlite")
    conn = sqlite3.connect(tmp_path / "idx.sqlite")
    labels = [r[0] for r in conn.execute("SELECT label FROM face_clusters WHERE label IS NOT NULL")]
    assert labels == ["Speaker A"]


def test_faces_list_and_get_cluster(tmp_path):
    config.set_db_override(tmp_path / "idx.sqlite")
    rng = np.random.RandomState(99)
    c1 = rng.randn(512).astype(np.float32)
    c1 /= np.linalg.norm(c1)
    c2 = rng.randn(512).astype(np.float32)
    c2 /= np.linalg.norm(c2)
    embs = []
    for _ in range(6):
        e = c1 + 0.02 * rng.randn(512).astype(np.float32)
        embs.append([e / np.linalg.norm(e)])
    for _ in range(8):
        e = c2 + 0.02 * rng.randn(512).astype(np.float32)
        embs.append([e / np.linalg.norm(e)])

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "9"*64, "/v.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid = conn.execute("SELECT id FROM videos").fetchone()[0]
    _seed_synthetic_detections(conn, video_id=vid, frame_dir=tmp_path, embeddings_per_frame=embs)
    conn.close()

    from reelgrep.faces import Faces, cluster_faces
    cluster_faces(min_cluster_size=5, db_path=tmp_path / "idx.sqlite")

    faces = Faces(db_path=tmp_path / "idx.sqlite")
    clusters = faces.list_clusters()
    assert len(clusters) == 2
    assert clusters[0].size >= clusters[1].size  # ranked desc by size
    detail = faces.get_cluster(clusters[0].id)
    assert detail.size == clusters[0].size
    members = faces.cluster_members(clusters[0].id)
    assert len(members) == clusters[0].size


def test_faces_find_by_label_and_find_by_embedding(tmp_path):
    config.set_db_override(tmp_path / "idx.sqlite")
    rng = np.random.RandomState(123)
    c = rng.randn(512).astype(np.float32)
    c /= np.linalg.norm(c)
    raw = [c + 0.02 * rng.randn(512).astype(np.float32) for _ in range(7)]
    embs = [[e / np.linalg.norm(e)] for e in raw]
    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "8"*64, "/v.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid = conn.execute("SELECT id FROM videos").fetchone()[0]
    _seed_synthetic_detections(conn, video_id=vid, frame_dir=tmp_path, embeddings_per_frame=embs)
    conn.close()

    from reelgrep.faces import Faces, cluster_faces
    cluster_faces(min_cluster_size=5, db_path=tmp_path / "idx.sqlite")
    faces = Faces(db_path=tmp_path / "idx.sqlite")
    clusters = faces.list_clusters()
    assert clusters, "cluster_faces should have produced at least one cluster"
    cluster_id = clusters[0].id
    faces.label_cluster(cluster_id, "Speaker A")

    by_label = faces.find_by_label("Speaker A")
    assert len(by_label) >= 1

    query = c + 0.01 * rng.randn(512).astype(np.float32)
    query /= np.linalg.norm(query)
    top = faces.find_by_embedding(query, top_k=3, min_similarity=0.5)
    assert 1 <= len(top) <= 3
    for _det, sim in top:
        assert sim >= 0.5


def test_orphan_labels_can_be_reattached_after_recluster(tmp_path):
    """A labeled cluster that gets orphaned by a recluster can be relabeled onto a new cluster."""
    config.set_db_override(tmp_path / "idx.sqlite")
    rng = np.random.RandomState(53)
    c1 = rng.randn(512).astype(np.float32)
    c1 /= np.linalg.norm(c1)
    embs = []
    for _ in range(8):
        e = c1 + 0.02 * rng.randn(512).astype(np.float32)
        embs.append([e / np.linalg.norm(e)])

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "5"*64, "/v.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid = conn.execute("SELECT id FROM videos").fetchone()[0]
    _seed_synthetic_detections(conn, video_id=vid, frame_dir=tmp_path, embeddings_per_frame=embs)
    conn.close()

    from reelgrep.faces import Faces, cluster_faces
    cluster_faces(min_cluster_size=5, db_path=tmp_path / "idx.sqlite")
    faces = Faces(db_path=tmp_path / "idx.sqlite")
    cluster_id = faces.list_clusters()[0].id
    faces.label_cluster(cluster_id, "Person A")

    # Now wipe and re-seed with completely different embeddings so the old centroid
    # no longer matches anything; "Person A" will be orphaned by carry-over.
    conn = sqlite3.connect(tmp_path / "idx.sqlite")
    conn.execute("DELETE FROM face_detections")
    rng2 = np.random.RandomState(91)
    c2 = rng2.randn(512).astype(np.float32)
    c2 /= np.linalg.norm(c2)
    # Re-seed orthogonal-ish embeddings (NEW video so the recluster sees a distinct centroid).
    for i in range(8):
        e = c2 + 0.02 * rng2.randn(512).astype(np.float32)
        e /= np.linalg.norm(e)
        # Insert directly under the SAME video for simplicity (frames already exist? add new ones).
        cur = conn.execute(
            "INSERT INTO frames(video_id, timestamp_ms, path, sampling_strategy) VALUES (?,?,?,?)",
            (vid, 100000 + i * 1000, str(tmp_path / f"orth_{i}.jpg"), "every_n"),
        )
        (tmp_path / f"orth_{i}.jpg").write_bytes(b"x")
        conn.execute(
            "INSERT INTO face_detections(frame_id, bbox_x, bbox_y, bbox_w, bbox_h, "
            "embedding, embedding_model, detected_at) VALUES (?,?,?,?,?,?,?,?)",
            (cur.lastrowid, 0, 0, 80, 100, e.tobytes(),
             "insightface_buffalo_l", "2026-05-18T00:00:00Z"),
        )
    conn.commit()
    conn.close()

    report = cluster_faces(min_cluster_size=5, db_path=tmp_path / "idx.sqlite")
    # The original "Person A" label snapshot was the OLD centroid (c1), and the new
    # cluster centroid is c2 — far outside the label_carry_threshold.
    assert "Person A" in report.labels_orphaned

    # Now the user can re-attach by labeling the new cluster — this must not collide.
    faces = Faces(db_path=tmp_path / "idx.sqlite")
    new_clusters = faces.list_clusters()
    assert len(new_clusters) >= 1, "expected at least one new cluster"
    faces.label_cluster(new_clusters[0].id, "Person A")  # must not raise
    assert faces.find_by_label("Person A")


def test_faces_label_collision_raises(tmp_path):
    """Labeling cluster B with a name already on cluster A raises FacesError."""
    config.set_db_override(tmp_path / "idx.sqlite")
    rng = np.random.RandomState(77)
    c1 = rng.randn(512).astype(np.float32)
    c1 /= np.linalg.norm(c1)
    c2 = rng.randn(512).astype(np.float32)
    c2 /= np.linalg.norm(c2)
    embs = []
    for _ in range(6):
        e = c1 + 0.02 * rng.randn(512).astype(np.float32)
        embs.append([e / np.linalg.norm(e)])
    for _ in range(8):
        e = c2 + 0.02 * rng.randn(512).astype(np.float32)
        embs.append([e / np.linalg.norm(e)])

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "7"*64, "/v.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid = conn.execute("SELECT id FROM videos").fetchone()[0]
    _seed_synthetic_detections(conn, video_id=vid, frame_dir=tmp_path, embeddings_per_frame=embs)
    conn.close()

    from reelgrep.faces import Faces, FacesError, cluster_faces
    cluster_faces(min_cluster_size=5, db_path=tmp_path / "idx.sqlite")
    faces = Faces(db_path=tmp_path / "idx.sqlite")
    clusters = faces.list_clusters()
    assert len(clusters) >= 2, f"expected >=2 clusters, got {len(clusters)}"
    faces.label_cluster(clusters[0].id, "X")
    with pytest.raises(FacesError, match="already on cluster"):
        faces.label_cluster(clusters[1].id, "X")


def test_purge_video_wipes_clusters_to_force_recompute(tmp_path):
    """After purging a video's detections, cluster tables are cleared so a recompute is required."""
    config.set_db_override(tmp_path / "idx.sqlite")
    rng = np.random.RandomState(31)
    c = rng.randn(512).astype(np.float32)
    c /= np.linalg.norm(c)
    embs = []
    for _ in range(6):
        e = c + 0.02 * rng.randn(512).astype(np.float32)
        embs.append([e / np.linalg.norm(e)])

    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "9"*64, "/v.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid = conn.execute("SELECT id FROM videos").fetchone()[0]
    _seed_synthetic_detections(conn, video_id=vid, frame_dir=tmp_path, embeddings_per_frame=embs)
    conn.close()

    from reelgrep.faces import Faces, cluster_faces
    cluster_faces(min_cluster_size=5, db_path=tmp_path / "idx.sqlite")
    faces = Faces(db_path=tmp_path / "idx.sqlite")
    assert len(faces.list_clusters()) >= 1
    n = faces.purge_video("/v.mp4")
    assert n >= 1
    # Both detections and clusters cleared.
    assert faces.detection_count() == 0
    assert faces.list_clusters() == []


def test_faces_purge_video_and_purge_all(tmp_path):
    config.set_db_override(tmp_path / "idx.sqlite")
    conn = connect(tmp_path / "idx.sqlite")
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "a"*64, "/a.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "b"*64, "/b.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    va = conn.execute("SELECT id FROM videos WHERE path='/a.mp4'").fetchone()[0]
    vb = conn.execute("SELECT id FROM videos WHERE path='/b.mp4'").fetchone()[0]
    e = np.ones(512, dtype=np.float32) / np.sqrt(512)
    _seed_synthetic_detections(conn, video_id=va, frame_dir=tmp_path, embeddings_per_frame=[[e]])
    _seed_synthetic_detections(conn, video_id=vb, frame_dir=tmp_path, embeddings_per_frame=[[e]])
    conn.close()

    from reelgrep.faces import Faces
    faces = Faces(db_path=tmp_path / "idx.sqlite")
    assert faces.detection_count() == 2
    n = faces.purge_video("/a.mp4")
    assert n == 1
    assert faces.detection_count() == 1
    faces.purge_all()
    assert faces.detection_count() == 0
