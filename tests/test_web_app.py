"""Tests for the reelgrep Starlette web backend."""

from __future__ import annotations

import pytest

from reelgrep import __version__, config, db
from reelgrep.web.app import create_app


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path / "home"))
    config.reset_settings()
    yield
    config.reset_settings()


@pytest.fixture
def seeded_db(tmp_path):
    """Seed a fully populated index DB and return (db_path, frame_paths)."""
    s = config.get_settings()
    config.ensure_dirs(s)

    # Real on-disk frame files so /file happy-path can stream them.
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_paths: list[str] = []
    for i in range(3):
        fp = frame_dir / f"f{i}.jpg"
        # Tiny JPEG SOI + EOI markers - real bytes are enough for FileResponse.
        fp.write_bytes(b"\xff\xd8\xff\xd9")
        frame_paths.append(str(fp))

    export_path = tmp_path / "exports" / "x.mp4"
    export_path.parent.mkdir()
    export_path.write_bytes(b"\x00" * 16)
    manifest_path = tmp_path / "exports" / "x.mp4.manifest.json"
    manifest_path.write_text("{}")

    conn = db.connect(s.db_path)
    db.migrate(conn)

    conn.execute(
        "INSERT INTO videos (file_hash, path, duration_ms, width, height, fps, container, "
        "video_codec, audio_codec, size_bytes, ingested_at, probe_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "blake2b:aaa",
            "/v/lecture.mp4",
            600000,
            1920,
            1080,
            30.0,
            "mp4",
            "h264",
            "aac",
            12345,
            "2026-05-17T12:00:00",
            "{}",
        ),
    )
    a_id = conn.execute(
        "SELECT id FROM videos WHERE file_hash='blake2b:aaa'"
    ).fetchone()[0]

    for i, fp in enumerate(frame_paths):
        conn.execute(
            "INSERT INTO frames (video_id, timestamp_ms, path, sampling_strategy, "
            "width, height) VALUES (?,?,?,?,?,?)",
            (a_id, i * 5000, fp, "every_n", 320, 180),
        )

    cues = [
        (a_id, "en", "whisper", None, 1000, 4000, "welcome to networking"),
        (a_id, "en", "whisper", None, 5000, 8000, "pods talk over CNI"),
    ]
    for c in cues:
        cur = conn.execute(
            "INSERT INTO subtitles (video_id, language, source, stream_index, "
            "start_ms, end_ms, text) VALUES (?,?,?,?,?,?,?)",
            c,
        )
        conn.execute(
            "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
            (cur.lastrowid, c[6]),
        )

    cur = conn.execute(
        "INSERT INTO person_searches (video_id, label, backend, positive_examples_json, "
        "negative_examples_json, config_json, threshold, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            a_id,
            "speaker",
            "face_embed",
            '["/p1.jpg"]',
            "[]",
            '{"model": "buffalo_l"}',
            0.3,
            "2026-05-17T12:30:00",
        ),
    )
    search_id = cur.lastrowid
    fid = conn.execute(
        "SELECT id FROM frames WHERE video_id = ? ORDER BY timestamp_ms ASC LIMIT 1",
        (a_id,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO person_matches (search_id, frame_id, confidence, bbox_json, "
        "reasoning) VALUES (?,?,?,?,?)",
        (search_id, fid, 0.82, "[10,20,30,40]", "cosine 0.87 to positive"),
    )

    conn.execute(
        "INSERT INTO export_artifacts (video_id, kind, path, start_ms, end_ms, "
        "manifest_path, created_at) VALUES (?,?,?,?,?,?,?)",
        (
            a_id,
            "clip",
            str(export_path),
            1000,
            4000,
            str(manifest_path),
            "2026-05-17T13:00:00",
        ),
    )

    # Second video (sparse, newer ingested_at so it sorts first).
    conn.execute(
        "INSERT INTO videos (file_hash, path, ingested_at, probe_json) VALUES (?,?,?,?)",
        ("blake2b:bbb", "/v/other.mp4", "2026-05-17T13:00:00", "{}"),
    )

    conn.commit()
    conn.close()
    return s.db_path, frame_paths, str(export_path)


@pytest.fixture
def client(seeded_db):
    from starlette.testclient import TestClient

    db_path, _, _ = seeded_db
    app = create_app(db_path=db_path)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_returns_version(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "version": __version__}


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


def test_list_videos_orders_by_ingested_at_desc(client):
    r = client.get("/api/videos")
    assert r.status_code == 200
    videos = r.json()["videos"]
    assert len(videos) == 2
    # bbb has ingested_at 13:00 (newer), aaa has 12:00
    assert videos[0]["file_hash"] == "blake2b:bbb"
    assert videos[1]["file_hash"] == "blake2b:aaa"
    aaa = videos[1]
    assert aaa["frames_count"] == 3
    assert aaa["subtitle_cues_count"] == 2
    assert aaa["exports_count"] == 1
    assert aaa["person_searches_count"] == 1
    assert aaa["width"] == 1920
    assert aaa["height"] == 1080


def test_get_video_detail(client):
    r = client.get("/api/videos/blake2b:aaa")
    assert r.status_code == 200
    v = r.json()
    assert v["file_hash"] == "blake2b:aaa"
    assert v["container"] == "mp4"
    assert v["video_codec"] == "h264"
    assert v["audio_codec"] == "aac"
    assert v["width"] == 1920
    assert v["height"] == 1080
    assert v["fps"] == 30.0
    assert v["size_bytes"] == 12345
    assert v["exports_by_kind"] == {"clip": 1}


def test_get_video_not_found(client):
    r = client.get("/api/videos/nonexistent")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def test_list_video_frames_orders_by_timestamp(client):
    r = client.get("/api/videos/blake2b:aaa/frames")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    frames = body["frames"]
    assert len(frames) == 3
    assert [f["timestamp_ms"] for f in frames] == [0, 5000, 10000]
    assert frames[0]["sampling_strategy"] == "every_n"


def test_list_video_frames_pagination(client):
    r = client.get("/api/videos/blake2b:aaa/frames?limit=1&offset=1")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert len(body["frames"]) == 1
    assert body["frames"][0]["timestamp_ms"] == 5000


def test_list_video_frames_unknown_video(client):
    r = client.get("/api/videos/nope/frames")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------


def test_list_subtitles_all(client):
    r = client.get("/api/videos/blake2b:aaa/subtitles")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["cues"]) == 2
    assert [c["start_ms"] for c in body["cues"]] == [1000, 5000]
    assert body["cues"][0]["source"] == "whisper"


def test_list_subtitles_fts_match_pods(client):
    r = client.get("/api/videos/blake2b:aaa/subtitles?q=pods")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert len(body["cues"]) == 1
    assert "pods" in body["cues"][0]["text"]


def test_list_subtitles_fts_match_networking(client):
    r = client.get("/api/videos/blake2b:aaa/subtitles?q=networking")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["cues"][0]["text"] == "welcome to networking"


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------


def test_list_searches(client):
    r = client.get("/api/searches")
    assert r.status_code == 200
    searches = r.json()["searches"]
    assert len(searches) == 1
    s = searches[0]
    assert s["video_hash"] == "blake2b:aaa"
    assert s["match_count"] == 1
    assert s["label"] == "speaker"
    assert s["backend"] == "face_embed"
    assert s["threshold"] == 0.3


def test_get_search_detail(client):
    listed = client.get("/api/searches").json()["searches"]
    search_id = listed[0]["id"]
    r = client.get(f"/api/searches/{search_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["video_hash"] == "blake2b:aaa"
    assert body["positive_examples"] == ["/p1.jpg"]
    assert body["negative_examples"] == []
    assert body["config"] == {"model": "buffalo_l"}
    assert len(body["matches"]) == 1
    m = body["matches"][0]
    assert m["bbox"] == [10, 20, 30, 40]
    assert m["reasoning"] == "cosine 0.87 to positive"
    assert m["confidence"] == 0.82
    assert m["frame_timestamp_ms"] == 0
    assert m["frame_path"].endswith("f0.jpg")


def test_get_search_not_found(client):
    r = client.get("/api/searches/99999")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_list_exports_all(client):
    r = client.get("/api/exports")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert len(body["exports"]) == 1
    e = body["exports"][0]
    assert e["kind"] == "clip"
    assert e["video_hash"] == "blake2b:aaa"
    assert e["start_ms"] == 1000
    assert e["end_ms"] == 4000
    assert e["manifest_path"].endswith(".manifest.json")


def test_list_exports_filter_kind_clip(client):
    r = client.get("/api/exports?kind=clip")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["exports"][0]["kind"] == "clip"


def test_list_exports_filter_kind_gif_empty(client):
    r = client.get("/api/exports?kind=gif")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["exports"] == []


def test_list_exports_invalid_kind(client):
    r = client.get("/api/exports?kind=bogus")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /file
# ---------------------------------------------------------------------------


def test_file_missing_query_param(client):
    r = client.get("/file")
    assert r.status_code == 400


def test_file_rejects_path_not_in_index(client):
    r = client.get("/file", params={"path": "/etc/passwd"})
    assert r.status_code == 403


def test_file_404_when_referenced_but_missing(client, seeded_db):
    db_path, frame_paths, _ = seeded_db
    # Delete the on-disk file but leave the DB row.
    import os

    os.unlink(frame_paths[0])
    r = client.get("/file", params={"path": frame_paths[0]})
    assert r.status_code == 404


def test_file_serves_frame_happy_path(client, seeded_db):
    _, frame_paths, _ = seeded_db
    r = client.get("/file", params={"path": frame_paths[1]})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image")
    assert r.content.startswith(b"\xff\xd8")


def test_file_serves_export_artifact(client, seeded_db):
    _, _, export_path = seeded_db
    r = client.get("/file", params={"path": export_path})
    assert r.status_code == 200
    assert r.content == b"\x00" * 16


def test_file_serves_manifest_path(client, seeded_db):
    _, _, export_path = seeded_db
    manifest_path = export_path + ".manifest.json"
    r = client.get("/file", params={"path": manifest_path})
    assert r.status_code == 200
    assert r.content == b"{}"


# ---------------------------------------------------------------------------
# Faces clusters
# ---------------------------------------------------------------------------


def _seed_face_clusters(db_path):
    """Insert face_detections + face_clusters + members into the seeded DB.

    Returns the cluster id of the labeled cluster.
    """
    import numpy as np

    from reelgrep.db import connect, migrate

    conn = connect(db_path)
    migrate(conn)
    try:
        # Grab the three frames that the base fixture already created on
        # video blake2b:aaa.
        frame_rows = conn.execute(
            "SELECT id FROM frames "
            "WHERE video_id = (SELECT id FROM videos WHERE file_hash='blake2b:aaa') "
            "ORDER BY timestamp_ms ASC"
        ).fetchall()
        frame_ids = [r[0] for r in frame_rows]
        assert len(frame_ids) >= 3, "expected seeded fixture to provide 3 frames"

        # Insert three face_detections, one per frame, all with a known
        # embedding so the cluster representative is real.
        emb = np.zeros(512, dtype=np.float32)
        emb[0] = 1.0
        emb_blob = emb.tobytes()
        det_ids: list[int] = []
        for i, fid in enumerate(frame_ids):
            cur = conn.execute(
                "INSERT INTO face_detections(frame_id, bbox_x, bbox_y, bbox_w, bbox_h, "
                "embedding, embedding_model, detected_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    fid,
                    10 + i,
                    20 + i,
                    40,
                    60,
                    emb_blob,
                    "insightface_buffalo_l",
                    "2026-05-18T00:00:00",
                ),
            )
            det_ids.append(cur.lastrowid)

        cur = conn.execute(
            "INSERT INTO face_clusters(label, rep_detection_id, size, computed_at) "
            "VALUES (?,?,?,?)",
            ("Alice", det_ids[0], len(det_ids), "2026-05-18T00:01:00"),
        )
        cluster_id = cur.lastrowid
        for i, det_id in enumerate(det_ids):
            conn.execute(
                "INSERT INTO face_cluster_members(cluster_id, detection_id, distance) "
                "VALUES (?,?,?)",
                (cluster_id, det_id, float(i) * 0.01),
            )
        conn.commit()
        return cluster_id
    finally:
        conn.close()


def test_faces_clusters_endpoint(client, seeded_db):
    """GET /api/faces/clusters returns ranked clusters."""
    db_path, _, _ = seeded_db
    cluster_id = _seed_face_clusters(db_path)

    r = client.get("/api/faces/clusters")
    assert r.status_code == 200
    body = r.json()
    assert "clusters" in body
    assert len(body["clusters"]) >= 1
    sample = next(c for c in body["clusters"] if c["id"] == cluster_id)
    assert {"id", "label", "size", "rep_detection_id", "computed_at"} <= set(
        sample.keys()
    )
    assert sample["label"] == "Alice"
    assert sample["size"] == 3


def test_faces_clusters_labeled_only_filter(client, seeded_db):
    """labeled_only=true filters out unlabeled clusters."""
    db_path, _, _ = seeded_db
    _seed_face_clusters(db_path)
    # Insert one extra unlabeled cluster.
    from reelgrep.db import connect, migrate

    conn = connect(db_path)
    migrate(conn)
    try:
        conn.execute(
            "INSERT INTO face_clusters(label, size, computed_at) VALUES (?,?,?)",
            (None, 0, "2026-05-18T00:02:00"),
        )
        conn.commit()
    finally:
        conn.close()

    r_all = client.get("/api/faces/clusters")
    r_labeled = client.get("/api/faces/clusters?labeled_only=true")
    assert r_all.status_code == 200
    assert r_labeled.status_code == 200
    assert len(r_labeled.json()["clusters"]) < len(r_all.json()["clusters"])
    assert all(c["label"] is not None for c in r_labeled.json()["clusters"])


def test_faces_cluster_detail_endpoint(client, seeded_db):
    """GET /api/faces/clusters/{id} returns cluster + members."""
    db_path, _, _ = seeded_db
    cluster_id = _seed_face_clusters(db_path)

    r = client.get(f"/api/faces/clusters/{cluster_id}")
    assert r.status_code == 200
    body = r.json()
    assert "cluster" in body and "members" in body
    assert body["cluster"]["id"] == cluster_id
    assert body["cluster"]["label"] == "Alice"
    assert len(body["members"]) == 3
    m = body["members"][0]
    assert {
        "id",
        "video_id",
        "video_path",
        "frame_id",
        "frame_path",
        "timestamp_ms",
        "bbox",
    } <= set(m.keys())
    assert m["bbox"] == [10, 20, 40, 60]
    assert m["video_path"] == "/v/lecture.mp4"


def test_faces_cluster_detail_404_when_missing(client, seeded_db):
    """GET /api/faces/clusters/{id} returns 404 for nonexistent cluster."""
    db_path, _, _ = seeded_db
    _seed_face_clusters(db_path)
    r = client.get("/api/faces/clusters/99999")
    assert r.status_code == 404


def test_faces_cluster_label_patch_endpoint(client, seeded_db):
    """PATCH /api/faces/clusters/{id} updates the label."""
    db_path, _, _ = seeded_db
    cluster_id = _seed_face_clusters(db_path)

    r = client.patch(
        f"/api/faces/clusters/{cluster_id}",
        json={"label": "Speaker A"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    listing = client.get("/api/faces/clusters").json()["clusters"]
    by_id = next(c for c in listing if c["id"] == cluster_id)
    assert by_id["label"] == "Speaker A"


def test_faces_cluster_label_patch_clears_when_null(client, seeded_db):
    """PATCH with label=None clears the label."""
    db_path, _, _ = seeded_db
    cluster_id = _seed_face_clusters(db_path)

    r = client.patch(f"/api/faces/clusters/{cluster_id}", json={"label": None})
    assert r.status_code == 200
    listing = client.get("/api/faces/clusters").json()["clusters"]
    by_id = next(c for c in listing if c["id"] == cluster_id)
    assert by_id["label"] is None


def test_faces_cluster_label_patch_404_when_missing(client, seeded_db):
    """PATCH on missing cluster returns 404."""
    db_path, _, _ = seeded_db
    _seed_face_clusters(db_path)
    r = client.patch("/api/faces/clusters/99999", json={"label": "x"})
    assert r.status_code == 404


def test_faces_cluster_label_patch_409_on_collision(client, seeded_db):
    """PATCH with a label already on another cluster returns 400."""
    db_path, _, _ = seeded_db
    cluster_id = _seed_face_clusters(db_path)
    # Insert a second cluster with label "Bob".
    from reelgrep.db import connect, migrate

    conn = connect(db_path)
    migrate(conn)
    try:
        conn.execute(
            "INSERT INTO face_clusters(label, size, computed_at) VALUES (?,?,?)",
            ("Bob", 0, "2026-05-18T00:02:00"),
        )
        conn.commit()
    finally:
        conn.close()

    # Try to relabel the Alice cluster as Bob — should collide.
    r = client.patch(
        f"/api/faces/clusters/{cluster_id}", json={"label": "Bob"}
    )
    assert r.status_code == 400
