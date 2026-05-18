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
