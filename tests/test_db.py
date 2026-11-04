"""Tests for reelgrep.db: connection, migration, schema integrity, FTS5."""

from __future__ import annotations

import sqlite3

import pytest

from reelgrep.db import SCHEMA_VERSION, connect, current_version, migrate

EXPECTED_TABLES = {
    "videos",
    "subtitles",
    "subtitles_fts",
    "frames",
    "scenes",
    "person_searches",
    "person_matches",
    "export_artifacts",
    "tags",
    "face_detections",
    "face_clusters",
    "face_cluster_members",
}


def test_fresh_db_starts_at_version_zero(tmp_path):
    db_path = tmp_path / "fresh.db"
    conn = connect(db_path)
    try:
        assert current_version(conn) == 0
    finally:
        conn.close()


def test_migrate_applies_schema_and_sets_version(tmp_path):
    conn = connect(tmp_path / "migrate.db")
    try:
        new_version = migrate(conn)
        assert new_version == SCHEMA_VERSION == 4
        assert current_version(conn) == 4
    finally:
        conn.close()


def test_migrate_creates_all_expected_tables(tmp_path):
    conn = connect(tmp_path / "tables.db")
    try:
        migrate(conn)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        names = {row["name"] for row in rows}
        # All 12 expected user tables present, plus schema_version.
        assert EXPECTED_TABLES.issubset(names)
        assert "schema_version" in names
        # FTS5 creates shadow tables (subtitles_fts_data/_idx/_docsize/_config); filter them
        # out and confirm the canonical count of 13 (12 user tables + schema_version).
        shadow_prefix = "subtitles_fts_"
        non_shadow = {name for name in names if not name.startswith(shadow_prefix)}
        assert len(non_shadow) == 13
    finally:
        conn.close()


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "idem.db")
    try:
        first = migrate(conn)
        second = migrate(conn)
        assert first == second == 4
        # Sanity: schema_version still has exactly one row.
        count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_foreign_keys_pragma_enabled(tmp_path):
    conn = connect(tmp_path / "fk.db")
    try:
        value = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert value == 1
    finally:
        conn.close()


def test_foreign_keys_enforced_on_subtitles(tmp_path):
    conn = connect(tmp_path / "fk_enforce.db")
    try:
        migrate(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO subtitles (video_id, source, start_ms, end_ms, text) "
                "VALUES (?, ?, ?, ?, ?)",
                (999, "embedded", 0, 1000, "orphan"),
            )
    finally:
        conn.close()


def test_check_constraint_on_subtitles_source(tmp_path):
    conn = connect(tmp_path / "check.db")
    try:
        migrate(conn)
        # Insert a parent video so we hit the CHECK, not the FK.
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) "
            "VALUES (?, ?, ?, ?)",
            ("abc123", "/tmp/v.mp4", "2026-01-01T00:00:00Z", "{}"),
        )
        video_id = conn.execute("SELECT id FROM videos WHERE file_hash='abc123'").fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO subtitles (video_id, source, start_ms, end_ms, text) "
                "VALUES (?, ?, ?, ?, ?)",
                (video_id, "garbage", 0, 1000, "bad source"),
            )
    finally:
        conn.close()


def test_fts5_round_trip(tmp_path):
    conn = connect(tmp_path / "fts.db")
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) "
            "VALUES (?, ?, ?, ?)",
            ("hashfts", "/tmp/fts.mp4", "2026-01-01T00:00:00Z", "{}"),
        )
        video_id = conn.execute(
            "SELECT id FROM videos WHERE file_hash='hashfts'"
        ).fetchone()[0]
        conn.executemany(
            "INSERT INTO subtitles (video_id, source, start_ms, end_ms, text) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (video_id, "embedded", 0, 1000, "foo bar baz"),
                (video_id, "embedded", 1000, 2000, "the quick brown fox"),
            ],
        )
        conn.execute(
            "INSERT INTO subtitles_fts(rowid, text) SELECT id, text FROM subtitles"
        )
        rows = conn.execute(
            "SELECT rowid FROM subtitles_fts WHERE subtitles_fts MATCH 'foo'"
        ).fetchall()
        matched = {row["rowid"] for row in rows}
        sub_ids = {
            row["id"]
            for row in conn.execute(
                "SELECT id FROM subtitles WHERE text LIKE '%foo%'"
            ).fetchall()
        }
        assert matched == sub_ids
        assert len(matched) == 1

        rows_fox = conn.execute(
            "SELECT rowid FROM subtitles_fts WHERE subtitles_fts MATCH 'fox'"
        ).fetchall()
        assert len(rows_fox) == 1
    finally:
        conn.close()


def test_row_factory_returns_sqlite_row(tmp_path):
    conn = connect(tmp_path / "rowfactory.db")
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) "
            "VALUES (?, ?, ?, ?)",
            ("rowhash", "/tmp/row.mp4", "2026-01-01T00:00:00Z", "{}"),
        )
        row = conn.execute(
            "SELECT file_hash, path FROM videos WHERE file_hash='rowhash'"
        ).fetchone()
        assert isinstance(row, sqlite3.Row)
        assert row["file_hash"] == "rowhash"
        assert row["path"] == "/tmp/row.mp4"
    finally:
        conn.close()


def test_schema_version_is_two_on_fresh(tmp_path):
    db_path = tmp_path / "fresh.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        assert current_version(conn) == 4
    finally:
        conn.close()


def test_subtitles_accepts_whisper_source(tmp_path):
    """Fresh DB must accept source='whisper' rows."""
    db_path = tmp_path / "v2.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) VALUES (?, ?, ?, ?)",
            ("blake2b:abc", "/x.mp4", "2026-05-17T00:00:00", "{}"),
        )
        vid = conn.execute(
            "SELECT id FROM videos WHERE file_hash='blake2b:abc'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO subtitles "
            "(video_id, language, source, stream_index, start_ms, end_ms, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (vid, "en", "whisper", None, 0, 2000, "hello world"),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT source FROM subtitles WHERE video_id = ?", (vid,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "whisper"
    finally:
        conn.close()


def test_subtitles_still_rejects_garbage_source(tmp_path):
    db_path = tmp_path / "v2reject.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) VALUES (?, ?, ?, ?)",
            ("blake2b:bad", "/y.mp4", "2026-05-17T00:00:00", "{}"),
        )
        vid = conn.execute(
            "SELECT id FROM videos WHERE file_hash='blake2b:bad'"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO subtitles "
                "(video_id, language, source, stream_index, start_ms, end_ms, text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (vid, "en", "garbage", None, 0, 1000, "x"),
            )
    finally:
        conn.close()


def test_in_place_upgrade_from_v1_to_v2(tmp_path):
    """Pre-existing v1 DB must migrate cleanly and accept 'whisper' afterward."""
    db_path = tmp_path / "upgrade.sqlite"
    # Build a v1 DB by hand (mimicking the old schema verbatim).
    raw = sqlite3.connect(str(db_path))
    raw.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version VALUES (1);
        CREATE TABLE videos (
          id INTEGER PRIMARY KEY, file_hash TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
          duration_ms INTEGER, width INTEGER, height INTEGER, fps REAL,
          container TEXT, video_codec TEXT, audio_codec TEXT, size_bytes INTEGER,
          ingested_at TEXT NOT NULL, probe_json TEXT NOT NULL
        );
        CREATE INDEX idx_videos_path ON videos(path);
        CREATE TABLE subtitles (
          id INTEGER PRIMARY KEY,
          video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
          language TEXT,
          source TEXT NOT NULL CHECK(source IN ('embedded','sidecar')),
          stream_index INTEGER,
          start_ms INTEGER NOT NULL,
          end_ms INTEGER NOT NULL,
          text TEXT NOT NULL
        );
        CREATE INDEX idx_subtitles_video_ts ON subtitles(video_id, start_ms);
        CREATE VIRTUAL TABLE subtitles_fts USING fts5(
          text, content='subtitles', content_rowid='id', tokenize='porter unicode61'
        );
        """
    )
    raw.execute(
        "INSERT INTO videos (file_hash, path, ingested_at, probe_json) VALUES (?, ?, ?, ?)",
        ("blake2b:keep", "/old.mp4", "2026-05-17T00:00:00", "{}"),
    )
    vid = raw.execute(
        "SELECT id FROM videos WHERE file_hash='blake2b:keep'"
    ).fetchone()[0]
    cur = raw.execute(
        "INSERT INTO subtitles "
        "(video_id, language, source, stream_index, start_ms, end_ms, text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (vid, "en", "embedded", 2, 0, 1500, "preserved cue"),
    )
    raw.execute(
        "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
        (cur.lastrowid, "preserved cue"),
    )
    raw.commit()
    raw.close()

    conn = connect(db_path)
    try:
        assert current_version(conn) == 1
        migrate(conn)
        # migrate() chains forward to SCHEMA_VERSION (currently 4), so a v1 DB
        # lands at v4 in one call. We still assert that v2 was reached along the
        # way by virtue of 'whisper' being accepted afterward.
        assert current_version(conn) == 4

        rows = conn.execute(
            "SELECT source, text FROM subtitles WHERE video_id = ?", (vid,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["source"] == "embedded"
        assert rows[0]["text"] == "preserved cue"

        fts_rows = conn.execute(
            "SELECT s.text FROM subtitles_fts JOIN subtitles s ON s.id = subtitles_fts.rowid "
            "WHERE subtitles_fts MATCH 'preserved'"
        ).fetchall()
        assert len(fts_rows) == 1

        conn.execute(
            "INSERT INTO subtitles "
            "(video_id, language, source, stream_index, start_ms, end_ms, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (vid, "en", "whisper", None, 2000, 4000, "new whisper cue"),
        )
        conn.commit()
        new_rows = conn.execute(
            "SELECT source FROM subtitles WHERE video_id = ? ORDER BY start_ms", (vid,)
        ).fetchall()
        assert [r["source"] for r in new_rows] == ["embedded", "whisper"]

        assert migrate(conn) == 4
    finally:
        conn.close()


def test_in_place_upgrade_idempotent_when_already_v2(tmp_path):
    db_path = tmp_path / "already.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        assert migrate(conn) == 4
        assert migrate(conn) == 4
    finally:
        conn.close()


def test_schema_version_is_three_on_fresh(tmp_path):
    db_path = tmp_path / "fresh_v3.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        assert current_version(conn) == 4
        assert SCHEMA_VERSION == 4
    finally:
        conn.close()


def test_subtitles_accepts_aligned_source(tmp_path):
    """Fresh v3 DB must accept source='aligned' rows."""
    db_path = tmp_path / "v3_aligned.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) VALUES (?, ?, ?, ?)",
            ("blake2b:aligned", "/a.mp4", "2026-05-17T00:00:00", "{}"),
        )
        vid = conn.execute(
            "SELECT id FROM videos WHERE file_hash='blake2b:aligned'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO subtitles "
            "(video_id, language, source, stream_index, start_ms, end_ms, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (vid, "en", "aligned", None, 0, 2000, "aligned cue"),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT source FROM subtitles WHERE video_id = ?", (vid,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "aligned"
    finally:
        conn.close()


def test_subtitles_still_rejects_garbage_source_v3(tmp_path):
    db_path = tmp_path / "v3reject.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) VALUES (?, ?, ?, ?)",
            ("blake2b:bogus", "/b.mp4", "2026-05-17T00:00:00", "{}"),
        )
        vid = conn.execute(
            "SELECT id FROM videos WHERE file_hash='blake2b:bogus'"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO subtitles "
                "(video_id, language, source, stream_index, start_ms, end_ms, text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (vid, "en", "bogus", None, 0, 1000, "x"),
            )
    finally:
        conn.close()


def test_in_place_upgrade_from_v2_to_v3(tmp_path):
    """Pre-existing v2 DB must migrate cleanly and accept 'aligned' afterward."""
    db_path = tmp_path / "v2_to_v3.sqlite"
    # Build a v2 DB by hand (mimicking the v2 schema verbatim).
    raw = sqlite3.connect(str(db_path))
    raw.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version VALUES (2);
        CREATE TABLE videos (
          id INTEGER PRIMARY KEY, file_hash TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
          duration_ms INTEGER, width INTEGER, height INTEGER, fps REAL,
          container TEXT, video_codec TEXT, audio_codec TEXT, size_bytes INTEGER,
          ingested_at TEXT NOT NULL, probe_json TEXT NOT NULL
        );
        CREATE INDEX idx_videos_path ON videos(path);
        CREATE TABLE subtitles (
          id INTEGER PRIMARY KEY,
          video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
          language TEXT,
          source TEXT NOT NULL CHECK(source IN ('embedded','sidecar','whisper')),
          stream_index INTEGER,
          start_ms INTEGER NOT NULL,
          end_ms INTEGER NOT NULL,
          text TEXT NOT NULL
        );
        CREATE INDEX idx_subtitles_video_ts ON subtitles(video_id, start_ms);
        CREATE VIRTUAL TABLE subtitles_fts USING fts5(
          text, content='subtitles', content_rowid='id', tokenize='porter unicode61'
        );
        """
    )
    raw.execute(
        "INSERT INTO videos (file_hash, path, ingested_at, probe_json) VALUES (?, ?, ?, ?)",
        ("blake2b:v2keep", "/v2old.mp4", "2026-05-17T00:00:00", "{}"),
    )
    vid = raw.execute(
        "SELECT id FROM videos WHERE file_hash='blake2b:v2keep'"
    ).fetchone()[0]
    cur = raw.execute(
        "INSERT INTO subtitles "
        "(video_id, language, source, stream_index, start_ms, end_ms, text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (vid, "en", "whisper", None, 0, 1500, "preserved whisper cue"),
    )
    raw.execute(
        "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
        (cur.lastrowid, "preserved whisper cue"),
    )
    raw.commit()
    raw.close()

    conn = connect(db_path)
    try:
        assert current_version(conn) == 2
        migrate(conn)
        assert current_version(conn) == 4

        rows = conn.execute(
            "SELECT source, text FROM subtitles WHERE video_id = ?", (vid,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["source"] == "whisper"
        assert rows[0]["text"] == "preserved whisper cue"

        fts_rows = conn.execute(
            "SELECT s.text FROM subtitles_fts JOIN subtitles s ON s.id = subtitles_fts.rowid "
            "WHERE subtitles_fts MATCH 'preserved'"
        ).fetchall()
        assert len(fts_rows) == 1

        conn.execute(
            "INSERT INTO subtitles "
            "(video_id, language, source, stream_index, start_ms, end_ms, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (vid, "en", "aligned", None, 2000, 4000, "new aligned cue"),
        )
        conn.commit()
        new_rows = conn.execute(
            "SELECT source FROM subtitles WHERE video_id = ? ORDER BY start_ms", (vid,)
        ).fetchall()
        assert [r["source"] for r in new_rows] == ["whisper", "aligned"]

        assert migrate(conn) == 4
    finally:
        conn.close()


def test_in_place_upgrade_from_v1_to_v3_chain(tmp_path):
    """Pre-existing v1 DB must chain v1->v2->v3 in a single migrate() call."""
    db_path = tmp_path / "v1_to_v3.sqlite"
    raw = sqlite3.connect(str(db_path))
    raw.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version VALUES (1);
        CREATE TABLE videos (
          id INTEGER PRIMARY KEY, file_hash TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
          duration_ms INTEGER, width INTEGER, height INTEGER, fps REAL,
          container TEXT, video_codec TEXT, audio_codec TEXT, size_bytes INTEGER,
          ingested_at TEXT NOT NULL, probe_json TEXT NOT NULL
        );
        CREATE INDEX idx_videos_path ON videos(path);
        CREATE TABLE subtitles (
          id INTEGER PRIMARY KEY,
          video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
          language TEXT,
          source TEXT NOT NULL CHECK(source IN ('embedded','sidecar')),
          stream_index INTEGER,
          start_ms INTEGER NOT NULL,
          end_ms INTEGER NOT NULL,
          text TEXT NOT NULL
        );
        CREATE INDEX idx_subtitles_video_ts ON subtitles(video_id, start_ms);
        CREATE VIRTUAL TABLE subtitles_fts USING fts5(
          text, content='subtitles', content_rowid='id', tokenize='porter unicode61'
        );
        """
    )
    raw.execute(
        "INSERT INTO videos (file_hash, path, ingested_at, probe_json) VALUES (?, ?, ?, ?)",
        ("blake2b:chain", "/chain.mp4", "2026-05-17T00:00:00", "{}"),
    )
    vid = raw.execute(
        "SELECT id FROM videos WHERE file_hash='blake2b:chain'"
    ).fetchone()[0]
    cur = raw.execute(
        "INSERT INTO subtitles "
        "(video_id, language, source, stream_index, start_ms, end_ms, text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (vid, "en", "embedded", 0, 0, 1000, "chain cue"),
    )
    raw.execute(
        "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
        (cur.lastrowid, "chain cue"),
    )
    raw.commit()
    raw.close()

    conn = connect(db_path)
    try:
        assert current_version(conn) == 1
        result = migrate(conn)
        assert result == 4
        assert current_version(conn) == 4

        rows = conn.execute(
            "SELECT source, text FROM subtitles WHERE video_id = ?", (vid,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["source"] == "embedded"
        assert rows[0]["text"] == "chain cue"

        # Both 'whisper' (added v2) and 'aligned' (added v3) must now be accepted.
        conn.execute(
            "INSERT INTO subtitles "
            "(video_id, language, source, stream_index, start_ms, end_ms, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (vid, "en", "whisper", None, 1000, 2000, "chain whisper cue"),
        )
        conn.execute(
            "INSERT INTO subtitles "
            "(video_id, language, source, stream_index, start_ms, end_ms, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (vid, "en", "aligned", None, 2000, 3000, "chain aligned cue"),
        )
        conn.commit()
        sources = [
            r["source"]
            for r in conn.execute(
                "SELECT source FROM subtitles WHERE video_id = ? ORDER BY start_ms", (vid,)
            ).fetchall()
        ]
        assert sources == ["embedded", "whisper", "aligned"]
    finally:
        conn.close()


def test_in_place_upgrade_idempotent_on_v3(tmp_path):
    db_path = tmp_path / "already_v3.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        assert current_version(conn) == 4
        assert migrate(conn) == 4
        assert migrate(conn) == 4
    finally:
        conn.close()


def _build_v3_database(db_path) -> None:
    """Build a v3-shaped SQLite database by hand (mirrors pre-v4 schema.sql)."""
    raw = sqlite3.connect(str(db_path))
    raw.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version VALUES (3);

        CREATE TABLE videos (
          id INTEGER PRIMARY KEY,
          file_hash TEXT UNIQUE NOT NULL,
          path TEXT NOT NULL,
          duration_ms INTEGER,
          width INTEGER,
          height INTEGER,
          fps REAL,
          container TEXT,
          video_codec TEXT,
          audio_codec TEXT,
          size_bytes INTEGER,
          ingested_at TEXT NOT NULL,
          probe_json TEXT NOT NULL
        );
        CREATE INDEX idx_videos_path ON videos(path);

        CREATE TABLE subtitles (
          id INTEGER PRIMARY KEY,
          video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
          language TEXT,
          source TEXT NOT NULL CHECK(source IN ('embedded','sidecar','whisper','aligned')),
          stream_index INTEGER,
          start_ms INTEGER NOT NULL,
          end_ms INTEGER NOT NULL,
          text TEXT NOT NULL
        );
        CREATE INDEX idx_subtitles_video_ts ON subtitles(video_id, start_ms);
        CREATE VIRTUAL TABLE subtitles_fts USING fts5(
          text, content='subtitles', content_rowid='id', tokenize='porter unicode61'
        );

        CREATE TABLE frames (
          id INTEGER PRIMARY KEY,
          video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
          timestamp_ms INTEGER NOT NULL,
          path TEXT NOT NULL,
          sampling_strategy TEXT NOT NULL CHECK(sampling_strategy IN ('every_n','scene','manual')),
          width INTEGER,
          height INTEGER
        );
        CREATE INDEX idx_frames_video_ts ON frames(video_id, timestamp_ms);

        CREATE TABLE scenes (
          id INTEGER PRIMARY KEY,
          video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
          start_ms INTEGER NOT NULL,
          end_ms INTEGER NOT NULL,
          score REAL
        );

        CREATE TABLE person_searches (
          id INTEGER PRIMARY KEY,
          video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
          label TEXT NOT NULL,
          backend TEXT NOT NULL,
          positive_examples_json TEXT NOT NULL,
          negative_examples_json TEXT NOT NULL,
          config_json TEXT NOT NULL,
          threshold REAL NOT NULL,
          created_at TEXT NOT NULL
        );

        CREATE TABLE person_matches (
          id INTEGER PRIMARY KEY,
          search_id INTEGER NOT NULL REFERENCES person_searches(id) ON DELETE CASCADE,
          frame_id INTEGER NOT NULL REFERENCES frames(id),
          confidence REAL NOT NULL,
          bbox_json TEXT,
          reasoning TEXT
        );

        CREATE TABLE export_artifacts (
          id INTEGER PRIMARY KEY,
          video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
          kind TEXT NOT NULL CHECK(kind IN ('screenshot','clip','gif','contact_sheet')),
          path TEXT NOT NULL,
          start_ms INTEGER,
          end_ms INTEGER,
          manifest_path TEXT,
          created_at TEXT NOT NULL
        );

        CREATE TABLE tags (
          id INTEGER PRIMARY KEY,
          video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
          frame_id INTEGER REFERENCES frames(id) ON DELETE CASCADE,
          key TEXT NOT NULL,
          value TEXT
        );
        """
    )
    raw.commit()
    raw.close()


def test_migration_v3_to_v4_creates_face_tables(tmp_path):
    """A v3 database upgraded to v4 should gain the three face tables with the documented schema."""
    db_path = tmp_path / "v3.sqlite"
    _build_v3_database(db_path)

    conn = connect(db_path)
    try:
        assert current_version(conn) == 3
        migrate(conn)
        assert current_version(conn) == 4

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"face_detections", "face_clusters", "face_cluster_members"} <= tables

        cols = {row[1] for row in conn.execute("PRAGMA table_info(face_detections)")}
        assert cols == {
            "id",
            "frame_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "embedding",
            "embedding_model",
            "detected_at",
        }

        cluster_cols = {row[1] for row in conn.execute("PRAGMA table_info(face_clusters)")}
        assert cluster_cols == {
            "id",
            "label",
            "rep_detection_id",
            "size",
            "computed_at",
        }

        member_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(face_cluster_members)")
        }
        assert member_cols == {"cluster_id", "detection_id", "distance"}
    finally:
        conn.close()


def test_migration_v3_to_v4_is_idempotent(tmp_path):
    """Running migrate twice on a v4 database is a no-op."""
    db_path = tmp_path / "v4.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        assert current_version(conn) == 4
        assert migrate(conn) == 4
        # schema_version still has exactly one row.
        count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_face_detection_cascade_delete(tmp_path):
    """Deleting a frame cascades to its face_detections."""
    conn = connect(tmp_path / "fk_face.sqlite")
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) "
            "VALUES (?, ?, ?, ?)",
            ("blake2b:" + "a" * 64, "/v.mp4", "2026-05-18T00:00:00Z", "{}"),
        )
        video_id = conn.execute(
            "SELECT id FROM videos WHERE file_hash = ?",
            ("blake2b:" + "a" * 64,),
        ).fetchone()[0]
        cur = conn.execute(
            "INSERT INTO frames (video_id, timestamp_ms, path, sampling_strategy) "
            "VALUES (?, ?, ?, ?)",
            (video_id, 500, "/f.jpg", "every_n"),
        )
        frame_id = cur.lastrowid
        conn.execute(
            "INSERT INTO face_detections "
            "(frame_id, bbox_x, bbox_y, bbox_w, bbox_h, embedding, embedding_model, detected_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                frame_id,
                10,
                20,
                100,
                120,
                b"\x00" * 2048,
                "insightface_buffalo_l",
                "2026-05-18T00:00:00Z",
            ),
        )
        conn.commit()
        assert (
            conn.execute("SELECT COUNT(*) FROM face_detections").fetchone()[0] == 1
        )
        conn.execute("DELETE FROM frames WHERE id = ?", (frame_id,))
        conn.commit()
        assert (
            conn.execute("SELECT COUNT(*) FROM face_detections").fetchone()[0] == 0
        )
    finally:
        conn.close()
