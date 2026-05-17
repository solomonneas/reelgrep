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
        assert new_version == SCHEMA_VERSION == 2
        assert current_version(conn) == 2
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
        # All 9 expected user tables present, plus schema_version.
        assert EXPECTED_TABLES.issubset(names)
        assert "schema_version" in names
        # FTS5 creates shadow tables (subtitles_fts_data/_idx/_docsize/_config); filter them
        # out and confirm the canonical count of 10 (9 user tables + schema_version).
        shadow_prefix = "subtitles_fts_"
        non_shadow = {name for name in names if not name.startswith(shadow_prefix)}
        assert len(non_shadow) == 10
    finally:
        conn.close()


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "idem.db")
    try:
        first = migrate(conn)
        second = migrate(conn)
        assert first == second == 2
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
        assert current_version(conn) == 2
    finally:
        conn.close()


def test_subtitles_accepts_whisper_source(tmp_path):
    """Fresh v2 DB must accept source='whisper' rows."""
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
        assert current_version(conn) == 2

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

        assert migrate(conn) == 2
    finally:
        conn.close()


def test_in_place_upgrade_idempotent_when_already_v2(tmp_path):
    db_path = tmp_path / "already.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
        assert migrate(conn) == 2
        assert migrate(conn) == 2
    finally:
        conn.close()
