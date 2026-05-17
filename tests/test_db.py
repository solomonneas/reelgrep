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
        assert new_version == SCHEMA_VERSION == 1
        assert current_version(conn) == 1
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
        assert first == second == 1
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
