"""SQLite connection helpers and schema migration for reelgrep."""

from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION", "connect", "current_version", "migrate"]


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the reelgrep SQLite database, creating the parent dir if missing."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def current_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version, or 0 if the schema_version table is absent."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    result = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    if result is None or result[0] is None:
        return 0
    return int(result[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply the bundled schema if the database is older than SCHEMA_VERSION."""
    version = current_version(conn)
    if version >= SCHEMA_VERSION:
        return version
    sql = files("reelgrep").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()
    return current_version(conn)
