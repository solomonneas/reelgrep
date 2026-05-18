"""SQLite connection + schema migrations for reelgrep."""

from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path

__all__ = ["SCHEMA_VERSION", "connect", "current_version", "migrate"]

SCHEMA_VERSION = 3

# Forward migrations: each key is the target version; the value is the
# .sql resource name applied to upgrade FROM (key - 1) TO key.
_FORWARD_MIGRATIONS: dict[int, str] = {
    2: "migration_v1_to_v2.sql",
    3: "migration_v2_to_v3.sql",
}


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the index database with foreign keys, WAL, and Row factory."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def current_version(conn: sqlite3.Connection) -> int:
    """Return the highest applied schema version, or 0 if none."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if not row:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the database forward to SCHEMA_VERSION; idempotent."""
    v = current_version(conn)
    if v >= SCHEMA_VERSION:
        return v
    if v == 0:
        sql = files("reelgrep").joinpath("schema.sql").read_text(encoding="utf-8")
        conn.executescript(sql)
        conn.commit()
        return current_version(conn)
    # Incremental forward steps from the current version up to SCHEMA_VERSION.
    for target in sorted(_FORWARD_MIGRATIONS):
        if target <= v:
            continue
        if target > SCHEMA_VERSION:
            break
        script = files("reelgrep").joinpath(_FORWARD_MIGRATIONS[target]).read_text(
            encoding="utf-8"
        )
        conn.executescript(script)
        conn.commit()
    return current_version(conn)
