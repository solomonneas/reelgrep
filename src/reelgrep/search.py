"""Library-level read API for the reelgrep index.

This module exposes :class:`Search`, the supported way for library
callers to query the SQLite index. Existing CLI commands and the web
backend delegate to :class:`Search` for shared read patterns; the
class itself is independent of Click, Starlette, and any other
framework concerns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from reelgrep.config import get_settings
from reelgrep.db import connect, migrate

__all__ = ["DetectionHit", "FrameRow", "Search", "SubtitleHit"]


@dataclass(frozen=True)
class SubtitleHit:
    """A subtitle cue returned from an FTS5 search."""

    id: int
    video_id: int
    start_ms: int
    end_ms: int
    text: str
    language: str | None
    source: str


@dataclass(frozen=True)
class DetectionHit:
    """A recorded person-detection search (one row per stored search definition)."""

    id: int
    video_id: int
    video_hash: str
    label: str
    model: str
    threshold: float
    created_at: str
    match_count: int


@dataclass(frozen=True)
class FrameRow:
    """A sampled frame row from the index."""

    id: int
    video_id: int
    timestamp_ms: int
    path: str
    sampling_strategy: str


class Search:
    """Read-only query API for the reelgrep index.

    Each method opens a short-lived SQLite connection scoped to one
    call. ``Search`` never mutates rows and never alters the
    process-level db_path override; passing ``db_path`` to the
    constructor only affects connections made by this instance.
    """

    def __init__(self, *, db_path: str | os.PathLike[str] | None = None) -> None:
        """Bind this Search to a specific database file.

        Parameters
        ----------
        db_path:
            Path to the index SQLite file. When ``None``, the path
            resolved by :func:`reelgrep.config.get_settings` is used
            (which honours ``REELGREP_DB`` / ``REELGREP_HOME`` and any
            active process override).
        """
        if db_path is None:
            self._db_path: Path = get_settings().db_path
        else:
            self._db_path = Path(db_path).expanduser().resolve()

    @property
    def db_path(self) -> Path:
        """The database file this Search reads from."""
        return self._db_path

    def subtitles(
        self,
        query: str,
        *,
        limit: int = 50,
        video_id: int | None = None,
    ) -> list[SubtitleHit]:
        """Run an FTS5 search across the subtitles table.

        Parameters
        ----------
        query:
            FTS5 ``MATCH`` expression (e.g. ``"hello"``, ``"pod*"``,
            ``"kubernetes networking"``). Tokenisation follows the
            schema's ``porter unicode61`` configuration.
        limit:
            Maximum number of rows to return.
        video_id:
            Restrict the search to a single video's cues when given.
        """
        if limit < 1:
            return []
        conn = connect(self._db_path)
        try:
            migrate(conn)
            if video_id is None:
                cursor = conn.execute(
                    """
                    SELECT s.id, s.video_id, s.start_ms, s.end_ms,
                           s.text, s.language, s.source
                    FROM subtitles_fts
                    JOIN subtitles AS s ON s.id = subtitles_fts.rowid
                    WHERE subtitles_fts MATCH ?
                    ORDER BY s.video_id, s.start_ms
                    LIMIT ?
                    """,
                    (query, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT s.id, s.video_id, s.start_ms, s.end_ms,
                           s.text, s.language, s.source
                    FROM subtitles_fts
                    JOIN subtitles AS s ON s.id = subtitles_fts.rowid
                    WHERE subtitles_fts MATCH ?
                      AND s.video_id = ?
                    ORDER BY s.start_ms
                    LIMIT ?
                    """,
                    (query, video_id, limit),
                )
            return [
                SubtitleHit(
                    id=int(row["id"]),
                    video_id=int(row["video_id"]),
                    start_ms=int(row["start_ms"]),
                    end_ms=int(row["end_ms"]),
                    text=row["text"],
                    language=row["language"],
                    source=row["source"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def detections(
        self,
        *,
        model: str | None = None,
        label: str | None = None,
        limit: int = 50,
        video_id: int | None = None,
    ) -> list[DetectionHit]:
        """Return recorded person-detection searches, optionally filtered.

        Parameters
        ----------
        model:
            Backend identifier the search was run with (e.g.
            ``"face_embed"``, ``"ollama_vision"``). Stored as
            ``person_searches.backend``.
        label:
            Caller-provided name assigned to the search.
        video_id:
            Restrict to searches recorded against a single video.
        limit:
            Maximum number of rows to return.
        """
        if limit < 1:
            return []
        clauses: list[str] = []
        params: list[object] = []
        if model is not None:
            clauses.append("ps.backend = ?")
            params.append(model)
        if label is not None:
            clauses.append("ps.label = ?")
            params.append(label)
        if video_id is not None:
            clauses.append("ps.video_id = ?")
            params.append(video_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        conn = connect(self._db_path)
        try:
            migrate(conn)
            cursor = conn.execute(
                f"""
                SELECT ps.id, ps.video_id, v.file_hash AS video_hash,
                       ps.label, ps.backend, ps.threshold, ps.created_at,
                       (SELECT COUNT(*) FROM person_matches pm
                          WHERE pm.search_id = ps.id) AS match_count
                FROM person_searches ps
                JOIN videos v ON v.id = ps.video_id
                {where}
                ORDER BY ps.created_at DESC
                LIMIT ?
                """,
                params,
            )
            return [
                DetectionHit(
                    id=int(row["id"]),
                    video_id=int(row["video_id"]),
                    video_hash=row["video_hash"],
                    label=row["label"],
                    model=row["backend"],
                    threshold=float(row["threshold"]),
                    created_at=row["created_at"],
                    match_count=int(row["match_count"]),
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def frames_at(
        self,
        *,
        video_id: int,
        ts_start_ms: int | None = None,
        ts_end_ms: int | None = None,
        limit: int = 200,
    ) -> list[FrameRow]:
        """Return frames for ``video_id``, optionally within a timestamp window.

        Parameters
        ----------
        video_id:
            The ``videos.id`` whose frames to list.
        ts_start_ms:
            Lower bound (inclusive) on ``timestamp_ms``. ``None`` means
            no lower bound.
        ts_end_ms:
            Upper bound (inclusive) on ``timestamp_ms``. ``None`` means
            no upper bound.
        limit:
            Maximum number of frames to return.
        """
        if limit < 1:
            return []
        clauses = ["video_id = ?"]
        params: list[object] = [video_id]
        if ts_start_ms is not None:
            clauses.append("timestamp_ms >= ?")
            params.append(ts_start_ms)
        if ts_end_ms is not None:
            clauses.append("timestamp_ms <= ?")
            params.append(ts_end_ms)
        params.append(limit)

        conn = connect(self._db_path)
        try:
            migrate(conn)
            cursor = conn.execute(
                f"""
                SELECT id, video_id, timestamp_ms, path, sampling_strategy
                FROM frames
                WHERE {' AND '.join(clauses)}
                ORDER BY timestamp_ms ASC
                LIMIT ?
                """,
                params,
            )
            return [
                FrameRow(
                    id=int(row["id"]),
                    video_id=int(row["video_id"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                    path=row["path"],
                    sampling_strategy=row["sampling_strategy"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()
