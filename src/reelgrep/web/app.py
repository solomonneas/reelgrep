"""Starlette app exposing the reelgrep SQLite index as JSON + safe file serving."""

from __future__ import annotations

import json
import mimetypes
import sqlite3
from pathlib import Path
from typing import Any

_VALID_EXPORT_KINDS = {"screenshot", "clip", "gif", "contact_sheet"}
_DEFAULT_LIMIT = 200
_MAX_LIMIT = 1000


def create_app(db_path: Path | None = None):
    """Build the Starlette application bound to the given SQLite index."""
    try:
        from starlette.applications import Starlette
        from starlette.exceptions import HTTPException
        from starlette.middleware import Middleware
        from starlette.middleware.cors import CORSMiddleware
        from starlette.requests import Request
        from starlette.responses import FileResponse, JSONResponse, Response
        from starlette.routing import Route
    except ImportError as exc:
        raise RuntimeError(
            "reelgrep web requires the [web] extra: pip install reelgrep[web]"
        ) from exc

    from reelgrep import __version__
    from reelgrep.config import ensure_dirs
    from reelgrep.db import connect, migrate

    if db_path is None:
        settings = ensure_dirs()
        resolved_db_path = settings.db_path
    else:
        resolved_db_path = Path(db_path)

    def _conn() -> sqlite3.Connection:
        c = connect(resolved_db_path)
        migrate(c)
        return c

    def _pagination(request: Request) -> tuple[int, int]:
        try:
            limit = int(request.query_params.get("limit", _DEFAULT_LIMIT))
        except ValueError as exc:
            raise HTTPException(400, "limit must be an integer") from exc
        try:
            offset = int(request.query_params.get("offset", 0))
        except ValueError as exc:
            raise HTTPException(400, "offset must be an integer") from exc
        if limit < 1:
            limit = 1
        if limit > _MAX_LIMIT:
            limit = _MAX_LIMIT
        if offset < 0:
            offset = 0
        return limit, offset

    def _video_by_hash(conn: sqlite3.Connection, file_hash: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT v.*, "
            " (SELECT COUNT(*) FROM frames f WHERE f.video_id = v.id) AS frames_count, "
            " (SELECT COUNT(*) FROM subtitles s WHERE s.video_id = v.id) AS subtitle_cues_count, "
            " (SELECT COUNT(*) FROM export_artifacts e WHERE e.video_id = v.id) AS exports_count, "
            " (SELECT COUNT(*) FROM person_searches p WHERE p.video_id = v.id) "
            "  AS person_searches_count "
            "FROM videos v WHERE v.file_hash = ?",
            (file_hash,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "video not found")
        return row

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "version": __version__})

    async def list_videos(request: Request) -> JSONResponse:
        conn = _conn()
        try:
            rows = conn.execute(
                "SELECT v.*, "
                " (SELECT COUNT(*) FROM frames f WHERE f.video_id = v.id) AS frames_count, "
                " (SELECT COUNT(*) FROM subtitles s WHERE s.video_id = v.id) "
                "  AS subtitle_cues_count, "
                " (SELECT COUNT(*) FROM export_artifacts e WHERE e.video_id = v.id) "
                "  AS exports_count, "
                " (SELECT COUNT(*) FROM person_searches p WHERE p.video_id = v.id) "
                "  AS person_searches_count "
                "FROM videos v ORDER BY v.ingested_at DESC"
            ).fetchall()
            return JSONResponse({"videos": [_row_to_video_summary(r) for r in rows]})
        finally:
            conn.close()

    async def get_video(request: Request) -> JSONResponse:
        file_hash = request.path_params["file_hash"]
        conn = _conn()
        try:
            row = _video_by_hash(conn, file_hash)
            kinds = conn.execute(
                "SELECT kind, COUNT(*) AS n FROM export_artifacts "
                "WHERE video_id = ? GROUP BY kind",
                (row["id"],),
            ).fetchall()
            exports_by_kind = {k["kind"]: int(k["n"]) for k in kinds}
            return JSONResponse(_row_to_video_detail(row, exports_by_kind))
        finally:
            conn.close()

    async def list_video_frames(request: Request) -> JSONResponse:
        file_hash = request.path_params["file_hash"]
        limit, offset = _pagination(request)
        conn = _conn()
        try:
            row = _video_by_hash(conn, file_hash)
            video_id = row["id"]
            total = conn.execute(
                "SELECT COUNT(*) FROM frames WHERE video_id = ?", (video_id,)
            ).fetchone()[0]
            frames = conn.execute(
                "SELECT id, timestamp_ms, path, sampling_strategy FROM frames "
                "WHERE video_id = ? ORDER BY timestamp_ms ASC LIMIT ? OFFSET ?",
                (video_id, limit, offset),
            ).fetchall()
            return JSONResponse(
                {
                    "frames": [_row_to_frame(f) for f in frames],
                    "total": int(total),
                }
            )
        finally:
            conn.close()

    async def list_video_subtitles(request: Request) -> JSONResponse:
        file_hash = request.path_params["file_hash"]
        limit, _offset = _pagination(request)
        q = request.query_params.get("q")
        conn = _conn()
        try:
            row = _video_by_hash(conn, file_hash)
            video_id = row["id"]
            if q:
                total = conn.execute(
                    "SELECT COUNT(*) FROM subtitles s "
                    "JOIN subtitles_fts fts ON fts.rowid = s.id "
                    "WHERE s.video_id = ? AND subtitles_fts MATCH ?",
                    (video_id, q),
                ).fetchone()[0]
                cues = conn.execute(
                    "SELECT s.id, s.start_ms, s.end_ms, s.text, s.language, s.source "
                    "FROM subtitles s "
                    "JOIN subtitles_fts fts ON fts.rowid = s.id "
                    "WHERE s.video_id = ? AND subtitles_fts MATCH ? "
                    "ORDER BY s.start_ms ASC LIMIT ?",
                    (video_id, q, limit),
                ).fetchall()
            else:
                total = conn.execute(
                    "SELECT COUNT(*) FROM subtitles WHERE video_id = ?",
                    (video_id,),
                ).fetchone()[0]
                cues = conn.execute(
                    "SELECT id, start_ms, end_ms, text, language, source FROM subtitles "
                    "WHERE video_id = ? ORDER BY start_ms ASC LIMIT ?",
                    (video_id, limit),
                ).fetchall()
            return JSONResponse(
                {
                    "cues": [_row_to_cue(c) for c in cues],
                    "total": int(total),
                }
            )
        finally:
            conn.close()

    async def list_searches(request: Request) -> JSONResponse:
        conn = _conn()
        try:
            rows = conn.execute(
                "SELECT ps.id, ps.video_id, v.file_hash AS video_hash, ps.label, ps.backend, "
                "       ps.threshold, ps.created_at, "
                "       (SELECT COUNT(*) FROM person_matches pm WHERE pm.search_id = ps.id) "
                "         AS match_count "
                "FROM person_searches ps "
                "JOIN videos v ON v.id = ps.video_id "
                "ORDER BY ps.created_at DESC"
            ).fetchall()
            return JSONResponse(
                {"searches": [_row_to_search_summary(r) for r in rows]}
            )
        finally:
            conn.close()

    async def get_search(request: Request) -> JSONResponse:
        search_id = request.path_params["search_id"]
        conn = _conn()
        try:
            row = conn.execute(
                "SELECT ps.id, ps.video_id, v.file_hash AS video_hash, ps.label, ps.backend, "
                "       ps.threshold, ps.created_at, "
                "       ps.positive_examples_json, ps.negative_examples_json, ps.config_json, "
                "       (SELECT COUNT(*) FROM person_matches pm WHERE pm.search_id = ps.id) "
                "         AS match_count "
                "FROM person_searches ps "
                "JOIN videos v ON v.id = ps.video_id "
                "WHERE ps.id = ?",
                (search_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(404, "search not found")
            matches = conn.execute(
                "SELECT pm.id, pm.frame_id, pm.confidence, pm.bbox_json, pm.reasoning, "
                "       f.path AS frame_path, f.timestamp_ms AS frame_timestamp_ms "
                "FROM person_matches pm "
                "JOIN frames f ON f.id = pm.frame_id "
                "WHERE pm.search_id = ? "
                "ORDER BY pm.confidence DESC",
                (search_id,),
            ).fetchall()
            return JSONResponse(_row_to_search_detail(row, matches))
        finally:
            conn.close()

    async def list_exports(request: Request) -> JSONResponse:
        limit, offset = _pagination(request)
        kind = request.query_params.get("kind")
        if kind is not None and kind not in _VALID_EXPORT_KINDS:
            raise HTTPException(400, f"invalid kind: {kind}")
        conn = _conn()
        try:
            if kind is None:
                total = conn.execute(
                    "SELECT COUNT(*) FROM export_artifacts"
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT e.id, v.file_hash AS video_hash, e.kind, e.path, "
                    "       e.start_ms, e.end_ms, e.manifest_path, e.created_at "
                    "FROM export_artifacts e "
                    "JOIN videos v ON v.id = e.video_id "
                    "ORDER BY e.created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                total = conn.execute(
                    "SELECT COUNT(*) FROM export_artifacts WHERE kind = ?",
                    (kind,),
                ).fetchone()[0]
                rows = conn.execute(
                    "SELECT e.id, v.file_hash AS video_hash, e.kind, e.path, "
                    "       e.start_ms, e.end_ms, e.manifest_path, e.created_at "
                    "FROM export_artifacts e "
                    "JOIN videos v ON v.id = e.video_id "
                    "WHERE e.kind = ? "
                    "ORDER BY e.created_at DESC LIMIT ? OFFSET ?",
                    (kind, limit, offset),
                ).fetchall()
            return JSONResponse(
                {
                    "exports": [_row_to_export(r) for r in rows],
                    "total": int(total),
                }
            )
        finally:
            conn.close()

    async def serve_file(request: Request) -> Response:
        raw = request.query_params.get("path", "")
        if not raw:
            raise HTTPException(400, "missing ?path")
        target = str(Path(raw).expanduser())
        # Also try the resolved form (handles symlink-canonicalization mismatches).
        try:
            resolved = str(Path(raw).expanduser().resolve())
        except OSError:
            resolved = target
        conn = _conn()
        try:
            allowed = conn.execute(
                "SELECT 1 FROM frames WHERE path IN (?, ?) "
                "UNION SELECT 1 FROM export_artifacts WHERE path IN (?, ?) "
                "UNION SELECT 1 FROM export_artifacts WHERE manifest_path IN (?, ?) LIMIT 1",
                (target, resolved, target, resolved, target, resolved),
            ).fetchone()
        finally:
            conn.close()
        if not allowed:
            raise HTTPException(403, "path not referenced in index")
        target_path = Path(target)
        if not target_path.exists():
            raise HTTPException(404, "file missing on disk")
        mime, _ = mimetypes.guess_type(str(target_path))
        return FileResponse(str(target_path), media_type=mime or "application/octet-stream")

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1", "http://localhost"],
            allow_methods=["GET"],
        ),
    ]

    routes = [
        Route("/api/health", health),
        Route("/api/videos", list_videos),
        Route("/api/videos/{file_hash}", get_video),
        Route("/api/videos/{file_hash}/frames", list_video_frames),
        Route("/api/videos/{file_hash}/subtitles", list_video_subtitles),
        Route("/api/searches", list_searches),
        Route("/api/searches/{search_id:int}", get_search),
        Route("/api/exports", list_exports),
        Route("/file", serve_file),
    ]

    return Starlette(debug=False, routes=routes, middleware=middleware)


def _row_to_video_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "file_hash": row["file_hash"],
        "path": row["path"],
        "duration_ms": row["duration_ms"],
        "width": row["width"],
        "height": row["height"],
        "fps": row["fps"],
        "ingested_at": row["ingested_at"],
        "frames_count": int(row["frames_count"]),
        "subtitle_cues_count": int(row["subtitle_cues_count"]),
        "exports_count": int(row["exports_count"]),
        "person_searches_count": int(row["person_searches_count"]),
    }


def _row_to_video_detail(
    row: sqlite3.Row, exports_by_kind: dict[str, int]
) -> dict[str, Any]:
    base = _row_to_video_summary(row)
    base.update(
        {
            "container": row["container"],
            "video_codec": row["video_codec"],
            "audio_codec": row["audio_codec"],
            "size_bytes": row["size_bytes"],
            "exports_by_kind": exports_by_kind,
        }
    )
    return base


def _row_to_frame(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "timestamp_ms": int(row["timestamp_ms"]),
        "path": row["path"],
        "sampling_strategy": row["sampling_strategy"],
    }


def _row_to_cue(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "start_ms": int(row["start_ms"]),
        "end_ms": int(row["end_ms"]),
        "text": row["text"],
        "language": row["language"],
        "source": row["source"],
    }


def _row_to_search_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "video_hash": row["video_hash"],
        "label": row["label"],
        "backend": row["backend"],
        "threshold": float(row["threshold"]),
        "created_at": row["created_at"],
        "match_count": int(row["match_count"]),
    }


def _row_to_search_detail(
    row: sqlite3.Row, matches: list[sqlite3.Row]
) -> dict[str, Any]:
    base = _row_to_search_summary(row)
    base.update(
        {
            "positive_examples": _json_list(row["positive_examples_json"]),
            "negative_examples": _json_list(row["negative_examples_json"]),
            "config": _json_obj(row["config_json"]),
            "matches": [_row_to_match(m) for m in matches],
        }
    )
    return base


def _row_to_match(row: sqlite3.Row) -> dict[str, Any]:
    bbox_raw = row["bbox_json"]
    bbox = None
    if bbox_raw:
        try:
            parsed = json.loads(bbox_raw)
            if isinstance(parsed, list):
                bbox = [int(x) for x in parsed]
        except (ValueError, TypeError):
            bbox = None
    return {
        "id": int(row["id"]),
        "frame_id": int(row["frame_id"]),
        "frame_path": row["frame_path"],
        "frame_timestamp_ms": int(row["frame_timestamp_ms"]),
        "confidence": float(row["confidence"]),
        "bbox": bbox,
        "reasoning": row["reasoning"],
    }


def _row_to_export(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "video_hash": row["video_hash"],
        "kind": row["kind"],
        "path": row["path"],
        "start_ms": int(row["start_ms"]) if row["start_ms"] is not None else None,
        "end_ms": int(row["end_ms"]) if row["end_ms"] is not None else None,
        "manifest_path": row["manifest_path"],
        "created_at": row["created_at"],
    }


def _json_list(raw: str | None) -> list[Any]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except ValueError:
        return []
    return value if isinstance(value, list) else []


def _json_obj(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}
