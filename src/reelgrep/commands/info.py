"""``reelgrep info`` command: show indexed details for a single video."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import click

from reelgrep import config, db
from reelgrep import timecode as tc
from reelgrep.hashing import file_hash

__all__ = ["info"]

_HEX_PREFIX_RE = re.compile(r"[0-9a-fA-F]{12,}")
_EXPORT_KINDS = ("screenshots", "clips", "gifs", "contact_sheets")
_KIND_PLURAL = {
    "screenshot": "screenshots",
    "clip": "clips",
    "gif": "gifs",
    "contact_sheet": "contact_sheets",
}


def _humanize_bytes(n: int | None) -> str:
    """Return ``n`` bytes as a short human-readable string."""
    if n is None:
        return "unknown"
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            return f"{v:.1f} {u}" if u != "B" else f"{int(v)} B"
        v /= 1024
    return f"{v:.1f} {units[-1]}"


def _fmt_duration(duration_ms: int | None) -> str:
    if duration_ms is None:
        return "-"
    return tc.format(int(duration_ms))


def _fmt_resolution(width: int | None, height: int | None) -> str:
    if width is None or height is None:
        return "-"
    return f"{width}x{height}"


def _fmt_fps(fps: float | None) -> str:
    if fps is None:
        return "-"
    return f"{fps:.2f}"


def _fmt_or_dash(value: object) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


_VIDEO_COLUMNS = (
    "id, file_hash, path, duration_ms, width, height, fps, container, "
    "video_codec, audio_codec, size_bytes, ingested_at"
)


def _resolve_rows(conn: sqlite3.Connection, target: str) -> list[sqlite3.Row]:
    """Resolve ``target`` to one or more video rows using the spec's lookup order."""
    if target.startswith("blake2b:"):
        rows = conn.execute(
            f"SELECT {_VIDEO_COLUMNS} FROM videos WHERE file_hash = ?",
            (target,),
        ).fetchall()
        return list(rows)

    if _HEX_PREFIX_RE.fullmatch(target):
        rows = conn.execute(
            f"SELECT {_VIDEO_COLUMNS} FROM videos "
            "WHERE file_hash LIKE 'blake2b:' || ? || '%'",
            (target.lower(),),
        ).fetchall()
        return list(rows)

    path = Path(target)
    if path.exists() and path.is_file():
        digest = file_hash(path)
        rows = conn.execute(
            f"SELECT {_VIDEO_COLUMNS} FROM videos WHERE file_hash = ?",
            (digest,),
        ).fetchall()
        if rows:
            return list(rows)

    rows = conn.execute(
        f"SELECT {_VIDEO_COLUMNS} FROM videos WHERE path LIKE '%' || ? || '%'",
        (target,),
    ).fetchall()
    return list(rows)


def _collect_counts(conn: sqlite3.Connection, video_id: int) -> dict[str, object]:
    sub_row = conn.execute(
        "SELECT "
        "COUNT(DISTINCT source || ':' || COALESCE(CAST(stream_index AS TEXT),'-')) AS tracks, "
        "COUNT(*) AS cues "
        "FROM subtitles WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    frames = conn.execute(
        "SELECT COUNT(*) FROM frames WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    scenes = conn.execute(
        "SELECT COUNT(*) FROM scenes WHERE video_id = ?", (video_id,)
    ).fetchone()[0]
    people = conn.execute(
        "SELECT COUNT(*) FROM person_searches WHERE video_id = ?", (video_id,)
    ).fetchone()[0]

    export_counts: dict[str, int] = dict.fromkeys(_EXPORT_KINDS, 0)
    for row in conn.execute(
        "SELECT kind, COUNT(*) FROM export_artifacts WHERE video_id = ? GROUP BY kind",
        (video_id,),
    ):
        kind_plural = _KIND_PLURAL.get(row[0])
        if kind_plural is not None:
            export_counts[kind_plural] = int(row[1])

    return {
        "tracks": int(sub_row["tracks"]) if sub_row is not None else 0,
        "cues": int(sub_row["cues"]) if sub_row is not None else 0,
        "frames": int(frames),
        "scenes": int(scenes),
        "people": int(people),
        "exports": export_counts,
    }


def _print_report(row: sqlite3.Row, counts: dict[str, object]) -> None:
    click.echo(f"File:        {row['path']}")
    click.echo(f"Hash:        {row['file_hash']}")
    click.echo(f"Container:   {_fmt_or_dash(row['container'])}")
    click.echo(f"Duration:    {_fmt_duration(row['duration_ms'])}")
    click.echo(f"Resolution:  {_fmt_resolution(row['width'], row['height'])}")
    click.echo(f"FPS:         {_fmt_fps(row['fps'])}")
    click.echo(
        f"Codecs:      {_fmt_or_dash(row['video_codec'])} / {_fmt_or_dash(row['audio_codec'])}"
    )
    click.echo(f"Size:        {_humanize_bytes(row['size_bytes'])}")
    click.echo(f"Ingested:    {row['ingested_at']}")
    click.echo("")
    click.echo(f"Subtitles:   {counts['tracks']} tracks, {counts['cues']} cues")
    click.echo(f"Frames:      {counts['frames']} sampled")
    click.echo(f"Scenes:      {counts['scenes']} detected")
    click.echo(f"Person searches: {counts['people']}")
    exports = counts["exports"]
    assert isinstance(exports, dict)
    parts = ", ".join(f"{kind}:{exports[kind]}" for kind in _EXPORT_KINDS)
    click.echo(f"Exports:     {parts}")


@click.command("info")
@click.argument("target")
def info(target: str) -> None:
    """Show indexed details for a video. Accepts a file path or a blake2b hash."""
    settings = config.get_settings()
    config.ensure_dirs(settings)
    conn = db.connect(settings.db_path)
    try:
        db.migrate(conn)
        rows = _resolve_rows(conn, target)
        if not rows:
            click.echo("not found", err=True)
            raise SystemExit(2)
        if len(rows) > 1:
            for idx, r in enumerate(rows):
                click.echo(f"{idx}: {r['path']}")
            click.echo("multiple matches; be more specific", err=True)
            raise SystemExit(2)
        row = rows[0]
        counts = _collect_counts(conn, int(row["id"]))
    finally:
        conn.close()

    _print_report(row, counts)
