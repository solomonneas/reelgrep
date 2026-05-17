"""``reelgrep search-subtitles`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from reelgrep import timecode
from reelgrep.config import ensure_dirs, get_settings
from reelgrep.db import connect, migrate
from reelgrep.ffmpeg_exec import FFmpegError, run_ffmpeg
from reelgrep.hashing import file_hash
from reelgrep.manifest import Manifest, Source, write

__all__ = ["search_subtitles"]


@click.command("search-subtitles")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("query")
@click.option(
    "--export-frames",
    is_flag=True,
    default=False,
    help="Export a frame for each match.",
)
@click.option(
    "--context-ms",
    type=int,
    default=0,
    show_default=True,
    help="Offset (ms) from cue start when grabbing the frame.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for exported frames (required with --export-frames).",
)
@click.option("--limit", type=int, default=100, show_default=True)
def search_subtitles(
    video_path: Path,
    query: str,
    export_frames: bool,
    context_ms: int,
    out_dir: Path | None,
    limit: int,
) -> None:
    """Search subtitle cues for a video using FTS5."""
    if export_frames and out_dir is None:
        raise click.BadParameter("--out is required when --export-frames is set.")
    if limit <= 0:
        raise click.BadParameter("--limit must be a positive integer.")

    resolved_video = video_path.resolve()
    digest = file_hash(resolved_video)

    settings = get_settings()
    ensure_dirs(settings)
    conn = connect(settings.db_path)
    try:
        migrate(conn)
        row = conn.execute(
            "SELECT id, duration_ms FROM videos WHERE file_hash = ?",
            (digest,),
        ).fetchone()
        if row is None:
            click.echo(
                "Video not ingested. Run 'reelgrep ingest <path>' first.",
                err=True,
            )
            raise SystemExit(2)

        video_id = row[0]
        duration_ms = row[1]

        cursor = conn.execute(
            """
            SELECT s.id, s.start_ms, s.end_ms, s.text, s.language
            FROM subtitles_fts
            JOIN subtitles AS s ON s.id = subtitles_fts.rowid
            WHERE subtitles_fts MATCH ?
              AND s.video_id = ?
            ORDER BY s.start_ms
            LIMIT ?
            """,
            (query, video_id, limit),
        )
        matches = [
            {
                "id": r[0],
                "start_ms": r[1],
                "end_ms": r[2],
                "text": r[3],
                "language": r[4],
            }
            for r in cursor.fetchall()
        ]
    finally:
        conn.close()

    if not matches:
        click.echo("no matches")
        return

    for m in matches:
        click.echo(f"{timecode.format(m['start_ms'])}  {m['text']}")

    results: list[dict[str, Any]] = []
    if export_frames:
        assert out_dir is not None
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = resolved_video.stem
        for m in matches:
            ts_ms = max(0, m["start_ms"] + context_ms)
            ts_label = timecode.format(ts_ms).replace(":", "-")
            frame_path = out_dir / f"{stem}_{ts_label}.jpg"
            try:
                run_ffmpeg(
                    [
                        "-ss",
                        f"{ts_ms / 1000:.3f}",
                        "-i",
                        str(resolved_video),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        "-y",
                        str(frame_path),
                    ]
                )
                results.append(
                    {
                        "start_ms": m["start_ms"],
                        "end_ms": m["end_ms"],
                        "text": m["text"],
                        "frame_path": str(frame_path),
                    }
                )
            except FFmpegError as exc:
                ts_str = timecode.format(m["start_ms"])
                click.echo(
                    f"warning: failed to export frame for match at {ts_str}: {exc}",
                    err=True,
                )
                results.append(
                    {
                        "start_ms": m["start_ms"],
                        "end_ms": m["end_ms"],
                        "text": m["text"],
                        "frame_path": None,
                    }
                )
    else:
        for m in matches:
            results.append(
                {
                    "start_ms": m["start_ms"],
                    "end_ms": m["end_ms"],
                    "text": m["text"],
                    "frame_path": None,
                }
            )

    manifest = Manifest(
        operation="search-subtitles",
        source=Source(
            path=str(resolved_video),
            file_hash=digest,
            duration_ms=duration_ms,
        ),
        parameters={
            "query": query,
            "context_ms": context_ms,
            "export_frames": export_frames,
            "limit": limit,
        },
        results=results,
    )

    if out_dir is not None:
        manifest_path = out_dir / "search-subtitles.manifest.json"
    else:
        manifest_path = resolved_video.with_suffix(resolved_video.suffix + ".search.manifest.json")
    write(manifest_path, manifest)

    click.echo(f"{len(matches)} matches")
