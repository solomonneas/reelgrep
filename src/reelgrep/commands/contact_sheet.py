"""``reelgrep contact-sheet`` command."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import click

from reelgrep.config import ensure_dirs, get_settings
from reelgrep.contact_sheet import build as build_sheet
from reelgrep.db import connect, migrate
from reelgrep.frames import sample_every
from reelgrep.hashing import file_hash
from reelgrep.manifest import Manifest, Source, sidecar_path, write
from reelgrep.probe import probe

__all__ = ["contact_sheet"]


@click.command("contact-sheet")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--cols", type=int, default=6, show_default=True)
@click.option("--rows", type=int, default=None)
@click.option("--every", "interval_seconds", type=float, default=None)
@click.option("--thumb-width", type=int, default=320, show_default=True)
@click.option("--use-cached", is_flag=True, default=False)
@click.option("--no-db", is_flag=True, default=False)
def contact_sheet(
    video_path: Path,
    out_path: Path,
    cols: int,
    rows: int | None,
    interval_seconds: float | None,
    thumb_width: int,
    use_cached: bool,
    no_db: bool,
) -> None:
    """Build a contact sheet of sampled frames from a video."""
    if cols < 1:
        raise click.BadParameter("must be >= 1", param_hint="--cols")
    if thumb_width < 32:
        raise click.BadParameter("must be >= 32", param_hint="--thumb-width")
    if rows is not None and rows < 1:
        raise click.BadParameter("must be >= 1", param_hint="--rows")

    resolved_video = video_path.resolve()
    resolved_out = out_path.resolve()
    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    ensure_dirs(settings)
    digest = file_hash(resolved_video)

    frames: list[tuple[Path, int]] = []
    interval_used: float | None = None
    duration_ms: int | None = None

    if use_cached:
        conn = connect(settings.db_path)
        try:
            migrate(conn)
            row = conn.execute(
                "SELECT id, duration_ms FROM videos WHERE file_hash = ?",
                (digest,),
            ).fetchone()
            if row is not None:
                video_id = row[0]
                duration_ms = row[1]
                cur = conn.execute(
                    "SELECT path, timestamp_ms FROM frames "
                    "WHERE video_id = ? ORDER BY timestamp_ms",
                    (video_id,),
                )
                frames = [(Path(r[0]), int(r[1])) for r in cur.fetchall()]
        finally:
            conn.close()

    if not frames:
        meta = probe(resolved_video)
        duration_ms = meta.duration_ms
        if interval_seconds is not None:
            interval_used = interval_seconds
        else:
            effective_rows_guess = rows if rows else 8
            interval_used = max(
                1.0,
                meta.duration_ms / 1000 / (cols * effective_rows_guess),
            )
        cache_dir = settings.cache_dir / "contact" / digest[8:24]
        sampled = sample_every(
            resolved_video,
            cache_dir,
            interval_seconds=interval_used,
        )
        frames = [(Path(f.path), int(f.timestamp_ms)) for f in sampled]

    if not frames:
        raise click.ClickException("no frames available to build contact sheet")

    effective_rows = rows if rows else math.ceil(len(frames) / cols)
    capacity = effective_rows * cols
    if len(frames) > capacity:
        frames = frames[:capacity]

    frame_paths = [p for p, _ in frames]
    timestamps_ms = [ts for _, ts in frames]

    build_sheet(
        frame_paths=frame_paths,
        out_path=resolved_out,
        cols=cols,
        rows=effective_rows,
        thumb_width=thumb_width,
        timestamps_ms=timestamps_ms,
    )

    manifest = Manifest(
        operation="contact-sheet",
        source=Source(
            path=str(resolved_video),
            file_hash=digest,
            duration_ms=duration_ms,
        ),
        parameters={
            "cols": cols,
            "rows": effective_rows,
            "thumb_width": thumb_width,
            "interval_seconds": interval_used if not use_cached else None,
            "use_cached": use_cached,
            "source_frame_count": len(frames),
        },
        results=[
            {
                "output_path": str(resolved_out),
                "tile_count": len(frames),
            }
        ],
    )
    manifest_path = sidecar_path(resolved_out)
    write(manifest_path, manifest)

    if not no_db:
        conn = connect(settings.db_path)
        try:
            migrate(conn)
            row = conn.execute(
                "SELECT id FROM videos WHERE file_hash = ?",
                (digest,),
            ).fetchone()
            if row is None:
                click.echo(
                    "warning: video not indexed; skipping export_artifacts row",
                    err=True,
                )
            else:
                video_id = row[0]
                created_at = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
                conn.execute(
                    "INSERT INTO export_artifacts "
                    "(video_id, kind, path, manifest_path, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        video_id,
                        "contact_sheet",
                        str(resolved_out),
                        str(manifest_path),
                        created_at,
                    ),
                )
                conn.commit()
        finally:
            conn.close()

    click.echo(f"contact sheet: {resolved_out}")
    click.echo(f"tiles:         {len(frames)} ({cols}x{effective_rows})")
    click.echo(f"manifest:      {manifest_path}")
