"""``reelgrep export-clip`` command."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click

from reelgrep import timecode
from reelgrep.config import ensure_dirs, get_settings
from reelgrep.db import connect, migrate
from reelgrep.ffmpeg_exec import run_ffmpeg
from reelgrep.hashing import file_hash
from reelgrep.manifest import Manifest, Source, sidecar_path, write
from reelgrep.probe import probe

__all__ = ["export_clip"]


@click.command("export-clip")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--start", "start_str", required=True)
@click.option("--end", "end_str", required=True)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--reencode", is_flag=True, default=False)
@click.option("--no-db", is_flag=True, default=False)
def export_clip(
    video_path: Path,
    start_str: str,
    end_str: str,
    out_path: Path,
    reencode: bool,
    no_db: bool,
) -> None:
    """Cut a sub-clip from a video by start/end timecodes."""
    try:
        start_ms = timecode.parse(start_str)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--start") from exc
    try:
        end_ms = timecode.parse(end_str)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--end") from exc

    if end_ms <= start_ms:
        raise click.BadParameter("--end must be after --start")

    resolved_video = video_path.resolve()
    resolved_out = out_path.resolve()
    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    digest = file_hash(resolved_video)
    meta = probe(resolved_video)

    if end_ms > meta.duration_ms:
        click.echo(
            f"warning: --end {timecode.format(end_ms)} exceeds source duration "
            f"{timecode.format(meta.duration_ms)}; clamping to duration",
            err=True,
        )
        end_ms = meta.duration_ms
        if end_ms <= start_ms:
            raise click.BadParameter("--start is past end of video")

    if reencode:
        args: list[str] = [
            "-i",
            str(resolved_video),
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-to",
            f"{end_ms / 1000:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-y",
            str(resolved_out),
        ]
    else:
        args = [
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-to",
            f"{end_ms / 1000:.3f}",
            "-i",
            str(resolved_video),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            str(resolved_out),
        ]

    run_ffmpeg(args)

    size_bytes = resolved_out.stat().st_size if resolved_out.exists() else None
    manifest = Manifest(
        operation="export-clip",
        source=Source(
            path=str(resolved_video),
            file_hash=digest,
            duration_ms=meta.duration_ms,
        ),
        parameters={
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start": timecode.format(start_ms),
            "end": timecode.format(end_ms),
            "reencode": reencode,
        },
        results=[
            {
                "output_path": str(resolved_out),
                "duration_ms": end_ms - start_ms,
                "size_bytes": size_bytes,
            }
        ],
    )
    manifest_path = sidecar_path(resolved_out)
    write(manifest_path, manifest)

    if not no_db:
        settings = get_settings()
        ensure_dirs(settings)
        conn = connect(settings.db_path)
        try:
            migrate(conn)
            row = conn.execute(
                "SELECT id FROM videos WHERE file_hash = ?",
                (digest,),
            ).fetchone()
            if row is None:
                click.echo(
                    "warning: video not in index; skipping export_artifacts row",
                    err=True,
                )
            else:
                video_id = row[0]
                created_at = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
                with conn:
                    conn.execute(
                        "INSERT INTO export_artifacts "
                        "(video_id, kind, path, start_ms, end_ms, manifest_path, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            video_id,
                            "clip",
                            str(resolved_out),
                            start_ms,
                            end_ms,
                            str(manifest_path),
                            created_at,
                        ),
                    )
        finally:
            conn.close()

    duration_s = (end_ms - start_ms) / 1000
    click.echo(f"exported: {resolved_out}")
    click.echo(
        f"range:    {timecode.format(start_ms)} -> {timecode.format(end_ms)} "
        f"({duration_s:.3f}s)"
    )
    click.echo(f"manifest: {manifest_path}")
