"""``reelgrep make-gif`` command: render a short animated WebP or GIF loop."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from reelgrep import timecode
from reelgrep.config import ensure_dirs, get_settings
from reelgrep.db import connect, migrate
from reelgrep.ffmpeg_exec import run_ffmpeg
from reelgrep.hashing import file_hash
from reelgrep.manifest import Manifest, Source, sidecar_path, write
from reelgrep.probe import probe

__all__ = ["make_gif"]


def _webp_args(
    video_path: Path,
    start_ms: int,
    duration_seconds: float,
    fps: int,
    width: int,
    out_path: Path,
) -> list[str]:
    return [
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-t",
        str(duration_seconds),
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:-2:flags=lanczos",
        "-c:v",
        "libwebp",
        "-loop",
        "0",
        "-lossless",
        "0",
        "-q:v",
        "75",
        "-an",
        "-y",
        str(out_path),
    ]


def _palettegen_args(
    video_path: Path,
    start_ms: int,
    duration_seconds: float,
    fps: int,
    width: int,
    palette: Path,
) -> list[str]:
    return [
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-t",
        str(duration_seconds),
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:-2:flags=lanczos,palettegen",
        "-y",
        str(palette),
    ]


def _paletteuse_args(
    video_path: Path,
    start_ms: int,
    duration_seconds: float,
    fps: int,
    width: int,
    palette: Path,
    out_path: Path,
) -> list[str]:
    return [
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-t",
        str(duration_seconds),
        "-i",
        str(video_path),
        "-i",
        str(palette),
        "-lavfi",
        f"fps={fps},scale={width}:-2:flags=lanczos [v]; [v][1:v] paletteuse",
        "-y",
        str(out_path),
    ]


@click.command("make-gif")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--start", "start_str", required=True)
@click.option("--duration", "duration_seconds", type=float, required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--fps", type=int, default=12, show_default=True)
@click.option("--width", type=int, default=480, show_default=True)
@click.option("--no-db", is_flag=True, default=False)
def make_gif(
    video_path: Path,
    start_str: str,
    duration_seconds: float,
    out_path: Path,
    fps: int,
    width: int,
    no_db: bool,
) -> None:
    """Render a short animated WebP or GIF loop from a video."""
    if duration_seconds <= 0:
        raise click.BadParameter("--duration must be > 0", param_hint="--duration")
    if fps <= 0:
        raise click.BadParameter("--fps must be > 0", param_hint="--fps")
    if width < 32:
        raise click.BadParameter("--width must be >= 32", param_hint="--width")

    start_ms = timecode.parse(start_str)

    resolved_video = video_path.resolve()
    resolved_out = out_path.resolve()

    suffix = resolved_out.suffix.lower()
    if suffix == ".webp":
        fmt = "webp"
    elif suffix == ".gif":
        fmt = "gif"
    else:
        resolved_out = resolved_out.with_suffix(".webp")
        fmt = "webp"

    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    meta = probe(resolved_video)
    digest = file_hash(resolved_video)
    settings = get_settings()
    ensure_dirs(settings)

    if fmt == "webp":
        run_ffmpeg(
            _webp_args(resolved_video, start_ms, duration_seconds, fps, width, resolved_out)
        )
    else:
        palette_dir = settings.cache_dir / "tmp"
        palette_dir.mkdir(parents=True, exist_ok=True)
        hex_part = digest[len("blake2b:") :] if digest.startswith("blake2b:") else digest
        palette = palette_dir / f"{hex_part[8:24]}_palette.png"
        try:
            run_ffmpeg(
                _palettegen_args(
                    resolved_video, start_ms, duration_seconds, fps, width, palette
                )
            )
            run_ffmpeg(
                _paletteuse_args(
                    resolved_video,
                    start_ms,
                    duration_seconds,
                    fps,
                    width,
                    palette,
                    resolved_out,
                )
            )
        finally:
            palette.unlink(missing_ok=True)

    size_bytes = resolved_out.stat().st_size if resolved_out.exists() else 0

    parameters: dict[str, Any] = {
        "start_ms": start_ms,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "width": width,
        "format": fmt,
    }
    results: list[dict[str, Any]] = [
        {
            "output_path": str(resolved_out),
            "size_bytes": size_bytes,
        }
    ]
    manifest = Manifest(
        operation="make-gif",
        source=Source(
            path=str(resolved_video),
            file_hash=digest,
            duration_ms=meta.duration_ms,
        ),
        parameters=parameters,
        results=results,
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
                    "warning: video not indexed; skipping export_artifacts row "
                    "(run 'reelgrep ingest <path>' first).",
                    err=True,
                )
            else:
                video_id = row[0]
                end_ms = start_ms + int(round(duration_seconds * 1000))
                created_at = datetime.now(UTC).isoformat(timespec="seconds")
                conn.execute(
                    "INSERT INTO export_artifacts "
                    "(video_id, kind, path, start_ms, end_ms, manifest_path, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        video_id,
                        "gif",
                        str(resolved_out),
                        start_ms,
                        end_ms,
                        str(manifest_path),
                        created_at,
                    ),
                )
                conn.commit()
        finally:
            conn.close()

    click.echo(f"rendered: {resolved_out}")
    click.echo(
        f"span:     {timecode.format(start_ms)} + {duration_seconds}s @ {fps}fps, {width}w"
    )
    click.echo(f"manifest: {manifest_path}")
