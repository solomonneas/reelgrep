"""``reelgrep ls`` command: list recently ingested videos."""

from __future__ import annotations

import click

from reelgrep import config, db
from reelgrep import timecode as tc

__all__ = ["ls"]

_PATH_MAX = 60


def _truncate_path(path: str, limit: int = _PATH_MAX) -> str:
    """Return ``path`` shortened to its last ``limit`` chars with a leading ellipsis."""
    if len(path) <= limit:
        return path
    return "..." + path[-(limit - 3) :]


def _truncate_hash(file_hash: str) -> str:
    """Return ``blake2b:<first 8 hex>...`` form of ``file_hash`` (19 chars total)."""
    if file_hash.startswith("blake2b:"):
        hex_part = file_hash[len("blake2b:") :]
    else:
        hex_part = file_hash
    return f"blake2b:{hex_part[:8]}..."


def _fmt_resolution(width: int | None, height: int | None) -> str:
    if width is None or height is None:
        return "-"
    return f"{width}x{height}"


def _fmt_fps(fps: float | None) -> str:
    if fps is None:
        return "-"
    return f"{fps:.2f}"


def _fmt_duration(duration_ms: int | None) -> str:
    if duration_ms is None:
        return "-"
    return tc.format(int(duration_ms))


@click.command("ls")
@click.option("--limit", type=int, default=20, show_default=True)
def ls(limit: int) -> None:
    """List recently ingested videos."""
    if limit <= 0:
        raise click.BadParameter("limit must be > 0", param_hint="--limit")

    settings = config.get_settings()
    config.ensure_dirs(settings)
    conn = db.connect(settings.db_path)
    try:
        db.migrate(conn)
        rows = conn.execute(
            "SELECT file_hash, path, duration_ms, width, height, fps, ingested_at "
            "FROM videos ORDER BY ingested_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        click.echo("no videos indexed")
        return

    header = f"{'HASH':<19}  {'DURATION':<12}  {'RES':<10} {'FPS':<6} {'INGESTED':<21} PATH"
    click.echo(header)
    for row in rows:
        hash_short = _truncate_hash(row["file_hash"])
        duration = _fmt_duration(row["duration_ms"])
        resolution = _fmt_resolution(row["width"], row["height"])
        fps = _fmt_fps(row["fps"])
        ingested = str(row["ingested_at"])
        path = _truncate_path(str(row["path"]))
        click.echo(
            f"{hash_short:<19}  {duration:<12}  {resolution:<10} {fps:<6} {ingested:<21} {path}"
        )
