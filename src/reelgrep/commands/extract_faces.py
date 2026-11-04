"""`reelgrep extract-faces` CLI command."""
from __future__ import annotations

import click

from reelgrep.config import get_settings
from reelgrep.db import connect, migrate
from reelgrep.faces import FacesError, InsightFaceMissingError, extract_faces

__all__ = ["extract_faces_command"]


@click.command("extract-faces")
@click.argument("video", required=False)
@click.option("--all", "all_", is_flag=True, help="Backfill across every ingested video.")
@click.option("--force", is_flag=True, help="Re-detect even when detections already exist.")
@click.pass_context
def extract_faces_command(
    ctx: click.Context, video: str | None, all_: bool, force: bool,
) -> None:
    """Detect + embed all faces in the sampled frames of a video (or every video with --all)."""
    if not all_ and not video:
        raise click.UsageError("either pass a VIDEO path or --all to backfill")

    try:
        if all_:
            db_path = get_settings().db_path
            conn = connect(db_path)
            migrate(conn)
            videos = [r[0] for r in conn.execute("SELECT path FROM videos ORDER BY id")]
            conn.close()
            total = 0
            for v in videos:
                r = extract_faces(v, force=force)
                total += r.detections_added
                if r.skipped_existing:
                    click.echo(f"  {v}: skipped (already detected)")
                else:
                    click.echo(f"  {v}: +{r.detections_added}")
            click.echo(f"videos processed: {len(videos)}")
            click.echo(f"detections: {total}")
        else:
            r = extract_faces(video, force=force)
            click.echo(f"video:      {r.video_path}")
            click.echo(f"frames:     {r.frames_scanned}")
            click.echo(f"detections: {r.detections_added}")
            click.echo(f"model:      {r.embedding_model}")
            if r.skipped_existing:
                click.echo("status:     skipped (use --force to re-detect)")
    except InsightFaceMissingError as exc:
        click.echo(f"error: {exc}", err=True)
        ctx.exit(1)
    except FacesError as exc:
        click.echo(f"error: {exc}", err=True)
        ctx.exit(1)
