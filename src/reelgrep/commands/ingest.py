"""``reelgrep ingest`` command: probe, extract subtitles, sample frames, persist."""

from __future__ import annotations

from pathlib import Path

import click

from reelgrep import timecode
from reelgrep.index import hash_slice as _hash_slice  # noqa: F401 - re-export
from reelgrep.index import ingest_video

__all__ = ["ingest"]


@click.command("ingest")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--every", "interval_seconds", type=float, default=5.0, show_default=True)
@click.option(
    "--scene/--no-scene",
    default=False,
    help="Forward-compat flag, scene sampling not wired in this command.",
)
@click.option("--no-subtitles", is_flag=True, default=False)
@click.option("--no-frames", is_flag=True, default=False)
@click.option("--force", is_flag=True, default=False)
@click.option(
    "--transcribe",
    "do_transcribe",
    is_flag=True,
    default=False,
    help="After ingest, run Whisper transcription if no embedded/sidecar subs were found.",
)
@click.option(
    "--transcribe-model",
    default="small",
    show_default=True,
    help="Whisper model size when --transcribe is set.",
)
def ingest(
    video_path: Path,
    interval_seconds: float,
    scene: bool,  # noqa: ARG001 - forward-compat flag, not wired yet
    no_subtitles: bool,
    no_frames: bool,
    force: bool,
    do_transcribe: bool,
    transcribe_model: str,
) -> None:
    """Ingest a video: probe, extract subtitles, sample frames, write to local index."""
    def _stderr(msg: str) -> None:
        click.echo(msg, err=True)

    result = ingest_video(
        video_path,
        backend="local",
        interval_seconds=interval_seconds,
        no_subtitles=no_subtitles,
        no_frames=no_frames,
        force=force,
        do_transcribe=do_transcribe,
        transcribe_model=transcribe_model,
        on_message=_stderr,
    )

    if result.already_ingested:
        click.echo(f"Already ingested: {result.file_hash} ({result.video_path})")
        return

    click.echo(f"ingested: {result.video_path}")
    click.echo(f"hash:     {result.file_hash}")
    click.echo(f"duration: {timecode.format(result.duration_ms)}")
    click.echo(
        f"subtitle tracks: {result.subtitle_track_count} (cues: {result.subtitle_cue_count})"
    )
    if result.transcribed_cue_count:
        click.echo(
            f"transcribed {result.transcribed_cue_count} cues with "
            f"whisper:{result.transcribe_model}"
        )
    click.echo(f"frames sampled: {result.frame_count}")
    click.echo(f"db:       {result.db_path}")
