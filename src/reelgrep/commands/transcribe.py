"""CLI command to transcribe a video with Whisper and write cues into the index."""

from __future__ import annotations

import json
from pathlib import Path

import click

from reelgrep.hashing import file_hash
from reelgrep.timecode import format as format_timecode
from reelgrep.transcribe import (
    TranscribeError,
    available_models,
    transcribe,
    transcribe_video,
)

__all__ = ["transcribe_cmd"]


@click.command("transcribe")
@click.argument(
    "video_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--model",
    "model_size",
    type=click.Choice(available_models()),
    default="small",
    show_default=True,
    help="Whisper model size. Larger = slower + more accurate.",
)
@click.option(
    "--language",
    default=None,
    help="ISO-639-1 code (en, es, fr, ...). Default: auto-detect.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cpu",
    show_default=True,
)
@click.option(
    "--compute-type",
    default=None,
    help="int8 / int8_float16 / float16 / float32. Auto-picked when omitted.",
)
@click.option("--beam-size", type=int, default=5, show_default=True)
@click.option(
    "--no-vad-filter",
    is_flag=True,
    default=False,
    help="Skip Silero VAD silence-trimming (faster, more hallucinations).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Replace existing whisper cues for this video.",
)
@click.option(
    "--no-db",
    is_flag=True,
    default=False,
    help="Print cues to stdout as JSON instead of writing to the index.",
)
def transcribe_cmd(
    video_path: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str | None,
    beam_size: int,
    no_vad_filter: bool,
    force: bool,
    no_db: bool,
) -> None:
    """Transcribe a video with Whisper and store the cues in the local index."""
    video_resolved = video_path.expanduser().resolve()

    if no_db:
        # Don't even open the DB - just run inference and print to stdout.
        try:
            track = transcribe(
                video_resolved,
                model_size=model_size,
                language=language,
                device=device,
                compute_type=compute_type,
                beam_size=beam_size,
                vad_filter=not no_vad_filter,
            )
        except TranscribeError as exc:
            click.echo(f"error: {exc}", err=True)
            raise click.exceptions.Exit(2) from exc
        click.echo(
            json.dumps(
                {
                    "video": str(video_resolved),
                    "file_hash": file_hash(video_resolved),
                    "language": track.language,
                    "model": model_size,
                    "cues": [
                        {"start_ms": c.start_ms, "end_ms": c.end_ms, "text": c.text}
                        for c in track.cues
                    ],
                },
                indent=2,
            )
        )
        return

    def _emit(msg: str) -> None:
        click.echo(msg, err=True)

    try:
        result = transcribe_video(
            video_resolved,
            model=model_size,
            language=language,
            force=force,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=not no_vad_filter,
            on_message=_emit,
        )
    except TranscribeError as exc:
        click.echo(f"error: {exc}", err=True)
        raise click.exceptions.Exit(2) from exc

    if result.already_transcribed:
        click.echo(
            f"video already has {result.cue_count} whisper cue(s); "
            f"pass --force to replace",
            err=True,
        )
        raise click.exceptions.Exit(0)

    if result.cue_count > 0:
        first_last = _fetch_cue_span(result.db_path, result.video_id)
        span = first_last if first_last is not None else "(no cues)"
    else:
        span = "(no cues)"

    click.echo(
        f"transcribed: {video_resolved}\n"
        f"language:    {result.language_detected}\n"
        f"model:       whisper:{result.model}\n"
        f"cues:        {result.cue_count}\n"
        f"span:        {span}"
    )


def _fetch_cue_span(db_path: Path, video_id: int) -> str | None:
    """Read first/last whisper cue timestamps for a friendly span echo.

    Returns ``None`` when no whisper cues exist for the video.
    """
    from reelgrep.db import connect

    conn = connect(db_path)
    try:
        first = conn.execute(
            "SELECT start_ms FROM subtitles "
            "WHERE video_id = ? AND source = 'whisper' "
            "ORDER BY start_ms ASC LIMIT 1",
            (video_id,),
        ).fetchone()
        last = conn.execute(
            "SELECT end_ms FROM subtitles "
            "WHERE video_id = ? AND source = 'whisper' "
            "ORDER BY end_ms DESC LIMIT 1",
            (video_id,),
        ).fetchone()
    finally:
        conn.close()
    if first is None or last is None:
        return None
    return f"{format_timecode(int(first[0]))} -> {format_timecode(int(last[0]))}"
