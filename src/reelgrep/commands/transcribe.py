"""CLI command to transcribe a video with Whisper and write cues into the index."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import click

from reelgrep.config import ensure_dirs, get_settings
from reelgrep.db import connect, migrate
from reelgrep.hashing import file_hash
from reelgrep.probe import probe
from reelgrep.timecode import format as format_timecode
from reelgrep.transcribe import TranscribeError, available_models, transcribe

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
    settings = get_settings()
    ensure_dirs(settings)
    video_resolved = video_path.expanduser().resolve()
    digest = file_hash(video_resolved)

    if no_db:
        # Don't even open the DB - just print to stdout.
        track = _do_transcribe(
            video_resolved,
            model_size,
            language,
            device,
            compute_type,
            beam_size,
            not no_vad_filter,
        )
        click.echo(
            json.dumps(
                {
                    "video": str(video_resolved),
                    "file_hash": digest,
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

    conn = connect(settings.db_path)
    try:
        migrate(conn)
        row = conn.execute(
            "SELECT id FROM videos WHERE file_hash = ?", (digest,)
        ).fetchone()
        if row is None:
            # Auto-create the videos row (metadata only).
            meta = probe(video_resolved)
            with conn:
                conn.execute(
                    """
                    INSERT INTO videos (
                        file_hash, path, duration_ms, width, height, fps,
                        container, video_codec, audio_codec, size_bytes,
                        ingested_at, probe_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        digest,
                        str(video_resolved),
                        meta.duration_ms,
                        meta.width,
                        meta.height,
                        meta.fps,
                        meta.format_name,
                        meta.video_codec,
                        meta.audio_codec,
                        meta.size_bytes,
                        datetime.now(UTC).isoformat(timespec="seconds"),
                        json.dumps(meta.raw),
                    ),
                )
            video_id = conn.execute(
                "SELECT id FROM videos WHERE file_hash = ?", (digest,)
            ).fetchone()[0]
        else:
            video_id = row[0]

        # Idempotency check.
        existing = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE video_id = ? AND source = 'whisper'",
            (video_id,),
        ).fetchone()[0]
        if existing > 0 and not force:
            click.echo(
                f"video already has {existing} whisper cue(s); pass --force to replace",
                err=True,
            )
            raise click.exceptions.Exit(0)
        if force and existing > 0:
            # Remove old whisper cues + their FTS rows.
            old_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM subtitles WHERE video_id = ? AND source = 'whisper'",
                    (video_id,),
                ).fetchall()
            ]
            with conn:
                conn.executemany(
                    "DELETE FROM subtitles_fts WHERE rowid = ?",
                    [(i,) for i in old_ids],
                )
                conn.execute(
                    "DELETE FROM subtitles WHERE video_id = ? AND source = 'whisper'",
                    (video_id,),
                )

        click.echo(
            f"transcribing {video_resolved.name} with whisper:{model_size}...",
            err=True,
        )
        track = _do_transcribe(
            video_resolved,
            model_size,
            language,
            device,
            compute_type,
            beam_size,
            not no_vad_filter,
        )

        with conn:
            for cue in track.cues:
                cur = conn.execute(
                    """
                    INSERT INTO subtitles (
                        video_id, language, source, stream_index,
                        start_ms, end_ms, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        video_id,
                        cue.language or track.language,
                        "whisper",
                        None,
                        cue.start_ms,
                        cue.end_ms,
                        cue.text,
                    ),
                )
                conn.execute(
                    "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
                    (cur.lastrowid, cue.text),
                )
    finally:
        conn.close()

    if track.cues:
        first = track.cues[0]
        last = track.cues[-1]
        span = f"{format_timecode(first.start_ms)} -> {format_timecode(last.end_ms)}"
    else:
        span = "(no cues)"

    click.echo(
        f"transcribed: {video_resolved}\n"
        f"language:    {track.language}\n"
        f"model:       whisper:{model_size}\n"
        f"cues:        {len(track.cues)}\n"
        f"span:        {span}"
    )


def _do_transcribe(
    video: Path,
    model_size: str,
    language: str | None,
    device: str,
    compute_type: str | None,
    beam_size: int,
    vad_filter: bool,
):
    """Wrap the core transcribe() call and surface TranscribeError as a click exit."""
    try:
        return transcribe(
            video,
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
    except TranscribeError as exc:
        click.echo(f"error: {exc}", err=True)
        raise click.exceptions.Exit(2) from exc


