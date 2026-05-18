"""CLI command to align an official prose transcript to existing video cues."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import click

from reelgrep.align import AlignError, align_video
from reelgrep.config import ensure_dirs, get_settings
from reelgrep.db import connect, migrate
from reelgrep.hashing import file_hash
from reelgrep.probe import probe
from reelgrep.subtitles import SubtitleCue
from reelgrep.transcribe import TranscribeError
from reelgrep.transcribe import transcribe as run_whisper
from reelgrep.transcript_loaders import TranscriptLoadError

__all__ = ["align_cmd"]


@click.command("align")
@click.argument(
    "video_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--transcript",
    "transcript_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the official prose transcript (.txt, .md, .pdf).",
)
@click.option(
    "--language",
    default=None,
    help="ISO 639-1 code; defaults to the existing cues' language.",
)
@click.option(
    "--min-similarity",
    type=float,
    default=0.55,
    show_default=True,
    help="Below this threshold a cue keeps its original Whisper text.",
)
@click.option(
    "--auto-transcribe/--no-auto-transcribe",
    default=True,
    show_default=True,
    help="Run whisper:tiny if no cues exist for the video yet.",
)
@click.option(
    "--whisper-model",
    default="tiny",
    show_default=True,
    help="Model used for the auto-transcribe fallback.",
)
@click.option(
    "--out",
    "srt_out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also write the aligned cues to this .srt file.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Replace existing aligned cues for this video.",
)
@click.option(
    "--no-db",
    is_flag=True,
    default=False,
    help="Print aligned cues as JSON instead of writing to the index.",
)
def align_cmd(
    video_path: Path,
    transcript_path: Path,
    language: str | None,
    min_similarity: float,
    auto_transcribe: bool,
    whisper_model: str,
    srt_out: Path | None,
    force: bool,
    no_db: bool,
) -> None:
    """Align an official transcript to existing cue timestamps."""
    settings = get_settings()
    ensure_dirs(settings)
    video_resolved = video_path.expanduser().resolve()
    digest = file_hash(video_resolved)

    # Always need the cues - even for --no-db we want existing Whisper cues.
    conn = connect(settings.db_path)
    try:
        migrate(conn)
        video_row = conn.execute(
            "SELECT id FROM videos WHERE file_hash = ?", (digest,)
        ).fetchone()
        if video_row is None:
            # Auto-create the videos row from probe data.
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
            video_id = video_row[0]

        # Load existing cues - prefer non-aligned (Whisper/embedded/sidecar).
        cue_rows = conn.execute(
            "SELECT language, source, start_ms, end_ms, text FROM subtitles "
            "WHERE video_id = ? AND source != 'aligned' ORDER BY start_ms",
            (video_id,),
        ).fetchall()

        if not cue_rows and auto_transcribe:
            click.echo(
                f"no cues found; auto-transcribing with whisper:{whisper_model}...",
                err=True,
            )
            try:
                wh_track = run_whisper(video_resolved, model_size=whisper_model)
            except TranscribeError as exc:
                click.echo(f"auto-transcribe failed: {exc}", err=True)
                raise click.exceptions.Exit(2) from exc
            with conn:
                for cue in wh_track.cues:
                    cur = conn.execute(
                        """
                        INSERT INTO subtitles (
                            video_id, language, source, stream_index,
                            start_ms, end_ms, text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            video_id,
                            cue.language or wh_track.language,
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
            cue_rows = conn.execute(
                "SELECT language, source, start_ms, end_ms, text FROM subtitles "
                "WHERE video_id = ? AND source != 'aligned' ORDER BY start_ms",
                (video_id,),
            ).fetchall()

        if not cue_rows:
            click.echo(
                "no cues to align against; run `reelgrep transcribe` first or "
                "pass --auto-transcribe",
                err=True,
            )
            raise click.exceptions.Exit(2)

        cues = [
            SubtitleCue(
                start_ms=int(r["start_ms"]),
                end_ms=int(r["end_ms"]),
                text=str(r["text"]),
                language=r["language"],
            )
            for r in cue_rows
        ]

        try:
            track, stats = align_video(
                video_resolved,
                transcript_path,
                cues,
                language=language,
                min_similarity=min_similarity,
            )
        except (AlignError, TranscriptLoadError) as exc:
            click.echo(f"error: {exc}", err=True)
            raise click.exceptions.Exit(2) from exc

        if no_db:
            click.echo(
                json.dumps(
                    {
                        "video": str(video_resolved),
                        "file_hash": digest,
                        "transcript": str(transcript_path),
                        "language": track.language,
                        "stats": stats.to_dict(),
                        "cues": [
                            {
                                "start_ms": c.start_ms,
                                "end_ms": c.end_ms,
                                "text": c.text,
                            }
                            for c in track.cues
                        ],
                    },
                    indent=2,
                )
            )
            return

        existing_aligned = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE video_id = ? AND source = 'aligned'",
            (video_id,),
        ).fetchone()[0]
        if existing_aligned > 0 and not force:
            click.echo(
                f"video already has {existing_aligned} aligned cue(s); "
                "pass --force to replace",
                err=True,
            )
            raise click.exceptions.Exit(0)
        if force and existing_aligned > 0:
            old_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM subtitles WHERE video_id = ? AND source = 'aligned'",
                    (video_id,),
                ).fetchall()
            ]
            with conn:
                conn.executemany(
                    "DELETE FROM subtitles_fts WHERE rowid = ?",
                    [(i,) for i in old_ids],
                )
                conn.execute(
                    "DELETE FROM subtitles WHERE video_id = ? AND source = 'aligned'",
                    (video_id,),
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
                        "aligned",
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

    if srt_out is not None:
        srt_out.parent.mkdir(parents=True, exist_ok=True)
        with srt_out.open("w", encoding="utf-8") as f:
            for i, cue in enumerate(track.cues, start=1):
                f.write(f"{i}\n")
                f.write(
                    f"{_srt_timestamp(cue.start_ms)} --> "
                    f"{_srt_timestamp(cue.end_ms)}\n"
                )
                f.write(f"{cue.text}\n\n")

    click.echo(
        f"aligned:        {video_resolved}\n"
        f"transcript:     {transcript_path}\n"
        f"language:       {track.language}\n"
        f"cues:           {len(track.cues)} "
        f"(matched {stats.matched_word_count}/{stats.transcript_word_count} "
        f"transcript words, coverage {stats.coverage:.1%})\n"
        f"avg similarity: {stats.avg_similarity:.2f}"
        + (f"\nsrt:            {srt_out}" if srt_out else "")
    )


def _srt_timestamp(ms: int) -> str:
    h = ms // 3600000
    m = (ms // 60000) % 60
    s = (ms // 1000) % 60
    msec = ms % 1000
    return f"{h:02}:{m:02}:{s:02},{msec:03}"
