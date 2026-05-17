"""``reelgrep ingest`` command: probe, extract subtitles, sample frames, persist."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import click

from reelgrep import timecode
from reelgrep.config import ensure_dirs, get_settings
from reelgrep.db import connect, migrate
from reelgrep.ffmpeg_exec import FFmpegError
from reelgrep.frames import sample_every
from reelgrep.hashing import file_hash
from reelgrep.probe import probe
from reelgrep.subtitles import extract_embedded, find_sidecars, parse_sidecar
from reelgrep.transcribe import TranscribeError
from reelgrep.transcribe import transcribe as run_whisper

__all__ = ["ingest"]


def _hash_slice(digest: str) -> str:
    """Return the deterministic cache subdirectory name for a file hash."""
    if digest.startswith("blake2b:"):
        hex_part = digest[len("blake2b:") :]
    else:
        hex_part = digest
    # Match the original spec's slicing on the full ``blake2b:<hex>`` digest,
    # which yields characters 8..24 of the prefixed string (the first 16 hex
    # characters of the digest).
    return digest[8:24] if digest.startswith("blake2b:") else hex_part[:16]


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
    resolved = video_path.resolve()
    digest = file_hash(resolved)

    settings = get_settings()
    ensure_dirs(settings)

    slice_name = _hash_slice(digest)
    subs_dir = settings.cache_dir / "subtitles" / slice_name
    frames_dir = settings.cache_dir / "frames" / slice_name

    conn = connect(settings.db_path)
    try:
        migrate(conn)

        existing = conn.execute(
            "SELECT id, path FROM videos WHERE file_hash = ?",
            (digest,),
        ).fetchone()

        if existing is not None and not force:
            click.echo(f"Already ingested: {digest} ({existing['path']})")
            return

        if existing is not None and force:
            with conn:
                conn.execute("DELETE FROM videos WHERE id = ?", (existing["id"],))
            shutil.rmtree(subs_dir, ignore_errors=True)
            shutil.rmtree(frames_dir, ignore_errors=True)

        meta = probe(resolved)
        ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
        probe_json = json.dumps(meta.raw)

        with conn:
            cursor = conn.execute(
                """
                INSERT INTO videos (
                    file_hash, path, duration_ms, width, height, fps,
                    container, video_codec, audio_codec, size_bytes,
                    ingested_at, probe_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    digest,
                    str(resolved),
                    meta.duration_ms,
                    meta.width,
                    meta.height,
                    meta.fps,
                    meta.format_name,
                    meta.video_codec,
                    meta.audio_codec,
                    meta.size_bytes,
                    ingested_at,
                    probe_json,
                ),
            )
            video_id = cursor.lastrowid

        tracks: list = []
        total_cues = 0
        transcribed_cues = 0

        if not no_subtitles:
            subs_dir.mkdir(parents=True, exist_ok=True)
            try:
                embedded = extract_embedded(resolved, subs_dir)
                tracks.extend(embedded)
            except FFmpegError as exc:
                click.echo(
                    f"warning: failed to extract embedded subtitles: {exc}",
                    err=True,
                )

            for sidecar_path in find_sidecars(resolved):
                try:
                    sidecar_track = parse_sidecar(sidecar_path)
                except Exception as exc:  # noqa: BLE001 - sidecar parsers raise varied errors
                    click.echo(
                        f"warning: failed to parse sidecar {sidecar_path}: {exc}",
                        err=True,
                    )
                    continue
                tracks.append(sidecar_track)

            with conn:
                for track in tracks:
                    for cue in track.cues:
                        sub_cursor = conn.execute(
                            """
                            INSERT INTO subtitles (
                                video_id, language, source, stream_index,
                                start_ms, end_ms, text
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                video_id,
                                cue.language or track.language,
                                track.source,
                                track.stream_index,
                                cue.start_ms,
                                cue.end_ms,
                                cue.text,
                            ),
                        )
                        conn.execute(
                            "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
                            (sub_cursor.lastrowid, cue.text),
                        )
                        total_cues += 1

        if do_transcribe and not no_subtitles:
            # Only transcribe when no embedded or sidecar subs were inserted.
            existing_subs = conn.execute(
                "SELECT COUNT(*) FROM subtitles WHERE video_id = ?",
                (video_id,),
            ).fetchone()[0]
            if existing_subs == 0:
                click.echo(
                    f"no subs found; transcribing with whisper:{transcribe_model}...",
                    err=True,
                )
                try:
                    whisper_track = run_whisper(
                        resolved, model_size=transcribe_model
                    )
                except TranscribeError as exc:
                    click.echo(f"transcribe failed: {exc}", err=True)
                else:
                    with conn:
                        for cue in whisper_track.cues:
                            wcur = conn.execute(
                                """
                                INSERT INTO subtitles (
                                    video_id, language, source, stream_index,
                                    start_ms, end_ms, text
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    video_id,
                                    cue.language or whisper_track.language,
                                    "whisper",
                                    None,
                                    cue.start_ms,
                                    cue.end_ms,
                                    cue.text,
                                ),
                            )
                            conn.execute(
                                "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
                                (wcur.lastrowid, cue.text),
                            )
                            transcribed_cues += 1

        sampled_frames: list = []
        if not no_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)
            sampled_frames = sample_every(
                resolved, frames_dir, interval_seconds=interval_seconds
            )
            with conn:
                for frame in sampled_frames:
                    conn.execute(
                        """
                        INSERT INTO frames (
                            video_id, timestamp_ms, path, sampling_strategy,
                            width, height
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            video_id,
                            frame.timestamp_ms,
                            frame.path,
                            frame.sampling_strategy,
                            frame.width,
                            frame.height,
                        ),
                    )
    finally:
        conn.close()

    click.echo(f"ingested: {resolved}")
    click.echo(f"hash:     {digest}")
    click.echo(f"duration: {timecode.format(meta.duration_ms)}")
    click.echo(f"subtitle tracks: {len(tracks)} (cues: {total_cues})")
    if transcribed_cues:
        click.echo(f"transcribed {transcribed_cues} cues with whisper:{transcribe_model}")
    click.echo(f"frames sampled: {len(sampled_frames)}")
    click.echo(f"db:       {settings.db_path}")
