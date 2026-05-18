"""Library-level ingest API for reelgrep.

This module exposes :func:`ingest_video`, the callable form of the
``reelgrep ingest`` command. The Click CLI in
:mod:`reelgrep.commands.ingest` is a thin wrapper around this function;
both share the exact same logic.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from reelgrep.backends import get_backend
from reelgrep.config import (
    ensure_dirs,
    get_db_override,
    get_settings,
    set_db_override,
)
from reelgrep.db import connect, migrate
from reelgrep.ffmpeg_exec import FFmpegError
from reelgrep.frames import sample_every
from reelgrep.hashing import file_hash
from reelgrep.probe import probe
from reelgrep.subtitles import (
    SubtitleTrack,
    extract_embedded,
    find_sidecars,
    parse_sidecar,
)
from reelgrep.transcribe import TranscribeError
from reelgrep.transcribe import transcribe as run_whisper

__all__ = ["IngestResult", "IngestWarning", "ingest_video", "hash_slice"]

logger = logging.getLogger(__name__)


def hash_slice(digest: str) -> str:
    """Return the deterministic cache subdirectory name for a file hash.

    Matches the slicing used by the historical CLI: characters 8..24 of the
    ``blake2b:<hex>`` digest (the first 16 hex characters of the digest).
    """
    if digest.startswith("blake2b:"):
        return digest[8:24]
    return digest[:16]


@dataclass(frozen=True)
class IngestWarning:
    """A non-fatal warning emitted during ingest."""

    stage: str
    message: str


@dataclass
class IngestResult:
    """Return value from :func:`ingest_video`."""

    video_path: Path
    file_hash: str
    db_path: Path
    duration_ms: int
    video_id: int | None
    already_ingested: bool
    subtitle_track_count: int
    subtitle_cue_count: int
    transcribed_cue_count: int
    transcribe_model: str | None
    frame_count: int
    warnings: list[IngestWarning] = field(default_factory=list)
    previously_indexed_path: Path | None = None


def ingest_video(
    path: str | os.PathLike,
    *,
    backend: str = "local",
    db_path: str | os.PathLike | None = None,
    interval_seconds: float = 5.0,
    no_subtitles: bool = False,
    no_frames: bool = False,
    force: bool = False,
    do_transcribe: bool = False,
    transcribe_model: str = "small",
    on_message: Callable[[str], None] | None = None,
) -> IngestResult:
    """Probe a video, extract subtitles + frames, and persist to the index.

    Parameters
    ----------
    path:
        Identifier for the video. Interpreted by the chosen ``backend``.
        For the default ``"local"`` backend this is the absolute or
        relative path to a file on disk.
    backend:
        Registered backend name from :mod:`reelgrep.backends`. The backend
        resolves ``path`` to an absolute local file path before probing.
    db_path:
        Optional override for the index database location. When ``None``,
        the resolved value from :func:`reelgrep.config.get_settings` is
        used (which honours ``REELGREP_DB`` and ``REELGREP_HOME``). When
        provided, the override is set via
        :func:`reelgrep.config.set_db_override` for the duration of this
        call and the prior override (if any) is restored on return.
    interval_seconds:
        Uniform frame-sampling interval in seconds. Ignored when
        ``no_frames`` is True.
    no_subtitles:
        Skip embedded + sidecar subtitle extraction. Also disables the
        Whisper transcription step regardless of ``do_transcribe``.
    no_frames:
        Skip uniform frame sampling.
    force:
        Re-ingest even when a row with the same file hash already exists.
        Deletes the previous row + cached subtitle/frame artifacts.
    do_transcribe:
        After subtitle extraction, run Whisper transcription when no
        embedded or sidecar cues were inserted.
    transcribe_model:
        Whisper model size identifier passed to
        :func:`reelgrep.transcribe.transcribe`.
    on_message:
        Optional callback invoked with progress + warning strings as they
        occur. Used by the Click wrapper to preserve interleaved stderr
        output. Each message is also recorded on the result so library
        callers can inspect them after the call.

    Returns
    -------
    :class:`IngestResult` describing what was persisted.
    """
    def _emit(message: str) -> None:
        if on_message is not None:
            on_message(message)

    prior_db_override: Path | None = None
    restore_db_override = False
    if db_path is not None:
        prior_db_override = get_db_override()
        set_db_override(db_path)
        restore_db_override = True

    try:
        settings = get_settings()
        ensure_dirs(settings)

        resolver = get_backend(backend)
        resolved = resolver.resolve(str(path))

        digest = file_hash(resolved)
        slice_name = hash_slice(digest)
        subs_dir = settings.cache_dir / "subtitles" / slice_name
        frames_dir = settings.cache_dir / "frames" / slice_name

        warnings_out: list[IngestWarning] = []

        conn = connect(settings.db_path)
        try:
            migrate(conn)

            existing = conn.execute(
                "SELECT id, path, duration_ms FROM videos WHERE file_hash = ?",
                (digest,),
            ).fetchone()

            if existing is not None and not force:
                return IngestResult(
                    video_path=resolved,
                    file_hash=digest,
                    db_path=settings.db_path,
                    duration_ms=int(existing["duration_ms"]),
                    video_id=int(existing["id"]),
                    already_ingested=True,
                    subtitle_track_count=0,
                    subtitle_cue_count=0,
                    transcribed_cue_count=0,
                    transcribe_model=None,
                    frame_count=0,
                    warnings=warnings_out,
                    previously_indexed_path=Path(existing["path"]),
                )

            if existing is not None and force:
                with conn:
                    conn.execute(
                        "DELETE FROM videos WHERE id = ?", (existing["id"],)
                    )
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

            tracks: list[SubtitleTrack] = []
            total_cues = 0
            transcribed_cues = 0

            if not no_subtitles:
                subs_dir.mkdir(parents=True, exist_ok=True)
                try:
                    embedded = extract_embedded(resolved, subs_dir)
                    tracks.extend(embedded)
                except FFmpegError as exc:
                    msg = f"failed to extract embedded subtitles: {exc}"
                    warnings_out.append(
                        IngestWarning(stage="embedded_subtitles", message=msg)
                    )
                    _emit(f"warning: {msg}")

                for sidecar_path in find_sidecars(resolved):
                    try:
                        sidecar_track = parse_sidecar(sidecar_path)
                    except Exception as exc:  # noqa: BLE001 - sidecar parsers raise varied errors
                        msg = f"failed to parse sidecar {sidecar_path}: {exc}"
                        warnings_out.append(
                            IngestWarning(stage="sidecar_subtitles", message=msg)
                        )
                        _emit(f"warning: {msg}")
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

            emitted_transcribe_model: str | None = None
            if do_transcribe and not no_subtitles:
                # Only transcribe when no embedded or sidecar subs were inserted.
                existing_subs = conn.execute(
                    "SELECT COUNT(*) FROM subtitles WHERE video_id = ?",
                    (video_id,),
                ).fetchone()[0]
                if existing_subs == 0:
                    emitted_transcribe_model = transcribe_model
                    logger.info(
                        "no subs found; transcribing with whisper:%s",
                        transcribe_model,
                    )
                    _emit(
                        f"no subs found; transcribing with whisper:{transcribe_model}..."
                    )
                    try:
                        whisper_track = run_whisper(
                            resolved, model_size=transcribe_model
                        )
                    except TranscribeError as exc:
                        msg = f"transcribe failed: {exc}"
                        warnings_out.append(
                            IngestWarning(stage="transcribe", message=msg)
                        )
                        _emit(msg)
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

        return IngestResult(
            video_path=resolved,
            file_hash=digest,
            db_path=settings.db_path,
            duration_ms=meta.duration_ms,
            video_id=int(video_id) if video_id is not None else None,
            already_ingested=False,
            subtitle_track_count=len(tracks),
            subtitle_cue_count=total_cues,
            transcribed_cue_count=transcribed_cues,
            transcribe_model=emitted_transcribe_model if transcribed_cues else None,
            frame_count=len(sampled_frames),
            warnings=warnings_out,
        )
    finally:
        if restore_db_override:
            set_db_override(prior_db_override)
