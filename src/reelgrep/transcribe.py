"""Whisper-based transcription of videos without embedded subtitles.

Two entry points are exposed here:

* :func:`transcribe` is the low-level inference primitive. Given a video
  path, it runs faster-whisper and returns a :class:`SubtitleTrack`
  in memory; it does not touch the index database.
* :func:`transcribe_video` is the library-grade wrapper most callers
  want. It runs the same inference and persists the resulting cues into
  the reelgrep index, ensuring the ``videos`` row exists and honouring
  the same ``db_path`` / idempotency / ``force`` discipline as the CLI.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from reelgrep.config import (
    ensure_dirs,
    get_db_override,
    get_settings,
    set_db_override,
)
from reelgrep.db import connect, migrate
from reelgrep.hashing import file_hash
from reelgrep.probe import probe
from reelgrep.subtitles import SubtitleCue, SubtitleTrack

__all__ = [
    "TranscribeError",
    "TranscribeResult",
    "available_models",
    "transcribe",
    "transcribe_video",
]

logger = logging.getLogger(__name__)


AVAILABLE_MODELS = (
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "large-v3-turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v3",
)


class TranscribeError(RuntimeError):
    """Raised when whisper transcription cannot run or fails."""


@dataclass(frozen=True)
class TranscribeResult:
    """Return value from :func:`transcribe_video`.

    Fields
    ------
    video_path:
        The resolved absolute path to the video that was transcribed.
    file_hash:
        Content hash of the video (``blake2b:<hex>``).
    video_id:
        Primary-key id of the row in the ``videos`` table.
    db_path:
        Filesystem path to the SQLite index that was written to.
    model:
        Whisper model size identifier that produced the cues. When
        ``already_transcribed`` is ``True`` and no inference was run,
        this is ``None`` because the stored cues may pre-date this call.
    language_detected:
        Best-guess ISO-639 language code. ``None`` when no inference
        was run (e.g. ``already_transcribed=True`` short-circuit).
    cue_count:
        Number of subtitle cues attributable to whisper for this video
        in the index after the call. When ``already_transcribed`` is
        ``True``, this is the count of pre-existing whisper rows
        (no new writes happened).
    duration_ms:
        The video's duration in milliseconds, as reported by ffprobe.
    already_transcribed:
        ``True`` when this video already had whisper cues at call time
        and ``force=False``, so no inference or writes occurred.
    """

    video_path: Path
    file_hash: str
    video_id: int
    db_path: Path
    model: str | None
    language_detected: str | None
    cue_count: int
    duration_ms: int
    already_transcribed: bool


def available_models() -> tuple[str, ...]:
    """Return the tuple of model size identifiers reelgrep recognizes."""
    return AVAILABLE_MODELS


def transcribe(
    video_path: str | Path,
    *,
    model_size: str = "small",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> SubtitleTrack:
    """Transcribe a video to a SubtitleTrack via faster-whisper."""
    if model_size not in AVAILABLE_MODELS:
        raise TranscribeError(
            f"unknown model_size {model_size!r}; choose one of {AVAILABLE_MODELS}"
        )
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError(
            "transcribe requires the [whisper] extra: "
            "pip install reelgrep[whisper] (installs faster-whisper + ctranslate2)"
        ) from exc

    resolved = Path(video_path).expanduser().resolve(strict=True)

    # Default compute_type per device. faster-whisper recommends int8 for CPU.
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "float16"

    logger.info(
        "loading whisper model %s on %s (%s); this may download ~%s on first use",
        model_size, device, compute_type, _approx_model_size(model_size),
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_iter, info = model.transcribe(
        str(resolved),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=False,
    )
    detected_language = getattr(info, "language", None) or language or "unknown"

    cues: list[SubtitleCue] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        cues.append(SubtitleCue(
            start_ms=int(round(seg.start * 1000)),
            end_ms=int(round(seg.end * 1000)),
            text=text,
            language=detected_language,
        ))

    return SubtitleTrack(
        source="whisper",
        stream_index=None,
        language=detected_language,
        format="whisper",
        cues=cues,
    )


def transcribe_video(
    path: str | os.PathLike[str],
    *,
    model: str = "small",
    language: str | None = None,
    db_path: str | os.PathLike[str] | None = None,
    force: bool = False,
    device: str = "cpu",
    compute_type: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    on_message: Callable[[str], None] | None = None,
) -> TranscribeResult:
    """Run Whisper on ``path`` and persist the cues to the reelgrep index.

    This is the public library-grade entry point. It composes:

    1. Resolve the video path (raises :class:`FileNotFoundError` if missing).
    2. Resolve the target db (honours an explicit ``db_path`` via
       :func:`reelgrep.config.set_db_override`, which is restored on
       return so the call has no sticky process-level side effects).
    3. Probe the video and ensure a row exists in ``videos``.
    4. Check for existing whisper cues. When present and ``force`` is
       ``False``, short-circuit and return a result with
       ``already_transcribed=True`` and ``cue_count`` reflecting the
       existing rows; no inference is run.
    5. When ``force`` is ``True`` and whisper cues already exist, delete
       them (and their FTS rows) before re-transcribing.
    6. Call :func:`transcribe` to produce a :class:`SubtitleTrack` and
       write each cue into ``subtitles`` + ``subtitles_fts``.

    Parameters
    ----------
    path:
        Path on disk to the video file. Must exist.
    model:
        Whisper model size identifier (see :func:`available_models`).
    language:
        ISO-639 language hint forwarded to whisper. ``None`` enables
        auto-detection.
    db_path:
        Optional override for the index database location. When
        ``None``, the resolved value from
        :func:`reelgrep.config.get_settings` is used. When provided,
        the file MUST exist; :class:`FileNotFoundError` is raised
        otherwise. Mirrors the :class:`reelgrep.search.Search`
        contract.
    force:
        When ``True``, replace any existing whisper cues for this
        video before re-transcribing. When ``False`` (the default),
        a video that already has whisper cues short-circuits and
        returns ``already_transcribed=True`` without running
        inference.
    device, compute_type, beam_size, vad_filter:
        Forwarded to :func:`transcribe`.
    on_message:
        Optional callback invoked with progress strings as they occur.
        Used by the Click wrapper to preserve interleaved stderr output;
        library callers can leave it unset.

    Returns
    -------
    :class:`TranscribeResult` summarising what was persisted.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist on disk, or if an explicit
        ``db_path`` does not point at an existing file.
    TranscribeError
        If model_size is unknown or the whisper extra is not
        installed.
    """
    resolved_video = Path(path).expanduser().resolve(strict=True)

    if db_path is not None:
        explicit_db = Path(db_path).expanduser().resolve()
        if not explicit_db.exists():
            raise FileNotFoundError(
                f"reelgrep index not found at {explicit_db}"
            )

    prior_db_override: Path | None = None
    restore_db_override = False
    if db_path is not None:
        prior_db_override = get_db_override()
        set_db_override(db_path)
        restore_db_override = True

    try:
        settings = get_settings()
        ensure_dirs(settings)

        digest = file_hash(resolved_video)

        conn = connect(settings.db_path)
        try:
            migrate(conn)

            row = conn.execute(
                "SELECT id FROM videos WHERE file_hash = ?", (digest,)
            ).fetchone()
            if row is None:
                meta = probe(resolved_video)
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
                            str(resolved_video),
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
                video_id = int(
                    conn.execute(
                        "SELECT id FROM videos WHERE file_hash = ?", (digest,)
                    ).fetchone()[0]
                )
                duration_ms = int(meta.duration_ms)
            else:
                video_id = int(row[0])
                duration_ms = int(
                    conn.execute(
                        "SELECT duration_ms FROM videos WHERE id = ?",
                        (video_id,),
                    ).fetchone()[0]
                )

            existing = conn.execute(
                "SELECT COUNT(*) FROM subtitles "
                "WHERE video_id = ? AND source = 'whisper'",
                (video_id,),
            ).fetchone()[0]

            if existing > 0 and not force:
                return TranscribeResult(
                    video_path=resolved_video,
                    file_hash=digest,
                    video_id=video_id,
                    db_path=settings.db_path,
                    model=None,
                    language_detected=None,
                    cue_count=int(existing),
                    duration_ms=duration_ms,
                    already_transcribed=True,
                )

            if force and existing > 0:
                old_ids = [
                    r[0]
                    for r in conn.execute(
                        "SELECT id FROM subtitles "
                        "WHERE video_id = ? AND source = 'whisper'",
                        (video_id,),
                    ).fetchall()
                ]
                with conn:
                    conn.executemany(
                        "DELETE FROM subtitles_fts WHERE rowid = ?",
                        [(i,) for i in old_ids],
                    )
                    conn.execute(
                        "DELETE FROM subtitles "
                        "WHERE video_id = ? AND source = 'whisper'",
                        (video_id,),
                    )

            if on_message is not None:
                on_message(
                    f"transcribing {resolved_video.name} with whisper:{model}..."
                )

            track = transcribe(
                resolved_video,
                model_size=model,
                language=language,
                device=device,
                compute_type=compute_type,
                beam_size=beam_size,
                vad_filter=vad_filter,
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

            return TranscribeResult(
                video_path=resolved_video,
                file_hash=digest,
                video_id=video_id,
                db_path=settings.db_path,
                model=model,
                language_detected=track.language,
                cue_count=len(track.cues),
                duration_ms=duration_ms,
                already_transcribed=False,
            )
        finally:
            conn.close()
    finally:
        if restore_db_override:
            set_db_override(prior_db_override)


def _approx_model_size(model_size: str) -> str:
    """Rough size hint used in the log line."""
    return {
        "tiny": "75MB", "tiny.en": "75MB",
        "base": "140MB", "base.en": "140MB",
        "small": "240MB", "small.en": "240MB",
        "medium": "770MB", "medium.en": "770MB",
        "large-v1": "2.9GB", "large-v2": "2.9GB", "large-v3": "2.9GB",
        "large-v3-turbo": "1.5GB",
        "distil-small.en": "180MB",
        "distil-medium.en": "400MB",
        "distil-large-v3": "1.2GB",
    }.get(model_size, "?")
