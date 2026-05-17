"""Whisper-based transcription of videos without embedded subtitles."""

from __future__ import annotations

import logging
from pathlib import Path

from reelgrep.subtitles import SubtitleCue, SubtitleTrack

__all__ = ["TranscribeError", "transcribe", "available_models"]

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
