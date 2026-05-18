"""Map an official prose transcript onto existing cue timestamps via sequence alignment."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from reelgrep.subtitles import SubtitleCue, SubtitleTrack
from reelgrep.transcript_loaders import load_transcript, split_into_words

__all__ = ["AlignError", "AlignmentStats", "align_transcript_to_cues", "align_video"]

logger = logging.getLogger(__name__)


class AlignError(RuntimeError):
    """Raised when alignment fails (empty transcript, no overlap, etc.)."""


class AlignmentStats:
    """Diagnostic counters from one alignment run."""

    def __init__(self) -> None:
        self.cue_count: int = 0
        self.transcript_word_count: int = 0
        self.cue_word_count: int = 0
        self.matched_word_count: int = 0
        self.avg_similarity: float = 0.0
        self.coverage: float = 0.0  # fraction of transcript words consumed

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats for logging or JSON output."""
        return {
            "cue_count": self.cue_count,
            "transcript_word_count": self.transcript_word_count,
            "cue_word_count": self.cue_word_count,
            "matched_word_count": self.matched_word_count,
            "avg_similarity": round(self.avg_similarity, 3),
            "coverage": round(self.coverage, 3),
        }


def align_transcript_to_cues(
    cues: list[SubtitleCue],
    transcript: str,
    *,
    language: str | None = None,
    min_similarity: float = 0.55,
) -> tuple[list[SubtitleCue], AlignmentStats]:
    """Replace each cue's text with the best-matching span from the transcript."""
    try:
        from rapidfuzz import fuzz
    except ImportError as exc:
        raise AlignError(
            "alignment requires the [align] extra: pip install reelgrep[align]"
        ) from exc

    if not cues:
        raise AlignError("no cues to align against (run reelgrep transcribe first)")
    if not transcript.strip():
        raise AlignError("transcript is empty")

    # Build a word-level index over the transcript while preserving char offsets so
    # we can reconstruct casing + punctuation later.
    word_re = re.compile(r"[a-zA-Z0-9']+")
    word_spans = [
        (m.group(0).lower(), m.start(), m.end()) for m in word_re.finditer(transcript)
    ]
    transcript_words = [w for w, _, _ in word_spans]
    if not transcript_words:
        raise AlignError("transcript has no words after tokenization")

    stats = AlignmentStats()
    stats.cue_count = len(cues)
    stats.transcript_word_count = len(transcript_words)

    cursor = 0
    new_cues: list[SubtitleCue] = []
    similarities: list[float] = []

    for cue in cues:
        cue_words = split_into_words(cue.text)
        stats.cue_word_count += len(cue_words)
        if not cue_words:
            new_cues.append(cue.model_copy(update={"language": language or cue.language}))
            continue

        target_len = len(cue_words)
        window_len = max(target_len * 3, 12)
        window_start = cursor
        window_end = min(cursor + window_len, len(transcript_words))

        if window_start >= len(transcript_words):
            # Out of transcript - keep original cue text
            new_cues.append(cue.model_copy(update={"language": language or cue.language}))
            continue

        # Slide a target-length window within the search window, score each.
        best_score = -1.0
        best_span = (window_start, window_start + target_len)
        cue_joined = " ".join(cue_words)
        for span_start in range(
            window_start, max(window_start + 1, window_end - target_len + 1)
        ):
            span_end = min(span_start + target_len, len(transcript_words))
            candidate = " ".join(transcript_words[span_start:span_end])
            if not candidate:
                continue
            score = fuzz.ratio(cue_joined, candidate)
            if score > best_score:
                best_score = score
                best_span = (span_start, span_end)

        norm_score = best_score / 100.0
        similarities.append(norm_score)

        if norm_score < min_similarity:
            # Low confidence - keep original Whisper text, do NOT advance cursor.
            new_cues.append(cue.model_copy(update={"language": language or cue.language}))
            continue

        # Reconstruct original substring from char offsets so casing/punctuation survive.
        ws, we = best_span
        if ws < len(word_spans) and we <= len(word_spans):
            char_start = word_spans[ws][1]
            char_end = word_spans[we - 1][2] if we > ws else char_start
            # Extend char_end to include trailing punctuation attached to the last
            # word (e.g. "talk." or "module,") so we don't drop sentence terminators.
            while char_end < len(transcript) and transcript[char_end] in ".,!?;:)\"']":
                char_end += 1
            text = transcript[char_start:char_end].strip()
        else:
            text = " ".join(transcript_words[ws:we])

        new_cues.append(
            SubtitleCue(
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=text,
                language=language or cue.language,
            )
        )
        cursor = we
        stats.matched_word_count += we - ws

    if similarities:
        stats.avg_similarity = sum(similarities) / len(similarities)
    stats.coverage = stats.matched_word_count / max(stats.transcript_word_count, 1)
    return new_cues, stats


def align_video(
    video_path: str | Path,
    transcript_path: str | Path,
    cues: list[SubtitleCue],
    *,
    language: str | None = None,
    min_similarity: float = 0.55,
) -> tuple[SubtitleTrack, AlignmentStats]:
    """High-level entry: load a transcript file and align it to the given cues."""
    transcript_text = load_transcript(transcript_path)
    new_cues, stats = align_transcript_to_cues(
        cues,
        transcript_text,
        language=language,
        min_similarity=min_similarity,
    )
    return (
        SubtitleTrack(
            source="aligned",
            stream_index=None,
            language=language or (cues[0].language if cues else None),
            format="aligned",
            cues=new_cues,
        ),
        stats,
    )
