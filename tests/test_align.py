"""Unit tests for reelgrep.align."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from reelgrep.align import (
    AlignError,
    AlignmentStats,
    align_transcript_to_cues,
    align_video,
)
from reelgrep.subtitles import SubtitleCue


def _cue(text: str, start_ms: int, end_ms: int) -> SubtitleCue:
    return SubtitleCue(start_ms=start_ms, end_ms=end_ms, text=text, language="en")


def test_empty_cues_raises() -> None:
    with pytest.raises(AlignError, match="no cues to align"):
        align_transcript_to_cues([], "Hello world.")


def test_empty_transcript_raises() -> None:
    with pytest.raises(AlignError, match="transcript is empty"):
        align_transcript_to_cues([_cue("hello", 0, 1000)], "   ")


def test_transcript_with_no_words_raises() -> None:
    with pytest.raises(AlignError, match="no words"):
        align_transcript_to_cues([_cue("hello", 0, 1000)], "!!! ??? ...")


def test_happy_path_strong_alignment_replaces_text() -> None:
    cues = [_cue("so we will now start on module one which focuses on databases", 0, 5000)]
    transcript = (
        "So we will now start on module one, which focuses on databases "
        "and database management systems."
    )
    new_cues, stats = align_transcript_to_cues(cues, transcript)
    assert len(new_cues) == 1
    aligned = new_cues[0]
    assert aligned.start_ms == 0
    assert aligned.end_ms == 5000
    assert "module one, which" in aligned.text
    assert stats.avg_similarity > 0.55


def test_low_similarity_falls_back_to_original_text() -> None:
    cues = [_cue("completely unrelated tax forms paperwork", 0, 2000)]
    transcript = "The quick brown fox jumps over a lazy dog repeatedly tonight."
    new_cues, stats = align_transcript_to_cues(cues, transcript, min_similarity=0.55)
    assert len(new_cues) == 1
    assert new_cues[0].text == "completely unrelated tax forms paperwork"
    assert new_cues[0].start_ms == 0
    assert new_cues[0].end_ms == 2000
    assert stats.avg_similarity < 0.55


def test_cursor_advances_across_sequential_cues() -> None:
    cues = [
        _cue("welcome to the lecture", 0, 1000),
        _cue("today we cover networking basics", 1000, 3000),
        _cue("the osi model has seven layers", 3000, 5000),
    ]
    transcript = (
        "Welcome to the lecture. Today we cover networking basics. "
        "The OSI model has seven layers."
    )
    new_cues, stats = align_transcript_to_cues(cues, transcript)
    assert len(new_cues) == 3
    assert "Welcome" in new_cues[0].text
    assert "Today we cover" in new_cues[1].text
    assert "OSI model" in new_cues[2].text
    assert stats.matched_word_count > 0


def test_alignment_stats_basic_counts() -> None:
    cues = [_cue("hello world", 0, 1000)]
    transcript = "Hello world, this is a transcript."
    _, stats = align_transcript_to_cues(cues, transcript)
    assert isinstance(stats, AlignmentStats)
    assert stats.cue_count == 1
    assert stats.transcript_word_count == 6
    assert stats.cue_word_count == 2
    assert 0.0 <= stats.coverage <= 1.0
    assert stats.avg_similarity > 0.0
    d = stats.to_dict()
    assert d["cue_count"] == 1
    assert "coverage" in d


def test_casing_and_punctuation_preserved() -> None:
    cues = [_cue("welcome to the talk", 0, 1000)]
    transcript = "Welcome to the talk."
    new_cues, _ = align_transcript_to_cues(cues, transcript)
    assert new_cues[0].text == "Welcome to the talk."


def test_language_passthrough() -> None:
    cues = [_cue("hello world", 0, 1000)]
    transcript = "Hello world."
    new_cues, _ = align_transcript_to_cues(cues, transcript, language="es")
    assert all(c.language == "es" for c in new_cues)


def test_empty_cue_text_kept_with_language_override() -> None:
    cues = [_cue("", 0, 1000)]
    transcript = "Hello world."
    new_cues, _ = align_transcript_to_cues(cues, transcript, language="fr")
    assert new_cues[0].text == ""
    assert new_cues[0].language == "fr"


def test_missing_rapidfuzz_raises_align_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy import to fail.
    monkeypatch.setitem(sys.modules, "rapidfuzz", None)
    cues = [_cue("hello", 0, 1000)]
    with pytest.raises(AlignError, match=r"\[align\] extra"):
        align_transcript_to_cues(cues, "Hello world.")


def test_align_video_returns_aligned_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript_text = "Hello world. This is the talk."

    import reelgrep.align as align_mod

    monkeypatch.setattr(align_mod, "load_transcript", lambda _p: transcript_text)

    cues = [_cue("hello world", 0, 1000), _cue("this is the talk", 1000, 2000)]
    video = tmp_path / "video.mp4"
    transcript = tmp_path / "transcript.txt"
    track, stats = align_video(video, transcript, cues, language="en")
    assert track.source == "aligned"
    assert track.format == "aligned"
    assert track.language == "en"
    assert len(track.cues) == 2
    assert stats.cue_count == 2


def test_align_video_infers_language_from_cues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript_text = "Hello world."
    import reelgrep.align as align_mod

    monkeypatch.setattr(align_mod, "load_transcript", lambda _p: transcript_text)
    cues = [_cue("hello world", 0, 1000)]
    track, _ = align_video(tmp_path / "v.mp4", tmp_path / "t.txt", cues)
    assert track.language == "en"


def test_out_of_transcript_cues_keep_original_text() -> None:
    cues = [
        _cue("welcome everyone today", 0, 1000),
        _cue("a totally different unrelated phrase here", 1000, 2000),
    ]
    transcript = "Welcome everyone today."
    new_cues, _ = align_transcript_to_cues(cues, transcript)
    assert "Welcome everyone today" in new_cues[0].text
    # Second cue runs out of transcript or fails similarity - keeps original.
    assert new_cues[1].text == "a totally different unrelated phrase here"
    assert new_cues[1].start_ms == 1000
    assert new_cues[1].end_ms == 2000


def test_coverage_within_bounds() -> None:
    cues = [_cue("hello world today", 0, 1000)]
    transcript = "Hello world today, the talk begins."
    _, stats = align_transcript_to_cues(cues, transcript)
    assert 0.0 <= stats.coverage <= 1.0
