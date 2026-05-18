"""Tests for the public ``reelgrep.transcribe.transcribe_video`` library API."""

from __future__ import annotations

import sqlite3
import sys
import types
from collections.abc import Iterator
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import pytest

from reelgrep import config
from reelgrep.probe import VideoMetadata
from reelgrep.search import Search
from reelgrep.subtitles import SubtitleCue, SubtitleTrack
from reelgrep.transcribe import (
    TranscribeError,
    TranscribeResult,
    transcribe_video,
)


@pytest.fixture(autouse=True)
def _reelgrep_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "rg-home"
    monkeypatch.setenv("REELGREP_HOME", str(home))
    for var in ("REELGREP_DB", "REELGREP_CACHE", "REELGREP_FFMPEG", "REELGREP_FFPROBE"):
        monkeypatch.delenv(var, raising=False)
    config.reset_settings()
    yield home
    config.reset_settings()


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    """A real file on disk so resolve(strict=True) succeeds."""
    p = tmp_path / "lecture.mp4"
    p.write_bytes(b"\x00fake-video-bytes")
    return p


def _make_meta(path: str) -> VideoMetadata:
    return VideoMetadata(
        path=path,
        format_name="mp4",
        duration_ms=60000,
        size_bytes=4096,
        width=1280,
        height=720,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        raw={"format": {"format_name": "mp4"}, "streams": []},
    )


def _make_track() -> SubtitleTrack:
    return SubtitleTrack(
        source="whisper",
        stream_index=None,
        language="en",
        format="whisper",
        cues=[
            SubtitleCue(
                start_ms=0, end_ms=2500, text="welcome to the lecture", language="en"
            ),
            SubtitleCue(
                start_ms=3000,
                end_ms=5500,
                text="today we cover kubernetes networking",
                language="en",
            ),
            SubtitleCue(
                start_ms=6000,
                end_ms=8000,
                text="pods talk to each other via CNI",
                language="en",
            ),
        ],
    )


@pytest.fixture
def patched_pipeline(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> dict[str, Any]:
    """Stub the heavy bits (faster-whisper + ffprobe) for the library tests.

    Both names live on the ``reelgrep.transcribe`` module since that's
    where ``transcribe_video`` resolves them.
    """
    meta = _make_meta(str(fake_video.resolve()))
    track = _make_track()
    monkeypatch.setattr("reelgrep.transcribe.probe", lambda p: meta)
    monkeypatch.setattr(
        "reelgrep.transcribe.transcribe", lambda *a, **k: track
    )
    return {"meta": meta, "track": track}


# ---------- result type shape ----------


def test_transcribe_result_is_frozen_dataclass() -> None:
    assert is_dataclass(TranscribeResult)
    fields = set(TranscribeResult.__dataclass_fields__)
    assert fields == {
        "video_path",
        "file_hash",
        "video_id",
        "db_path",
        "model",
        "language_detected",
        "cue_count",
        "duration_ms",
        "already_transcribed",
    }
    sample = TranscribeResult(
        video_path=Path("/tmp/x.mp4"),
        file_hash="blake2b:xxx",
        video_id=1,
        db_path=Path("/tmp/idx.sqlite"),
        model="small",
        language_detected="en",
        cue_count=0,
        duration_ms=0,
        already_transcribed=False,
    )
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
        sample.cue_count = 7  # type: ignore[misc]


# ---------- happy path: writes rows and Search can read them ----------


def test_writes_rows_and_search_finds_them(
    fake_video: Path, patched_pipeline: dict[str, Any], tmp_path: Path
) -> None:
    target_db = tmp_path / "explicit.sqlite"
    # Bootstrap an empty schema so the explicit-path existence check passes.
    from reelgrep.db import connect, migrate

    boot = connect(target_db)
    try:
        migrate(boot)
    finally:
        boot.close()

    result = transcribe_video(
        fake_video, model="small", db_path=target_db
    )

    assert isinstance(result, TranscribeResult)
    assert result.video_path == fake_video.resolve()
    assert result.file_hash.startswith("blake2b:")
    assert result.video_id >= 1
    assert result.db_path == target_db.resolve()
    assert result.model == "small"
    assert result.language_detected == "en"
    assert result.cue_count == 3
    assert result.duration_ms == 60000
    assert result.already_transcribed is False

    # Search must find the persisted cues.
    s = Search(db_path=target_db)
    hits = s.subtitles("kubernetes")
    assert len(hits) == 1
    assert hits[0].text == "today we cover kubernetes networking"
    assert hits[0].video_id == result.video_id
    assert hits[0].source == "whisper"


# ---------- default db_path resolution ----------


def test_default_db_path_uses_settings(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    settings_db = config.get_settings().db_path
    result = transcribe_video(fake_video)
    assert result.db_path == settings_db
    assert settings_db.exists()


# ---------- idempotency ----------


def test_second_call_is_idempotent_and_reports_existing_count(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    first = transcribe_video(fake_video)
    assert first.already_transcribed is False
    assert first.cue_count == 3

    second = transcribe_video(fake_video)
    assert second.already_transcribed is True
    assert second.cue_count == 3
    assert second.video_id == first.video_id
    assert second.file_hash == first.file_hash
    assert second.model is None  # no inference ran, so no model was applied
    assert second.language_detected is None

    # DB still has exactly three whisper rows - no duplicates were written.
    conn = sqlite3.connect(str(second.db_path))
    try:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source = 'whisper'"
        ).fetchone()
    finally:
        conn.close()
    assert count == 3


def test_force_replaces_existing_whisper_cues(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    first = transcribe_video(fake_video)
    assert first.already_transcribed is False

    forced = transcribe_video(fake_video, force=True)
    assert forced.already_transcribed is False
    assert forced.cue_count == 3  # the stub track has 3 cues

    # And the row count in DB is still 3 (not 6) - old rows were deleted.
    conn = sqlite3.connect(str(forced.db_path))
    try:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source = 'whisper'"
        ).fetchone()
        # FTS rows should also be consistent (one per subtitle row).
        (fts_count,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles_fts"
        ).fetchone()
    finally:
        conn.close()
    assert count == 3
    assert fts_count == 3


# ---------- existence checks ----------


def test_missing_video_path_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError):
        transcribe_video(missing)


def test_explicit_missing_db_path_raises_file_not_found(
    fake_video: Path, patched_pipeline: dict[str, Any], tmp_path: Path
) -> None:
    missing_db = tmp_path / "does-not-exist.sqlite"
    with pytest.raises(FileNotFoundError):
        transcribe_video(fake_video, db_path=missing_db)


# ---------- error surface ----------


def test_unknown_model_raises_transcribe_error(
    fake_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown whisper model surfaces as TranscribeError, not click.Exit."""
    meta = _make_meta(str(fake_video.resolve()))
    monkeypatch.setattr("reelgrep.transcribe.probe", lambda p: meta)
    # Ensure faster_whisper is "missing" so the model_size check is what surfaces.
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    with pytest.raises(TranscribeError, match="unknown model_size"):
        transcribe_video(fake_video, model="not-a-real-model")


# ---------- db_path is not sticky across calls ----------


def test_explicit_db_path_does_not_leak_to_subsequent_calls(
    fake_video: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Passing db_path uses set_db_override only for the duration of the call."""
    meta = _make_meta(str(fake_video.resolve()))
    track = _make_track()
    monkeypatch.setattr("reelgrep.transcribe.probe", lambda p: meta)
    monkeypatch.setattr("reelgrep.transcribe.transcribe", lambda *a, **k: track)

    target_db = tmp_path / "explicit.sqlite"
    from reelgrep.db import connect, migrate

    boot = connect(target_db)
    try:
        migrate(boot)
    finally:
        boot.close()

    settings_db_before = config.get_settings().db_path
    transcribe_video(fake_video, db_path=target_db)

    # A second call with a different (fresh) video must NOT write to target_db.
    second_video = tmp_path / "movie2.mp4"
    second_video.write_bytes(b"\x01different-bytes")
    # Override the resolved meta path for the second video to match.
    monkeypatch.setattr(
        "reelgrep.transcribe.probe",
        lambda p: _make_meta(str(Path(p).resolve())),
    )
    second = transcribe_video(second_video)

    assert second.db_path == settings_db_before
    # And the explicit-target db still has exactly one video row.
    conn = sqlite3.connect(str(target_db))
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
    finally:
        conn.close()
    assert vcount == 1


# ---------- on_message callback ----------


def test_on_message_fires_before_inference(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    captured: list[str] = []
    transcribe_video(fake_video, on_message=captured.append)
    assert any("transcribing" in m for m in captured)
    assert any("whisper:small" in m for m in captured)


def test_on_message_not_called_when_already_transcribed(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    transcribe_video(fake_video)
    captured: list[str] = []
    second = transcribe_video(fake_video, on_message=captured.append)
    assert second.already_transcribed is True
    # No progress message should fire because no inference ran.
    assert captured == []


# ---------- whisper-extra missing surfaces cleanly ----------


def test_missing_whisper_extra_raises_transcribe_error(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    """When faster_whisper can't be imported, TranscribeError is raised."""
    import builtins

    meta = _make_meta(str(fake_video.resolve()))
    monkeypatch.setattr("reelgrep.transcribe.probe", lambda p: meta)
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            raise ImportError("No module named 'faster_whisper'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(TranscribeError, match=r"\[whisper\] extra"):
        transcribe_video(fake_video)


# ---------- a fake whisper end-to-end smoke ----------


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    def __init__(self, language: str = "en"):
        self.language = language


class _FakeWhisperModel:
    segments_to_return: list[_FakeSegment] = []
    info_to_return: _FakeInfo | None = None

    def __init__(self, model_size: str, **kwargs: Any) -> None:
        self.model_size = model_size

    def transcribe(self, audio_path: str, **kwargs: Any):
        return (
            iter(type(self).segments_to_return),
            type(self).info_to_return or _FakeInfo(),
        )


def test_end_to_end_through_fake_whisper_module(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    """Drive transcribe_video through the same faster_whisper mock used by
    the low-level tests in test_transcribe.py. Confirms the wiring from
    library entry point → low-level transcribe() → SubtitleTrack → DB rows.
    """
    fake_mod = types.ModuleType("faster_whisper")
    fake_mod.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)
    _FakeWhisperModel.segments_to_return = [
        _FakeSegment(0.0, 1.5, "hello world"),
        _FakeSegment(2.0, 3.5, "kubernetes"),
    ]
    _FakeWhisperModel.info_to_return = _FakeInfo(language="en")

    meta = _make_meta(str(fake_video.resolve()))
    monkeypatch.setattr("reelgrep.transcribe.probe", lambda p: meta)

    result = transcribe_video(fake_video)
    assert result.cue_count == 2
    assert result.language_detected == "en"
    assert result.already_transcribed is False

    s = Search(db_path=result.db_path)
    hits = s.subtitles("kubernetes")
    assert len(hits) == 1
    assert hits[0].text == "kubernetes"
    assert hits[0].source == "whisper"
