"""Tests for the public ``reelgrep.index`` library API."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from reelgrep.config import get_settings, reset_settings
from reelgrep.frames import Frame
from reelgrep.index import IngestResult, ingest_video
from reelgrep.probe import VideoMetadata
from reelgrep.subtitles import SubtitleCue, SubtitleTrack


@pytest.fixture(autouse=True)
def _reelgrep_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "rg-home"
    monkeypatch.setenv("REELGREP_HOME", str(home))
    monkeypatch.delenv("REELGREP_DB", raising=False)
    monkeypatch.delenv("REELGREP_CACHE", raising=False)
    reset_settings()
    yield home
    reset_settings()


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    v = tmp_path / "movie.mp4"
    v.write_bytes(b"\x00fake-video-bytes")
    return v


def _make_meta() -> VideoMetadata:
    return VideoMetadata(
        path="filled-in",
        format_name="mp4",
        duration_ms=90000,
        size_bytes=12345,
        width=1920,
        height=1080,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        raw={"format": {"format_name": "mp4"}, "streams": []},
    )


def _make_track() -> SubtitleTrack:
    return SubtitleTrack(
        source="sidecar",
        stream_index=None,
        language="en",
        format="srt",
        cues=[
            SubtitleCue(start_ms=0, end_ms=2000, text="hello", language=None),
            SubtitleCue(start_ms=2000, end_ms=4000, text="kubernetes", language=None),
        ],
    )


def _make_frames(tmp_path: Path) -> list[Frame]:
    (tmp_path / "f0.jpg").write_bytes(b"")
    (tmp_path / "f5.jpg").write_bytes(b"")
    return [
        Frame(timestamp_ms=0, path=str(tmp_path / "f0.jpg"), sampling_strategy="every_n"),
        Frame(timestamp_ms=5000, path=str(tmp_path / "f5.jpg"), sampling_strategy="every_n"),
    ]


@pytest.fixture
def patched_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    fake_meta = _make_meta()
    track = _make_track()
    fake_frames = _make_frames(tmp_path)

    monkeypatch.setattr(
        "reelgrep.index.probe",
        lambda p: fake_meta.model_copy(update={"path": str(p)}),
    )
    monkeypatch.setattr("reelgrep.index.extract_embedded", lambda *a, **k: [])
    monkeypatch.setattr("reelgrep.index.find_sidecars", lambda p: [])
    monkeypatch.setattr("reelgrep.index.parse_sidecar", lambda p: track)
    monkeypatch.setattr("reelgrep.index.sample_every", lambda *a, **k: fake_frames)
    return {"meta": fake_meta, "track": track, "frames": fake_frames}


def _open_db(db_path: Path | None = None) -> sqlite3.Connection:
    settings = get_settings()
    target = Path(db_path) if db_path is not None else settings.db_path
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    return conn


def test_ingest_video_returns_result_and_writes_rows(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    """Calling ingest_video directly populates the index and returns an IngestResult."""
    result = ingest_video(fake_video)

    assert isinstance(result, IngestResult)
    assert result.already_ingested is False
    assert result.file_hash.startswith("blake2b:")
    assert result.video_path == fake_video.resolve()
    assert result.duration_ms == 90000
    assert result.frame_count == 2
    assert result.subtitle_track_count == 0
    assert result.subtitle_cue_count == 0
    assert result.transcribed_cue_count == 0
    assert result.warnings == []
    assert result.video_id is not None

    conn = _open_db(result.db_path)
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
        (fcount,) = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        (scount,) = conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()
        row = conn.execute(
            "SELECT file_hash, duration_ms, width, height FROM videos"
        ).fetchone()
    finally:
        conn.close()
    assert vcount == 1
    assert fcount == 2
    assert scount == 0
    assert row["file_hash"] == result.file_hash
    assert row["duration_ms"] == 90000
    assert row["width"] == 1920
    assert row["height"] == 1080


def test_ingest_video_explicit_db_path(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    patched_pipeline: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Passing db_path routes writes to that file via set_db_override.

    The override is also restored on return so subsequent calls without
    db_path resolve through the normal settings chain (no sticky state).
    """
    target_db = tmp_path / "explicit.sqlite"

    # Capture the settings-resolved db path before any override.
    reset_settings()
    settings_db_path = get_settings().db_path

    result = ingest_video(fake_video, db_path=target_db)

    assert result.db_path == target_db.resolve()
    assert target_db.exists()

    conn = _open_db(target_db)
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
    finally:
        conn.close()
    assert vcount == 1

    # A second call without db_path must NOT route to the previous override.
    second_video = tmp_path / "movie2.mp4"
    second_video.write_bytes(b"\x01different-bytes")
    second = ingest_video(second_video)

    assert second.db_path == settings_db_path
    # And the explicit-target db must not have grown a second row.
    conn = _open_db(target_db)
    try:
        (vcount2,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
    finally:
        conn.close()
    assert vcount2 == 1


def test_ingest_video_second_call_is_idempotent(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    first = ingest_video(fake_video)
    assert first.already_ingested is False
    assert first.previously_indexed_path is None

    second = ingest_video(fake_video)
    assert second.already_ingested is True
    assert second.file_hash == first.file_hash
    assert second.video_id == first.video_id
    # Echoes the path stored on the prior ingest, even if the caller
    # passed the same input path - this is the row's recorded path.
    assert second.previously_indexed_path == first.video_path

    conn = _open_db(second.db_path)
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
    finally:
        conn.close()
    assert vcount == 1


def test_ingest_video_unknown_backend_raises(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    with pytest.raises(KeyError):
        ingest_video(fake_video, backend="this-backend-does-not-exist")


def test_ingest_video_invokes_on_message_callback(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    tmp_path: Path,
) -> None:
    """Warnings stream through on_message in the order they occur."""
    from reelgrep.ffmpeg_exec import FFmpegError

    fake_meta = _make_meta()
    fake_frames = _make_frames(tmp_path)

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise FFmpegError(1, "boom", ["ffmpeg"])

    monkeypatch.setattr(
        "reelgrep.index.probe",
        lambda p: fake_meta.model_copy(update={"path": str(p)}),
    )
    monkeypatch.setattr("reelgrep.index.extract_embedded", boom)
    monkeypatch.setattr("reelgrep.index.find_sidecars", lambda p: [])
    monkeypatch.setattr("reelgrep.index.sample_every", lambda *a, **k: fake_frames)

    captured: list[str] = []
    result = ingest_video(fake_video, on_message=captured.append)

    assert any("embedded subtitles" in m for m in captured)
    assert len(result.warnings) == 1
    assert result.warnings[0].stage == "embedded_subtitles"
