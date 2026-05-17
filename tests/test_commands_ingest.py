"""Tests for the ``reelgrep ingest`` command."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from reelgrep.commands.ingest import ingest
from reelgrep.config import get_settings, reset_settings
from reelgrep.ffmpeg_exec import FFmpegError
from reelgrep.frames import Frame
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
        "reelgrep.commands.ingest.probe",
        lambda p: fake_meta.model_copy(update={"path": str(p)}),
    )
    monkeypatch.setattr(
        "reelgrep.commands.ingest.extract_embedded", lambda *a, **k: []
    )
    monkeypatch.setattr("reelgrep.commands.ingest.find_sidecars", lambda p: [])
    monkeypatch.setattr("reelgrep.commands.ingest.parse_sidecar", lambda p: track)
    monkeypatch.setattr(
        "reelgrep.commands.ingest.sample_every", lambda *a, **k: fake_frames
    )
    return {"meta": fake_meta, "track": track, "frames": fake_frames}


def _open_db() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    return conn


def test_happy_path_no_sidecars(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    runner = CliRunner()
    result = runner.invoke(ingest, [str(fake_video)])
    assert result.exit_code == 0, result.output
    assert "ingested:" in result.output
    assert "hash:     blake2b:" in result.output

    conn = _open_db()
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
        (fcount,) = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        (scount,) = conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()
    finally:
        conn.close()
    assert vcount == 1
    assert fcount == 2
    assert scount == 0


def test_sidecar_present_indexes_cues_and_fts(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    tmp_path: Path,
) -> None:
    fake_meta = _make_meta()
    track = _make_track()
    fake_frames = _make_frames(tmp_path)
    sidecar = tmp_path / "movie.en.srt"
    sidecar.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        "reelgrep.commands.ingest.probe",
        lambda p: fake_meta.model_copy(update={"path": str(p)}),
    )
    monkeypatch.setattr(
        "reelgrep.commands.ingest.extract_embedded", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "reelgrep.commands.ingest.find_sidecars", lambda p: [sidecar]
    )
    monkeypatch.setattr("reelgrep.commands.ingest.parse_sidecar", lambda p: track)
    monkeypatch.setattr(
        "reelgrep.commands.ingest.sample_every", lambda *a, **k: fake_frames
    )

    runner = CliRunner()
    result = runner.invoke(ingest, [str(fake_video)])
    assert result.exit_code == 0, result.output

    conn = _open_db()
    try:
        (scount,) = conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()
        fts_rows = conn.execute(
            "SELECT rowid FROM subtitles_fts WHERE subtitles_fts MATCH 'kubernetes'"
        ).fetchall()
    finally:
        conn.close()
    assert scount == 2
    assert len(fts_rows) == 1


def test_no_subtitles_flag_skips_subtitle_writes(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    tmp_path: Path,
) -> None:
    fake_meta = _make_meta()
    track = _make_track()
    fake_frames = _make_frames(tmp_path)
    sidecar = tmp_path / "movie.en.srt"
    sidecar.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        "reelgrep.commands.ingest.probe",
        lambda p: fake_meta.model_copy(update={"path": str(p)}),
    )
    monkeypatch.setattr(
        "reelgrep.commands.ingest.extract_embedded", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "reelgrep.commands.ingest.find_sidecars", lambda p: [sidecar]
    )
    monkeypatch.setattr("reelgrep.commands.ingest.parse_sidecar", lambda p: track)
    monkeypatch.setattr(
        "reelgrep.commands.ingest.sample_every", lambda *a, **k: fake_frames
    )

    runner = CliRunner()
    result = runner.invoke(ingest, [str(fake_video), "--no-subtitles"])
    assert result.exit_code == 0, result.output

    conn = _open_db()
    try:
        (scount,) = conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()
        (fcount,) = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
    finally:
        conn.close()
    assert scount == 0
    assert fcount == 2


def test_no_frames_flag_skips_frame_writes(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    runner = CliRunner()
    result = runner.invoke(ingest, [str(fake_video), "--no-frames"])
    assert result.exit_code == 0, result.output

    conn = _open_db()
    try:
        (fcount,) = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
    finally:
        conn.close()
    assert fcount == 0
    assert vcount == 1


def test_idempotency_second_invocation_warns_and_no_dup(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    runner = CliRunner()
    first = runner.invoke(ingest, [str(fake_video)])
    assert first.exit_code == 0, first.output

    second = runner.invoke(ingest, [str(fake_video)])
    assert second.exit_code == 0, second.output
    assert "Already ingested" in second.output

    conn = _open_db()
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
    finally:
        conn.close()
    assert vcount == 1


def test_force_replaces_existing_and_wipes_cache(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    patched_pipeline: dict[str, Any],
) -> None:
    runner = CliRunner()
    first = runner.invoke(ingest, [str(fake_video)])
    assert first.exit_code == 0, first.output

    # Plant marker files inside the cache subdirs that should be wiped on force.
    settings = get_settings()
    from reelgrep.commands.ingest import _hash_slice
    from reelgrep.hashing import file_hash

    digest = file_hash(fake_video.resolve())
    slice_name = _hash_slice(digest)
    subs_dir = settings.cache_dir / "subtitles" / slice_name
    frames_dir = settings.cache_dir / "frames" / slice_name
    subs_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    sub_marker = subs_dir / "marker.txt"
    frame_marker = frames_dir / "marker.txt"
    sub_marker.write_text("subs-marker", encoding="utf-8")
    frame_marker.write_text("frames-marker", encoding="utf-8")

    rmtree_calls: list[Path] = []
    real_rmtree = shutil.rmtree

    def tracking_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        rmtree_calls.append(Path(path))
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        "reelgrep.commands.ingest.shutil.rmtree", tracking_rmtree
    )

    second = runner.invoke(ingest, [str(fake_video), "--force"])
    assert second.exit_code == 0, second.output

    assert subs_dir in rmtree_calls
    assert frames_dir in rmtree_calls
    assert not sub_marker.exists()
    assert not frame_marker.exists()

    conn = _open_db()
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
        (fcount,) = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        (scount,) = conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()
    finally:
        conn.close()
    assert vcount == 1
    assert fcount == 2
    assert scount == 0


def test_nonexistent_path_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    missing = tmp_path / "does-not-exist.mp4"
    result = runner.invoke(ingest, [str(missing)])
    assert result.exit_code != 0


def test_extract_embedded_ffmpeg_error_is_warned_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    tmp_path: Path,
) -> None:
    fake_meta = _make_meta()
    fake_frames = _make_frames(tmp_path)

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise FFmpegError(1, "boom", ["ffmpeg"])

    monkeypatch.setattr(
        "reelgrep.commands.ingest.probe",
        lambda p: fake_meta.model_copy(update={"path": str(p)}),
    )
    monkeypatch.setattr("reelgrep.commands.ingest.extract_embedded", boom)
    monkeypatch.setattr("reelgrep.commands.ingest.find_sidecars", lambda p: [])
    monkeypatch.setattr(
        "reelgrep.commands.ingest.parse_sidecar", lambda p: _make_track()
    )
    monkeypatch.setattr(
        "reelgrep.commands.ingest.sample_every", lambda *a, **k: fake_frames
    )

    runner = CliRunner()
    result = runner.invoke(ingest, [str(fake_video)])
    assert result.exit_code == 0, result.output
    # In Click 8.2+, stderr is merged into result.output by default.
    assert "warning" in result.stderr.lower()
    assert "embedded subtitles" in result.stderr

    conn = _open_db()
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
        (fcount,) = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
        (scount,) = conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()
    finally:
        conn.close()
    assert vcount == 1
    assert fcount == 2
    assert scount == 0
