"""Tests for the ``reelgrep transcribe`` command."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from reelgrep import config, hashing
from reelgrep.commands.transcribe import transcribe_cmd
from reelgrep.probe import VideoMetadata
from reelgrep.subtitles import SubtitleCue, SubtitleTrack
from reelgrep.transcribe import TranscribeError


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "rg-home"
    monkeypatch.setenv("REELGREP_HOME", str(home))
    for var in ("REELGREP_DB", "REELGREP_CACHE", "REELGREP_FFMPEG", "REELGREP_FFPROBE"):
        monkeypatch.delenv(var, raising=False)
    config.reset_settings()
    yield home
    config.reset_settings()


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
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
    meta = _make_meta(str(fake_video.resolve()))
    track = _make_track()
    monkeypatch.setattr(
        "reelgrep.commands.transcribe.probe", lambda p: meta
    )
    monkeypatch.setattr(
        "reelgrep.commands.transcribe.transcribe", lambda *a, **k: track
    )
    return {"meta": meta, "track": track}


def _open_db() -> sqlite3.Connection:
    settings = config.get_settings()
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    return conn


# ---------- help / arg validation ----------


def test_help_lists_options() -> None:
    runner = CliRunner()
    result = runner.invoke(transcribe_cmd, ["--help"])
    assert result.exit_code == 0
    assert "Transcribe" in result.output
    assert "--model" in result.output
    assert "--language" in result.output


def test_nonexistent_path_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    missing = tmp_path / "nope.mp4"
    result = runner.invoke(transcribe_cmd, [str(missing)])
    assert result.exit_code != 0


def test_unknown_model_is_rejected_by_click(fake_video: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(transcribe_cmd, [str(fake_video), "--model", "bogus"])
    assert result.exit_code != 0
    assert "bogus" in result.output or "Invalid value" in result.output


# ---------- DB-writing paths ----------


def test_auto_creates_video_row_and_indexes_cues(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    runner = CliRunner()
    result = runner.invoke(transcribe_cmd, [str(fake_video)])
    assert result.exit_code == 0, result.output

    digest = hashing.file_hash(fake_video.resolve())
    conn = _open_db()
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
        rows = conn.execute(
            "SELECT file_hash, source FROM videos v "
            "JOIN subtitles s ON s.video_id = v.id"
        ).fetchall()
        (scount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source = 'whisper'"
        ).fetchone()
        fts = conn.execute(
            "SELECT rowid FROM subtitles_fts WHERE subtitles_fts MATCH 'kubernetes'"
        ).fetchall()
    finally:
        conn.close()

    assert vcount == 1
    assert scount == 3
    assert len(fts) == 1
    assert all(r["file_hash"] == digest for r in rows)
    assert all(r["source"] == "whisper" for r in rows)


def test_reuses_existing_video_row(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    # Pre-insert a video row with the matching hash.
    from reelgrep.config import ensure_dirs, get_settings
    from reelgrep.db import connect, migrate

    settings = get_settings()
    ensure_dirs(settings)
    pre = connect(settings.db_path)
    try:
        migrate(pre)
        with pre:
            pre.execute(
                """
                INSERT INTO videos (
                    file_hash, path, duration_ms, width, height, fps,
                    container, video_codec, audio_codec, size_bytes,
                    ingested_at, probe_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hashing.file_hash(fake_video.resolve()),
                    str(fake_video.resolve()),
                    60000, 1280, 720, 30.0, "mp4", "h264", "aac",
                    4096, "2026-01-01T00:00:00+00:00", "{}",
                ),
            )
    finally:
        pre.close()

    runner = CliRunner()
    result = runner.invoke(transcribe_cmd, [str(fake_video)])
    assert result.exit_code == 0, result.output

    conn = _open_db()
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
        (scount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source = 'whisper'"
        ).fetchone()
    finally:
        conn.close()
    assert vcount == 1
    assert scount == 3


def test_idempotent_second_invocation_warns(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    runner = CliRunner()
    first = runner.invoke(transcribe_cmd, [str(fake_video)])
    assert first.exit_code == 0, first.output

    second = runner.invoke(transcribe_cmd, [str(fake_video)])
    assert second.exit_code == 0, second.output
    assert "already has 3 whisper cue(s)" in second.stderr
    assert "--force" in second.stderr

    conn = _open_db()
    try:
        (scount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source = 'whisper'"
        ).fetchone()
    finally:
        conn.close()
    assert scount == 3


def test_force_replaces_existing_whisper_cues(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    runner = CliRunner()
    first = runner.invoke(transcribe_cmd, [str(fake_video)])
    assert first.exit_code == 0, first.output

    second = runner.invoke(transcribe_cmd, [str(fake_video), "--force"])
    assert second.exit_code == 0, second.output

    conn = _open_db()
    try:
        (scount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source = 'whisper'"
        ).fetchone()
        fts = conn.execute(
            "SELECT rowid FROM subtitles_fts WHERE subtitles_fts MATCH 'kubernetes'"
        ).fetchall()
    finally:
        conn.close()
    assert scount == 3
    assert len(fts) == 1


def test_summary_output_fields(
    fake_video: Path, patched_pipeline: dict[str, Any]
) -> None:
    runner = CliRunner()
    result = runner.invoke(transcribe_cmd, [str(fake_video)])
    assert result.exit_code == 0, result.output
    for label in ("transcribed:", "language:", "model:", "cues:", "span:"):
        assert label in result.output


def test_empty_cues_handled(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    meta = _make_meta(str(fake_video.resolve()))
    empty_track = SubtitleTrack(
        source="whisper",
        stream_index=None,
        language="en",
        format="whisper",
        cues=[],
    )
    monkeypatch.setattr("reelgrep.commands.transcribe.probe", lambda p: meta)
    monkeypatch.setattr(
        "reelgrep.commands.transcribe.transcribe", lambda *a, **k: empty_track
    )

    runner = CliRunner()
    result = runner.invoke(transcribe_cmd, [str(fake_video)])
    assert result.exit_code == 0, result.output
    assert "cues:        0" in result.output
    assert "span:        (no cues)" in result.output

    conn = _open_db()
    try:
        (scount,) = conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()
    finally:
        conn.close()
    assert scount == 0


# ---------- error paths ----------


def test_transcribe_error_surfaces_as_exit_2(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    meta = _make_meta(str(fake_video.resolve()))

    def boom(*_a: Any, **_k: Any) -> SubtitleTrack:
        raise TranscribeError("ctranslate2 missing")

    monkeypatch.setattr("reelgrep.commands.transcribe.probe", lambda p: meta)
    monkeypatch.setattr("reelgrep.commands.transcribe.transcribe", boom)

    runner = CliRunner()
    result = runner.invoke(transcribe_cmd, [str(fake_video)])
    assert result.exit_code == 2
    assert "error: ctranslate2 missing" in result.stderr


# ---------- --no-db path ----------


def test_no_db_prints_json_and_skips_db(
    monkeypatch: pytest.MonkeyPatch, fake_video: Path
) -> None:
    track = _make_track()
    monkeypatch.setattr(
        "reelgrep.commands.transcribe.transcribe", lambda *a, **k: track
    )
    # probe should NOT be needed - if it gets called, the test should fail
    def fail_probe(_p: Path) -> Any:
        raise AssertionError("probe should not run when --no-db is set")
    monkeypatch.setattr("reelgrep.commands.transcribe.probe", fail_probe)

    runner = CliRunner()
    result = runner.invoke(transcribe_cmd, [str(fake_video), "--no-db"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)
    assert payload["video"] == str(fake_video.resolve())
    assert payload["language"] == "en"
    assert payload["model"] == "small"
    assert payload["file_hash"].startswith("blake2b:")
    assert len(payload["cues"]) == 3
    assert payload["cues"][1]["text"] == "today we cover kubernetes networking"

    # DB should not have any subtitle rows (the DB file may exist from the
    # autouse fixture path, but no whisper writes should have happened).
    settings = config.get_settings()
    if settings.db_path.exists():
        conn = sqlite3.connect(str(settings.db_path))
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='subtitles'"
            ).fetchone()
            if row is not None:
                (scount,) = conn.execute(
                    "SELECT COUNT(*) FROM subtitles"
                ).fetchone()
                assert scount == 0
        finally:
            conn.close()
