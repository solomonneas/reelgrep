"""Tests for the ``reelgrep align`` command."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from reelgrep import config, hashing
from reelgrep.align import AlignError, AlignmentStats
from reelgrep.commands.align import align_cmd
from reelgrep.probe import VideoMetadata
from reelgrep.subtitles import SubtitleCue, SubtitleTrack
from reelgrep.transcript_loaders import TranscriptLoadError


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
    v = tmp_path / "lecture.mp4"
    v.write_bytes(b"\x00fake-video-bytes")
    return v


@pytest.fixture
def fake_transcript(tmp_path: Path) -> Path:
    t = tmp_path / "official.txt"
    t.write_text(
        "Welcome to the lecture. Today we cover Kubernetes networking. "
        "Pods talk to each other over the pod network. "
        "Services give pods a stable address.\n",
        encoding="utf-8",
    )
    return t


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


def _make_whisper_track() -> SubtitleTrack:
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


def _make_aligned_track(language: str = "en") -> SubtitleTrack:
    return SubtitleTrack(
        source="aligned",
        stream_index=None,
        language=language,
        format="aligned",
        cues=[
            SubtitleCue(
                start_ms=0,
                end_ms=2500,
                text="Welcome to the lecture.",
                language=language,
            ),
            SubtitleCue(
                start_ms=3000,
                end_ms=5500,
                text="Today we cover Kubernetes networking.",
                language=language,
            ),
            SubtitleCue(
                start_ms=6000,
                end_ms=8000,
                text="Pods talk to each other over the pod network.",
                language=language,
            ),
        ],
    )


def _make_stats(
    cue_count: int = 3,
    transcript_word_count: int = 20,
    matched_word_count: int = 15,
) -> AlignmentStats:
    stats = AlignmentStats()
    stats.cue_count = cue_count
    stats.transcript_word_count = transcript_word_count
    stats.cue_word_count = 18
    stats.matched_word_count = matched_word_count
    stats.avg_similarity = 0.82
    stats.coverage = (
        matched_word_count / transcript_word_count if transcript_word_count else 0.0
    )
    return stats


def _open_db() -> sqlite3.Connection:
    settings = config.get_settings()
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _insert_video_and_whisper_cues(fake_video: Path) -> int:
    """Pre-populate the DB with a video row + 3 Whisper cues; return video_id."""
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
            video_id = pre.execute(
                "SELECT id FROM videos WHERE file_hash = ?",
                (hashing.file_hash(fake_video.resolve()),),
            ).fetchone()[0]
            for cue in _make_whisper_track().cues:
                cur = pre.execute(
                    """
                    INSERT INTO subtitles (
                        video_id, language, source, stream_index,
                        start_ms, end_ms, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        video_id, cue.language, "whisper", None,
                        cue.start_ms, cue.end_ms, cue.text,
                    ),
                )
                pre.execute(
                    "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
                    (cur.lastrowid, cue.text),
                )
    finally:
        pre.close()
    return video_id


# ---------- help / arg validation ----------


def test_help_lists_options() -> None:
    runner = CliRunner()
    result = runner.invoke(align_cmd, ["--help"])
    assert result.exit_code == 0
    assert "Align an official transcript" in result.output
    assert "--transcript" in result.output
    assert "--force" in result.output
    assert "--no-db" in result.output
    assert "--auto-transcribe" in result.output


def test_nonexistent_video_errors(tmp_path: Path, fake_transcript: Path) -> None:
    runner = CliRunner()
    missing = tmp_path / "nope.mp4"
    result = runner.invoke(
        align_cmd, [str(missing), "--transcript", str(fake_transcript)]
    )
    assert result.exit_code != 0


def test_nonexistent_transcript_errors(tmp_path: Path, fake_video: Path) -> None:
    runner = CliRunner()
    missing = tmp_path / "nope.txt"
    result = runner.invoke(
        align_cmd, [str(fake_video), "--transcript", str(missing)]
    )
    assert result.exit_code != 0


# ---------- auto-transcribe path ----------


def test_auto_transcribe_path_indexes_whisper_and_aligned(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    fake_transcript: Path,
) -> None:
    meta = _make_meta(str(fake_video.resolve()))
    monkeypatch.setattr("reelgrep.commands.align.probe", lambda p: meta)
    monkeypatch.setattr(
        "reelgrep.commands.align.run_whisper",
        lambda *a, **k: _make_whisper_track(),
    )
    monkeypatch.setattr(
        "reelgrep.commands.align.align_video",
        lambda *a, **k: (_make_aligned_track(), _make_stats()),
    )

    runner = CliRunner()
    result = runner.invoke(
        align_cmd, [str(fake_video), "--transcript", str(fake_transcript)]
    )
    assert result.exit_code == 0, result.output

    conn = _open_db()
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
        (scount,) = conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()
        (wcount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source='whisper'"
        ).fetchone()
        (acount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source='aligned'"
        ).fetchone()
        fts = conn.execute(
            "SELECT rowid FROM subtitles_fts WHERE subtitles_fts MATCH 'kubernetes'"
        ).fetchall()
    finally:
        conn.close()

    assert vcount == 1
    assert scount == 6
    assert wcount == 3
    assert acount == 3
    assert len(fts) >= 1


def test_no_auto_transcribe_exits_two_when_no_cues(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    fake_transcript: Path,
) -> None:
    meta = _make_meta(str(fake_video.resolve()))
    monkeypatch.setattr("reelgrep.commands.align.probe", lambda p: meta)

    def _should_not_run(*_a: Any, **_k: Any) -> None:
        raise AssertionError("run_whisper should not be called with --no-auto-transcribe")

    monkeypatch.setattr("reelgrep.commands.align.run_whisper", _should_not_run)

    runner = CliRunner()
    result = runner.invoke(
        align_cmd,
        [str(fake_video), "--transcript", str(fake_transcript), "--no-auto-transcribe"],
    )
    assert result.exit_code == 2, result.output
    assert "no cues to align against" in result.stderr


# ---------- existing-cues path ----------


def test_existing_whisper_cues_get_aligned(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    fake_transcript: Path,
) -> None:
    _insert_video_and_whisper_cues(fake_video)
    monkeypatch.setattr(
        "reelgrep.commands.align.align_video",
        lambda *a, **k: (_make_aligned_track(), _make_stats()),
    )

    runner = CliRunner()
    result = runner.invoke(
        align_cmd, [str(fake_video), "--transcript", str(fake_transcript)]
    )
    assert result.exit_code == 0, result.output

    conn = _open_db()
    try:
        (wcount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source='whisper'"
        ).fetchone()
        (acount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source='aligned'"
        ).fetchone()
    finally:
        conn.close()
    assert wcount == 3  # untouched
    assert acount == 3


# ---------- idempotency / --force ----------


def test_idempotent_second_invocation_warns(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    fake_transcript: Path,
) -> None:
    _insert_video_and_whisper_cues(fake_video)
    monkeypatch.setattr(
        "reelgrep.commands.align.align_video",
        lambda *a, **k: (_make_aligned_track(), _make_stats()),
    )

    runner = CliRunner()
    first = runner.invoke(
        align_cmd, [str(fake_video), "--transcript", str(fake_transcript)]
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        align_cmd, [str(fake_video), "--transcript", str(fake_transcript)]
    )
    assert second.exit_code == 0, second.output
    assert "already has 3 aligned cue(s)" in second.stderr
    assert "--force" in second.stderr

    conn = _open_db()
    try:
        (acount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source='aligned'"
        ).fetchone()
    finally:
        conn.close()
    assert acount == 3


def test_force_replaces_aligned_cues(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    fake_transcript: Path,
) -> None:
    _insert_video_and_whisper_cues(fake_video)
    monkeypatch.setattr(
        "reelgrep.commands.align.align_video",
        lambda *a, **k: (_make_aligned_track(), _make_stats()),
    )

    runner = CliRunner()
    first = runner.invoke(
        align_cmd, [str(fake_video), "--transcript", str(fake_transcript)]
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        align_cmd,
        [str(fake_video), "--transcript", str(fake_transcript), "--force"],
    )
    assert second.exit_code == 0, second.output

    conn = _open_db()
    try:
        (acount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source='aligned'"
        ).fetchone()
        fts = conn.execute(
            "SELECT rowid FROM subtitles_fts WHERE subtitles_fts MATCH 'kubernetes'"
        ).fetchall()
    finally:
        conn.close()
    assert acount == 3
    # FTS still has the kubernetes hit from the whisper row (untouched by --force).
    assert len(fts) >= 1


# ---------- --no-db path ----------


def test_no_db_prints_json_and_skips_aligned_writes(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    fake_transcript: Path,
) -> None:
    _insert_video_and_whisper_cues(fake_video)
    monkeypatch.setattr(
        "reelgrep.commands.align.align_video",
        lambda *a, **k: (_make_aligned_track(), _make_stats()),
    )

    runner = CliRunner()
    result = runner.invoke(
        align_cmd,
        [str(fake_video), "--transcript", str(fake_transcript), "--no-db"],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)
    assert payload["video"] == str(fake_video.resolve())
    assert payload["language"] == "en"
    assert payload["file_hash"].startswith("blake2b:")
    assert "stats" in payload
    assert payload["stats"]["matched_word_count"] == 15
    assert len(payload["cues"]) == 3
    assert payload["cues"][0]["text"] == "Welcome to the lecture."

    conn = _open_db()
    try:
        (acount,) = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE source='aligned'"
        ).fetchone()
    finally:
        conn.close()
    assert acount == 0


# ---------- error surfacing ----------


def test_align_error_exit_two(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    fake_transcript: Path,
) -> None:
    _insert_video_and_whisper_cues(fake_video)

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AlignError("transcript empty")

    monkeypatch.setattr("reelgrep.commands.align.align_video", _boom)

    runner = CliRunner()
    result = runner.invoke(
        align_cmd, [str(fake_video), "--transcript", str(fake_transcript)]
    )
    assert result.exit_code == 2
    assert "error: transcript empty" in result.stderr


def test_transcript_load_error_exit_two(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path,
    fake_transcript: Path,
) -> None:
    _insert_video_and_whisper_cues(fake_video)

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise TranscriptLoadError("unsupported")

    monkeypatch.setattr("reelgrep.commands.align.align_video", _boom)

    runner = CliRunner()
    result = runner.invoke(
        align_cmd, [str(fake_video), "--transcript", str(fake_transcript)]
    )
    assert result.exit_code == 2
    assert "error: unsupported" in result.stderr


# ---------- --out SRT path ----------


def test_out_writes_srt_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_video: Path,
    fake_transcript: Path,
) -> None:
    _insert_video_and_whisper_cues(fake_video)
    monkeypatch.setattr(
        "reelgrep.commands.align.align_video",
        lambda *a, **k: (_make_aligned_track(), _make_stats()),
    )
    srt_out = tmp_path / "out" / "lecture.srt"

    runner = CliRunner()
    result = runner.invoke(
        align_cmd,
        [
            str(fake_video),
            "--transcript", str(fake_transcript),
            "--out", str(srt_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert srt_out.exists()

    content = srt_out.read_text(encoding="utf-8")
    # Expect 3 entries.
    blocks = [b for b in content.strip().split("\n\n") if b.strip()]
    assert len(blocks) == 3
    # Timestamps formatted HH:MM:SS,mmm.
    import re
    ts_pattern = re.compile(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}")
    for block in blocks:
        assert ts_pattern.search(block), f"missing timestamp in block: {block!r}"
    # The summary should mention the srt path.
    assert "srt:" in result.output
