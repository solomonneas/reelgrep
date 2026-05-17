"""Tests for ``reelgrep info`` and ``reelgrep ls`` commands."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from reelgrep import config, db
from reelgrep.commands.info import _humanize_bytes, info
from reelgrep.commands.ls import ls


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point REELGREP_HOME at a per-test tmp dir and reset cached settings."""
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path))
    config.reset_settings()
    yield
    config.reset_settings()


@pytest.fixture
def seeded_db():
    """Seed the DB with two videos and assorted children for Video A."""
    config.reset_settings()
    s = config.get_settings()
    config.ensure_dirs(s)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    conn.execute(
        "INSERT INTO videos (file_hash, path, duration_ms, width, height, fps, container, "
        "video_codec, audio_codec, size_bytes, ingested_at, probe_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "blake2b:aaaa1111" + "0" * 56,
            "/videos/lecture.mp4",
            5500,
            1920,
            1080,
            29.97,
            "mp4",
            "h264",
            "aac",
            1234567890,
            "2026-05-17T12:00:00",
            "{}",
        ),
    )
    a_id = conn.execute(
        "SELECT id FROM videos WHERE file_hash LIKE 'blake2b:aaaa%'"
    ).fetchone()[0]
    for src, idx in [("embedded", 2), ("sidecar", None)]:
        for i in range(3):
            conn.execute(
                "INSERT INTO subtitles (video_id, language, source, stream_index, "
                "start_ms, end_ms, text) VALUES (?,?,?,?,?,?,?)",
                (a_id, "en", src, idx, i * 1000, (i + 1) * 1000, f"cue {i}"),
            )
    for i in range(5):
        conn.execute(
            "INSERT INTO frames (video_id, timestamp_ms, path, sampling_strategy) "
            "VALUES (?,?,?,?)",
            (a_id, i * 5000, f"/tmp/f{i}.jpg", "every_n"),
        )
    conn.execute(
        "INSERT INTO scenes (video_id, start_ms, end_ms, score) VALUES (?,?,?,?)",
        (a_id, 0, 10000, 0.5),
    )
    conn.execute(
        "INSERT INTO scenes (video_id, start_ms, end_ms, score) VALUES (?,?,?,?)",
        (a_id, 10000, 20000, 0.6),
    )
    conn.execute(
        "INSERT INTO person_searches (video_id, label, backend, positive_examples_json, "
        "negative_examples_json, config_json, threshold, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (a_id, "speaker", "face_embed", "[]", "[]", "{}", 0.3, "2026-05-17T12:30:00"),
    )
    for kind in ["clip", "clip", "gif", "contact_sheet"]:
        conn.execute(
            "INSERT INTO export_artifacts (video_id, kind, path, created_at) "
            "VALUES (?,?,?,?)",
            (a_id, kind, f"/tmp/x.{kind}", "2026-05-17T13:00:00"),
        )
    conn.execute(
        "INSERT INTO videos (file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            "blake2b:bbbb2222" + "0" * 56,
            "/videos/very/long/path/that/keeps/going/and/going/and/going/movie.mkv",
            5400000,
            "2026-05-17T13:00:00",
            "{}",
        ),
    )
    conn.commit()
    conn.close()


def test_ls_happy_path(seeded_db):
    runner = CliRunner()
    result = runner.invoke(ls, [])
    assert result.exit_code == 0, result.output
    assert "blake2b:aaaa1111..." in result.output
    assert "blake2b:bbbb2222..." in result.output
    idx_a = result.output.index("blake2b:aaaa1111...")
    idx_b = result.output.index("blake2b:bbbb2222...")
    assert idx_b < idx_a


def test_ls_limit_one(seeded_db):
    runner = CliRunner()
    result = runner.invoke(ls, ["--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "blake2b:bbbb2222..." in result.output
    assert "blake2b:aaaa1111..." not in result.output


def test_ls_limit_zero_rejects(seeded_db):
    runner = CliRunner()
    result = runner.invoke(ls, ["--limit", "0"])
    assert result.exit_code != 0
    assert "limit" in result.output.lower()


def test_ls_empty_db():
    runner = CliRunner()
    result = runner.invoke(ls, [])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "no videos indexed"


def test_ls_path_truncation(seeded_db):
    runner = CliRunner()
    result = runner.invoke(ls, [])
    assert result.exit_code == 0, result.output
    assert "..." in result.output
    assert "movie.mkv" in result.output


def test_info_by_full_hash(seeded_db):
    runner = CliRunner()
    full_hash = "blake2b:aaaa1111" + "0" * 56
    result = runner.invoke(info, [full_hash])
    assert result.exit_code == 0, result.output
    assert full_hash in result.output
    assert "Subtitles:   2 tracks, 6 cues" in result.output
    assert "Frames:      5 sampled" in result.output
    assert "Scenes:      2 detected" in result.output
    assert "Person searches: 1" in result.output
    assert "clips:2" in result.output
    assert "gifs:1" in result.output
    assert "contact_sheets:1" in result.output
    assert "screenshots:0" in result.output


def test_info_by_hex_prefix(seeded_db):
    runner = CliRunner()
    result = runner.invoke(info, ["aaaa11110000"])
    assert result.exit_code == 0, result.output
    assert "blake2b:aaaa1111" in result.output
    assert "/videos/lecture.mp4" in result.output


def test_info_by_path_fallback(seeded_db):
    runner = CliRunner()
    result = runner.invoke(info, ["/videos/lecture.mp4"])
    assert result.exit_code == 0, result.output
    assert "/videos/lecture.mp4" in result.output
    assert "blake2b:aaaa1111" in result.output


def test_info_not_found(seeded_db):
    runner = CliRunner()
    result = runner.invoke(info, ["/nonexistent/foo.mp4"])
    assert result.exit_code == 2
    assert "not found" in result.stderr


def test_info_substring_unique_match(seeded_db):
    runner = CliRunner()
    result = runner.invoke(info, ["movie.mkv"])
    assert result.exit_code == 0, result.output
    assert "blake2b:bbbb2222" in result.output


def test_info_substring_multiple_matches(seeded_db):
    runner = CliRunner()
    result = runner.invoke(info, ["/videos"])
    assert result.exit_code == 2
    assert "multiple matches" in result.stderr


def test_humanize_bytes():
    assert _humanize_bytes(1024) == "1.0 KB"
    assert _humanize_bytes(1234567890) == "1.1 GB"
    assert _humanize_bytes(None) == "unknown"
    assert _humanize_bytes(0) == "0 B"
    assert _humanize_bytes(500) == "500 B"
