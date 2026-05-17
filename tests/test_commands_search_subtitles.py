"""Tests for ``reelgrep search-subtitles`` command."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from reelgrep import config, db, hashing, manifest
from reelgrep.commands.search_subtitles import search_subtitles

ENV_VARS = (
    "REELGREP_HOME",
    "REELGREP_DB",
    "REELGREP_CACHE",
    "REELGREP_FFMPEG",
    "REELGREP_FFPROBE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path))
    config.reset_settings()
    yield
    config.reset_settings()


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    config.reset_settings()
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"\x00fake")
    h = hashing.file_hash(video)
    s = config.get_settings()
    config.ensure_dirs(s)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    conn.execute(
        "INSERT INTO videos (file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (h, str(video), 600000, "2026-05-17T00:00:00", "{}"),
    )
    vid = conn.execute(
        "SELECT id FROM videos WHERE file_hash = ?", (h,)
    ).fetchone()[0]
    cues = [
        (vid, "en", "sidecar", None, 1000, 4000, "Welcome to kubernetes networking"),
        (vid, "en", "sidecar", None, 5000, 8000, "Pods talk to each other over the pod network"),
        (vid, "en", "sidecar", None, 9000, 12000, "Services give pods a stable address"),
    ]
    for c in cues:
        cur = conn.execute(
            "INSERT INTO subtitles "
            "(video_id, language, source, stream_index, start_ms, end_ms, text) "
            "VALUES (?,?,?,?,?,?,?)",
            c,
        )
        conn.execute(
            "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
            (cur.lastrowid, c[6]),
        )
    conn.commit()
    conn.close()
    return video


def _make_recording_ffmpeg(
    calls: list[list[str]],
    *,
    fail_on_call: int | None = None,
) -> Any:
    def fake_run(
        args: Sequence[str],
        *,
        timeout: float | None = None,
        binary: str | None = None,
        capture_stdout: bool = False,
    ) -> None:
        from reelgrep.ffmpeg_exec import FFmpegError

        call_index = len(calls)
        calls.append(list(args))
        if fail_on_call is not None and call_index == fail_on_call:
            raise FFmpegError(1, "boom", ["ffmpeg", *args])
        out_path = Path(args[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\xff\xd8stub-jpg")
        return None

    return fake_run


def test_export_frames_requires_out(seeded_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        search_subtitles,
        [str(seeded_db), "kubernetes", "--export-frames"],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr if result.stderr_bytes is not None else "")
    assert "--out" in combined


def test_limit_zero_rejected(seeded_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        search_subtitles,
        [str(seeded_db), "kubernetes", "--limit", "0"],
    )
    assert result.exit_code != 0
    assert "limit" in result.output.lower() or "limit" in str(result.exception).lower()


def test_video_not_ingested_exits_2(tmp_path: Path) -> None:
    config.reset_settings()
    s = config.get_settings()
    config.ensure_dirs(s)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    conn.close()

    other = tmp_path / "unknown.mp4"
    other.write_bytes(b"\x00ghost")

    runner = CliRunner()
    result = runner.invoke(search_subtitles, [str(other), "kubernetes"])
    assert result.exit_code == 2
    assert "not ingested" in result.stderr


def test_single_match(seeded_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(search_subtitles, [str(seeded_db), "kubernetes"])
    assert result.exit_code == 0
    assert "Welcome to kubernetes networking" in result.output
    assert "00:00:01.000" in result.output
    assert "1 matches" in result.output


def test_multi_match_with_stemming(seeded_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(search_subtitles, [str(seeded_db), "pod"])
    assert result.exit_code == 0
    assert "Pods talk to each other over the pod network" in result.output
    assert "Services give pods a stable address" in result.output
    assert "2 matches" in result.output


def test_no_match_writes_no_manifest(seeded_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(search_subtitles, [str(seeded_db), "xyzzy"])
    assert result.exit_code == 0
    assert "no matches" in result.output
    sibling = seeded_db.with_suffix(seeded_db.suffix + ".search.manifest.json")
    assert not sibling.exists()


def test_limit_one(seeded_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        search_subtitles, [str(seeded_db), "pod", "--limit", "1"]
    )
    assert result.exit_code == 0
    assert "Pods talk to each other over the pod network" in result.output
    assert "Services give pods a stable address" not in result.output
    assert "1 matches" in result.output


def test_export_frames_success(
    seeded_db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "reelgrep.commands.search_subtitles.run_ffmpeg",
        _make_recording_ffmpeg(calls),
    )
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        search_subtitles,
        [
            str(seeded_db),
            "pod",
            "--export-frames",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    # Each call should produce a .jpg in out_dir
    jpgs = sorted(out_dir.glob("*.jpg"))
    assert len(jpgs) == 2
    for jpg in jpgs:
        assert jpg.exists()

    manifest_path = out_dir / "search-subtitles.manifest.json"
    assert manifest_path.exists()
    m = manifest.read(manifest_path)
    assert m.operation == "search-subtitles"
    assert len(m.results) == 2
    assert m.results[0]["frame_path"] is not None
    assert m.results[1]["frame_path"] is not None


def test_export_frames_second_call_fails(
    seeded_db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "reelgrep.commands.search_subtitles.run_ffmpeg",
        _make_recording_ffmpeg(calls, fail_on_call=1),
    )
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        search_subtitles,
        [
            str(seeded_db),
            "pod",
            "--export-frames",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert "warning" in result.stderr.lower() or "failed" in result.stderr.lower()

    manifest_path = out_dir / "search-subtitles.manifest.json"
    assert manifest_path.exists()
    m = manifest.read(manifest_path)
    assert len(m.results) == 2
    frame_paths = [r["frame_path"] for r in m.results]
    assert sum(1 for fp in frame_paths if fp is not None) == 1
    assert sum(1 for fp in frame_paths if fp is None) == 1


def test_context_ms_offsets_ss_arg(
    seeded_db: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "reelgrep.commands.search_subtitles.run_ffmpeg",
        _make_recording_ffmpeg(calls),
    )
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        search_subtitles,
        [
            str(seeded_db),
            "kubernetes",
            "--export-frames",
            "--out",
            str(out_dir),
            "--context-ms",
            "2000",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    args = calls[0]
    ss_index = args.index("-ss")
    ss_value = args[ss_index + 1]
    # cue 1 starts at 1000 ms, context_ms=2000 -> 3000 ms -> 3.000
    assert ss_value == f"{(1000 + 2000) / 1000:.3f}"


def test_default_manifest_path_sibling(seeded_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(search_subtitles, [str(seeded_db), "pod"])
    assert result.exit_code == 0
    manifest_path = seeded_db.with_suffix(
        seeded_db.suffix + ".search.manifest.json"
    )
    assert manifest_path.exists()
    m = manifest.read(manifest_path)
    assert m.operation == "search-subtitles"
    assert len(m.results) == 2


def test_final_line_match_count(seeded_db: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(search_subtitles, [str(seeded_db), "kubernetes"])
    assert result.exit_code == 0
    lines = [ln for ln in result.output.strip().splitlines() if ln.strip()]
    assert lines[-1] == "1 matches"
