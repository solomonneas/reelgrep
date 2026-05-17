"""Tests for ``reelgrep export-clip``."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from reelgrep.commands.export_clip import export_clip
from reelgrep.config import get_settings, reset_settings
from reelgrep.db import connect, migrate
from reelgrep.manifest import read as read_manifest
from reelgrep.manifest import sidecar_path
from reelgrep.probe import VideoMetadata, probe


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path / "home"))
    reset_settings()
    yield
    reset_settings()


def _video_file(tmp_path: Path) -> Path:
    p = tmp_path / "src.mp4"
    p.write_bytes(b"\x00" * 1024)
    return p


def _fake_meta(path: Path, duration_ms: int) -> VideoMetadata:
    return VideoMetadata(
        path=str(path),
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        duration_ms=duration_ms,
        size_bytes=1024,
        width=160,
        height=120,
        fps=12.0,
        video_codec="h264",
        audio_codec=None,
        video_streams=[],
        audio_streams=[],
        raw={},
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration_ms: int,
    recorder: list[list[str]] | None = None,
    write_size: int = 256,
) -> None:
    def fake_probe(path: str | Path, **_: Any) -> VideoMetadata:
        return _fake_meta(Path(path), duration_ms)

    def fake_run_ffmpeg(args: Any, **_: Any) -> None:
        arg_list = list(args)
        if recorder is not None:
            recorder.append(arg_list)
        # Find the output path: last positional arg.
        out = Path(arg_list[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * write_size)

    monkeypatch.setattr("reelgrep.commands.export_clip.probe", fake_probe)
    monkeypatch.setattr("reelgrep.commands.export_clip.run_ffmpeg", fake_run_ffmpeg)


def test_bad_timecode_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    _install_fakes(monkeypatch, duration_ms=60_000)
    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [str(video), "--start", "abc", "--end", "00:01:00", "--out", str(out)],
    )
    assert result.exit_code != 0


def test_end_before_start_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    _install_fakes(monkeypatch, duration_ms=600_000)
    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [str(video), "--start", "00:02:00", "--end", "00:01:00", "--out", str(out)],
    )
    assert result.exit_code != 0
    assert "--end must be after --start" in result.output


def test_end_past_duration_is_clamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    _install_fakes(monkeypatch, duration_ms=5_000)
    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [str(video), "--start", "0:00:01", "--end", "0:00:10", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "clamping" in result.output


def test_start_past_end_after_clamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    _install_fakes(monkeypatch, duration_ms=1_000)
    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [str(video), "--start", "0:00:05", "--end", "0:00:10", "--out", str(out)],
    )
    assert result.exit_code != 0
    assert "past end" in result.output


def test_stream_copy_command_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    recorder: list[list[str]] = []
    _install_fakes(monkeypatch, duration_ms=300_000, recorder=recorder)
    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [
            str(video),
            "--start",
            "0:01:00",
            "--end",
            "0:01:30",
            "--out",
            str(out),
            "--no-db",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(recorder) == 1
    args = recorder[0]
    expected = [
        "-ss",
        "60.000",
        "-to",
        "90.000",
        "-i",
        str(video.resolve()),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-y",
        str(out.resolve()),
    ]
    assert args == expected


def test_reencode_command_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    recorder: list[list[str]] = []
    _install_fakes(monkeypatch, duration_ms=300_000, recorder=recorder)
    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [
            str(video),
            "--start",
            "0:01:00",
            "--end",
            "0:01:30",
            "--out",
            str(out),
            "--reencode",
            "--no-db",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(recorder) == 1
    args = recorder[0]
    # -i must come before -ss for accurate seeking.
    assert args.index("-i") < args.index("-ss")
    # Check the encoder block in order.
    for token in [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
    ]:
        assert token in args
    assert args.index("-c:v") + 1 == args.index("libx264")
    assert args.index("-preset") + 1 == args.index("veryfast")
    assert args.index("-crf") + 1 == args.index("20")
    assert args.index("-c:a") + 1 == args.index("aac")
    assert args.index("-b:a") + 1 == args.index("128k")


def test_manifest_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    _install_fakes(monkeypatch, duration_ms=300_000)
    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [
            str(video),
            "--start",
            "0:01:00",
            "--end",
            "0:01:30",
            "--out",
            str(out),
            "--no-db",
        ],
    )
    assert result.exit_code == 0, result.output

    manifest = read_manifest(sidecar_path(out.resolve()))
    assert manifest.operation == "export-clip"
    assert manifest.parameters["start_ms"] == 60_000
    assert manifest.parameters["end_ms"] == 90_000
    assert manifest.parameters["reencode"] is False
    assert manifest.results[0]["output_path"] == str(out.resolve())
    assert manifest.results[0]["duration_ms"] == 30_000
    assert manifest.results[0]["size_bytes"] == 256


def test_no_db_flag_skips_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    _install_fakes(monkeypatch, duration_ms=300_000)
    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [
            str(video),
            "--start",
            "0:01:00",
            "--end",
            "0:01:30",
            "--out",
            str(out),
            "--no-db",
        ],
    )
    assert result.exit_code == 0, result.output

    settings = get_settings()
    if settings.db_path.exists():
        conn = connect(settings.db_path)
        try:
            migrate(conn)
            count = conn.execute("SELECT COUNT(*) FROM export_artifacts").fetchone()[0]
            assert count == 0
        finally:
            conn.close()


def test_db_row_written_when_video_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    _install_fakes(monkeypatch, duration_ms=300_000)

    # Pre-insert a videos row keyed by the real hash of the source file.
    from reelgrep.hashing import file_hash

    digest = file_hash(video)
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    try:
        migrate(conn)
        with conn:
            conn.execute(
                "INSERT INTO videos (file_hash, path, duration_ms, ingested_at, probe_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (digest, str(video.resolve()), 300_000, "2026-01-01T00:00:00Z", "{}"),
            )
        video_id = conn.execute(
            "SELECT id FROM videos WHERE file_hash = ?", (digest,)
        ).fetchone()[0]
    finally:
        conn.close()

    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [
            str(video),
            "--start",
            "0:01:00",
            "--end",
            "0:01:30",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    conn = connect(settings.db_path)
    try:
        rows = conn.execute(
            "SELECT video_id, kind, path, start_ms, end_ms, manifest_path "
            "FROM export_artifacts"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    row = rows[0]
    assert row["video_id"] == video_id
    assert row["kind"] == "clip"
    assert row["path"] == str(out.resolve())
    assert row["start_ms"] == 60_000
    assert row["end_ms"] == 90_000
    assert row["manifest_path"] == str(sidecar_path(out.resolve()))


def test_warning_when_video_not_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video_file(tmp_path)
    out = tmp_path / "out.mp4"
    _install_fakes(monkeypatch, duration_ms=300_000)

    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [
            str(video),
            "--start",
            "0:01:00",
            "--end",
            "0:01:30",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "video not in index" in result.output
    # Manifest still written.
    manifest = read_manifest(sidecar_path(out.resolve()))
    assert manifest.operation == "export-clip"
    # No row in export_artifacts.
    settings = get_settings()
    conn = connect(settings.db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM export_artifacts").fetchone()[0]
    finally:
        conn.close()
    assert count == 0


@pytest.mark.integration
def test_real_ffmpeg_export_reencode(tmp_path: Path) -> None:
    src = tmp_path / "blue.mp4"
    subprocess.run(
        [
            "/usr/bin/ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:d=5:r=12",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    assert src.exists() and src.stat().st_size > 0

    out = tmp_path / "out.mp4"
    runner = CliRunner()
    result = runner.invoke(
        export_clip,
        [
            str(src),
            "--start",
            "0:00:01",
            "--end",
            "0:00:03",
            "--out",
            str(out),
            "--reencode",
            "--no-db",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0

    meta = probe(out)
    duration_s = meta.duration_ms / 1000
    assert 1.5 <= duration_s <= 2.5, f"unexpected duration: {duration_s}s"
