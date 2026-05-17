"""Tests for the ``reelgrep make-gif`` command."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from reelgrep import manifest as manifest_mod
from reelgrep.commands.make_gif import make_gif
from reelgrep.config import reset_settings
from reelgrep.db import connect, migrate
from reelgrep.probe import VideoMetadata

ENV_VARS = (
    "REELGREP_HOME",
    "REELGREP_DB",
    "REELGREP_CACHE",
    "REELGREP_FFMPEG",
    "REELGREP_FFPROBE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path))
    reset_settings()
    yield
    reset_settings()


def _make_video_file(path: Path, contents: bytes = b"not really a video") -> Path:
    path.write_bytes(contents)
    return path


def _fake_metadata(video_path: Path) -> VideoMetadata:
    return VideoMetadata(
        path=str(video_path),
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        duration_ms=10_000,
        size_bytes=video_path.stat().st_size,
        width=640,
        height=360,
        fps=24.0,
        video_codec="h264",
        audio_codec="aac",
        raw={"format": {}, "streams": []},
    )


class _FFmpegRecorder:
    """Records ffmpeg calls and writes a placeholder file at the requested output."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.call_count: int = 0

    def __call__(
        self,
        args: list[str],
        *,
        timeout: float | None = None,
        binary: str | None = None,
        capture_stdout: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        arg_list = list(args)
        self.calls.append(arg_list)
        self.call_count += 1
        # Last arg is the output path (per all command shapes in make_gif).
        out_arg = arg_list[-1]
        out_path = Path(out_arg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"stub-output")
        return subprocess.CompletedProcess(args=[binary or "ffmpeg", *arg_list], returncode=0)


@pytest.fixture
def patched_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> _FFmpegRecorder:
    recorder = _FFmpegRecorder()
    monkeypatch.setattr("reelgrep.commands.make_gif.run_ffmpeg", recorder)
    return recorder


@pytest.fixture
def patched_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def _probe(path: Any, **_kwargs: Any) -> VideoMetadata:
        return _fake_metadata(Path(path))

    monkeypatch.setattr("reelgrep.commands.make_gif.probe", _probe)


@pytest.fixture
def patched_hash(monkeypatch: pytest.MonkeyPatch) -> str:
    digest = "blake2b:" + "ab" * 32
    monkeypatch.setattr("reelgrep.commands.make_gif.file_hash", lambda _p: digest)
    return digest


def _invoke(args: list[str]) -> Any:
    runner = CliRunner()
    return runner.invoke(make_gif, args, catch_exceptions=False)


def test_duration_zero_raises_bad_parameter(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.webp"
    result = _invoke(
        [str(video), "--start", "0:00:00", "--duration", "0", "--out", str(out)]
    )
    assert result.exit_code != 0
    assert "--duration" in (result.stderr or "")
    assert patched_ffmpeg.call_count == 0


def test_fps_zero_raises_bad_parameter(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.webp"
    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:00",
            "--duration",
            "1",
            "--out",
            str(out),
            "--fps",
            "0",
        ]
    )
    assert result.exit_code != 0
    assert "--fps" in (result.stderr or "")
    assert patched_ffmpeg.call_count == 0


def test_width_too_small_raises_bad_parameter(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.webp"
    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:00",
            "--duration",
            "1",
            "--out",
            str(out),
            "--width",
            "16",
        ]
    )
    assert result.exit_code != 0
    assert "--width" in (result.stderr or "")
    assert patched_ffmpeg.call_count == 0


def test_webp_single_pass(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.webp"
    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:01.500",
            "--duration",
            "2",
            "--out",
            str(out),
            "--no-db",
        ]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert patched_ffmpeg.call_count == 1
    args = patched_ffmpeg.calls[0]
    assert "-c:v" in args
    assert args[args.index("-c:v") + 1] == "libwebp"
    assert out.exists()


def test_gif_two_pass(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.gif"
    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:00",
            "--duration",
            "1",
            "--out",
            str(out),
            "--no-db",
        ]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert patched_ffmpeg.call_count == 2
    first_args = patched_ffmpeg.calls[0]
    second_args = patched_ffmpeg.calls[1]
    assert any("palettegen" in a for a in first_args)
    assert any("paletteuse" in a for a in second_args)


def test_unknown_extension_rewrites_to_webp(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.xyz"
    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:00",
            "--duration",
            "1",
            "--out",
            str(out),
            "--no-db",
        ]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert patched_ffmpeg.call_count == 1
    args = patched_ffmpeg.calls[0]
    out_arg = args[-1]
    assert out_arg.endswith(".webp")
    rewritten = tmp_path / "out.webp"
    assert rewritten.exists()


def test_manifest_written(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.webp"
    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:02",
            "--duration",
            "1.5",
            "--out",
            str(out),
            "--fps",
            "15",
            "--width",
            "320",
            "--no-db",
        ]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    actual_out_path = Path(patched_ffmpeg.calls[0][-1])
    sidecar = manifest_mod.sidecar_path(actual_out_path)
    assert sidecar.exists()
    manifest = manifest_mod.read(sidecar)
    assert manifest.operation == "make-gif"
    assert manifest.parameters["start_ms"] == 2000
    assert manifest.parameters["duration_seconds"] == 1.5
    assert manifest.parameters["fps"] == 15
    assert manifest.parameters["width"] == 320
    assert manifest.parameters["format"] in {"webp", "gif"}


def test_no_db_skips_export_artifacts(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.webp"
    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:00",
            "--duration",
            "1",
            "--out",
            str(out),
            "--no-db",
        ]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    db_path = tmp_path / "index.sqlite"
    if not db_path.exists():
        return
    conn = connect(db_path)
    try:
        migrate(conn)
        rows = conn.execute(
            "SELECT COUNT(*) FROM export_artifacts WHERE kind='gif'"
        ).fetchone()
        assert rows[0] == 0
    finally:
        conn.close()


def _seed_video_row(db_path: Path, digest: str, video_path: Path) -> int:
    conn = connect(db_path)
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) "
            "VALUES (?, ?, ?, ?)",
            (digest, str(video_path), "2026-01-01T00:00:00Z", "{}"),
        )
        conn.commit()
        row = conn.execute("SELECT id FROM videos WHERE file_hash=?", (digest,)).fetchone()
        return int(row[0])
    finally:
        conn.close()


def test_db_row_inserted_when_video_indexed(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.webp"
    db_path = tmp_path / "index.sqlite"
    _seed_video_row(db_path, patched_hash, video.resolve())

    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:00",
            "--duration",
            "1",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT kind, path FROM export_artifacts WHERE kind='gif'"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "gif"


def test_missing_video_row_warns_and_skips(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.webp"
    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:00",
            "--duration",
            "1",
            "--out",
            str(out),
        ]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert "warning" in (result.stderr or "").lower()
    db_path = tmp_path / "index.sqlite"
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT COUNT(*) FROM export_artifacts WHERE kind='gif'"
        ).fetchone()
        assert rows[0] == 0
    finally:
        conn.close()


def test_gif_palette_cleanup(
    tmp_path: Path,
    patched_ffmpeg: _FFmpegRecorder,
    patched_probe: None,
    patched_hash: str,
) -> None:
    video = _make_video_file(tmp_path / "v.mp4")
    out = tmp_path / "out.gif"
    result = _invoke(
        [
            str(video),
            "--start",
            "0:00:00",
            "--duration",
            "1",
            "--out",
            str(out),
            "--no-db",
        ]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    hex_part = patched_hash[len("blake2b:") :]
    palette = tmp_path / "cache" / "tmp" / f"{hex_part[8:24]}_palette.png"
    assert not palette.exists()


@pytest.mark.integration
def test_integration_real_webp(tmp_path: Path) -> None:
    """End-to-end render using a real ffmpeg-generated source clip."""
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    src = tmp_path / "src.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=160x120:d=4:r=12",
        "-pix_fmt",
        "yuv420p",
        str(src),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert src.exists()

    out = tmp_path / "loop.webp"
    result = _invoke(
        [
            str(src),
            "--start",
            "0:00:01",
            "--duration",
            "2",
            "--out",
            str(out),
            "--no-db",
        ]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert out.exists()
    assert out.stat().st_size > 0
    with open(out, "rb") as fh:
        magic = fh.read(4)
    assert magic == b"RIFF"
