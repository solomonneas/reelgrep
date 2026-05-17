"""Tests for reelgrep.ffmpeg_exec."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from reelgrep import ffmpeg_exec
from reelgrep.config import get_settings, reset_settings
from reelgrep.ffmpeg_exec import (
    FFmpegError,
    FFmpegTimeout,
    run_ffmpeg,
    run_ffprobe,
)

ENV_VARS = ("REELGREP_FFMPEG", "REELGREP_FFPROBE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    yield
    reset_settings()


class _MockRun:
    """Capture subprocess.run invocations and return a configured CompletedProcess."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
        raise_timeout: bool = False,
        timeout_value: float | None = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.raise_timeout = raise_timeout
        self.timeout_value = timeout_value
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(cmd), dict(kwargs)))
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(cmd, self.timeout_value or 0.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def _patch_run(monkeypatch: pytest.MonkeyPatch, mock: _MockRun) -> None:
    monkeypatch.setattr(ffmpeg_exec.subprocess, "run", mock)


def test_run_ffmpeg_builds_default_command(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun()
    _patch_run(monkeypatch, mock)
    run_ffmpeg(["-version"])
    assert len(mock.calls) == 1
    cmd, _ = mock.calls[0]
    assert cmd == [get_settings().ffmpeg_binary, "-hide_banner", "-nostdin", "-version"]


def test_run_ffmpeg_binary_override(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun()
    _patch_run(monkeypatch, mock)
    run_ffmpeg(["-i", "in.mp4"], binary="/opt/ffmpeg")
    cmd, _ = mock.calls[0]
    assert cmd == ["/opt/ffmpeg", "-hide_banner", "-nostdin", "-i", "in.mp4"]


def test_run_ffmpeg_raises_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun(returncode=1, stderr=b"boom")
    _patch_run(monkeypatch, mock)
    with pytest.raises(FFmpegError) as exc_info:
        run_ffmpeg(["-version"])
    exc = exc_info.value
    assert exc.returncode == 1
    assert "boom" in exc.stderr
    rendered = str(exc)
    expected_cmd = " ".join(
        [get_settings().ffmpeg_binary, "-hide_banner", "-nostdin", "-version"]
    )
    assert expected_cmd in rendered
    assert "boom" in rendered


def test_run_ffmpeg_raises_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun(raise_timeout=True, timeout_value=5.0)
    _patch_run(monkeypatch, mock)
    with pytest.raises(FFmpegTimeout) as exc_info:
        run_ffmpeg(["-version"], timeout=5.0)
    exc = exc_info.value
    assert exc.timeout == 5.0
    assert exc.command == [
        get_settings().ffmpeg_binary,
        "-hide_banner",
        "-nostdin",
        "-version",
    ]


def test_run_ffmpeg_capture_stdout_true(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun()
    _patch_run(monkeypatch, mock)
    run_ffmpeg(["-version"], capture_stdout=True)
    _, kwargs = mock.calls[0]
    assert kwargs.get("capture_output") is True
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs


def test_run_ffmpeg_capture_stdout_false_discards_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun()
    _patch_run(monkeypatch, mock)
    run_ffmpeg(["-version"])
    _, kwargs = mock.calls[0]
    assert kwargs.get("stdout") is subprocess.DEVNULL
    assert kwargs.get("stderr") is subprocess.PIPE
    assert "capture_output" not in kwargs


def test_run_ffprobe_builds_default_command(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun(stdout=b"{}")
    _patch_run(monkeypatch, mock)
    run_ffprobe(["-show_format", "foo.mp4"])
    cmd, kwargs = mock.calls[0]
    assert cmd == [
        get_settings().ffprobe_binary,
        "-hide_banner",
        "-show_format",
        "foo.mp4",
    ]
    assert kwargs.get("capture_output") is True


def test_run_ffprobe_returns_decoded_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = "ffprobe version 7.0\n"
    mock = _MockRun(stdout=payload.encode("utf-8"))
    _patch_run(monkeypatch, mock)
    out = run_ffprobe(["-version"])
    assert out == payload


def test_run_ffprobe_raises_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun(returncode=2, stderr=b"missing file")
    _patch_run(monkeypatch, mock)
    with pytest.raises(FFmpegError) as exc_info:
        run_ffprobe(["-show_format", "nope.mp4"])
    exc = exc_info.value
    assert exc.returncode == 2
    assert "missing file" in exc.stderr


def test_run_ffprobe_binary_override(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun(stdout=b"")
    _patch_run(monkeypatch, mock)
    run_ffprobe(["-version"], binary="/opt/ffprobe")
    cmd, _ = mock.calls[0]
    assert cmd[0] == "/opt/ffprobe"


def test_run_ffprobe_raises_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = _MockRun(raise_timeout=True, timeout_value=3.0)
    _patch_run(monkeypatch, mock)
    with pytest.raises(FFmpegTimeout) as exc_info:
        run_ffprobe(["-version"], timeout=3.0)
    assert exc_info.value.timeout == 3.0


@pytest.mark.integration
def test_run_ffprobe_real_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REELGREP_FFPROBE", "/usr/bin/ffprobe")
    reset_settings()
    out = run_ffprobe(["-version"])
    assert "ffprobe version" in out
