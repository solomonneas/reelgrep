"""Thin, deterministic wrappers around the ffmpeg and ffprobe binaries."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

from .config import get_settings

__all__ = ["FFmpegError", "FFmpegTimeout", "run_ffmpeg", "run_ffprobe"]

_STDERR_TAIL = 4000


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg or ffprobe invocation exits with a non-zero status."""

    def __init__(self, returncode: int, stderr: str, command: list[str]) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.command = list(command)
        super().__init__(self._format())

    def _format(self) -> str:
        tail = self.stderr[-_STDERR_TAIL:] if self.stderr else ""
        joined = " ".join(self.command)
        return f"ffmpeg command failed (rc={self.returncode}): {joined}\nstderr: {tail}"

    def __str__(self) -> str:
        return self._format()


class FFmpegTimeout(TimeoutError):
    """Raised when an ffmpeg or ffprobe invocation exceeds its timeout."""

    def __init__(self, command: list[str], timeout: float | None) -> None:
        self.command = list(command)
        self.timeout = timeout
        joined = " ".join(self.command)
        super().__init__(f"ffmpeg command timed out after {timeout}s: {joined}")


def run_ffmpeg(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    binary: str | None = None,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    """Run ffmpeg with deterministic flags, raising FFmpegError on failure."""
    exe = binary if binary is not None else get_settings().ffmpeg_binary
    cmd: list[str] = [exe, "-hide_banner", "-nostdin", *args]

    run_kwargs: dict[str, object] = {"timeout": timeout}
    if capture_stdout:
        run_kwargs["capture_output"] = True
    else:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.PIPE

    try:
        result = subprocess.run(cmd, check=False, **run_kwargs)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegTimeout(cmd, timeout if timeout is not None else exc.timeout) from exc

    if result.returncode != 0:
        stderr_bytes = result.stderr if isinstance(result.stderr, (bytes, bytearray)) else b""
        stderr_text = bytes(stderr_bytes).decode("utf-8", errors="replace")
        raise FFmpegError(result.returncode, stderr_text, cmd)

    return result


def run_ffprobe(
    args: Sequence[str],
    *,
    timeout: float | None = 30.0,
    binary: str | None = None,
) -> str:
    """Run ffprobe and return its stdout text, raising FFmpegError on failure."""
    exe = binary if binary is not None else get_settings().ffprobe_binary
    cmd: list[str] = [exe, "-hide_banner", *args]

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegTimeout(cmd, timeout if timeout is not None else exc.timeout) from exc

    if result.returncode != 0:
        stderr_bytes = result.stderr if isinstance(result.stderr, (bytes, bytearray)) else b""
        stderr_text = bytes(stderr_bytes).decode("utf-8", errors="replace")
        raise FFmpegError(result.returncode, stderr_text, cmd)

    stdout_bytes = result.stdout if isinstance(result.stdout, (bytes, bytearray)) else b""
    return bytes(stdout_bytes).decode("utf-8", errors="replace")
