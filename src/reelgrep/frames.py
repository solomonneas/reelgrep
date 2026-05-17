"""Frame extraction helpers: uniform sampling and scene-change detection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .ffmpeg_exec import run_ffmpeg
from .probe import probe
from .timecode import format as format_timecode

__all__ = ["Frame", "sample_every", "sample_scenes"]

_PTS_TIME_RE = re.compile(r"pts_time:([\d.]+)")


class Frame(BaseModel):
    """A single extracted frame with its timestamp and on-disk path."""

    model_config = ConfigDict(extra="forbid")

    timestamp_ms: int
    path: str
    sampling_strategy: Literal["every_n", "scene", "manual"]
    width: int | None = None
    height: int | None = None


def _frame_filename(stem: str, ts_ms: int) -> str:
    """Return the deterministic JPEG filename for a given stem and timestamp."""
    return f"{stem}_{format_timecode(ts_ms)}.jpg"


def _decode_stderr(raw: bytes | str | None) -> str:
    """Decode ffmpeg stderr bytes (or str/None) into a string."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return bytes(raw).decode("utf-8", errors="replace")


def sample_every(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    interval_seconds: float = 5.0,
    ffmpeg_binary: str | None = None,
    ffprobe_binary: str | None = None,
) -> list[Frame]:
    """Extract one JPEG every interval_seconds across the video duration."""
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")

    resolved = Path(video_path).expanduser().resolve(strict=True)
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    metadata = probe(resolved, ffprobe_binary=ffprobe_binary)
    duration_ms = metadata.duration_ms
    if duration_ms <= 0:
        raise ValueError(f"video duration must be positive, got {duration_ms}")

    step_ms = int(interval_seconds * 1000)
    timestamps: list[int] = []
    ts = 0
    while ts <= duration_ms:
        timestamps.append(ts)
        ts += step_ms

    stem = resolved.stem
    frames: list[Frame] = []
    for ts_ms in timestamps:
        filename = _frame_filename(stem, ts_ms)
        out_file = out_path_dir / filename
        seconds = f"{ts_ms / 1000:.3f}"
        run_ffmpeg(
            [
                "-ss",
                seconds,
                "-i",
                str(resolved),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-y",
                str(out_file),
            ],
            binary=ffmpeg_binary,
        )
        frames.append(
            Frame(
                timestamp_ms=ts_ms,
                path=str(out_file.resolve()),
                sampling_strategy="every_n",
            )
        )
    return frames


def sample_scenes(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    threshold: float = 0.30,
    max_frames: int = 200,
    ffmpeg_binary: str | None = None,
    ffprobe_binary: str | None = None,  # noqa: ARG001
) -> list[Frame]:
    """Extract JPEGs at scene-change boundaries detected by ffmpeg's select filter."""
    if threshold <= 0 or threshold >= 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    resolved = Path(video_path).expanduser().resolve(strict=True)
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    stem = resolved.stem
    pattern = str(out_path_dir / f"{stem}_scene_%05d.jpg")
    result = run_ffmpeg(
        [
            "-i",
            str(resolved),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-vsync",
            "vfr",
            "-f",
            "image2",
            pattern,
        ],
        binary=ffmpeg_binary,
        capture_stdout=True,
    )

    stderr_text = _decode_stderr(result.stderr)
    pts_matches = _PTS_TIME_RE.findall(stderr_text)
    timestamps_ms = [int(float(m) * 1000) for m in pts_matches]

    if max_frames > 0:
        timestamps_ms = timestamps_ms[:max_frames]

    frames: list[Frame] = []
    for index, ts_ms in enumerate(timestamps_ms, start=1):
        scene_file = out_path_dir / f"{stem}_scene_{index:05d}.jpg"
        if not scene_file.exists():
            continue
        final_name = _frame_filename(stem, ts_ms)
        final_file = out_path_dir / final_name
        if final_file.exists() and final_file != scene_file:
            final_file.unlink()
        scene_file.rename(final_file)
        frames.append(
            Frame(
                timestamp_ms=ts_ms,
                path=str(final_file.resolve()),
                sampling_strategy="scene",
            )
        )

    if max_frames > 0:
        for leftover in out_path_dir.glob(f"{stem}_scene_*.jpg"):
            leftover.unlink()

    return frames
