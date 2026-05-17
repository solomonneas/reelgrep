"""ffprobe wrapper that returns typed video metadata models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .ffmpeg_exec import run_ffprobe

__all__ = [
    "VideoStream",
    "AudioStream",
    "VideoMetadata",
    "parse_fps",
    "parse_probe_json",
    "probe",
]


class VideoStream(BaseModel):
    """A single video stream as reported by ffprobe."""

    model_config = ConfigDict(extra="ignore")

    codec_name: str
    width: int
    height: int
    r_frame_rate: str
    bit_rate: int | None = None
    pix_fmt: str | None = None


class AudioStream(BaseModel):
    """A single audio stream as reported by ffprobe."""

    model_config = ConfigDict(extra="ignore")

    codec_name: str
    sample_rate: int | None = None
    channels: int | None = None
    bit_rate: int | None = None


class VideoMetadata(BaseModel):
    """Normalized metadata for a single media file."""

    model_config = ConfigDict(extra="ignore")

    path: str
    format_name: str
    duration_ms: int
    size_bytes: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    video_streams: list[VideoStream] = Field(default_factory=list)
    audio_streams: list[AudioStream] = Field(default_factory=list)
    raw: dict[str, Any]


def parse_fps(rate: str) -> float | None:
    """Convert an ffprobe r_frame_rate fraction to a float, or None if invalid."""
    if not rate:
        return None
    try:
        if "/" in rate:
            num_str, den_str = rate.split("/", 1)
            num = float(num_str)
            den = float(den_str)
            if den == 0:
                return None
            value = num / den
            if value <= 0:
                return None
            return value
        value = float(rate)
        if value <= 0:
            return None
        return value
    except (ValueError, TypeError):
        return None


def _coerce_int(value: Any) -> int | None:
    """Best-effort int coercion that tolerates strings and None."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion that tolerates strings and None."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_probe_json(raw: dict[str, Any], path: str) -> VideoMetadata:
    """Convert a raw ffprobe JSON dict into a VideoMetadata model."""
    fmt = raw.get("format") or {}
    streams = raw.get("streams") or []

    video_streams: list[VideoStream] = []
    audio_streams: list[AudioStream] = []

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            video_streams.append(
                VideoStream(
                    codec_name=str(stream.get("codec_name") or ""),
                    width=_coerce_int(stream.get("width")) or 0,
                    height=_coerce_int(stream.get("height")) or 0,
                    r_frame_rate=str(stream.get("r_frame_rate") or ""),
                    bit_rate=_coerce_int(stream.get("bit_rate")),
                    pix_fmt=stream.get("pix_fmt"),
                )
            )
        elif codec_type == "audio":
            audio_streams.append(
                AudioStream(
                    codec_name=str(stream.get("codec_name") or ""),
                    sample_rate=_coerce_int(stream.get("sample_rate")),
                    channels=_coerce_int(stream.get("channels")),
                    bit_rate=_coerce_int(stream.get("bit_rate")),
                )
            )

    primary_video = video_streams[0] if video_streams else None
    primary_audio = audio_streams[0] if audio_streams else None

    duration_seconds = _coerce_float(fmt.get("duration"))
    duration_ms = int(round(duration_seconds * 1000)) if duration_seconds is not None else 0

    fps = parse_fps(primary_video.r_frame_rate) if primary_video is not None else None

    return VideoMetadata(
        path=path,
        format_name=str(fmt.get("format_name") or ""),
        duration_ms=duration_ms,
        size_bytes=_coerce_int(fmt.get("size")),
        width=primary_video.width if primary_video is not None else None,
        height=primary_video.height if primary_video is not None else None,
        fps=fps,
        video_codec=primary_video.codec_name if primary_video is not None else None,
        audio_codec=primary_audio.codec_name if primary_audio is not None else None,
        video_streams=video_streams,
        audio_streams=audio_streams,
        raw=raw,
    )


def probe(path: str | Path, *, ffprobe_binary: str | None = None) -> VideoMetadata:
    """Run ffprobe on the given path and return parsed VideoMetadata."""
    resolved = Path(path).expanduser().resolve(strict=True)
    stdout = run_ffprobe(
        [
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(resolved),
        ],
        binary=ffprobe_binary,
    )
    raw = json.loads(stdout)
    return parse_probe_json(raw, str(resolved))
