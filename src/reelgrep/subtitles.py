"""Subtitle discovery, extraction, and parsing for sidecar and embedded tracks."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import pysrt
from pydantic import BaseModel, ConfigDict
from webvtt import WebVTT

from .ffmpeg_exec import run_ffmpeg
from .probe import probe

__all__ = [
    "SubtitleCue",
    "SubtitleTrack",
    "parse_sidecar",
    "find_sidecars",
    "extract_embedded",
]


_SUPPORTED_EMBEDDED_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt"}
_LANG_SUFFIX_RE = re.compile(r"^.+\.([A-Za-z]{2,3})$")
_VTT_TIMESTAMP_RE = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})[.,](\d{1,3})$"
)


def _vtt_timestamp_to_ms(value: str) -> int:
    """Convert a WebVTT/SRT-style timestamp like `00:00:03.500` to milliseconds."""
    match = _VTT_TIMESTAMP_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid WebVTT timestamp: {value!r}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis_raw = match.group(4)
    millis = int(millis_raw.ljust(3, "0")[:3])
    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis


class SubtitleCue(BaseModel):
    """A single subtitle cue with millisecond timestamps and text."""

    model_config = ConfigDict(extra="forbid")

    start_ms: int
    end_ms: int
    text: str
    language: str | None = None


class SubtitleTrack(BaseModel):
    """A subtitle track, either extracted from an embedded stream or a sidecar file."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["embedded", "sidecar", "whisper", "aligned"]
    stream_index: int | None = None
    language: str | None = None
    format: Literal[
        "srt", "vtt", "ass", "ssa", "mov_text", "whisper", "aligned", "unknown"
    ] = "unknown"
    cues: list[SubtitleCue]


def _detect_language_from_filename(path: Path) -> str | None:
    """Extract a two/three-letter language tag from a `<name>.<lang>.<ext>` basename."""
    stem = path.stem
    match = _LANG_SUFFIX_RE.match(stem)
    if match:
        return match.group(1).lower()
    return None


def _parse_srt(path: Path, language: str | None) -> list[SubtitleCue]:
    """Parse an SRT file into a list of cues, stripping inline tags when possible."""
    items = pysrt.open(str(path), encoding="utf-8")
    cues: list[SubtitleCue] = []
    for item in items:
        text_value = getattr(item, "text_without_tags", None)
        if text_value is None:
            text_value = item.text
        cues.append(
            SubtitleCue(
                start_ms=int(item.start.ordinal),
                end_ms=int(item.end.ordinal),
                text=str(text_value).strip(),
                language=language,
            )
        )
    return cues


def _parse_vtt(path: Path, language: str | None) -> list[SubtitleCue]:
    """Parse a WebVTT file into a list of cues using webvtt-py."""
    cues: list[SubtitleCue] = []
    for caption in WebVTT().read(str(path)):
        # webvtt-py's start_in_seconds/end_in_seconds round to whole seconds, so
        # parse the raw timestamp strings to preserve millisecond precision.
        start_ms = _vtt_timestamp_to_ms(caption.start)
        end_ms = _vtt_timestamp_to_ms(caption.end)
        cues.append(
            SubtitleCue(
                start_ms=start_ms,
                end_ms=end_ms,
                text=str(caption.text).strip(),
                language=language,
            )
        )
    return cues


def parse_sidecar(path: str | Path) -> SubtitleTrack:
    """Parse a sidecar subtitle file (.srt/.vtt/.ass/.ssa) into a SubtitleTrack."""
    p = Path(path)
    suffix = p.suffix.lower()
    language = _detect_language_from_filename(p)

    if suffix == ".srt":
        cues = _parse_srt(p, language)
        return SubtitleTrack(
            source="sidecar",
            stream_index=None,
            language=language,
            format="srt",
            cues=cues,
        )
    if suffix == ".vtt":
        cues = _parse_vtt(p, language)
        return SubtitleTrack(
            source="sidecar",
            stream_index=None,
            language=language,
            format="vtt",
            cues=cues,
        )
    if suffix in {".ass", ".ssa"}:
        fmt: Literal["ass", "ssa"] = "ass" if suffix == ".ass" else "ssa"
        warnings.warn(
            f"{fmt} sidecar parsing not yet supported: {p}",
            RuntimeWarning,
            stacklevel=2,
        )
        return SubtitleTrack(
            source="sidecar",
            stream_index=None,
            language=language,
            format=fmt,
            cues=[],
        )
    raise ValueError(f"Unsupported subtitle file extension: {suffix or '(none)'} ({p})")


def find_sidecars(video_path: str | Path) -> list[Path]:
    """Return sorted sidecar paths in the video's directory matching its stem."""
    video = Path(video_path)
    directory = video.parent
    if not directory.exists():
        return []
    video_stem = video.stem
    matches: list[Path] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in {".srt", ".vtt"}:
            continue
        # name like "<video_stem>.srt" or "<video_stem>.<lang>.srt"
        name_without_ext = entry.name[: -len(entry.suffix)]
        if name_without_ext == video_stem:
            matches.append(entry)
            continue
        if name_without_ext.startswith(video_stem + "."):
            # ensure the next segment is a language-like tag, not just any
            # filename starting with the same prefix. We allow any single
            # trailing segment here.
            remainder = name_without_ext[len(video_stem) + 1 :]
            if remainder and "." not in remainder:
                matches.append(entry)
    return sorted(matches)


def extract_embedded(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    ffmpeg_binary: str | None = None,
    ffprobe_binary: str | None = None,
) -> list[SubtitleTrack]:
    """Extract supported embedded subtitle streams to SRT files and parse them."""
    video = Path(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = probe(video, ffprobe_binary=ffprobe_binary)
    streams = (metadata.raw or {}).get("streams") or []

    tracks: list[SubtitleTrack] = []
    for stream in streams:
        if stream.get("codec_type") != "subtitle":
            continue
        codec_name = str(stream.get("codec_name") or "").lower()
        try:
            index = int(stream.get("index"))
        except (TypeError, ValueError):
            continue
        tags = stream.get("tags") or {}
        language = tags.get("language") if isinstance(tags, dict) else None

        if codec_name not in _SUPPORTED_EMBEDDED_CODECS:
            warnings.warn(
                f"Unsupported embedded subtitle codec {codec_name!r} on stream {index}, "
                f"skipping",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        out_path = out / f"{video.stem}.s{index}.srt"
        run_ffmpeg(
            [
                "-i",
                str(video),
                "-map",
                f"0:{index}",
                "-c:s",
                "srt",
                "-y",
                str(out_path),
            ],
            binary=ffmpeg_binary,
        )
        cues = _parse_srt(out_path, language)
        tracks.append(
            SubtitleTrack(
                source="embedded",
                stream_index=index,
                language=language,
                format="srt",
                cues=cues,
            )
        )
    return tracks
