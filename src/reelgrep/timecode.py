"""Timecode parsing and formatting helpers."""

from __future__ import annotations

import re

__all__ = ["parse", "format"]

_INT_RE = re.compile(r"^\d+$")
_FRAC_RE = re.compile(r"^(\d+)\.(\d{1,3})$")


def _parse_int(token: str) -> int:
    if not _INT_RE.match(token):
        raise ValueError(f"invalid numeric component: {token!r}")
    return int(token)


def _split_seconds(token: str) -> tuple[int, int]:
    if "." in token:
        m = _FRAC_RE.match(token)
        if not m:
            raise ValueError(f"invalid seconds component: {token!r}")
        whole = int(m.group(1))
        frac_raw = m.group(2)
        ms = int(frac_raw.ljust(3, "0"))
        return whole, ms
    return _parse_int(token), 0


def parse(s: str) -> int:
    """Parse a timecode string into milliseconds."""
    if not isinstance(s, str):
        raise ValueError(f"expected string, got {type(s).__name__}")
    stripped = s.strip()
    if not stripped:
        raise ValueError("empty timecode")
    if stripped.startswith("-"):
        raise ValueError(f"negative timecode not allowed: {s!r}")

    parts = stripped.split(":")
    if len(parts) > 3:
        raise ValueError(f"too many ':' separators: {s!r}")
    for p in parts:
        if p == "":
            raise ValueError(f"empty component in timecode: {s!r}")

    if len(parts) == 1:
        secs, ms = _split_seconds(parts[0])
        return secs * 1000 + ms

    if len(parts) == 2:
        minutes = _parse_int(parts[0])
        secs, ms = _split_seconds(parts[1])
        if secs >= 60:
            raise ValueError(f"seconds out of range in {s!r}")
        return minutes * 60_000 + secs * 1000 + ms

    hours = _parse_int(parts[0])
    minutes = _parse_int(parts[1])
    secs, ms = _split_seconds(parts[2])
    if minutes >= 60:
        raise ValueError(f"minutes out of range in {s!r}")
    if secs >= 60:
        raise ValueError(f"seconds out of range in {s!r}")
    return hours * 3_600_000 + minutes * 60_000 + secs * 1000 + ms


def format(ms: int) -> str:  # noqa: A001
    """Format milliseconds as HH:MM:SS.mmm."""
    if not isinstance(ms, int) or isinstance(ms, bool):
        raise ValueError(f"expected int, got {type(ms).__name__}")
    if ms < 0:
        raise ValueError(f"negative milliseconds not allowed: {ms}")
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
