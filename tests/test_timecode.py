from __future__ import annotations

import pytest

from reelgrep.timecode import format as fmt_timecode
from reelgrep.timecode import parse

ROUND_TRIP = [
    (0, "00:00:00.000"),
    (1, "00:00:00.001"),
    (1000, "00:00:01.000"),
    (61500, "00:01:01.500"),
    (3661000, "01:01:01.000"),
    (90061500, "25:01:01.500"),
]


@pytest.mark.parametrize(("ms", "s"), ROUND_TRIP)
def test_format_matches_table(ms: int, s: str) -> None:
    assert fmt_timecode(ms) == s


@pytest.mark.parametrize(("ms", "s"), ROUND_TRIP)
def test_parse_matches_table(ms: int, s: str) -> None:
    assert parse(s) == ms


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5", 5000),
        ("1:30", 90000),
        ("1:30.250", 90250),
        ("  00:01:00  ", 60000),
        ("100:00:00", 360_000_000),
    ],
)
def test_parse_acceptance(text: str, expected: int) -> None:
    assert parse(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "abc",
        "1:2:3:4",
        "1::30",
        "-1",
        "00:00:-1",
        "00:60:00",
        "00:00:60",
        "0:0:0.9999",
    ],
)
def test_parse_malformed_raises(text: str) -> None:
    with pytest.raises(ValueError):
        parse(text)


def test_format_negative_raises() -> None:
    with pytest.raises(ValueError):
        fmt_timecode(-1)
