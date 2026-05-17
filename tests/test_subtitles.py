"""Tests for reelgrep.subtitles."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from reelgrep.config import reset_settings
from reelgrep.subtitles import (
    SubtitleCue,
    SubtitleTrack,
    extract_embedded,
    find_sidecars,
    parse_sidecar,
)

ENV_VARS = (
    "REELGREP_HOME",
    "REELGREP_DB",
    "REELGREP_CACHE",
    "REELGREP_FFMPEG",
    "REELGREP_FFPROBE",
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_SRT = FIXTURES / "sample.srt"
SAMPLE_VTT = FIXTURES / "sample.vtt"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    yield
    reset_settings()


def test_parse_sidecar_srt() -> None:
    track = parse_sidecar(SAMPLE_SRT)
    assert track.source == "sidecar"
    assert track.format == "srt"
    assert track.stream_index is None
    assert track.language is None
    assert len(track.cues) == 3
    first = track.cues[0]
    assert first.start_ms == 1000
    assert first.end_ms == 3500
    assert first.text == "Welcome to the lecture."
    assert all(c.language is None for c in track.cues)


def test_parse_sidecar_vtt() -> None:
    track = parse_sidecar(SAMPLE_VTT)
    assert track.source == "sidecar"
    assert track.format == "vtt"
    assert track.stream_index is None
    assert track.language is None
    assert len(track.cues) == 3
    first = track.cues[0]
    assert first.start_ms == 1000
    assert first.end_ms == 3500
    assert first.text == "Welcome to the lecture."
    second = track.cues[1]
    assert second.start_ms == 4000
    assert second.end_ms == 7250


def test_parse_sidecar_language_from_filename(tmp_path: Path) -> None:
    tagged = tmp_path / "movie.en.srt"
    shutil.copy(SAMPLE_SRT, tagged)
    track = parse_sidecar(tagged)
    assert track.language == "en"
    assert all(c.language == "en" for c in track.cues)


def test_parse_sidecar_no_language_when_plain(tmp_path: Path) -> None:
    plain = tmp_path / "movie.srt"
    shutil.copy(SAMPLE_SRT, plain)
    track = parse_sidecar(plain)
    assert track.language is None
    assert all(c.language is None for c in track.cues)


def test_parse_sidecar_unknown_extension_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "subs.xyz"
    bogus.write_text("not a subtitle\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_sidecar(bogus)


def test_parse_sidecar_ass_warns_and_returns_empty(tmp_path: Path) -> None:
    ass_path = tmp_path / "x.ass"
    ass_path.write_text("[Script Info]\nTitle: stub\n", encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        track = parse_sidecar(ass_path)
    assert track.format == "ass"
    assert track.cues == []
    assert track.source == "sidecar"


def test_find_sidecars(tmp_path: Path) -> None:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")
    expected = [
        tmp_path / "movie.srt",
        tmp_path / "movie.en.srt",
        tmp_path / "movie.vtt",
    ]
    for p in expected:
        p.write_text("stub", encoding="utf-8")
    (tmp_path / "other.srt").write_text("stub", encoding="utf-8")
    (tmp_path / "movie.txt").write_text("stub", encoding="utf-8")

    found = find_sidecars(video)
    assert found == sorted(expected)


def test_subtitle_cue_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SubtitleCue(start_ms=0, end_ms=1, text="x", language=None, bogus="nope")


def test_subtitle_track_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SubtitleTrack(
            source="sidecar",
            stream_index=None,
            language=None,
            format="srt",
            cues=[],
            bogus="nope",
        )


class _FakeMetadata:
    """Minimal stand-in for VideoMetadata exposing only `.raw`."""

    def __init__(self, streams: list[dict[str, Any]]) -> None:
        self.raw = {"streams": streams}


def _make_fake_probe(streams: list[dict[str, Any]]) -> Any:
    def fake_probe(_path: Any, *, ffprobe_binary: str | None = None) -> _FakeMetadata:
        return _FakeMetadata(streams)

    return fake_probe


def _make_fake_run_ffmpeg(srt_payload: str, calls: list[list[str]]) -> Any:
    def fake_run(args: list[str], *, binary: str | None = None, **_kwargs: Any) -> None:
        calls.append(list(args))
        out_path = Path(args[-1])
        out_path.write_text(srt_payload, encoding="utf-8")
        return None

    return fake_run


def test_extract_embedded_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    streams = [
        {
            "codec_type": "subtitle",
            "codec_name": "subrip",
            "index": 2,
            "tags": {"language": "eng"},
        }
    ]
    monkeypatch.setattr("reelgrep.subtitles.probe", _make_fake_probe(streams))
    payload = SAMPLE_SRT.read_text(encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "reelgrep.subtitles.run_ffmpeg", _make_fake_run_ffmpeg(payload, calls)
    )

    tracks = extract_embedded("/fake/movie.mp4", tmp_path)
    assert len(tracks) == 1
    track = tracks[0]
    assert track.source == "embedded"
    assert track.stream_index == 2
    assert track.language == "eng"
    assert track.format == "srt"
    assert len(track.cues) == 3
    assert all(c.language == "eng" for c in track.cues)
    assert track.cues[0].start_ms == 1000
    assert track.cues[0].end_ms == 3500
    assert len(calls) == 1
    out_arg = calls[0][-1]
    assert out_arg.endswith("movie.s2.srt")


def test_extract_embedded_skips_unsupported_codec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    streams = [
        {
            "codec_type": "subtitle",
            "codec_name": "hdmv_pgs_subtitle",
            "index": 3,
            "tags": {"language": "eng"},
        }
    ]
    monkeypatch.setattr("reelgrep.subtitles.probe", _make_fake_probe(streams))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "reelgrep.subtitles.run_ffmpeg", _make_fake_run_ffmpeg("", calls)
    )

    with pytest.warns(RuntimeWarning):
        tracks = extract_embedded("/fake/movie.mp4", tmp_path)
    assert tracks == []
    assert len(calls) == 0


def test_extract_embedded_no_subtitle_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    streams = [
        {"codec_type": "video", "codec_name": "h264", "index": 0},
        {"codec_type": "audio", "codec_name": "aac", "index": 1},
    ]
    monkeypatch.setattr("reelgrep.subtitles.probe", _make_fake_probe(streams))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "reelgrep.subtitles.run_ffmpeg", _make_fake_run_ffmpeg("", calls)
    )

    tracks = extract_embedded("/fake/movie.mp4", tmp_path)
    assert tracks == []
    assert len(calls) == 0
