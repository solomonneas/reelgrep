"""Tests for reelgrep.probe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from reelgrep import ffmpeg_exec
from reelgrep import probe as probe_mod
from reelgrep.config import reset_settings
from reelgrep.probe import (
    AudioStream,
    VideoMetadata,
    VideoStream,
    parse_fps,
    parse_probe_json,
    probe,
)

ENV_VARS = (
    "REELGREP_HOME",
    "REELGREP_DB",
    "REELGREP_CACHE",
    "REELGREP_FFMPEG",
    "REELGREP_FFPROBE",
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "probe_sample.json"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    yield
    reset_settings()


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_parse_fps_simple_integer_fraction() -> None:
    assert parse_fps("24/1") == 24.0


def test_parse_fps_ntsc_fraction() -> None:
    assert parse_fps("30000/1001") == pytest.approx(29.97, rel=1e-3)


def test_parse_fps_zero_over_zero_returns_none() -> None:
    assert parse_fps("0/0") is None


def test_parse_fps_denominator_zero_returns_none() -> None:
    assert parse_fps("24/0") is None


def test_parse_fps_bare_number() -> None:
    assert parse_fps("24") == 24.0


def test_parse_fps_garbage_returns_none() -> None:
    assert parse_fps("abc") is None


def test_parse_fps_empty_string_returns_none() -> None:
    assert parse_fps("") is None


def test_parse_probe_json_from_fixture() -> None:
    raw = _load_fixture()
    meta = parse_probe_json(raw, "/tmp/sample.mp4")
    assert isinstance(meta, VideoMetadata)
    assert meta.path == "/tmp/sample.mp4"
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 24.0
    assert meta.video_codec == "h264"
    assert "mp4" in meta.format_name
    assert 2900 <= meta.duration_ms <= 3100
    assert meta.raw == raw
    assert len(meta.video_streams) >= 1
    primary = meta.video_streams[0]
    assert isinstance(primary, VideoStream)
    assert primary.width == 320
    assert primary.r_frame_rate == "24/1"


def test_parse_probe_json_missing_video_stream() -> None:
    raw: dict[str, Any] = {
        "format": {
            "format_name": "wav",
            "duration": "1.500000",
            "size": "44100",
        },
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "sample_rate": "44100",
                "channels": 2,
            }
        ],
    }
    meta = parse_probe_json(raw, "/tmp/audio.wav")
    assert meta.width is None
    assert meta.height is None
    assert meta.fps is None
    assert meta.video_codec is None
    assert meta.video_streams == []
    assert len(meta.audio_streams) == 1
    audio = meta.audio_streams[0]
    assert isinstance(audio, AudioStream)
    assert audio.codec_name == "pcm_s16le"
    assert audio.sample_rate == 44100
    assert audio.channels == 2
    assert meta.audio_codec == "pcm_s16le"
    assert meta.duration_ms == 1500


def test_parse_probe_json_missing_audio_stream() -> None:
    raw: dict[str, Any] = {
        "format": {
            "format_name": "mp4",
            "duration": "5.000000",
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
            }
        ],
    }
    meta = parse_probe_json(raw, "/tmp/video.mp4")
    assert meta.audio_codec is None
    assert meta.audio_streams == []
    assert meta.video_codec == "h264"
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.fps == 30.0


def test_probe_raises_filenotfound_for_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        probe("/nonexistent/path.mp4")


def test_probe_end_to_end_with_patched_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create a real file so Path.resolve(strict=True) succeeds.
    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"\x00\x00\x00\x00")

    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")

    def fake_run_ffprobe(args: list[str], *, binary: str | None = None) -> str:
        # Last arg should be the resolved path string.
        assert args[-1].endswith("fake.mp4")
        return fixture_text

    monkeypatch.setattr(probe_mod, "run_ffprobe", fake_run_ffprobe)

    meta = probe(video_path)
    assert isinstance(meta, VideoMetadata)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 24.0
    assert meta.video_codec == "h264"
    assert "mp4" in meta.format_name
    assert 2900 <= meta.duration_ms <= 3100


def test_probe_propagates_ffmpeg_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"\x00")

    def raising_run_ffprobe(args: list[str], *, binary: str | None = None) -> str:
        raise ffmpeg_exec.FFmpegError(1, "decode failure", list(args))

    monkeypatch.setattr(probe_mod, "run_ffprobe", raising_run_ffprobe)

    with pytest.raises(ffmpeg_exec.FFmpegError):
        probe(video_path)


@pytest.mark.integration
def test_probe_against_real_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REELGREP_FFPROBE", "/usr/bin/ffprobe")
    reset_settings()

    video_path = tmp_path / "blue.mp4"
    subprocess.run(
        [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=3:r=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    meta = probe(video_path)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == pytest.approx(24.0, rel=1e-3)
    assert 2900 <= meta.duration_ms <= 3100
    assert meta.video_codec == "h264"
