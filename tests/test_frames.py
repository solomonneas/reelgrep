"""Tests for reelgrep.frames."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from reelgrep import frames as frames_mod
from reelgrep.config import reset_settings
from reelgrep.frames import Frame, sample_every, sample_scenes
from reelgrep.timecode import format as format_timecode

ENV_VARS = (
    "REELGREP_HOME",
    "REELGREP_DB",
    "REELGREP_CACHE",
    "REELGREP_FFMPEG",
    "REELGREP_FFPROBE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    reset_settings()
    yield
    reset_settings()


class _FakeMetadata:
    """Stand-in for probe.VideoMetadata with the fields frames.py reads."""

    def __init__(self, duration_ms: int, fps: float | None = 24.0) -> None:
        self.duration_ms = duration_ms
        self.fps = fps


class _RecordingFFmpeg:
    """Capture run_ffmpeg invocations and optionally seed output files."""

    def __init__(
        self,
        *,
        touch_index: int | None = None,
        stderr: bytes = b"",
        seed_pattern_count: int = 0,
        pattern_stem: str | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.touch_index = touch_index
        self.stderr = stderr
        self.seed_pattern_count = seed_pattern_count
        self.pattern_stem = pattern_stem

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(list(args))
        self.kwargs.append(dict(kwargs))

        if self.touch_index is not None:
            out_path = Path(args[self.touch_index])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"\xff\xd8\xff\xe0fake")

        if self.seed_pattern_count > 0 and self.pattern_stem is not None:
            pattern = args[-1]
            parent = Path(pattern).parent
            parent.mkdir(parents=True, exist_ok=True)
            for i in range(1, self.seed_pattern_count + 1):
                scene_file = parent / f"{self.pattern_stem}_scene_{i:05d}.jpg"
                scene_file.write_bytes(b"\xff\xd8\xff\xe0fake")

        return subprocess.CompletedProcess(
            args=[*args],
            returncode=0,
            stdout=b"",
            stderr=self.stderr,
        )


def _make_video_file(tmp_path: Path, name: str = "clip.mp4") -> Path:
    """Create an empty file that ffmpeg won't actually read (probe is mocked)."""
    path = tmp_path / name
    path.write_bytes(b"fakevideo")
    return path


# ---------------- Frame model ----------------


def test_frame_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Frame(
            timestamp_ms=0,
            path="/tmp/x.jpg",
            sampling_strategy="every_n",
            bogus="nope",
        )


def test_frame_accepts_each_sampling_strategy() -> None:
    for strategy in ("every_n", "scene", "manual"):
        frame = Frame(timestamp_ms=0, path="/tmp/x.jpg", sampling_strategy=strategy)
        assert frame.sampling_strategy == strategy


def test_frame_rejects_unknown_strategy() -> None:
    with pytest.raises(ValidationError):
        Frame(timestamp_ms=0, path="/tmp/x.jpg", sampling_strategy="weird")


def test_frame_optional_width_height_default_none() -> None:
    frame = Frame(timestamp_ms=100, path="/tmp/x.jpg", sampling_strategy="manual")
    assert frame.width is None
    assert frame.height is None


# ---------------- sample_every ----------------


def test_sample_every_generates_expected_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path, "clip.mp4")
    out_dir = tmp_path / "frames"

    monkeypatch.setattr(frames_mod, "probe", lambda p, **kw: _FakeMetadata(duration_ms=10000))

    mock_ffmpeg = _RecordingFFmpeg(touch_index=-1)
    monkeypatch.setattr(frames_mod, "run_ffmpeg", mock_ffmpeg)

    result = sample_every(video, out_dir, interval_seconds=2.0)

    timestamps = [f.timestamp_ms for f in result]
    assert timestamps == [0, 2000, 4000, 6000, 8000, 10000]
    assert len(result) == 6

    for frame in result:
        assert frame.sampling_strategy == "every_n"
        assert Path(frame.path).exists()
        assert frame.width is None
        assert frame.height is None

    expected_ss = ["0.000", "2.000", "4.000", "6.000", "8.000", "10.000"]
    for call, ss in zip(mock_ffmpeg.calls, expected_ss, strict=True):
        assert "-ss" in call
        idx = call.index("-ss")
        assert call[idx + 1] == ss


def test_sample_every_filename_uses_timecode_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path, "myclip.mp4")
    out_dir = tmp_path / "frames"

    monkeypatch.setattr(frames_mod, "probe", lambda p, **kw: _FakeMetadata(duration_ms=4000))
    monkeypatch.setattr(frames_mod, "run_ffmpeg", _RecordingFFmpeg(touch_index=-1))

    result = sample_every(video, out_dir, interval_seconds=2.0)

    first = result[0]
    assert Path(first.path).name == f"myclip_{format_timecode(0)}.jpg"


def test_sample_every_creates_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path, "c.mp4")
    out_dir = tmp_path / "deep" / "nested" / "dir"

    monkeypatch.setattr(frames_mod, "probe", lambda p, **kw: _FakeMetadata(duration_ms=1000))
    monkeypatch.setattr(frames_mod, "run_ffmpeg", _RecordingFFmpeg(touch_index=-1))

    sample_every(video, out_dir, interval_seconds=1.0)
    assert out_dir.is_dir()


def test_sample_every_zero_interval_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path)
    out_dir = tmp_path / "frames"
    monkeypatch.setattr(frames_mod, "probe", lambda p, **kw: _FakeMetadata(duration_ms=5000))
    monkeypatch.setattr(frames_mod, "run_ffmpeg", _RecordingFFmpeg(touch_index=-1))

    with pytest.raises(ValueError):
        sample_every(video, out_dir, interval_seconds=0)


def test_sample_every_negative_interval_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path)
    out_dir = tmp_path / "frames"
    monkeypatch.setattr(frames_mod, "probe", lambda p, **kw: _FakeMetadata(duration_ms=5000))
    monkeypatch.setattr(frames_mod, "run_ffmpeg", _RecordingFFmpeg(touch_index=-1))

    with pytest.raises(ValueError):
        sample_every(video, out_dir, interval_seconds=-2.0)


def test_sample_every_zero_duration_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path)
    out_dir = tmp_path / "frames"
    monkeypatch.setattr(frames_mod, "probe", lambda p, **kw: _FakeMetadata(duration_ms=0))
    monkeypatch.setattr(frames_mod, "run_ffmpeg", _RecordingFFmpeg(touch_index=-1))

    with pytest.raises(ValueError):
        sample_every(video, out_dir, interval_seconds=1.0)


def test_sample_every_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sample_every(tmp_path / "missing.mp4", tmp_path / "out", interval_seconds=1.0)


# ---------------- sample_scenes ----------------


_FAKE_SHOWINFO = (
    b"[Parsed_showinfo_1 @ 0x] n:0 pts:30000 pts_time:1.234567 pos:1234\n"
    b"[Parsed_showinfo_1 @ 0x] n:1 pts:60000 pts_time:2.500000 pos:5678\n"
    b"[Parsed_showinfo_1 @ 0x] n:2 pts:90000 pts_time:3.750000 pos:9012\n"
)


def test_sample_scenes_parses_pts_time_and_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path, "scenes.mp4")
    out_dir = tmp_path / "scenes_out"

    mock_ffmpeg = _RecordingFFmpeg(
        stderr=_FAKE_SHOWINFO,
        seed_pattern_count=3,
        pattern_stem="scenes",
    )
    monkeypatch.setattr(frames_mod, "run_ffmpeg", mock_ffmpeg)

    result = sample_scenes(video, out_dir, threshold=0.3)

    assert [f.timestamp_ms for f in result] == [1234, 2500, 3750]
    for frame in result:
        assert frame.sampling_strategy == "scene"
        assert Path(frame.path).exists()
        expected_name = f"scenes_{format_timecode(frame.timestamp_ms)}.jpg"
        assert Path(frame.path).name == expected_name

    leftovers = list(out_dir.glob("scenes_scene_*.jpg"))
    assert leftovers == []


def test_sample_scenes_threshold_zero_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path)
    monkeypatch.setattr(frames_mod, "run_ffmpeg", _RecordingFFmpeg())
    with pytest.raises(ValueError):
        sample_scenes(video, tmp_path / "out", threshold=0.0)


def test_sample_scenes_threshold_one_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path)
    monkeypatch.setattr(frames_mod, "run_ffmpeg", _RecordingFFmpeg())
    with pytest.raises(ValueError):
        sample_scenes(video, tmp_path / "out", threshold=1.0)


def test_sample_scenes_caps_at_max_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path, "scenes.mp4")
    out_dir = tmp_path / "scenes_out"

    mock_ffmpeg = _RecordingFFmpeg(
        stderr=_FAKE_SHOWINFO,
        seed_pattern_count=3,
        pattern_stem="scenes",
    )
    monkeypatch.setattr(frames_mod, "run_ffmpeg", mock_ffmpeg)

    result = sample_scenes(video, out_dir, threshold=0.3, max_frames=2)
    assert len(result) == 2
    assert [f.timestamp_ms for f in result] == [1234, 2500]


def test_sample_scenes_no_matches_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path, "noisy.mp4")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        frames_mod,
        "run_ffmpeg",
        _RecordingFFmpeg(stderr=b"[Parsed_showinfo_1 @ 0x] no matches here\n"),
    )

    result = sample_scenes(video, out_dir, threshold=0.3)
    assert result == []


def test_sample_scenes_creates_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_file(tmp_path, "x.mp4")
    out_dir = tmp_path / "fresh" / "nested"

    monkeypatch.setattr(frames_mod, "run_ffmpeg", _RecordingFFmpeg(stderr=b""))
    sample_scenes(video, out_dir, threshold=0.3)
    assert out_dir.is_dir()


# ---------------- integration ----------------


@pytest.mark.integration
def test_sample_every_integration_real_ffmpeg(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")

    # Generate slightly over 4s (4.5s) so the natural-step series ends at 4000ms
    # without seeking to the exact end of the clip (ffmpeg returns no frame there).
    video = tmp_path / "blue.mp4"
    lavfi_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=160x120:d=4.5:r=12",
        str(video),
    ]
    proc = subprocess.run(lavfi_cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg failed to generate fixture: {proc.stderr!r}")

    out_dir = tmp_path / "frames"
    frames_out = sample_every(video, out_dir, interval_seconds=1.0)

    assert len(frames_out) == 5
    assert [f.timestamp_ms for f in frames_out] == [0, 1000, 2000, 3000, 4000]
    for frame in frames_out:
        p = Path(frame.path)
        assert p.exists(), p
        data = p.read_bytes()
        assert len(data) > 0
        assert data[:3] == b"\xff\xd8\xff"


def test_env_isolation_no_reelgrep_vars_present() -> None:
    for name in ENV_VARS:
        assert name not in os.environ
