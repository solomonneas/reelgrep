"""Tests for the whisper transcription wrapper."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from reelgrep.transcribe import TranscribeError, available_models, transcribe


@pytest.fixture
def fake_video(tmp_path):
    """Create a placeholder media file on disk so resolve(strict=True) succeeds."""
    p = tmp_path / "lecture.mp4"
    p.write_bytes(b"\x00fake")
    return p


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    def __init__(self, language: str = "en"):
        self.language = language


class _FakeWhisperModel:
    last_kwargs: dict | None = None
    segments_to_return: list[_FakeSegment] = []
    info_to_return: _FakeInfo | None = None

    def __init__(self, model_size, **kwargs):
        type(self).last_kwargs = {"model_size": model_size, **kwargs}

    def transcribe(self, audio_path, **kwargs):
        type(self).last_kwargs = {
            **(type(self).last_kwargs or {}),
            **kwargs,
            "audio_path": audio_path,
        }
        return (
            iter(type(self).segments_to_return),
            type(self).info_to_return or _FakeInfo(),
        )


@pytest.fixture
def fake_whisper(monkeypatch):
    """Install a fake faster_whisper module that returns canned segments."""
    fake_mod = types.ModuleType("faster_whisper")
    fake_mod.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)
    _FakeWhisperModel.segments_to_return = []
    _FakeWhisperModel.info_to_return = None
    _FakeWhisperModel.last_kwargs = None
    return _FakeWhisperModel


def test_available_models_contains_expected_sizes():
    models = available_models()
    assert isinstance(models, tuple)
    assert "small" in models
    assert "large-v3" in models
    assert "large-v3-turbo" in models


def test_unknown_model_size_raises_before_import(fake_video, monkeypatch):
    # Ensure even with faster_whisper missing we still get the model_size error first.
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    with pytest.raises(TranscribeError, match="unknown model_size"):
        transcribe(fake_video, model_size="huge-v9")


def test_missing_faster_whisper_raises_helpful_error(fake_video, monkeypatch):
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            raise ImportError("No module named 'faster_whisper'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(TranscribeError, match=r"\[whisper\] extra"):
        transcribe(fake_video)


def test_nonexistent_video_raises_file_not_found(tmp_path, fake_whisper):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(FileNotFoundError):
        transcribe(missing)


def test_happy_path_three_segments_blank_skipped(fake_video, fake_whisper):
    fake_whisper.segments_to_return = [
        _FakeSegment(0.0, 2.5, "hello world"),
        _FakeSegment(3.0, 5.5, "this is kubernetes"),
        _FakeSegment(6.0, 8.0, "  "),
    ]
    fake_whisper.info_to_return = _FakeInfo(language="en")

    track = transcribe(fake_video)

    assert track.source == "whisper"
    assert track.language == "en"
    assert track.format == "whisper"
    assert track.stream_index is None
    assert len(track.cues) == 2
    assert track.cues[0].start_ms == 0
    assert track.cues[0].end_ms == 2500
    assert track.cues[0].text == "hello world"
    assert track.cues[0].language == "en"
    assert track.cues[1].start_ms == 3000
    assert track.cues[1].end_ms == 5500
    assert track.cues[1].text == "this is kubernetes"


def test_millisecond_rounding(fake_video, fake_whisper):
    fake_whisper.segments_to_return = [_FakeSegment(1.2345, 2.7891, "x")]
    fake_whisper.info_to_return = _FakeInfo(language="en")

    track = transcribe(fake_video)

    assert len(track.cues) == 1
    assert track.cues[0].start_ms == 1234
    assert track.cues[0].end_ms == 2789


def test_whitespace_stripping(fake_video, fake_whisper):
    fake_whisper.segments_to_return = [_FakeSegment(0.0, 1.0, "   hello   ")]
    fake_whisper.info_to_return = _FakeInfo(language="en")

    track = transcribe(fake_video)

    assert len(track.cues) == 1
    assert track.cues[0].text == "hello"


def test_empty_segment_list(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    fake_whisper.info_to_return = _FakeInfo(language="en")

    track = transcribe(fake_video)

    assert track.cues == []
    assert track.language == "en"
    assert track.source == "whisper"


def test_language_passthrough(fake_video, fake_whisper):
    fake_whisper.segments_to_return = [_FakeSegment(0.0, 1.0, "hola")]
    fake_whisper.info_to_return = _FakeInfo(language="es")

    track = transcribe(fake_video, language="es")

    assert fake_whisper.last_kwargs["language"] == "es"
    assert track.language == "es"


def test_language_falls_back_to_requested_when_info_lacks_it(fake_video, fake_whisper):
    fake_whisper.segments_to_return = [_FakeSegment(0.0, 1.0, "hola")]
    fake_whisper.info_to_return = _FakeInfo(language="")  # falsy

    track = transcribe(fake_video, language="es")

    assert track.language == "es"
    assert track.cues[0].language == "es"


def test_auto_detect_language(fake_video, fake_whisper):
    fake_whisper.segments_to_return = [
        _FakeSegment(0.0, 1.0, "bonjour"),
        _FakeSegment(1.0, 2.0, "le monde"),
    ]
    fake_whisper.info_to_return = _FakeInfo(language="fr")

    track = transcribe(fake_video, language=None)

    assert fake_whisper.last_kwargs["language"] is None
    assert track.language == "fr"
    for cue in track.cues:
        assert cue.language == "fr"


def test_model_size_passthrough(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    transcribe(fake_video, model_size="large-v3-turbo")
    assert fake_whisper.last_kwargs["model_size"] == "large-v3-turbo"


def test_compute_type_default_for_cpu(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    transcribe(fake_video)
    assert fake_whisper.last_kwargs["device"] == "cpu"
    assert fake_whisper.last_kwargs["compute_type"] == "int8"


def test_compute_type_default_for_cuda(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    transcribe(fake_video, device="cuda")
    assert fake_whisper.last_kwargs["device"] == "cuda"
    assert fake_whisper.last_kwargs["compute_type"] == "float16"


def test_compute_type_explicit_overrides(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    transcribe(fake_video, device="cpu", compute_type="float32")
    assert fake_whisper.last_kwargs["compute_type"] == "float32"


def test_beam_size_default(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    transcribe(fake_video)
    assert fake_whisper.last_kwargs["beam_size"] == 5


def test_beam_size_passthrough(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    transcribe(fake_video, beam_size=1)
    assert fake_whisper.last_kwargs["beam_size"] == 1


def test_vad_filter_default_true(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    transcribe(fake_video)
    assert fake_whisper.last_kwargs["vad_filter"] is True


def test_vad_filter_passthrough(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    transcribe(fake_video, vad_filter=False)
    assert fake_whisper.last_kwargs["vad_filter"] is False


def test_audio_path_is_resolved_absolute_string(fake_video, fake_whisper):
    fake_whisper.segments_to_return = []
    transcribe(fake_video)
    assert fake_whisper.last_kwargs["audio_path"] == str(fake_video.resolve())
