"""Tests for the Ollama vision-LLM person/object-finding backend."""

from __future__ import annotations

import base64
import json as _json
import sys
from pathlib import Path

import pytest

from reelgrep.frames import Frame
from reelgrep.models import ModelError
from reelgrep.models.ollama_vision import OllamaVisionModel


class FakeResponse:
    """Stand-in for httpx.Response used by the test FakeClient."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = "",
    ) -> None:
        """Capture status, optional JSON body, and raw text."""
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self) -> dict:
        """Return the JSON body or raise like httpx would on invalid JSON."""
        if self._json is None:
            raise _json.JSONDecodeError("no body", "", 0)
        return self._json


class FakeClient:
    """Context-managed fake httpx Client capturing posts and returning queued responses."""

    def __init__(
        self,
        *,
        response: FakeResponse | None = None,
        responses: list[FakeResponse] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        """Configure a single response, a queue of responses, or a raised exception."""
        self._single = response
        self._queue = list(responses) if responses else None
        self._raise = raise_exc
        self.posts: list[tuple[str, dict]] = []

    def __enter__(self) -> FakeClient:
        """Enter context."""
        return self

    def __exit__(self, *_a) -> None:
        """Exit context (no-op)."""
        return None

    def post(self, url: str, json: dict) -> FakeResponse:
        """Record the POST and return a queued or single response (or raise)."""
        self.posts.append((url, json))
        if self._raise:
            raise self._raise
        if self._queue is not None:
            return self._queue.pop(0)
        assert self._single is not None
        return self._single


def _write_image(path: Path, payload: bytes = b"\xff\xd8\xff\xe0fake-jpeg-bytes") -> Path:
    """Write a small byte blob to act as a reference or candidate image file."""
    path.write_bytes(payload)
    return path


def _make_frame(path: Path, ts_ms: int = 0) -> Frame:
    """Build a Frame backed by an existing on-disk image."""
    return Frame(timestamp_ms=ts_ms, path=str(path), sampling_strategy="every_n")


def _ollama_response(match: bool, confidence: float, reasoning: str = "ok") -> FakeResponse:
    """Build a FakeResponse mirroring Ollama's /api/chat envelope."""
    body = _json.dumps({"match": match, "confidence": confidence, "reasoning": reasoning})
    return FakeResponse(200, {"message": {"content": body}})


def test_lazy_import_raises_modelerror_without_httpx(monkeypatch, tmp_path):
    """find() raises ModelError mentioning the [vision] extra when httpx is missing."""
    pos = _write_image(tmp_path / "pos.jpg")
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))

    # Simulate httpx being uninstalled.
    monkeypatch.setitem(sys.modules, "httpx", None)

    model = OllamaVisionModel()
    with pytest.raises(ModelError, match=r"\[vision\] extra"):
        model.find([frame], [pos], [], threshold=0.5)


def test_config_dict_defaults():
    """config_dict() returns the four configured fields with documented defaults."""
    model = OllamaVisionModel()
    cfg = model.config_dict()
    assert cfg == {
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen2-vl:7b",
        "timeout": 60.0,
        "label": "subject",
    }


def test_env_var_defaults(monkeypatch):
    """OLLAMA_URL and OLLAMA_VISION_MODEL env vars seed the instance fields."""
    monkeypatch.setenv("OLLAMA_URL", "http://o.local")
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "llava:13b")
    model = OllamaVisionModel()
    cfg = model.config_dict()
    assert cfg["base_url"] == "http://o.local"
    assert cfg["model"] == "llava:13b"


def test_explicit_base_url_strips_trailing_slash():
    """Trailing slash on an explicit base_url is stripped."""
    model = OllamaVisionModel(base_url="http://o.local/")
    assert model.config_dict()["base_url"] == "http://o.local"


def test_empty_positives_raises(tmp_path):
    """find() with no positive examples raises ModelError."""
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))
    model = OllamaVisionModel()
    with pytest.raises(ModelError, match="at least one positive"):
        model.find([frame], [], [], threshold=0.5)


def test_happy_path_three_frames_all_match(monkeypatch, tmp_path):
    """Three frames + always-match response yield three matches at confidence 0.85."""
    pos = _write_image(tmp_path / "pos.jpg")
    frames = [
        _make_frame(_write_image(tmp_path / f"f{i}.jpg"), ts_ms=i * 1000) for i in range(3)
    ]
    body = '{"match": true, "confidence": 0.85, "reasoning": "subject visible"}'
    response = FakeResponse(200, {"message": {"content": body}})
    fake = FakeClient(response=response)
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    matches = model.find(frames, [pos], [], threshold=0.5)

    assert len(matches) == 3
    assert all(m.confidence == 0.85 for m in matches)
    assert all(m.reasoning == "subject visible" for m in matches)


def test_threshold_filters_low_confidence(monkeypatch, tmp_path):
    """Confidence below threshold is dropped, returning an empty list."""
    pos = _write_image(tmp_path / "pos.jpg")
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))
    fake = FakeClient(response=_ollama_response(True, 0.3))
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    matches = model.find([frame], [pos], [], threshold=0.5)
    assert matches == []


def test_explicit_non_match_excluded_even_when_confident(monkeypatch, tmp_path):
    """match=false skips the frame regardless of confidence."""
    pos = _write_image(tmp_path / "pos.jpg")
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))
    fake = FakeClient(response=_ollama_response(False, 0.9))
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    matches = model.find([frame], [pos], [], threshold=0.5)
    assert matches == []


def test_non_json_content_is_logged_and_skipped(monkeypatch, tmp_path, caplog):
    """Non-JSON content is logged at WARNING and the frame is skipped (no exception)."""
    pos = _write_image(tmp_path / "pos.jpg")
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))
    fake = FakeClient(response=FakeResponse(200, {"message": {"content": "I think maybe"}}))
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    with caplog.at_level("WARNING", logger="reelgrep.models.ollama_vision"):
        matches = model.find([frame], [pos], [], threshold=0.5)

    assert matches == []
    assert any("non-JSON" in rec.message for rec in caplog.records)


def test_http_error_raises_modelerror(monkeypatch, tmp_path):
    """A 5xx response raises ModelError with the status code in the message."""
    pos = _write_image(tmp_path / "pos.jpg")
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))
    fake = FakeClient(response=FakeResponse(503, text="busy"))
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    with pytest.raises(ModelError, match="ollama HTTP 503"):
        model.find([frame], [pos], [], threshold=0.5)


def test_connection_failure_raises_modelerror(monkeypatch, tmp_path):
    """A raised exception from client.post is wrapped in ModelError."""
    pos = _write_image(tmp_path / "pos.jpg")
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))
    fake = FakeClient(raise_exc=Exception("connection refused"))
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    with pytest.raises(ModelError, match="ollama request failed"):
        model.find([frame], [pos], [], threshold=0.5)


def test_sort_descending_and_top_k(monkeypatch, tmp_path):
    """Matches sort by confidence descending and top_k truncates the list."""
    pos = _write_image(tmp_path / "pos.jpg")
    frames = [
        _make_frame(_write_image(tmp_path / f"f{i}.jpg"), ts_ms=i * 1000) for i in range(3)
    ]
    responses = [
        _ollama_response(True, 0.6),
        _ollama_response(True, 0.9),
        _ollama_response(True, 0.8),
    ]
    fake = FakeClient(responses=responses)
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    matches = model.find(frames, [pos], [], threshold=0.5, top_k=2)

    assert [m.confidence for m in matches] == [0.9, 0.8]


def test_reasoning_truncated_to_300_chars(monkeypatch, tmp_path):
    """Long reasoning strings are truncated to exactly 300 characters."""
    pos = _write_image(tmp_path / "pos.jpg")
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))
    long_reason = "x" * 500
    fake = FakeClient(response=_ollama_response(True, 0.9, reasoning=long_reason))
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    matches = model.find([frame], [pos], [], threshold=0.5)

    assert len(matches) == 1
    assert len(matches[0].reasoning) == 300
    assert matches[0].reasoning == "x" * 300


def test_prompt_structure_with_negatives(monkeypatch, tmp_path):
    """With negatives, the prompt mentions KNOWN-NOT-MATCH and ends with CANDIDATE frame."""
    pos = _write_image(tmp_path / "pos.jpg")
    neg = _write_image(tmp_path / "neg.jpg")
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))
    fake = FakeClient(response=_ollama_response(True, 0.9))
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    model.find([frame], [pos], [neg], threshold=0.5)

    assert len(fake.posts) == 1
    payload = fake.posts[0][1]
    messages = payload["messages"]

    system_content = messages[0]["content"]
    assert "KNOWN-MATCH" in system_content
    assert "KNOWN-NOT-MATCH" in system_content

    # System + KNOWN-MATCH user + KNOWN-NOT-MATCH user + CANDIDATE user.
    assert len(messages) == 4
    assert messages[1]["content"] == "KNOWN-MATCH reference images:"
    assert messages[2]["content"] == "KNOWN-NOT-MATCH reference images:"
    assert messages[-1]["content"] == "CANDIDATE frame:"

    for msg in messages[1:]:
        assert msg["images"]
        assert all(isinstance(img, str) and img for img in msg["images"])


def test_prompt_structure_without_negatives(monkeypatch, tmp_path):
    """Without negatives, the prompt omits KNOWN-NOT-MATCH and uses three messages."""
    pos = _write_image(tmp_path / "pos.jpg")
    frame = _make_frame(_write_image(tmp_path / "f.jpg"))
    fake = FakeClient(response=_ollama_response(True, 0.9))
    monkeypatch.setattr(OllamaVisionModel, "_client", lambda self: fake)

    model = OllamaVisionModel()
    model.find([frame], [pos], [], threshold=0.5)

    payload = fake.posts[0][1]
    messages = payload["messages"]

    system_content = messages[0]["content"]
    assert "KNOWN-MATCH" in system_content
    assert "KNOWN-NOT-MATCH" not in system_content

    # System + KNOWN-MATCH user + CANDIDATE user.
    assert len(messages) == 3
    assert messages[1]["content"] == "KNOWN-MATCH reference images:"
    assert messages[2]["content"] == "CANDIDATE frame:"
    assert all("KNOWN-NOT-MATCH" not in m["content"] for m in messages)


def test_encode_roundtrips_to_original_bytes(tmp_path):
    """_encode returns base64 that decodes back to the original file bytes."""
    payload = b"\xff\xd8\xff\xe0arbitrary-binary-\x00\x01\x02"
    path = _write_image(tmp_path / "ref.jpg", payload=payload)
    model = OllamaVisionModel()
    encoded = model._encode(path)
    assert isinstance(encoded, str)
    assert base64.b64decode(encoded.encode("ascii")) == payload
