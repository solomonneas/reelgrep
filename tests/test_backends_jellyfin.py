"""Tests for the Jellyfin backend using a fake transport (no real HTTP)."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from reelgrep.backends import BackendError, get_backend, list_backends
from reelgrep.backends.jellyfin import JellyfinBackend

HEX_ID = "0123456789abcdef0123456789abcdef"


def make_transport(responses: dict[str, dict[str, Any]]):
    """Returns a (transport, calls) pair backed by a static path -> response map."""
    calls: list[dict[str, Any]] = []

    def transport(*, method: str, path: str, query: dict[str, Any] | None) -> dict[str, Any]:
        calls.append({"method": method, "path": path, "query": query or {}})
        return responses.get(path, {})

    return transport, calls


def test_registry_lists_jellyfin() -> None:
    assert "jellyfin" in list_backends()


def test_get_backend_returns_instance() -> None:
    backend = get_backend(
        "jellyfin", base_url="http://j.local", api_key="x", transport=lambda **_: {}
    )
    assert isinstance(backend, JellyfinBackend)


def test_empty_uri_raises() -> None:
    backend = JellyfinBackend(transport=lambda **_: {})
    with pytest.raises(BackendError, match="empty query"):
        backend.resolve("")


def test_hex_id_skips_search() -> None:
    transport, calls = make_transport(
        {f"/Items/{HEX_ID}": {"Id": HEX_ID, "Path": "/videos/movie.mkv"}}
    )
    backend = JellyfinBackend(transport=transport)

    result = backend.resolve(HEX_ID)

    assert len(calls) == 1
    assert calls[0]["path"] == f"/Items/{HEX_ID}"
    assert calls[0]["method"] == "GET"
    assert result == Path("/videos/movie.mkv").resolve()


def test_search_term_then_resolve() -> None:
    found_id = "deadbeefdeadbeefdeadbeefdeadbeef"
    transport, calls = make_transport(
        {
            "/Items": {"Items": [{"Id": found_id, "Name": "The Movie"}]},
            f"/Items/{found_id}": {"Id": found_id, "Path": "/videos/the-movie.mkv"},
        }
    )
    backend = JellyfinBackend(transport=transport)

    result = backend.resolve("The Movie")

    assert len(calls) == 2
    assert calls[0]["path"] == "/Items"
    assert calls[0]["query"]["searchTerm"] == "The Movie"
    assert calls[1]["path"] == f"/Items/{found_id}"
    assert result == Path("/videos/the-movie.mkv").resolve()


def test_search_no_matches() -> None:
    transport, _ = make_transport({"/Items": {"Items": []}})
    backend = JellyfinBackend(transport=transport)
    with pytest.raises(BackendError, match="no jellyfin items match"):
        backend.resolve("nothing here")


def test_search_ambiguous() -> None:
    transport, _ = make_transport(
        {
            "/Items": {
                "Items": [
                    {"Id": "a" * 32, "Name": "Movie One"},
                    {"Id": "b" * 32, "Name": "Movie Two"},
                    {"Id": "c" * 32, "Name": "Movie Three"},
                ]
            }
        }
    )
    backend = JellyfinBackend(transport=transport)
    with pytest.raises(BackendError, match="ambiguous jellyfin search") as exc:
        backend.resolve("Movie")
    msg = str(exc.value)
    assert "Movie One" in msg
    assert "Movie Two" in msg
    assert "Movie Three" in msg


def test_item_missing_path_field() -> None:
    transport, _ = make_transport({f"/Items/{HEX_ID}": {"Id": HEX_ID}})
    backend = JellyfinBackend(transport=transport)
    with pytest.raises(BackendError, match="has no Path field"):
        backend.resolve(HEX_ID)


def test_search_term_query_holds_raw_value() -> None:
    """The fake transport sees the raw dict; urlencode only happens in the http layer."""
    found_id = "f" * 32
    transport, calls = make_transport(
        {
            "/Items": {"Items": [{"Id": found_id, "Name": "Lecture"}]},
            f"/Items/{found_id}": {"Id": found_id, "Path": "/videos/lecture.mp4"},
        }
    )
    backend = JellyfinBackend(transport=transport)
    backend.resolve("Lecture 1: Intro & Tour")

    assert calls[0]["query"]["searchTerm"] == "Lecture 1: Intro & Tour"
    assert calls[0]["query"]["Recursive"] == "true"
    assert calls[0]["query"]["IncludeItemTypes"] == "Movie,Episode,Video"


def test_http_request_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JELLYFIN_URL", raising=False)
    monkeypatch.delenv("JELLYFIN_API_KEY", raising=False)
    backend = JellyfinBackend()
    with pytest.raises(BackendError, match="JELLYFIN_URL"):
        backend.resolve(HEX_ID)


def test_http_request_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JELLYFIN_URL", raising=False)
    monkeypatch.delenv("JELLYFIN_API_KEY", raising=False)
    backend = JellyfinBackend(base_url="http://j.local")
    with pytest.raises(BackendError, match="JELLYFIN_API_KEY"):
        backend.resolve(HEX_ID)


def test_env_var_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JELLYFIN_URL", "http://j.local")
    monkeypatch.setenv("JELLYFIN_API_KEY", "secret")
    backend = JellyfinBackend()
    assert backend._base_url == "http://j.local"
    assert backend._api_key == "secret"


def test_trailing_slash_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JELLYFIN_URL", "http://j.local/")
    monkeypatch.setenv("JELLYFIN_API_KEY", "secret")
    backend = JellyfinBackend()
    assert backend._base_url == "http://j.local"


def test_http_404_surfaces_as_backend_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: Any, **_kwargs: Any):
        raise urllib.error.HTTPError(
            "http://j.local/Items/" + HEX_ID, 404, "Not Found", {}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = JellyfinBackend(base_url="http://j.local", api_key="secret")
    with pytest.raises(BackendError, match="HTTP 404"):
        backend.resolve(HEX_ID)


def test_connection_failure_surfaces_as_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(*_args: Any, **_kwargs: Any):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = JellyfinBackend(base_url="http://j.local", api_key="secret")
    with pytest.raises(BackendError, match="jellyfin connection failed"):
        backend.resolve(HEX_ID)
