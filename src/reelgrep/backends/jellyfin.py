"""Jellyfin backend: resolve item names or IDs to local file paths via the Jellyfin HTTP API."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from reelgrep.backends import BackendError, BaseBackend, register

__all__ = ["JellyfinBackend", "JellyfinTransport"]


class JellyfinTransport(Protocol):
    """A callable transport for unit tests: substitutes the live HTTP layer."""

    def __call__(
        self, *, method: str, path: str, query: dict[str, Any] | None
    ) -> dict[str, Any]: ...


_HEX_ID = re.compile(r"^[0-9a-fA-F]{32}$")


@register("jellyfin")
class JellyfinBackend(BaseBackend):
    """Resolves Jellyfin item identifiers or search terms to local paths."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        transport: JellyfinTransport | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("JELLYFIN_URL") or "").rstrip("/")
        self._api_key = api_key or os.environ.get("JELLYFIN_API_KEY")
        self._timeout = timeout
        self._verify_ssl = verify_ssl
        self._transport = transport

    def resolve(self, uri: str) -> Path:
        """Resolve a Jellyfin ItemId (32-hex) or free-text search term to a local Path."""
        if not uri:
            raise BackendError("empty query")
        item_id = uri if _HEX_ID.match(uri) else self._lookup_id(uri)
        item = self._request(
            method="GET",
            path=f"/Items/{item_id}",
            query={"Fields": "Path"},
        )
        path_value = item.get("Path") if isinstance(item, dict) else None
        if not path_value:
            raise BackendError(
                f"jellyfin item {item_id!r} has no Path field (transcoded only?)"
            )
        return Path(path_value).expanduser().resolve()

    # --- internals ---

    def _lookup_id(self, search_term: str) -> str:
        """Search Jellyfin for an item by free-text term and return its Id."""
        result = self._request(
            method="GET",
            path="/Items",
            query={
                "searchTerm": search_term,
                "Recursive": "true",
                "IncludeItemTypes": "Movie,Episode,Video",
                "Limit": "5",
            },
        )
        items = result.get("Items") if isinstance(result, dict) else None
        if not items:
            raise BackendError(f"no jellyfin items match: {search_term!r}")
        if len(items) > 1:
            names = [it.get("Name", "<unnamed>") for it in items[:5]]
            raise BackendError(f"ambiguous jellyfin search {search_term!r}: {names}")
        item_id = items[0].get("Id")
        if not item_id:
            raise BackendError("jellyfin returned an item with no Id")
        return item_id

    def _request(
        self, *, method: str, path: str, query: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Dispatch a request through the injected transport or fall back to real HTTP."""
        if self._transport is not None:
            return self._transport(method=method, path=path, query=query)
        if not self._base_url:
            raise BackendError("JELLYFIN_URL is not set")
        if not self._api_key:
            raise BackendError("JELLYFIN_API_KEY is not set")
        return self._http_request(method=method, path=path, query=query)

    def _http_request(
        self, *, method: str, path: str, query: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Perform the real HTTP request against the configured Jellyfin server."""
        url = self._base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query, doseq=True)
        req = urllib.request.Request(url, method=method)
        assert self._api_key is not None  # gated by _request
        req.add_header("X-Emby-Token", self._api_key)
        req.add_header("Accept", "application/json")
        ctx = (
            ssl.create_default_context()
            if self._verify_ssl
            else ssl._create_unverified_context()
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise BackendError(
                f"jellyfin {method} {path} -> HTTP {exc.code}: {text[:200]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendError(f"jellyfin connection failed: {exc.reason}") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise BackendError(f"jellyfin returned non-JSON: {body[:200]!r}") from exc
