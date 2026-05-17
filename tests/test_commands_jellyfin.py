"""Tests for the `reelgrep jellyfin resolve` CLI command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from reelgrep.commands.jellyfin import jellyfin


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("JELLYFIN_URL", "JELLYFIN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


def test_jellyfin_group_lists_resolve():
    runner = CliRunner()
    result = runner.invoke(jellyfin, ["--help"])
    assert result.exit_code == 0
    assert "resolve" in result.output


def test_resolve_help_works():
    runner = CliRunner()
    result = runner.invoke(jellyfin, ["resolve", "--help"])
    assert result.exit_code == 0
    assert "ItemId" in result.output or "item" in result.output.lower()


def test_resolve_no_config_fails_cleanly(monkeypatch):
    runner = CliRunner()
    result = runner.invoke(jellyfin, ["resolve", "Some Lecture"])
    assert result.exit_code == 2
    assert "error" in result.stderr.lower()


def test_resolve_happy_path(monkeypatch):
    from reelgrep.backends import jellyfin as backend_mod

    def fake_init(self, *, base_url=None, api_key=None, timeout=30.0,
                  verify_ssl=True, transport=None):
        self._transport = lambda **_: None
        self._base_url = "http://j.local"
        self._api_key = "x"
        self._timeout = timeout
        self._verify_ssl = verify_ssl
        self._calls = []

    def fake_resolve(self, uri):
        from pathlib import Path
        return Path(f"/videos/{uri}.mkv")

    monkeypatch.setattr(backend_mod.JellyfinBackend, "__init__", fake_init)
    monkeypatch.setattr(backend_mod.JellyfinBackend, "resolve", fake_resolve)

    runner = CliRunner()
    result = runner.invoke(jellyfin, ["resolve", "lecture1"])
    assert result.exit_code == 0, result.output
    assert "/videos/lecture1.mkv" in result.output
