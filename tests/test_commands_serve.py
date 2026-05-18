"""Tests for ``reelgrep serve`` (the local browser UI launcher)."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from reelgrep.commands.serve import serve
from reelgrep.config import reset_settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "rg-home"
    monkeypatch.setenv("REELGREP_HOME", str(home))
    for var in ("REELGREP_DB", "REELGREP_CACHE", "REELGREP_FFMPEG", "REELGREP_FFPROBE"):
        monkeypatch.delenv(var, raising=False)
    reset_settings()
    yield home
    reset_settings()


@pytest.fixture
def fake_web_app(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Provide a stub ``reelgrep.web.app`` module with a recordable ``create_app``."""
    captured: dict[str, MagicMock] = {}

    def _create_app(db_path: Path | None = None) -> MagicMock:
        app = MagicMock(name="StarletteApp")
        app.routes = []
        captured["app"] = app
        captured["db_path"] = db_path  # type: ignore[assignment]
        return app

    fake_module = types.ModuleType("reelgrep.web.app")
    fake_module.create_app = _create_app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "reelgrep.web.app", fake_module)

    # Stash the captured dict on the module so tests can introspect it.
    fake_module._captured = captured  # type: ignore[attr-defined]
    return fake_module  # type: ignore[return-value]


def test_help_shows_browser_ui_options():
    runner = CliRunner()
    result = runner.invoke(serve, ["--help"])
    assert result.exit_code == 0, result.output
    assert "browser UI" in result.output
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--no-open-browser" in result.output


def test_missing_web_extra_raises_click_exception(monkeypatch: pytest.MonkeyPatch):
    # Force `import uvicorn` to fail with ImportError.
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    runner = CliRunner()
    result = runner.invoke(serve, ["--no-open-browser"])
    assert result.exit_code != 0
    assert "[web] extra" in result.output


def test_happy_path_invokes_uvicorn_with_expected_kwargs(
    monkeypatch: pytest.MonkeyPatch, fake_web_app: types.ModuleType
):
    import uvicorn

    run_mock = MagicMock(name="uvicorn.run")
    monkeypatch.setattr(uvicorn, "run", run_mock)

    runner = CliRunner()
    result = runner.invoke(
        serve,
        ["--no-open-browser", "--port", "9999", "--host", "127.0.0.1"],
    )
    assert result.exit_code == 0, result.output
    assert run_mock.call_count == 1

    _args, kwargs = run_mock.call_args
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9999
    assert kwargs["reload"] is False
    assert kwargs.get("log_level") == "info"

    assert "reelgrep web UI at http://127.0.0.1:9999/" in result.output
    assert "Ctrl+C to stop" in result.output


def test_open_browser_triggers_webbrowser_open(
    monkeypatch: pytest.MonkeyPatch, fake_web_app: types.ModuleType
):
    import threading
    import webbrowser

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", MagicMock(name="uvicorn.run"))
    open_mock = MagicMock(name="webbrowser.open")
    monkeypatch.setattr(webbrowser, "open", open_mock)

    # Replace the time.sleep used by the opener thread so the test doesn't actually pause.
    import time
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    # Replace threading.Thread with a shim that calls ``target`` synchronously.
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None, **_kwargs):
            self._target = target

        def start(self) -> None:
            if self._target is not None:
                self._target()

    monkeypatch.setattr(threading, "Thread", _ImmediateThread)

    runner = CliRunner()
    result = runner.invoke(
        serve, ["--open-browser", "--port", "9001", "--host", "127.0.0.1"]
    )
    assert result.exit_code == 0, result.output
    assert open_mock.call_count == 1
    assert open_mock.call_args.args[0] == "http://127.0.0.1:9001/"


def test_missing_frontend_assets_raises(
    monkeypatch: pytest.MonkeyPatch, fake_web_app: types.ModuleType, tmp_path: Path
):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", MagicMock(name="uvicorn.run"))

    empty_static = tmp_path / "empty-static"
    empty_static.mkdir()
    # No index.html in this directory.

    from reelgrep.commands import serve as serve_mod

    monkeypatch.setattr(serve_mod, "_static_dir", lambda: empty_static)

    runner = CliRunner()
    result = runner.invoke(serve, ["--no-open-browser"])
    assert result.exit_code != 0
    assert "frontend assets missing" in result.output


def test_serve_is_registered_in_main_cli():
    from reelgrep.cli import main

    assert "serve" in main.commands
    assert main.commands["serve"] is serve


@pytest.mark.integration
def test_serve_smoke_invocation(
    monkeypatch: pytest.MonkeyPatch, fake_web_app: types.ModuleType, tmp_path: Path
):
    """Smoke test: serve runs end-to-end with uvicorn.run stubbed out."""
    from reelgrep.config import get_settings
    from reelgrep.db import connect, migrate

    # Seed a DB with one video row at the configured default path.
    settings = get_settings()
    settings.home.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json) "
            "VALUES (?, ?, ?, ?)",
            ("blake2b:deadbeef", str(tmp_path / "v.mkv"),
             "2026-01-01T00:00:00Z", "{}"),
        )
        conn.commit()
    finally:
        conn.close()

    import uvicorn
    monkeypatch.setattr(uvicorn, "run", MagicMock(return_value=None))

    runner = CliRunner()
    result = runner.invoke(serve, ["--no-open-browser"])
    assert result.exit_code == 0, result.output
