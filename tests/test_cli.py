"""Smoke tests for the CLI entry point: command registration and global options."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from reelgrep import __version__
from reelgrep.cli import main


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path))
    for var in ("REELGREP_DB", "REELGREP_CACHE", "REELGREP_FFMPEG", "REELGREP_FFPROBE"):
        monkeypatch.delenv(var, raising=False)
    from reelgrep.config import reset_settings
    reset_settings()
    yield
    reset_settings()  # clears _db_override too


def test_version_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_all_commands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for name in ("align", "ingest", "export-clip", "make-gif", "search-subtitles",
                 "contact-sheet", "find-person", "serve", "transcribe", "info",
                 "ls", "jellyfin"):
        assert name in result.output


@pytest.mark.parametrize("subcmd", [
    "align", "ingest", "export-clip", "make-gif", "search-subtitles",
    "contact-sheet", "find-person", "serve", "transcribe", "info", "ls",
])
def test_subcommand_help_works(subcmd):
    runner = CliRunner()
    result = runner.invoke(main, [subcmd, "--help"])
    assert result.exit_code == 0, result.output
    assert subcmd.replace("-", "_") in result.output.replace("-", "_") or subcmd in result.output


def test_db_override_sets_env(tmp_path, monkeypatch):
    runner = CliRunner()
    db_path = tmp_path / "custom.sqlite"
    # ls works against an empty DB; if --db is honored, it builds the DB at the override path.
    result = runner.invoke(main, ["--db", str(db_path), "ls"])
    assert result.exit_code == 0, result.output
    assert "no videos indexed" in result.output
    assert db_path.exists()


def test_ls_uses_default_db_when_no_override(tmp_path):
    from reelgrep.config import get_settings
    runner = CliRunner()
    result = runner.invoke(main, ["ls"])
    assert result.exit_code == 0, result.output
    # Default db_path is <REELGREP_HOME>/index.sqlite which is under tmp_path here.
    expected_db = get_settings().db_path
    assert expected_db.exists()
