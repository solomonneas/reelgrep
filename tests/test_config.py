"""Tests for reelgrep.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from reelgrep.config import Settings, ensure_dirs, get_settings, reset_settings

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


def test_defaults_when_no_env_set() -> None:
    s = get_settings()
    expected_home = Path("~/.local/share/reelgrep").expanduser().resolve()
    assert s.home == expected_home
    assert s.db_path == expected_home / "index.sqlite"
    assert s.cache_dir == expected_home / "cache"
    assert s.ffmpeg_binary == "ffmpeg"
    assert s.ffprobe_binary == "ffprobe"


def test_home_override_redirects_derived_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path))
    s = get_settings()
    assert s.home == tmp_path.resolve()
    assert s.db_path == tmp_path.resolve() / "index.sqlite"
    assert s.cache_dir == tmp_path.resolve() / "cache"


def test_explicit_db_wins_over_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    db = tmp_path / "custom" / "videos.db"
    monkeypatch.setenv("REELGREP_HOME", str(home))
    monkeypatch.setenv("REELGREP_DB", str(db))
    s = get_settings()
    assert s.home == home.resolve()
    assert s.db_path == db.resolve()
    assert s.cache_dir == home.resolve() / "cache"


def test_explicit_cache_wins_over_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    cache = tmp_path / "elsewhere" / "frames"
    monkeypatch.setenv("REELGREP_HOME", str(home))
    monkeypatch.setenv("REELGREP_CACHE", str(cache))
    s = get_settings()
    assert s.home == home.resolve()
    assert s.cache_dir == cache.resolve()
    assert s.db_path == home.resolve() / "index.sqlite"


def test_ffmpeg_env_reflected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REELGREP_FFMPEG", "/usr/local/bin/ffmpeg")
    s = get_settings()
    assert s.ffmpeg_binary == "/usr/local/bin/ffmpeg"


def test_ffprobe_env_reflected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REELGREP_FFPROBE", "/opt/bin/ffprobe")
    s = get_settings()
    assert s.ffprobe_binary == "/opt/bin/ffprobe"


def test_settings_is_frozen() -> None:
    s = get_settings()
    with pytest.raises(ValidationError):
        s.home = Path("/tmp/other")  # type: ignore[misc]


def test_settings_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Settings(extra_field="x")  # type: ignore[call-arg]


def test_ensure_dirs_creates_home_and_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "rg-home"
    monkeypatch.setenv("REELGREP_HOME", str(home))
    s = ensure_dirs()
    assert s.home.is_dir()
    assert s.cache_dir.is_dir()
    assert not s.db_path.exists()


def test_ensure_dirs_accepts_explicit_settings(tmp_path: Path) -> None:
    home = tmp_path / "explicit"
    s = Settings(
        home=home,
        db_path=home / "index.sqlite",
        cache_dir=home / "cache",
    )
    returned = ensure_dirs(s)
    assert returned is s
    assert (home).is_dir()
    assert (home / "cache").is_dir()


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path / "one"))
    first = get_settings()
    second = get_settings()
    assert first is second
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path / "two"))
    still_cached = get_settings()
    assert still_cached is first
    reset_settings()
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path / "three"))
    fresh = get_settings()
    assert fresh is not first
    assert fresh.home == (tmp_path / "three").resolve()
