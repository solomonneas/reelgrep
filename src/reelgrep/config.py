"""Runtime settings for reelgrep, sourced from env vars with sensible defaults."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Settings",
    "get_settings",
    "reset_settings",
    "ensure_dirs",
    "set_db_override",
    "get_db_override",
]


_DEFAULT_HOME = Path("~/.local/share/reelgrep").expanduser().resolve()
_db_override: Path | None = None


def _expand(value: str) -> Path:
    return Path(value).expanduser().resolve()


class Settings(BaseModel):
    """Immutable runtime configuration for reelgrep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    home: Path = _DEFAULT_HOME
    db_path: Path = _DEFAULT_HOME / "index.sqlite"
    cache_dir: Path = _DEFAULT_HOME / "cache"
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"


def _build_settings() -> Settings:
    home_env = os.environ.get("REELGREP_HOME")
    db_env = os.environ.get("REELGREP_DB")
    cache_env = os.environ.get("REELGREP_CACHE")
    ffmpeg_env = os.environ.get("REELGREP_FFMPEG")
    ffprobe_env = os.environ.get("REELGREP_FFPROBE")

    home = _expand(home_env) if home_env else _DEFAULT_HOME
    if _db_override is not None:
        db_path = _db_override
    else:
        db_path = _expand(db_env) if db_env else home / "index.sqlite"
    cache_dir = _expand(cache_env) if cache_env else home / "cache"

    return Settings(
        home=home,
        db_path=db_path,
        cache_dir=cache_dir,
        ffmpeg_binary=ffmpeg_env or "ffmpeg",
        ffprobe_binary=ffprobe_env or "ffprobe",
    )


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return _build_settings()


def get_settings() -> Settings:
    """Return the active Settings instance, cached after first call."""
    return _cached_settings()


def reset_settings() -> None:
    """Clear the cached Settings and any process-level overrides."""
    global _db_override
    _db_override = None
    _cached_settings.cache_clear()


def set_db_override(path: Path | str | None) -> None:
    """Set a process-level db_path that wins over REELGREP_DB; pass None to clear."""
    global _db_override
    _db_override = _expand(str(path)) if path is not None else None
    _cached_settings.cache_clear()


def get_db_override() -> Path | None:
    """Return the current process-level db_path override, or None when unset."""
    return _db_override


def ensure_dirs(settings: Settings | None = None) -> Settings:
    """Create the home and cache directories on disk and return the settings."""
    s = settings if settings is not None else get_settings()
    s.home.mkdir(parents=True, exist_ok=True)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s
