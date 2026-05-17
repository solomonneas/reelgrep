"""Runtime settings for reelgrep, sourced from env vars with sensible defaults."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

__all__ = ["Settings", "get_settings", "reset_settings", "ensure_dirs"]


_DEFAULT_HOME = Path("~/.local/share/reelgrep").expanduser().resolve()


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
    """Clear the cached Settings so the next get_settings() rereads env vars."""
    _cached_settings.cache_clear()


def ensure_dirs(settings: Settings | None = None) -> Settings:
    """Create the home and cache directories on disk and return the settings."""
    s = settings if settings is not None else get_settings()
    s.home.mkdir(parents=True, exist_ok=True)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s
