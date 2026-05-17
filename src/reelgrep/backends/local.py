"""Local filesystem backend: identity path resolution."""

from __future__ import annotations

from pathlib import Path

from reelgrep.backends import BackendError, BaseBackend, register

__all__ = ["LocalBackend"]


@register("local")
class LocalBackend(BaseBackend):
    """Trivially resolves a local file path, asserting it exists."""

    def resolve(self, uri: str) -> Path:
        """Expand and resolve a local path, raising BackendError if missing/non-file."""
        path = Path(uri).expanduser().resolve()
        if not path.exists():
            raise BackendError(f"path does not exist: {path}")
        if not path.is_file():
            raise BackendError(f"path is not a file: {path}")
        return path
