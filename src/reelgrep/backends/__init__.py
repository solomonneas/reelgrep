"""Pluggable backends for resolving video identifiers to local file paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

__all__ = ["BaseBackend", "BackendError", "register", "get_backend", "list_backends"]


class BackendError(RuntimeError):
    """Raised when a backend cannot resolve a URI to a real path."""


class BaseBackend(ABC):
    """Abstract base class for source-of-video backends."""

    name: str = ""  # set by @register

    @abstractmethod
    def resolve(self, uri: str) -> Path:
        """Resolve a backend-specific URI/identifier to an absolute local file path."""


_REGISTRY: dict[str, type[BaseBackend]] = {}


def register(name: str):
    """Class decorator that registers a BaseBackend subclass under `name`."""

    def decorator(cls: type[BaseBackend]) -> type[BaseBackend]:
        if not issubclass(cls, BaseBackend):
            raise TypeError(f"{cls.__name__} must subclass BaseBackend")
        if name in _REGISTRY:
            raise ValueError(f"backend {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(name: str, **kwargs) -> BaseBackend:
    """Instantiate a backend by registered name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend: {name!r} (available: {sorted(_REGISTRY)})")
    return _REGISTRY[name](**kwargs)


def list_backends() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(_REGISTRY)


# Bundled backends self-register on import.
from reelgrep.backends import local as _local  # noqa: E402, F401

try:
    from reelgrep.backends import jellyfin as _jellyfin  # noqa: E402, F401
except ImportError:
    pass
