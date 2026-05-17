"""Pluggable models for person and visual search inside videos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from reelgrep.frames import Frame

__all__ = [
    "Match",
    "BasePersonModel",
    "ModelError",
    "register",
    "get_person_model",
    "list_person_models",
]


class ModelError(RuntimeError):
    """Raised when a model backend cannot complete a request."""


class Match(BaseModel):
    """One frame-level match returned by a person model."""

    model_config = ConfigDict(extra="forbid")

    frame: Frame
    confidence: float
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h
    reasoning: str = ""


class BasePersonModel(ABC):
    """Abstract base for person-finding model backends."""

    name: str = ""  # set by @register

    @abstractmethod
    def find(
        self,
        frames: list[Frame],
        positive_examples: list[Path],
        negative_examples: list[Path],
        *,
        threshold: float,
        top_k: int | None = None,
    ) -> list[Match]:
        """Score frames against examples; return matches sorted by confidence desc."""

    def config_dict(self) -> dict[str, Any]:
        """Return the backend's effective configuration for manifest provenance."""
        return {}


_REGISTRY: dict[str, type[BasePersonModel]] = {}


def register(name: str):
    """Class decorator registering a BasePersonModel subclass under `name`."""

    def decorator(cls: type[BasePersonModel]) -> type[BasePersonModel]:
        if not issubclass(cls, BasePersonModel):
            raise TypeError(f"{cls.__name__} must subclass BasePersonModel")
        if name in _REGISTRY:
            raise ValueError(f"person model {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_person_model(name: str, **kwargs) -> BasePersonModel:
    """Instantiate a registered person model by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown person model: {name!r} (available: {sorted(_REGISTRY)})")
    return _REGISTRY[name](**kwargs)


def list_person_models() -> list[str]:
    """Return the sorted list of registered person model names."""
    return sorted(_REGISTRY)


# Bundled backends self-register on import. Wrap in try/except because
# face_embed and ollama_vision are in separate task scopes.
try:
    from reelgrep.models import face_embed as _face_embed  # noqa: E402, F401
except ImportError:
    pass

try:
    from reelgrep.models import ollama_vision as _ollama  # noqa: E402, F401
except ImportError:
    pass
