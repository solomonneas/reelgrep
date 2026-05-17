"""Tests for the backends registry and the local backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from reelgrep.backends import (
    _REGISTRY,
    BackendError,
    BaseBackend,
    get_backend,
    list_backends,
    register,
)
from reelgrep.backends.local import LocalBackend


@pytest.fixture
def registry_snapshot():
    """Snapshot _REGISTRY keys and remove any entries added during the test."""
    before = set(_REGISTRY)
    yield
    added = set(_REGISTRY) - before
    for key in added:
        del _REGISTRY[key]


def test_list_backends_includes_local():
    assert "local" in list_backends()


def test_get_backend_returns_local_instance():
    backend = get_backend("local")
    assert isinstance(backend, LocalBackend)
    assert backend.name == "local"


def test_get_backend_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="unknown backend"):
        get_backend("nonexistent")


def test_local_resolve_returns_resolved_path(tmp_path):
    target = tmp_path / "video.mp4"
    target.write_bytes(b"x")
    result = LocalBackend().resolve(str(target))
    assert result == target.resolve()


def test_local_resolve_follows_symlink(tmp_path):
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x")
    link = tmp_path / "link.mp4"
    link.symlink_to(real)
    result = LocalBackend().resolve(str(link))
    assert result == real.resolve()


def test_local_resolve_missing_path_raises(tmp_path):
    missing = tmp_path / "nope.mp4"
    with pytest.raises(BackendError, match="path does not exist"):
        LocalBackend().resolve(str(missing))


def test_local_resolve_directory_raises(tmp_path):
    with pytest.raises(BackendError, match="path is not a file"):
        LocalBackend().resolve(str(tmp_path))


def test_local_resolve_expands_tilde():
    # `~` should expand to $HOME before resolution; nonexistent suffix should
    # produce a BackendError whose message contains the expanded absolute path.
    home = Path("~").expanduser().resolve()
    with pytest.raises(BackendError) as exc:
        LocalBackend().resolve("~/__reelgrep_does_not_exist__.mp4")
    assert str(home) in str(exc.value)


def test_register_duplicate_raises_valueerror(registry_snapshot):
    with pytest.raises(ValueError, match="already registered"):

        @register("local")
        class _Dup(BaseBackend):
            def resolve(self, uri: str) -> Path:
                return Path(uri)


def test_register_non_basebackend_raises_typeerror(registry_snapshot):
    with pytest.raises(TypeError, match="must subclass BaseBackend"):

        @register("bogus")
        class _NotABackend:
            def resolve(self, uri: str) -> Path:
                return Path(uri)
