"""Tests for the pluggable person-model registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from reelgrep.frames import Frame
from reelgrep.models import (
    _REGISTRY,
    BasePersonModel,
    Match,
    get_person_model,
    list_person_models,
    register,
)


def _make_frame() -> Frame:
    """Build a minimal Frame for tests that need one."""
    return Frame(timestamp_ms=0, path="/x.jpg", sampling_strategy="every_n")


@pytest.fixture
def registry_snapshot():
    """Snapshot registry keys before a test and remove any added entries after."""
    before = set(_REGISTRY)
    yield
    for added in set(_REGISTRY) - before:
        _REGISTRY.pop(added, None)


def test_match_round_trips_fields():
    """Match preserves all assigned fields including bbox and reasoning."""
    frame = _make_frame()
    m = Match(frame=frame, confidence=0.9, bbox=(1, 2, 3, 4), reasoning="hit")
    assert m.frame == frame
    assert m.confidence == 0.9
    assert m.bbox == (1, 2, 3, 4)
    assert m.reasoning == "hit"


def test_match_rejects_extra_fields():
    """Match has extra='forbid' and refuses unknown keys."""
    with pytest.raises(ValidationError):
        Match(frame=_make_frame(), confidence=0.5, bogus="nope")


def test_match_bbox_accepts_four_ints():
    """bbox of length 4 is accepted."""
    m = Match(frame=_make_frame(), confidence=0.1, bbox=(0, 0, 10, 10))
    assert m.bbox == (0, 0, 10, 10)


def test_match_bbox_rejects_wrong_length():
    """bbox tuple length is enforced by the type annotation."""
    with pytest.raises(ValidationError):
        Match(frame=_make_frame(), confidence=0.1, bbox=(1, 2, 3))
    with pytest.raises(ValidationError):
        Match(frame=_make_frame(), confidence=0.1, bbox=(1, 2, 3, 4, 5))


def test_match_bbox_defaults_to_none():
    """bbox is optional and defaults to None; reasoning defaults to ''."""
    m = Match(frame=_make_frame(), confidence=0.5)
    assert m.bbox is None
    assert m.reasoning == ""


def test_list_person_models_returns_sorted_strings():
    """list_person_models returns a sorted list of strings (possibly empty)."""
    names = list_person_models()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
    assert names == sorted(names)


def test_register_duplicate_name_raises(registry_snapshot):
    """Registering two classes under the same name raises ValueError."""

    @register("dup_for_test")
    class A(BasePersonModel):
        def find(self, frames, positive_examples, negative_examples, *, threshold, top_k=None):
            return []

    with pytest.raises(ValueError, match="already registered"):

        @register("dup_for_test")
        class B(BasePersonModel):
            def find(self, frames, positive_examples, negative_examples, *, threshold, top_k=None):
                return []


def test_register_non_subclass_raises(registry_snapshot):
    """The decorator rejects classes that don't subclass BasePersonModel."""
    with pytest.raises(TypeError, match="must subclass BasePersonModel"):

        @register("bad_for_test")
        class NotAModel:  # noqa: D401 - inline test class
            """Not a BasePersonModel."""


def test_get_person_model_unknown_raises():
    """get_person_model raises KeyError listing available names."""
    with pytest.raises(KeyError) as excinfo:
        get_person_model("nonexistent_xyz_123")
    msg = str(excinfo.value)
    assert "nonexistent_xyz_123" in msg
    assert "available" in msg


def test_register_and_retrieve_end_to_end(registry_snapshot):
    """A stub backend can be registered and instantiated via get_person_model."""

    @register("stub_for_test")
    class StubModel(BasePersonModel):
        def find(
            self,
            frames: list[Frame],
            positive_examples: list[Path],
            negative_examples: list[Path],
            *,
            threshold: float,
            top_k: int | None = None,
        ) -> list[Match]:
            return []

    assert "stub_for_test" in list_person_models()
    instance = get_person_model("stub_for_test")
    assert isinstance(instance, StubModel)
    assert isinstance(instance, BasePersonModel)
    assert instance.name == "stub_for_test"


def test_config_dict_default_returns_empty(registry_snapshot):
    """BasePersonModel.config_dict() defaults to an empty dict."""

    @register("cfg_for_test")
    class CfgStub(BasePersonModel):
        def find(self, frames, positive_examples, negative_examples, *, threshold, top_k=None):
            return []

    instance = get_person_model("cfg_for_test")
    assert instance.config_dict() == {}
