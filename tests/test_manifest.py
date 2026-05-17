"""Tests for reelgrep.manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import reelgrep
from reelgrep.manifest import Manifest, Source, read, sidecar_path, write


def _make_manifest(**overrides) -> Manifest:
    defaults = dict(
        operation="find-person",
        source=Source(
            path="/abs/path/to/video.mkv",
            file_hash="blake2b:abcd1234",
            duration_ms=5432100,
        ),
        parameters={"person": "alice", "threshold": 0.4},
        results=[{"timestamp_ms": 1000, "confidence": 0.92}],
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def test_round_trip(tmp_path: Path) -> None:
    manifest = _make_manifest()
    out = tmp_path / "out.json"
    written = write(out, manifest)
    assert written == out.resolve()
    loaded = read(out)
    assert loaded == manifest


def test_tool_default() -> None:
    manifest = _make_manifest()
    assert manifest.tool == "reelgrep"


def test_tool_version_default_matches_package() -> None:
    manifest = _make_manifest()
    assert manifest.tool_version == reelgrep.__version__


def test_manifest_version_default() -> None:
    manifest = _make_manifest()
    assert manifest.manifest_version == "1.0"


def test_manifest_version_literal_enforced() -> None:
    with pytest.raises(ValidationError):
        _make_manifest(manifest_version="2.0")


def test_created_at_format() -> None:
    manifest = _make_manifest()
    assert "T" in manifest.created_at
    assert re.search(r"T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2}|Z)$", manifest.created_at)


def test_source_file_hash_rejects_bad_prefix() -> None:
    with pytest.raises(ValidationError):
        Source(path="/x", file_hash="sha256:abc")


def test_source_file_hash_accepts_blake2b() -> None:
    src = Source(path="/x", file_hash="blake2b:abc")
    assert src.file_hash == "blake2b:abc"


def test_missing_source_raises() -> None:
    with pytest.raises(ValidationError):
        Manifest(operation="x", parameters={}, results=[])


def test_missing_operation_raises() -> None:
    with pytest.raises(ValidationError):
        Manifest(
            source=Source(path="/x", file_hash="blake2b:a"),
            parameters={},
            results=[],
        )


def test_missing_parameters_raises() -> None:
    with pytest.raises(ValidationError):
        Manifest(
            operation="x",
            source=Source(path="/x", file_hash="blake2b:a"),
            results=[],
        )


def test_missing_results_raises() -> None:
    with pytest.raises(ValidationError):
        Manifest(
            operation="x",
            source=Source(path="/x", file_hash="blake2b:a"),
            parameters={},
        )


def test_extra_field_on_manifest_raises() -> None:
    with pytest.raises(ValidationError):
        _make_manifest(bogus="x")


def test_extra_field_on_source_raises() -> None:
    with pytest.raises(ValidationError):
        Source(path="/x", file_hash="blake2b:a", bogus="y")


def test_sidecar_path_mp4() -> None:
    assert sidecar_path("/tmp/clip.mp4") == Path("/tmp/clip.mp4.manifest.json")


def test_sidecar_path_multiple_dots() -> None:
    assert sidecar_path("/tmp/foo.bar.mp4") == Path("/tmp/foo.bar.mp4.manifest.json")


def test_sidecar_path_no_extension() -> None:
    assert sidecar_path("/tmp/no-ext") == Path("/tmp/no-ext.manifest.json")


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "out.json"
    manifest = _make_manifest()
    written = write(nested, manifest)
    assert written.exists()
    assert nested.exists()


def test_read_raises_on_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"manifest_version": "1.0"}', encoding="utf-8")
    with pytest.raises(ValidationError):
        read(bad)


def test_written_file_is_indented(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    write(out, _make_manifest())
    text = out.read_text(encoding="utf-8")
    assert '\n  "tool": "reelgrep"' in text


def test_written_file_ends_with_newline(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    write(out, _make_manifest())
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")


def test_written_file_is_valid_utf8_json(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    write(out, _make_manifest())
    raw = out.read_bytes()
    decoded = raw.decode("utf-8")
    data = json.loads(decoded)
    assert data["tool"] == "reelgrep"
    assert data["manifest_version"] == "1.0"
