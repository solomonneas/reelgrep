"""Tests for reelgrep.hashing."""

from __future__ import annotations

import hashlib

import pytest

from reelgrep.hashing import CHUNK_SIZE, file_hash


def test_deterministic(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    assert file_hash(p) == file_hash(p)


def test_prefix(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    assert file_hash(p).startswith("blake2b:")


def test_hex_length(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    digest = file_hash(p).split(":", 1)[1]
    assert len(digest) == 64
    int(digest, 16)


def test_empty_file_known_value(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    expected = hashlib.blake2b(b"", digest_size=32).hexdigest()
    assert file_hash(p) == f"blake2b:{expected}"


def test_multi_chunk(tmp_path):
    p = tmp_path / "big.bin"
    size = CHUNK_SIZE * 2 + 17
    data = bytes((i * 31) % 256 for i in range(size))
    p.write_bytes(data)
    expected = hashlib.blake2b(data, digest_size=32).hexdigest()
    assert file_hash(p) == f"blake2b:{expected}"


def test_different_content_different_hash(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")
    assert file_hash(a) != file_hash(b)


def test_str_and_path_equivalent(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"interchangeable")
    assert file_hash(p) == file_hash(str(p))


def test_file_not_found(tmp_path):
    missing = tmp_path / "does-not-exist.bin"
    with pytest.raises(FileNotFoundError):
        file_hash(missing)


def test_is_a_directory(tmp_path):
    with pytest.raises(IsADirectoryError):
        file_hash(tmp_path)


def test_chunk_size_constant():
    assert CHUNK_SIZE == 1024 * 1024
