"""Streaming BLAKE2b file hashing for local dedup."""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["file_hash", "CHUNK_SIZE"]

CHUNK_SIZE = 1024 * 1024


def file_hash(path: str | Path) -> str:
    """Return a ``blake2b:<hex>`` digest of the file at ``path``, streamed in 1 MiB chunks."""
    hasher = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return f"blake2b:{hasher.hexdigest()}"
