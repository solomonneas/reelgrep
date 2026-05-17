"""JSON sidecar manifest for reelgrep export operations."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reelgrep import __version__

__all__ = ["Source", "Manifest", "write", "read", "sidecar_path"]


class Source(BaseModel):
    """Source media descriptor for a manifest."""

    model_config = ConfigDict(extra="forbid")

    path: str
    file_hash: str
    duration_ms: int | None = None

    @field_validator("file_hash")
    @classmethod
    def _check_file_hash_prefix(cls, v: str) -> str:
        if not v.startswith("blake2b:"):
            raise ValueError("file_hash must start with 'blake2b:'")
        return v


class Manifest(BaseModel):
    """Reelgrep operation manifest written alongside every export."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal["1.0"] = "1.0"
    tool: Literal["reelgrep"] = "reelgrep"
    tool_version: str = Field(default_factory=lambda: __version__)
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).astimezone().isoformat(timespec="seconds")
    )
    operation: str
    source: Source
    parameters: dict[str, Any]
    results: list[dict[str, Any]]

    @field_validator("operation")
    @classmethod
    def _check_operation_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("operation must be non-empty")
        return v


def write(path: str | Path, manifest: Manifest) -> Path:
    """Serialize manifest to JSON and write it to path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = manifest.model_dump_json(indent=2) + "\n"
    p.write_text(text, encoding="utf-8")
    return p.resolve()


def read(path: str | Path) -> Manifest:
    """Read a manifest JSON file and validate it."""
    text = Path(path).read_text(encoding="utf-8")
    return Manifest.model_validate_json(text)


def sidecar_path(output_path: str | Path) -> Path:
    """Return the sidecar manifest path for an output file."""
    p = Path(output_path)
    return p.with_suffix(p.suffix + ".manifest.json")
