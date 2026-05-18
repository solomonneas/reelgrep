"""Load prose transcripts from .txt, .md, or .pdf into clean text."""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["TranscriptLoadError", "load_transcript", "split_into_words"]


class TranscriptLoadError(RuntimeError):
    """Raised when a transcript cannot be read."""


def load_transcript(path: str | Path) -> str:
    """Read a .txt, .md, or .pdf transcript and return normalized prose."""
    p = Path(path).expanduser().resolve(strict=True)
    suffix = p.suffix.lower()
    if suffix in (".txt", ".md", ".markdown"):
        text = p.read_text(encoding="utf-8", errors="replace")
        return _normalize(text, strip_markdown=suffix in (".md", ".markdown"))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise TranscriptLoadError(
                "PDF transcripts require the [align] extra: pip install reelgrep[align]"
            ) from exc
        reader = PdfReader(str(p))
        pages = [page.extract_text() or "" for page in reader.pages]
        return _normalize("\n".join(pages), strip_markdown=False)
    raise TranscriptLoadError(
        f"unsupported transcript format: {suffix} (supported: .txt, .md, .pdf)"
    )


def _normalize(text: str, *, strip_markdown: bool) -> str:
    """Collapse whitespace and (optionally) strip markdown formatting."""
    if strip_markdown:
        # Strip ATX headers, emphasis markers, link wrappers - keep the link text.
        text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Collapse internal whitespace, preserve paragraph breaks (double newline).
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_words(text: str) -> list[str]:
    """Tokenize prose into lowercase alphanumeric words (apostrophes preserved)."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())
