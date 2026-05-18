"""Unit tests for reelgrep.transcript_loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from reelgrep.transcript_loaders import (
    TranscriptLoadError,
    load_transcript,
    split_into_words,
)


def test_load_txt_collapses_whitespace_preserves_paragraph(tmp_path: Path) -> None:
    src = tmp_path / "lecture.txt"
    src.write_text("Welcome to the lecture.\n\nToday we cover networking.\n", encoding="utf-8")
    result = load_transcript(src)
    assert result == "Welcome to the lecture.\n\nToday we cover networking."


def test_load_md_strips_headers_and_emphasis(tmp_path: Path) -> None:
    src = tmp_path / "module.md"
    src.write_text(
        "# Module 1\n\n**Welcome** to the _lecture_.\n\n## Part 1\n\nText here.\n",
        encoding="utf-8",
    )
    result = load_transcript(src)
    assert "Module 1" in result
    assert "**" not in result
    assert "Welcome to the lecture." in result
    assert "## " not in result
    assert "Part 1" in result
    assert "Text here." in result


def test_load_md_strips_links_keeps_link_text(tmp_path: Path) -> None:
    src = tmp_path / "links.md"
    src.write_text("See the [OSI model](https://example.com) for layering.", encoding="utf-8")
    result = load_transcript(src)
    assert "OSI model" in result
    assert "https://example.com" not in result
    assert "[" not in result
    assert "(" not in result


def test_load_md_strips_inline_code_and_blockquotes(tmp_path: Path) -> None:
    src = tmp_path / "code.md"
    src.write_text("Run `npm install`.\n\n> Quoted line.\n", encoding="utf-8")
    result = load_transcript(src)
    assert "Run npm install." in result
    assert "`" not in result
    assert "Quoted line." in result
    assert "> " not in result


def test_load_pdf_with_monkeypatched_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "fake.pdf"
    src.write_bytes(b"%PDF-1.4\n")  # not a real PDF; reader is mocked

    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("Page one text."), FakePage("Page two text.")]

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)
    result = load_transcript(src)
    assert "Page one text." in result
    assert "Page two text." in result


@pytest.mark.integration
def test_load_pdf_real_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    # Building a real PDF requires a writer (pypdf only reads). Skip unless
    # a fixture is provided.
    fixture = Path(__file__).parent / "fixtures" / "transcript.pdf"
    if not fixture.exists():
        pytest.skip("no real PDF fixture available")
    result = load_transcript(fixture)
    assert result.strip()


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    src = tmp_path / "file.docx"
    src.write_text("ignored", encoding="utf-8")
    with pytest.raises(TranscriptLoadError, match="unsupported transcript format"):
        load_transcript(src)


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.txt"
    with pytest.raises(FileNotFoundError):
        load_transcript(missing)


def test_whitespace_normalization(tmp_path: Path) -> None:
    src = tmp_path / "ws.txt"
    src.write_text("Hello\t\t  world.\n   Indented line.\n\n\n\nEnd.\n", encoding="utf-8")
    result = load_transcript(src)
    assert "Hello world." in result
    assert "\t" not in result
    assert "  " not in result  # no double spaces
    # Triple blank lines collapsed to one paragraph break
    assert "\n\n\n" not in result


def test_split_into_words_basic() -> None:
    assert split_into_words("Hello, world's lecture!") == ["hello", "world's", "lecture"]


def test_split_into_words_empty() -> None:
    assert split_into_words("") == []


def test_split_into_words_preserves_numbers() -> None:
    assert split_into_words("OSI model layer 4") == ["osi", "model", "layer", "4"]


def test_split_into_words_punctuation_only() -> None:
    assert split_into_words("!@#$%^&*()") == []


def test_load_transcript_accepts_str_path(tmp_path: Path) -> None:
    src = tmp_path / "p.txt"
    src.write_text("hello", encoding="utf-8")
    assert load_transcript(str(src)) == "hello"


def test_pdf_loader_raises_when_pypdf_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "x.pdf"
    src.write_bytes(b"%PDF-1.4\n")
    import sys

    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(TranscriptLoadError, match=r"\[align\] extra"):
        load_transcript(src)
