"""Tests for reelgrep.contact_sheet."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from reelgrep.contact_sheet import build


def _make_frame(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (1, 1)) -> Path:
    img = Image.new("RGB", size, color=color)
    img.save(path, format="PNG")
    return path


@pytest.fixture
def six_frames(tmp_path: Path) -> list[Path]:
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    paths: list[Path] = []
    for i, color in enumerate(colors):
        p = tmp_path / f"frame_{i}.png"
        _make_frame(p, color)
        paths.append(p)
    return paths


def _expected_dims(
    cols: int,
    rows: int,
    thumb_width: int,
    tile_height: int,
    margin: int,
    label_height: int,
) -> tuple[int, int]:
    tile_w = thumb_width + 2 * margin
    tile_h = tile_height + 2 * margin + label_height
    return cols * tile_w, rows * tile_h


def test_build_3x2_dimensions(tmp_path: Path, six_frames: list[Path]) -> None:
    out = tmp_path / "sheet.png"
    thumb_width = 320
    margin = 8
    label_height = 24
    result = build(
        six_frames,
        out,
        cols=3,
        rows=2,
        thumb_width=thumb_width,
        margin=margin,
        label_height=label_height,
    )
    assert result == out
    assert result.exists()
    # 1x1 source scaled to thumb_width=320 -> thumb height also 320.
    expected_tile_h = 320
    expected_w, expected_h = _expected_dims(
        3, 2, thumb_width, expected_tile_h, margin, label_height
    )
    with Image.open(result) as img:
        assert img.size == (expected_w, expected_h)


def test_build_auto_rows(tmp_path: Path, six_frames: list[Path]) -> None:
    out = tmp_path / "auto.png"
    result = build(six_frames, out, cols=3, thumb_width=64, margin=4, label_height=16)
    assert result.exists()
    with Image.open(result) as img:
        # rows = ceil(6 / 3) = 2
        tile_w = 64 + 2 * 4
        tile_h = 64 + 2 * 4 + 16
        assert img.size == (3 * tile_w, 2 * tile_h)


def test_build_grid_too_small(tmp_path: Path) -> None:
    paths = [_make_frame(tmp_path / f"f{i}.png", (10 * i, 0, 0)) for i in range(4)]
    out = tmp_path / "tiny.png"
    with pytest.raises(ValueError, match="grid too small"):
        build(paths, out, cols=3, rows=1, thumb_width=64)


def test_build_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build([], tmp_path / "empty.png")


def test_build_cols_zero_raises(tmp_path: Path, six_frames: list[Path]) -> None:
    with pytest.raises(ValueError):
        build(six_frames, tmp_path / "z.png", cols=0)


def test_build_thumb_width_too_small(tmp_path: Path, six_frames: list[Path]) -> None:
    with pytest.raises(ValueError):
        build(six_frames, tmp_path / "z.png", cols=3, thumb_width=16)


def test_build_timestamps_draws_labels(tmp_path: Path, six_frames: list[Path]) -> None:
    out = tmp_path / "ts.png"
    thumb_width = 64
    margin = 4
    label_height = 24
    background = (16, 16, 20)
    timestamps = [0, 5000, 10000, 15000, 20000, 25000]
    build(
        six_frames,
        out,
        cols=3,
        rows=2,
        thumb_width=thumb_width,
        margin=margin,
        label_height=label_height,
        timestamps_ms=timestamps,
        background=background,
    )
    with Image.open(out) as img:
        # Label band of cell (0,0): below the thumb, within the tile.
        tile_w = thumb_width + 2 * margin
        thumb_h = 64
        label_band_top = margin + thumb_h + 2 * margin
        label_band_bottom = label_band_top + label_height
        crop = img.crop((0, label_band_top, tile_w, label_band_bottom)).convert("RGB")
        raw = crop.tobytes()
    pixels = [tuple(raw[i : i + 3]) for i in range(0, len(raw), 3)]
    bg_total = sum(background)
    diffs = [abs(sum(px) - bg_total) for px in pixels]
    # Some pixels must differ from background (text was drawn).
    assert max(diffs) > 30
    assert sum(1 for d in diffs if d > 10) > 5


def test_build_creates_nested_parent_dir(tmp_path: Path, six_frames: list[Path]) -> None:
    out = tmp_path / "a" / "b" / "c" / "sheet.png"
    result = build(six_frames, out, cols=3, rows=2, thumb_width=64)
    assert result.exists()
    assert result.parent.is_dir()


def test_build_unknown_extension_rewrites_to_jpg(tmp_path: Path, six_frames: list[Path]) -> None:
    out = tmp_path / "sheet.xyz"
    result = build(six_frames, out, cols=3, rows=2, thumb_width=64)
    assert result.suffix == ".jpg"
    assert result.exists()
    assert not out.exists()
    with Image.open(result) as img:
        assert img.format == "JPEG"


def test_build_mixed_aspect_ratios(tmp_path: Path) -> None:
    sources = [
        ("a.png", (255, 0, 0), (10, 10)),  # 1:1
        ("b.png", (0, 255, 0), (20, 10)),  # 2:1
        ("c.png", (0, 0, 255), (10, 20)),  # 1:2
        ("d.png", (255, 255, 0), (30, 20)),  # 3:2
    ]
    paths: list[Path] = []
    for name, color, size in sources:
        p = tmp_path / name
        _make_frame(p, color, size=size)
        paths.append(p)

    thumb_width = 64
    margin = 4
    label_height = 16
    out = tmp_path / "mixed.png"
    result = build(
        paths,
        out,
        cols=2,
        rows=2,
        thumb_width=thumb_width,
        margin=margin,
        label_height=label_height,
    )
    assert result.exists()

    # Max thumb height: tallest is c.png (1:2 aspect) -> 64 * (20/10) = 128.
    thumb_heights = []
    for _, _, (sw, sh) in sources:
        scale = thumb_width / sw
        thumb_heights.append(max(1, int(round(sh * scale))))
    tile_height = max(thumb_heights)
    expected_w, expected_h = _expected_dims(2, 2, thumb_width, tile_height, margin, label_height)
    with Image.open(result) as img:
        assert img.size == (expected_w, expected_h)
    # Also confirm auto-rows math sanity check helper.
    assert math.ceil(len(paths) / 2) == 2
