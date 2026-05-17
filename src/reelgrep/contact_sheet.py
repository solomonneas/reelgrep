"""Build a contact sheet image from a sequence of frame images."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from reelgrep.timecode import format as format_ms

__all__ = ["build"]

_JPEG_EXTS = {".jpg", ".jpeg"}
_KNOWN_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
_LABEL_COLOR = (240, 240, 240)
_MAX_LABEL_CHARS = 30


def _truncate(name: str, limit: int = _MAX_LABEL_CHARS) -> str:
    if len(name) <= limit:
        return name
    if limit <= 3:
        return name[:limit]
    return name[: limit - 3] + "..."


def _measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _resolve_out_path(out_path: str | Path) -> Path:
    resolved = Path(out_path).expanduser()
    if resolved.suffix.lower() not in _KNOWN_EXTS:
        resolved = resolved.with_suffix(".jpg")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def build(
    frame_paths: Sequence[str | Path],
    out_path: str | Path,
    *,
    cols: int = 6,
    rows: int | None = None,
    thumb_width: int = 320,
    margin: int = 8,
    label_height: int = 24,
    font_path: str | None = None,
    timestamps_ms: Sequence[int] | None = None,
    background: tuple[int, int, int] = (16, 16, 20),
) -> Path:
    """Render a grid contact sheet of frames with labels and return the saved path."""
    if not frame_paths:
        raise ValueError("frame_paths is empty")
    if cols < 1:
        raise ValueError("cols must be >= 1")
    if thumb_width < 32:
        raise ValueError("thumb_width must be >= 32")

    count = len(frame_paths)
    if rows is None:
        rows = math.ceil(count / cols)
    if rows * cols < count:
        raise ValueError("grid too small for frame count")

    images: list[Image.Image] = []
    thumb_sizes: list[tuple[int, int]] = []
    try:
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            src_w, src_h = img.size
            if src_w <= 0 or src_h <= 0:
                raise ValueError(f"invalid frame dimensions for {p!r}")
            scale = thumb_width / src_w
            thumb_h = max(1, int(round(src_h * scale)))
            images.append(img)
            thumb_sizes.append((thumb_width, thumb_h))

        tile_height = max(h for _, h in thumb_sizes)
        tile_w = thumb_width + 2 * margin
        tile_h = tile_height + 2 * margin + label_height

        canvas_w = cols * tile_w
        canvas_h = rows * tile_h
        canvas = Image.new("RGB", (canvas_w, canvas_h), background)

        if font_path:
            try:
                font = ImageFont.truetype(font_path, 14)
            except OSError:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(canvas)

        for i, src_img in enumerate(images):
            row = i // cols
            col = i % cols
            cell_x = col * tile_w + margin
            cell_y = row * tile_h + margin

            _, thumb_h = thumb_sizes[i]
            resized = src_img.resize((thumb_width, thumb_h), Image.Resampling.LANCZOS)

            thumb_y_offset = (tile_height - thumb_h) // 2
            paste_x = cell_x
            paste_y = cell_y + thumb_y_offset
            canvas.paste(resized, (paste_x, paste_y))

            if timestamps_ms is not None and i < len(timestamps_ms):
                label = format_ms(int(timestamps_ms[i]))
            else:
                label = _truncate(Path(frame_paths[i]).name)

            text_w, text_h = _measure_text(draw, label, font)
            label_band_top = cell_y + tile_height + 2 * margin
            label_band_h = label_height
            text_x = cell_x + (thumb_width - text_w) // 2
            text_y = label_band_top + max(0, (label_band_h - text_h) // 2) - margin
            draw.text((text_x, text_y), label, fill=_LABEL_COLOR, font=font)
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass

    resolved = _resolve_out_path(out_path)
    save_kwargs: dict[str, object] = {}
    if resolved.suffix.lower() in _JPEG_EXTS:
        save_kwargs["format"] = "JPEG"
        save_kwargs["quality"] = 88
    canvas.save(resolved, **save_kwargs)
    return resolved
