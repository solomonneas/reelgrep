"""Library-level face detection, clustering, and labeling API."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from reelgrep.config import get_settings
from reelgrep.db import connect, migrate

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512
DEFAULT_EMBEDDING_MODEL = "insightface_buffalo_l"

__all__ = [
    "EMBEDDING_DIM",
    "ExtractFacesResult",
    "FacesError",
    "InsightFaceMissingError",
    "extract_faces",
]


class FacesError(RuntimeError):
    """Base class for reelgrep.faces errors."""


class InsightFaceMissingError(FacesError):
    """Raised when the [face] extra is not installed but is required."""


@dataclass(frozen=True)
class ExtractFacesResult:
    """Summary returned by :func:`extract_faces`."""

    video_path: str
    video_id: int
    frames_scanned: int
    detections_added: int
    embedding_model: str
    skipped_existing: bool = False


def _load_insightface_app() -> Any:
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception as exc:
        raise InsightFaceMissingError(
            "insightface is not installed. Install reelgrep with the [face] extra: "
            "pipx install 'reelgrep[face]'"
        ) from exc
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise InsightFaceMissingError(
            "cv2 (opencv) is not installed. Install reelgrep with the [face] extra."
        ) from exc
    return cv2


def _assert_little_endian() -> None:
    if sys.byteorder != "little":
        raise FacesError(
            f"reelgrep face embeddings are stored little-endian; this host is "
            f"{sys.byteorder}-endian and is not supported in v0.5.0."
        )


def extract_faces(
    video: str | Path,
    *,
    db_path: Path | None = None,
    force: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> ExtractFacesResult:
    """Detect and embed all faces in every sampled frame of ``video``.

    Persists detections into the ``face_detections`` table. Idempotent for the
    same ``embedding_model`` unless ``force`` is True.
    """
    _assert_little_endian()

    settings = get_settings()
    resolved_db = Path(db_path) if db_path is not None else settings.db_path
    if not resolved_db.exists():
        raise FacesError(f"index does not exist at {resolved_db}; run `reelgrep ingest` first")

    conn = connect(resolved_db)
    migrate(conn)
    conn.execute("PRAGMA foreign_keys = ON")

    video_path = str(Path(video).resolve())
    # The video may have been ingested with an unresolved path; try both.
    row = conn.execute("SELECT id FROM videos WHERE path = ?", (str(video),)).fetchone()
    if row is None:
        row = conn.execute("SELECT id FROM videos WHERE path = ?", (video_path,)).fetchone()
    if row is None:
        conn.close()
        raise FacesError(f"video {video} is not in the index; run `reelgrep ingest` first")
    video_id = row[0]

    if not force:
        existing = conn.execute(
            "SELECT COUNT(*) FROM face_detections fd "
            "JOIN frames f ON f.id = fd.frame_id "
            "WHERE f.video_id = ? AND fd.embedding_model = ?",
            (video_id, embedding_model),
        ).fetchone()[0]
        if existing > 0:
            conn.close()
            return ExtractFacesResult(
                video_path=str(video),
                video_id=video_id,
                frames_scanned=0,
                detections_added=0,
                embedding_model=embedding_model,
                skipped_existing=True,
            )

    frames = conn.execute(
        "SELECT id, path FROM frames WHERE video_id = ? ORDER BY timestamp_ms",
        (video_id,),
    ).fetchall()

    app = _load_insightface_app()
    cv2 = _load_cv2()
    now = datetime.now(UTC).isoformat(timespec="seconds")

    detections_added = 0
    frames_scanned = 0
    for frame_id, frame_path in frames:
        if not Path(frame_path).exists():
            logger.warning("frame missing on disk: %s", frame_path)
            continue
        frames_scanned += 1
        img = cv2.imread(frame_path)
        if img is None:
            logger.warning("cv2 could not read frame: %s", frame_path)
            continue
        for face in app.get(img):
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            if embedding.shape != (EMBEDDING_DIM,):
                raise FacesError(
                    f"unexpected embedding shape {embedding.shape}, "
                    f"expected ({EMBEDDING_DIM},)"
                )
            conn.execute(
                "INSERT INTO face_detections(frame_id, bbox_x, bbox_y, bbox_w, bbox_h, "
                "embedding, embedding_model, detected_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    frame_id,
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1),
                    embedding.tobytes(),
                    embedding_model,
                    now,
                ),
            )
            detections_added += 1

    conn.commit()
    conn.close()
    return ExtractFacesResult(
        video_path=str(video),
        video_id=video_id,
        frames_scanned=frames_scanned,
        detections_added=detections_added,
        embedding_model=embedding_model,
    )
