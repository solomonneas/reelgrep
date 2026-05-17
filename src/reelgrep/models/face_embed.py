"""Face-embedding person-finding backend using insightface (ArcFace)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reelgrep.frames import Frame
from reelgrep.models import BasePersonModel, Match, ModelError, register

__all__ = ["FaceEmbedModel"]

logger = logging.getLogger(__name__)


@register("face_embed")
class FaceEmbedModel(BasePersonModel):
    """Default person-finding backend; cosine to positive centroid minus nearest negative."""

    def __init__(
        self,
        *,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
        ctx_id: int = -1,
    ) -> None:
        self._model_pack = model_pack
        self._det_size = det_size
        self._providers = providers or ["CPUExecutionProvider"]
        self._ctx_id = ctx_id
        self._app: Any = None

    def config_dict(self) -> dict[str, Any]:
        """Return the backend's effective configuration."""
        return {
            "model_pack": self._model_pack,
            "det_size": list(self._det_size),
            "providers": self._providers,
            "ctx_id": self._ctx_id,
        }

    def _load(self) -> Any:
        """Lazily instantiate the insightface FaceAnalysis app."""
        if self._app is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ModelError(
                "face_embed backend requires the [face] extra: "
                "pip install reelgrep[face]"
            ) from exc
        app = FaceAnalysis(name=self._model_pack, providers=self._providers)
        app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
        self._app = app
        return app

    def _read_image(self, path: Path) -> Any:
        """Read an image off disk via opencv."""
        try:
            import cv2
        except ImportError as exc:
            raise ModelError("face_embed requires opencv (installed via [face])") from exc
        return cv2.imread(str(path))

    def _normalize(self, vec: Any) -> Any:
        """Return a unit-norm copy of the input vector."""
        import numpy as np

        n = float(np.linalg.norm(vec)) or 1.0
        return vec / n

    def _embeddings_for(self, paths: list[Path]) -> list[Any]:
        """Compute one or more face embeddings for each image path."""
        app = self._load()
        out: list[Any] = []
        for p in paths:
            img = self._read_image(p)
            if img is None:
                logger.warning("face_embed: could not read %s", p)
                continue
            faces = app.get(img)
            if not faces:
                logger.warning("face_embed: no faces detected in %s", p)
                continue
            for face in faces:
                emb = getattr(face, "normed_embedding", None)
                if emb is None:
                    emb = self._normalize(face.embedding)
                out.append(emb)
        return out

    def find(
        self,
        frames: list[Frame],
        positive_examples: list[Path],
        negative_examples: list[Path],
        *,
        threshold: float,
        top_k: int | None = None,
    ) -> list[Match]:
        """Score frames against positive/negative examples and return matches."""
        try:
            import numpy as np
        except ImportError as exc:
            raise ModelError("face_embed requires numpy") from exc

        if not positive_examples:
            raise ModelError("face_embed requires at least one positive example")
        pos_embs = self._embeddings_for(positive_examples)
        if not pos_embs:
            raise ModelError("face_embed: no faces detected in any positive example")
        pos_centroid = self._normalize(np.mean(np.stack(pos_embs), axis=0))

        neg_embs = self._embeddings_for(negative_examples) if negative_examples else []

        app = self._load()
        matches: list[Match] = []
        for frame in frames:
            frame_path = Path(frame.path)
            if not frame_path.exists():
                continue
            img = self._read_image(frame_path)
            if img is None:
                continue
            faces = app.get(img)
            best: Match | None = None
            for face in faces:
                emb = getattr(face, "normed_embedding", None)
                if emb is None:
                    emb = self._normalize(face.embedding)
                pos_score = float(np.dot(emb, pos_centroid))
                neg_score = max(
                    (float(np.dot(emb, n)) for n in neg_embs),
                    default=0.0,
                )
                score = pos_score - neg_score
                if score >= threshold:
                    bbox: tuple[int, int, int, int] | None = None
                    if hasattr(face, "bbox") and face.bbox is not None:
                        x1, y1, x2, y2 = (int(v) for v in face.bbox)
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                    cand = Match(
                        frame=frame,
                        confidence=score,
                        bbox=bbox,
                        reasoning=(
                            f"cosine {pos_score:.2f} to positive, "
                            f"{neg_score:.2f} to nearest negative"
                        ),
                    )
                    if best is None or cand.confidence > best.confidence:
                        best = cand
            if best is not None:
                matches.append(best)

        matches.sort(key=lambda m: m.confidence, reverse=True)
        if top_k is not None:
            matches = matches[:top_k]
        return matches
