"""Library-level face detection, clustering, and labeling API."""
from __future__ import annotations

import logging
import sqlite3
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
    "ClusterReport",
    "ExtractFacesResult",
    "FaceCluster",
    "FaceDetection",
    "Faces",
    "FacesError",
    "InsightFaceMissingError",
    "cluster_faces",
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


@dataclass(frozen=True)
class ClusterReport:
    clusters_found: int
    detections_clustered: int
    noise_detections: int
    labels_carried_over: int
    labels_orphaned: list[str]
    computed_at: str


def _load_hdbscan() -> Any:
    try:
        import hdbscan  # type: ignore
    except Exception as exc:
        raise FacesError(
            "hdbscan is required for clustering. Install reelgrep with the [face] extra: "
            "pipx install 'reelgrep[face]'"
        ) from exc
    return hdbscan


def _decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def cluster_faces(
    *,
    min_cluster_size: int = 5,
    db_path: Path | None = None,
    label_carry_threshold: float = 0.4,
) -> ClusterReport:
    """(Re)compute clusters over the current face_detections pool.

    Preserves labels by centroid proximity.
    """
    _assert_little_endian()
    settings = get_settings()
    resolved_db = Path(db_path) if db_path is not None else settings.db_path
    if not resolved_db.exists():
        raise FacesError(f"index does not exist at {resolved_db}")

    hdbscan = _load_hdbscan()
    conn = connect(resolved_db)
    migrate(conn)
    conn.execute("PRAGMA foreign_keys = ON")

    rows = conn.execute("SELECT id, embedding FROM face_detections").fetchall()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if not rows:
        conn.commit()
        conn.close()
        return ClusterReport(0, 0, 0, 0, [], now)
    det_ids = np.array([r[0] for r in rows], dtype=np.int64)
    embeddings = np.stack([_normalize(_decode_embedding(r[1])) for r in rows])

    # Snapshot labeled centroids BEFORE wiping clusters.
    label_snapshots: list[tuple[str, np.ndarray]] = []
    for cid, label in conn.execute(
        "SELECT id, label FROM face_clusters WHERE label IS NOT NULL"
    ):
        member_blobs = [b[0] for b in conn.execute(
            "SELECT fd.embedding FROM face_cluster_members m "
            "JOIN face_detections fd ON fd.id = m.detection_id WHERE m.cluster_id = ?",
            (cid,),
        )]
        if not member_blobs:
            continue
        centroid = _normalize(np.mean(
            np.stack([_normalize(_decode_embedding(b)) for b in member_blobs]), axis=0,
        ))
        label_snapshots.append((label, centroid))

    conn.execute("DELETE FROM face_cluster_members")
    conn.execute("DELETE FROM face_clusters")

    # hdbscan's generic algorithm requires float64 input.
    X = embeddings.astype(np.float64)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="cosine",
        cluster_selection_method="eom",
        algorithm="generic",
    )
    labels = clusterer.fit_predict(X)
    # Fallback: HDBSCAN cannot recognize a single isolated cluster without help.
    # When the default pass marks everything as noise, retry with allow_single_cluster
    # so a tight blob of one identity still produces a cluster.
    if (labels >= 0).sum() == 0:
        fallback = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric="cosine",
            cluster_selection_method="eom",
            algorithm="generic",
            allow_single_cluster=True,
        )
        labels = fallback.fit_predict(X)

    cluster_id_by_label: dict[int, int] = {}
    centroids_by_new_cluster: dict[int, np.ndarray] = {}
    for cl in sorted(set(labels.tolist())):
        if cl == -1:
            continue
        mask = labels == cl
        cluster_embs = embeddings[mask]
        centroid = _normalize(cluster_embs.mean(axis=0))
        sims_to_centroid = cluster_embs @ centroid
        rep_local_idx = int(np.argmax(sims_to_centroid))
        rep_det_id = int(det_ids[mask][rep_local_idx])
        cur = conn.execute(
            "INSERT INTO face_clusters(rep_detection_id, size, computed_at) VALUES (?,?,?)",
            (rep_det_id, int(mask.sum()), now),
        )
        cluster_id_by_label[int(cl)] = cur.lastrowid
        centroids_by_new_cluster[cur.lastrowid] = centroid
        for i, det_id in enumerate(det_ids[mask].tolist()):
            distance = float(1.0 - sims_to_centroid[i])
            conn.execute(
                "INSERT INTO face_cluster_members(cluster_id, detection_id, distance) "
                "VALUES (?,?,?)",
                (cur.lastrowid, int(det_id), distance),
            )

    # Reattach labels by centroid proximity.
    labels_carried = 0
    orphaned: list[str] = []
    taken: set[int] = set()
    for label, snap_centroid in label_snapshots:
        best_cid, best_sim = None, -1.0
        for new_cid, centroid in centroids_by_new_cluster.items():
            if new_cid in taken:
                continue
            sim = float(snap_centroid @ centroid)
            if sim > best_sim:
                best_sim = sim
                best_cid = new_cid
        if best_cid is not None and (1.0 - best_sim) <= label_carry_threshold:
            conn.execute(
                "UPDATE face_clusters SET label = ? WHERE id = ?", (label, best_cid)
            )
            taken.add(best_cid)
            labels_carried += 1
        else:
            orphaned.append(label)
            conn.execute(
                "INSERT INTO face_clusters(label, size, computed_at) VALUES (?,?,?)",
                (label, 0, now),
            )

    conn.commit()
    conn.close()
    return ClusterReport(
        clusters_found=len(cluster_id_by_label),
        detections_clustered=int((labels != -1).sum()),
        noise_detections=int((labels == -1).sum()),
        labels_carried_over=labels_carried,
        labels_orphaned=orphaned,
        computed_at=now,
    )


@dataclass(frozen=True)
class FaceDetection:
    id: int
    frame_id: int
    video_id: int
    video_path: str
    timestamp_ms: int
    bbox: tuple[int, int, int, int]
    frame_path: str
    embedding_model: str
    detected_at: str


@dataclass(frozen=True)
class FaceCluster:
    id: int
    label: str | None
    size: int
    rep_detection_id: int | None
    computed_at: str


class Faces:
    """Read-and-update API over the face_detections / face_clusters tables."""

    def __init__(self, *, db_path: Path | None = None) -> None:
        settings = get_settings()
        self._db_path = Path(db_path) if db_path is not None else settings.db_path
        if not self._db_path.exists():
            raise FacesError(f"index does not exist at {self._db_path}")

    def _conn(self) -> sqlite3.Connection:
        conn = connect(self._db_path)
        migrate(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def list_clusters(
        self, *, labeled_only: bool = False, limit: int | None = None,
    ) -> list[FaceCluster]:
        sql = "SELECT id, label, size, rep_detection_id, computed_at FROM face_clusters"
        params: tuple[Any, ...] = ()
        if labeled_only:
            sql += " WHERE label IS NOT NULL"
        sql += " ORDER BY size DESC, id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        conn = self._conn()
        try:
            return [FaceCluster(*row) for row in conn.execute(sql, params)]
        finally:
            conn.close()

    def get_cluster(self, cluster_id: int) -> FaceCluster:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT id, label, size, rep_detection_id, computed_at "
                "FROM face_clusters WHERE id = ?",
                (cluster_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise FacesError(f"no cluster with id {cluster_id}")
        return FaceCluster(*row)

    def cluster_members(
        self, cluster_id: int, *, limit: int | None = None,
    ) -> list[FaceDetection]:
        sql = (
            "SELECT fd.id, fd.frame_id, f.video_id, v.path, f.timestamp_ms, "
            "fd.bbox_x, fd.bbox_y, fd.bbox_w, fd.bbox_h, f.path, "
            "fd.embedding_model, fd.detected_at "
            "FROM face_cluster_members m "
            "JOIN face_detections fd ON fd.id = m.detection_id "
            "JOIN frames f ON f.id = fd.frame_id "
            "JOIN videos v ON v.id = f.video_id "
            "WHERE m.cluster_id = ? ORDER BY m.distance ASC"
        )
        params: tuple[Any, ...] = (cluster_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (cluster_id, int(limit))
        conn = self._conn()
        try:
            return [
                FaceDetection(
                    id=r[0], frame_id=r[1], video_id=r[2], video_path=r[3],
                    timestamp_ms=r[4], bbox=(r[5], r[6], r[7], r[8]),
                    frame_path=r[9], embedding_model=r[10], detected_at=r[11],
                )
                for r in conn.execute(sql, params)
            ]
        finally:
            conn.close()

    def find_by_label(self, label: str) -> list[FaceDetection]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT id FROM face_clusters WHERE label = ?", (label,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return []
        return self.cluster_members(row[0])

    def find_by_embedding(
        self, embedding: np.ndarray, *, top_k: int = 25, min_similarity: float = 0.5,
    ) -> list[tuple[FaceDetection, float]]:
        if embedding.shape != (EMBEDDING_DIM,):
            raise FacesError(
                f"embedding shape {embedding.shape}, expected ({EMBEDDING_DIM},)"
            )
        q = _normalize(embedding.astype(np.float32))
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT fd.id, fd.frame_id, f.video_id, v.path, f.timestamp_ms, "
                "fd.bbox_x, fd.bbox_y, fd.bbox_w, fd.bbox_h, f.path, "
                "fd.embedding_model, fd.detected_at, fd.embedding "
                "FROM face_detections fd "
                "JOIN frames f ON f.id = fd.frame_id "
                "JOIN videos v ON v.id = f.video_id"
            ).fetchall()
        finally:
            conn.close()
        scored: list[tuple[FaceDetection, float]] = []
        for r in rows:
            emb = _normalize(_decode_embedding(r[12]))
            sim = float(q @ emb)
            if sim < min_similarity:
                continue
            scored.append((
                FaceDetection(
                    id=r[0], frame_id=r[1], video_id=r[2], video_path=r[3],
                    timestamp_ms=r[4], bbox=(r[5], r[6], r[7], r[8]),
                    frame_path=r[9], embedding_model=r[10], detected_at=r[11],
                ),
                sim,
            ))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def label_cluster(self, cluster_id: int, label: str | None) -> None:
        # get_cluster raises if missing.
        self.get_cluster(cluster_id)
        conn = self._conn()
        try:
            if label is not None:
                clash = conn.execute(
                    "SELECT id FROM face_clusters WHERE label = ? AND id != ?",
                    (label, cluster_id),
                ).fetchone()
                if clash is not None:
                    raise FacesError(
                        f"label {label!r} already on cluster {clash[0]}; "
                        f"clear that one or pick another label"
                    )
            conn.execute("UPDATE face_clusters SET label = ? WHERE id = ?",
                         (label, cluster_id))
            conn.commit()
        finally:
            conn.close()

    def detection_count(self) -> int:
        conn = self._conn()
        try:
            return conn.execute("SELECT COUNT(*) FROM face_detections").fetchone()[0]
        finally:
            conn.close()

    def purge_video(self, video: str | Path) -> int:
        video_str = str(video)
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT id FROM videos WHERE path = ?", (video_str,),
            ).fetchone()
            if row is None:
                # try the resolved form too
                row = conn.execute(
                    "SELECT id FROM videos WHERE path = ?",
                    (str(Path(video).resolve()),),
                ).fetchone()
            if row is None:
                return 0
            cur = conn.execute(
                "DELETE FROM face_detections WHERE frame_id IN "
                "(SELECT id FROM frames WHERE video_id = ?)",
                (row[0],),
            )
            conn.commit()
            return cur.rowcount or 0
        finally:
            conn.close()

    def purge_all(self) -> None:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM face_cluster_members")
            conn.execute("DELETE FROM face_clusters")
            conn.execute("DELETE FROM face_detections")
            conn.commit()
        finally:
            conn.close()
