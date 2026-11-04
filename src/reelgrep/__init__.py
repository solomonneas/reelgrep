"""reelgrep - local video search and media analysis.

Public library API. Downstream consumers should import names directly from
the top-level ``reelgrep`` package rather than from submodules; submodule
paths are not part of the stability contract.

Surface area:

* Indexing: :func:`ingest_video`, :class:`IngestResult`, :class:`IngestWarning`
* Search: :class:`Search`, :class:`SubtitleHit`, :class:`DetectionHit`,
  :class:`FrameRow`, :class:`SearchError`, :class:`InvalidQueryError`
* Transcription: :func:`transcribe_video`, :class:`TranscribeResult`,
  :class:`TranscribeError`
* Backend registry: :class:`BaseBackend`, :func:`register_backend`,
  :func:`get_backend`, :func:`list_backends`
* Person-model registry: :class:`BasePersonModel`,
  :func:`register_person_model`, :func:`get_person_model`,
  :func:`list_person_models`
"""

from __future__ import annotations

from reelgrep.backends import (
    BaseBackend,
    get_backend,
    list_backends,
)
from reelgrep.backends import register as register_backend
from reelgrep.faces import (
    ClusterReport,
    ExtractFacesResult,
    FaceCluster,
    FaceDetection,
    Faces,
    FacesError,
    InsightFaceMissingError,
    cluster_faces,
    extract_faces,
)
from reelgrep.index import (
    IngestResult,
    IngestWarning,
    ingest_video,
)
from reelgrep.models import (
    BasePersonModel,
    get_person_model,
    list_person_models,
)
from reelgrep.models import register as register_person_model
from reelgrep.search import (
    DetectionHit,
    FrameRow,
    InvalidQueryError,
    Search,
    SearchError,
    SubtitleHit,
)
from reelgrep.transcribe import (
    TranscribeError,
    TranscribeResult,
    transcribe_video,
)

__version__ = "0.5.0"

__all__ = [
    "BaseBackend",
    "BasePersonModel",
    "ClusterReport",
    "DetectionHit",
    "ExtractFacesResult",
    "FaceCluster",
    "FaceDetection",
    "Faces",
    "FacesError",
    "FrameRow",
    "IngestResult",
    "IngestWarning",
    "InsightFaceMissingError",
    "InvalidQueryError",
    "Search",
    "SearchError",
    "SubtitleHit",
    "TranscribeError",
    "TranscribeResult",
    "__version__",
    "cluster_faces",
    "extract_faces",
    "get_backend",
    "get_person_model",
    "ingest_video",
    "list_backends",
    "list_person_models",
    "register_backend",
    "register_person_model",
    "transcribe_video",
]
