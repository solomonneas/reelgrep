"""Lock the public API surface re-exported from the top-level ``reelgrep`` package.

These tests intentionally avoid exercising any of the underlying logic
(indexing, search, transcription) - existing tests cover that. Their sole
job is to detect accidental renames, drops, or shadowing of public names.
"""

from __future__ import annotations

import reelgrep

EXPECTED_PUBLIC_NAMES = {
    # Indexing
    "ingest_video",
    "IngestResult",
    "IngestWarning",
    # Search
    "Search",
    "SubtitleHit",
    "DetectionHit",
    "FrameRow",
    "SearchError",
    "InvalidQueryError",
    # Transcription
    "transcribe_video",
    "TranscribeResult",
    "TranscribeError",
    # Backend registry
    "BaseBackend",
    "register_backend",
    "get_backend",
    "list_backends",
    # Person-model registry
    "BasePersonModel",
    "register_person_model",
    "get_person_model",
    "list_person_models",
    # Faces
    "extract_faces",
    "cluster_faces",
    "Faces",
    "FaceDetection",
    "FaceCluster",
    "ClusterReport",
    "ExtractFacesResult",
    "FacesError",
    "InsightFaceMissingError",
    # Metadata
    "__version__",
}


def test_all_matches_expected_set() -> None:
    """``reelgrep.__all__`` is exactly the locked public surface."""
    assert set(reelgrep.__all__) == EXPECTED_PUBLIC_NAMES


def test_all_names_resolvable() -> None:
    """Every name in ``__all__`` is importable as an attribute of the package."""
    for name in reelgrep.__all__:
        assert hasattr(reelgrep, name), f"reelgrep.{name} is missing"
        # Sanity: not a stale stub
        assert getattr(reelgrep, name) is not None, f"reelgrep.{name} is None"


def test_all_names_in_dir() -> None:
    """``dir(reelgrep)`` includes every public name."""
    listing = set(dir(reelgrep))
    for name in reelgrep.__all__:
        assert name in listing, f"{name} missing from dir(reelgrep)"


def test_top_level_imports_work() -> None:
    """The documented public import pattern resolves without error."""
    from reelgrep import (  # noqa: F401
        BaseBackend,
        BasePersonModel,
        DetectionHit,
        FrameRow,
        IngestResult,
        IngestWarning,
        InvalidQueryError,
        Search,
        SearchError,
        SubtitleHit,
        TranscribeError,
        TranscribeResult,
        get_backend,
        get_person_model,
        ingest_video,
        list_backends,
        list_person_models,
        register_backend,
        register_person_model,
        transcribe_video,
    )


def test_search_identity() -> None:
    """Top-level ``Search`` is the same object as the submodule definition."""
    import reelgrep.search

    assert reelgrep.Search is reelgrep.search.Search
    assert reelgrep.SearchError is reelgrep.search.SearchError
    assert reelgrep.SubtitleHit is reelgrep.search.SubtitleHit


def test_register_backend_alias_identity() -> None:
    """``register_backend`` aliases ``reelgrep.backends.register`` (same callable)."""
    import reelgrep.backends

    assert reelgrep.register_backend is reelgrep.backends.register


def test_register_person_model_alias_identity() -> None:
    """``register_person_model`` aliases ``reelgrep.models.register`` (same callable)."""
    import reelgrep.models

    assert reelgrep.register_person_model is reelgrep.models.register


def test_ingest_and_transcribe_identity() -> None:
    """Indexing and transcription entry points are the same objects as in submodules."""
    import reelgrep.index
    import reelgrep.transcribe

    assert reelgrep.ingest_video is reelgrep.index.ingest_video
    assert reelgrep.IngestResult is reelgrep.index.IngestResult
    assert reelgrep.transcribe_video is reelgrep.transcribe.transcribe_video
    assert reelgrep.TranscribeResult is reelgrep.transcribe.TranscribeResult


def test_version_is_string() -> None:
    """``__version__`` survives the re-export and is a non-empty string."""
    assert isinstance(reelgrep.__version__, str)
    assert reelgrep.__version__


def test_end_to_end_via_top_level_only() -> None:
    """A consumer can use the registry surface with top-level imports only."""
    from reelgrep import (
        BaseBackend,
        Search,
        get_backend,
        list_backends,
        register_backend,
    )

    # Use a unique name to avoid clobbering the bundled "local"/"jellyfin" backends
    # on repeated test runs within the same interpreter session.
    name = "dummy_public_api_test"

    @register_backend(name)
    class _Dummy(BaseBackend):
        def resolve(self, uri):  # noqa: ANN001
            from pathlib import Path

            return Path(uri)

    try:
        assert name in list_backends()
        instance = get_backend(name)
        assert isinstance(instance, BaseBackend)
        assert instance.name == name

        # Search is constructible from the top-level import (just verify the class
        # is callable - the constructor's behavior is covered by search tests).
        assert callable(Search)
    finally:
        # Clean up the registry so the test is idempotent if re-run in the same
        # process (e.g. via pytest --lf without a fresh interpreter).
        import reelgrep.backends as _backends

        _backends._REGISTRY.pop(name, None)


def test_faces_public_surface() -> None:
    """The faces library surface is importable from top-level reelgrep."""
    import reelgrep

    expected = {
        "ClusterReport",
        "ExtractFacesResult",
        "FaceCluster",
        "FaceDetection",
        "Faces",
        "FacesError",
        "InsightFaceMissingError",
        "cluster_faces",
        "extract_faces",
    }
    missing_from_all = expected - set(reelgrep.__all__)
    assert not missing_from_all, f"missing from __all__: {missing_from_all}"
    for name in expected:
        assert hasattr(reelgrep, name), f"missing attribute on top-level package: {name}"


def test_top_level_import_does_not_require_face_extra():
    """Importing reelgrep + accessing the faces surface must NOT require the [face] extra."""
    import reelgrep
    # These attributes exist whether or not insightface/hdbscan is installed.
    assert hasattr(reelgrep, "extract_faces")
    assert hasattr(reelgrep, "cluster_faces")
    assert hasattr(reelgrep, "Faces")
    # numpy must be importable as a base dep.
    import numpy  # noqa: F401
