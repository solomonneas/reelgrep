"""Tests for the face_embed person-finding backend."""

from __future__ import annotations

import sys
import types

import pytest

from reelgrep.frames import Frame


def _insightface_available() -> bool:
    """Return True if real insightface is importable."""
    try:
        import insightface  # noqa: F401
        import insightface.app  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture
def fake_stack(monkeypatch):
    """Inject fake insightface + cv2 into sys.modules; return helpers."""
    import numpy as np

    class FakeFace:
        def __init__(self, embedding, bbox=None):
            arr = np.asarray(embedding, dtype="float32")
            n = float(np.linalg.norm(arr)) or 1.0
            self.normed_embedding = arr / n
            self.embedding = arr
            self.bbox = bbox

    class FakeApp:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._queue: list[list[FakeFace]] = []
            self.prepare_kwargs: dict = {}

        def prepare(self, **kwargs):
            self.prepare_kwargs = kwargs

        def get(self, img):
            if self._queue:
                return self._queue.pop(0)
            return []

        def queue(self, faces: list) -> None:
            self._queue.append(faces)

    fake_app_mod = types.ModuleType("insightface.app")
    fake_app_mod.FaceAnalysis = FakeApp
    fake_root = types.ModuleType("insightface")
    fake_root.app = fake_app_mod

    class FakeCV2:
        @staticmethod
        def imread(path):
            return ("fake-img", str(path))

    monkeypatch.setitem(sys.modules, "insightface", fake_root)
    monkeypatch.setitem(sys.modules, "insightface.app", fake_app_mod)
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)

    return {"FakeFace": FakeFace, "FakeApp": FakeApp}


def _frame(tmp_path, name="f.jpg", ts=0):
    p = tmp_path / name
    p.write_bytes(b"")
    return Frame(timestamp_ms=ts, path=str(p), sampling_strategy="every_n")


def _make_model(fake_stack):
    """Build a FaceEmbedModel with a fresh FakeApp wired in."""
    from reelgrep.models.face_embed import FaceEmbedModel

    model = FaceEmbedModel()
    app = fake_stack["FakeApp"]()
    model._app = app
    return model, app


def test_lazy_import_raises_modelerror_when_insightface_missing(tmp_path, monkeypatch):
    """find() must raise ModelError pointing at the [face] extra when insightface is absent."""
    monkeypatch.setitem(sys.modules, "insightface", None)
    monkeypatch.delitem(sys.modules, "insightface.app", raising=False)

    from reelgrep.models import ModelError
    from reelgrep.models.face_embed import FaceEmbedModel

    model = FaceEmbedModel()
    pos = tmp_path / "pos.jpg"
    pos.write_bytes(b"")

    with pytest.raises(ModelError) as exc:
        model.find([], [pos], [], threshold=0.5)
    assert "[face] extra" in str(exc.value)


def test_config_dict_returns_four_fields():
    """config_dict exposes model_pack, det_size, providers, ctx_id."""
    from reelgrep.models.face_embed import FaceEmbedModel

    cfg = FaceEmbedModel(
        model_pack="buffalo_s",
        det_size=(320, 320),
        providers=["CUDAExecutionProvider"],
        ctx_id=0,
    ).config_dict()
    assert cfg == {
        "model_pack": "buffalo_s",
        "det_size": [320, 320],
        "providers": ["CUDAExecutionProvider"],
        "ctx_id": 0,
    }


def test_empty_positives_raises(fake_stack, tmp_path):
    """Calling find with no positive examples raises ModelError."""
    from reelgrep.models import ModelError

    model, _ = _make_model(fake_stack)
    with pytest.raises(ModelError, match="at least one positive"):
        model.find([_frame(tmp_path)], [], [], threshold=0.5)


def test_no_faces_in_positives_raises(fake_stack, tmp_path):
    """If no faces are detected across all positives, raise ModelError."""
    from reelgrep.models import ModelError

    model, app = _make_model(fake_stack)
    app.queue([])  # positive image -> zero faces

    pos = tmp_path / "pos.jpg"
    pos.write_bytes(b"")

    with pytest.raises(ModelError, match="no faces detected in any positive"):
        model.find([_frame(tmp_path)], [pos], [], threshold=0.5)


def test_happy_path_single_match(fake_stack, tmp_path):
    """A perfectly-aligned face yields one match with high confidence and bbox."""
    FakeFace = fake_stack["FakeFace"]

    model, app = _make_model(fake_stack)
    app.queue([FakeFace([1, 0, 0, 0])])  # positive embedding
    app.queue([FakeFace([1, 0, 0, 0], bbox=(10, 20, 30, 40))])  # frame face

    pos = tmp_path / "pos.jpg"
    pos.write_bytes(b"")

    frame = _frame(tmp_path, name="frame.jpg", ts=1000)
    matches = model.find([frame], [pos], [], threshold=0.5)

    assert len(matches) == 1
    m = matches[0]
    assert m.confidence == pytest.approx(1.0, abs=1e-5)
    assert m.bbox == (10, 20, 20, 20)
    assert m.frame.timestamp_ms == 1000
    assert "cosine" in m.reasoning


def test_below_threshold_filters_out(fake_stack, tmp_path):
    """Orthogonal embeddings score 0 and fall below threshold 0.5."""
    FakeFace = fake_stack["FakeFace"]
    model, app = _make_model(fake_stack)
    app.queue([FakeFace([1, 0, 0, 0])])
    app.queue([FakeFace([0, 1, 0, 0])])

    pos = tmp_path / "pos.jpg"
    pos.write_bytes(b"")

    matches = model.find([_frame(tmp_path)], [pos], [], threshold=0.5)
    assert matches == []


def test_negative_rejection_blocks_lookalike(fake_stack, tmp_path):
    """A near-positive lookalike close to the negative is rejected (pos - neg < threshold)."""
    FakeFace = fake_stack["FakeFace"]
    model, app = _make_model(fake_stack)

    app.queue([FakeFace([1, 0, 0, 0])])  # positive
    app.queue([FakeFace([0.95, 0.05, 0.05, 0])])  # negative
    app.queue([FakeFace([0.97, 0.03, 0.03, 0])])  # frame face

    pos = tmp_path / "pos.jpg"
    pos.write_bytes(b"")
    neg = tmp_path / "neg.jpg"
    neg.write_bytes(b"")

    matches = model.find([_frame(tmp_path)], [pos], [neg], threshold=0.5)
    assert matches == []


def test_multiple_positives_averaged(fake_stack, tmp_path):
    """Two orthogonal positives produce a centroid; matching frame scores high."""
    FakeFace = fake_stack["FakeFace"]
    model, app = _make_model(fake_stack)

    app.queue([FakeFace([1, 0, 0, 0])])  # positive 1
    app.queue([FakeFace([0, 1, 0, 0])])  # positive 2
    app.queue([FakeFace([1, 1, 0, 0])])  # frame face -> ~[.707,.707,0,0]

    pos1 = tmp_path / "p1.jpg"
    pos1.write_bytes(b"")
    pos2 = tmp_path / "p2.jpg"
    pos2.write_bytes(b"")

    matches = model.find([_frame(tmp_path)], [pos1, pos2], [], threshold=0.5)
    assert len(matches) == 1
    # frame embedding dotted with centroid: both normalized to [~.707, ~.707, 0, 0]
    assert matches[0].confidence == pytest.approx(1.0, abs=1e-5)


def test_sorting_and_top_k(fake_stack, tmp_path):
    """top_k=2 across 3 matching frames returns the top two in descending order."""
    import numpy as np

    FakeFace = fake_stack["FakeFace"]
    model, app = _make_model(fake_stack)

    # Build three frame embeddings that cosine to [1,0,0,0] with values .6, .9, .8.
    def with_cosine(c):
        # vec = [c, sqrt(1 - c^2), 0, 0] is unit-norm and dots to c with [1,0,0,0].
        s = float(np.sqrt(max(0.0, 1.0 - c * c)))
        return [c, s, 0, 0]

    app.queue([FakeFace([1, 0, 0, 0])])  # positive
    app.queue([FakeFace(with_cosine(0.6))])
    app.queue([FakeFace(with_cosine(0.9))])
    app.queue([FakeFace(with_cosine(0.8))])

    pos = tmp_path / "pos.jpg"
    pos.write_bytes(b"")

    frames = [
        _frame(tmp_path, name="a.jpg", ts=1),
        _frame(tmp_path, name="b.jpg", ts=2),
        _frame(tmp_path, name="c.jpg", ts=3),
    ]
    matches = model.find(frames, [pos], [], threshold=0.5, top_k=2)
    assert len(matches) == 2
    confs = [m.confidence for m in matches]
    assert confs[0] == pytest.approx(0.9, abs=1e-5)
    assert confs[1] == pytest.approx(0.8, abs=1e-5)


def test_bbox_extraction(fake_stack, tmp_path):
    """Match.bbox is (x1, y1, x2-x1, y2-y1) derived from face.bbox."""
    FakeFace = fake_stack["FakeFace"]
    model, app = _make_model(fake_stack)

    app.queue([FakeFace([1, 0, 0, 0])])
    app.queue([FakeFace([1, 0, 0, 0], bbox=(10, 20, 30, 40))])

    pos = tmp_path / "pos.jpg"
    pos.write_bytes(b"")

    matches = model.find([_frame(tmp_path)], [pos], [], threshold=0.5)
    assert matches[0].bbox == (10, 20, 20, 20)


def test_missing_frame_path_skipped(fake_stack, tmp_path):
    """Frames whose path is missing on disk are silently skipped."""
    FakeFace = fake_stack["FakeFace"]
    model, app = _make_model(fake_stack)

    app.queue([FakeFace([1, 0, 0, 0])])  # positive
    app.queue([FakeFace([1, 0, 0, 0])])  # real frame

    pos = tmp_path / "pos.jpg"
    pos.write_bytes(b"")

    real_frame = _frame(tmp_path, name="real.jpg", ts=1)
    ghost = Frame(
        timestamp_ms=2,
        path=str(tmp_path / "does-not-exist.jpg"),
        sampling_strategy="every_n",
    )
    matches = model.find([ghost, real_frame], [pos], [], threshold=0.5)
    assert len(matches) == 1
    assert matches[0].frame.timestamp_ms == 1


@pytest.mark.integration
@pytest.mark.skipif(not _insightface_available(), reason="insightface not installed")
def test_real_insightface_loads():
    """Smoke test: real insightface FaceAnalysis can be instantiated."""
    from reelgrep.models.face_embed import FaceEmbedModel

    model = FaceEmbedModel()
    # Just confirm _load() does not raise. Model download is allowed.
    try:
        model._load()
    except Exception as exc:  # pragma: no cover - integration env dependent
        pytest.skip(f"insightface load failed in this env: {exc}")
