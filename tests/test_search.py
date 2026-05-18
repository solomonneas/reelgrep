"""Tests for the public ``reelgrep.search`` library API."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import is_dataclass
from pathlib import Path

import pytest

from reelgrep import config, db
from reelgrep.search import DetectionHit, FrameRow, Search, SubtitleHit


@pytest.fixture(autouse=True)
def _reelgrep_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "rg-home"
    monkeypatch.setenv("REELGREP_HOME", str(home))
    monkeypatch.delenv("REELGREP_DB", raising=False)
    monkeypatch.delenv("REELGREP_CACHE", raising=False)
    config.reset_settings()
    yield home
    config.reset_settings()


def _seed(db_path: Path) -> dict[str, int]:
    """Seed two videos with subtitles, frames, and person searches.

    Returns a dict of useful row ids for test assertions.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(db_path)
    db.migrate(conn)

    conn.execute(
        "INSERT INTO videos (file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("blake2b:aaa", "/v/one.mp4", 600000, "2026-05-17T10:00:00", "{}"),
    )
    v1 = conn.execute(
        "SELECT id FROM videos WHERE file_hash='blake2b:aaa'"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO videos (file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("blake2b:bbb", "/v/two.mp4", 900000, "2026-05-17T11:00:00", "{}"),
    )
    v2 = conn.execute(
        "SELECT id FROM videos WHERE file_hash='blake2b:bbb'"
    ).fetchone()[0]

    cues = [
        (v1, "en", "sidecar", None, 1000, 4000, "Welcome to kubernetes networking"),
        (v1, "en", "sidecar", None, 5000, 8000, "Pods talk to each other over the pod network"),
        (v2, "en", "whisper", None, 2000, 5000, "rivers run downhill"),
    ]
    for c in cues:
        cur = conn.execute(
            "INSERT INTO subtitles "
            "(video_id, language, source, stream_index, start_ms, end_ms, text) "
            "VALUES (?,?,?,?,?,?,?)",
            c,
        )
        conn.execute(
            "INSERT INTO subtitles_fts(rowid, text) VALUES (?, ?)",
            (cur.lastrowid, c[6]),
        )

    frames = [
        (v1, 0, "/cache/f0.jpg", "every_n"),
        (v1, 5000, "/cache/f5.jpg", "every_n"),
        (v1, 10000, "/cache/f10.jpg", "every_n"),
        (v1, 20000, "/cache/f20.jpg", "every_n"),
        (v2, 0, "/cache/v2-f0.jpg", "every_n"),
    ]
    frame_ids: list[int] = []
    for f in frames:
        cur = conn.execute(
            "INSERT INTO frames "
            "(video_id, timestamp_ms, path, sampling_strategy) "
            "VALUES (?,?,?,?)",
            f,
        )
        frame_ids.append(int(cur.lastrowid))

    searches = [
        (v1, "alice", "face_embed", 0.30, "2026-05-17T10:30:00"),
        (v1, "bob", "face_embed", 0.30, "2026-05-17T10:40:00"),
        (v1, "alice", "ollama_vision", 0.65, "2026-05-17T10:50:00"),
        (v2, "alice", "face_embed", 0.30, "2026-05-17T11:30:00"),
    ]
    search_ids: list[int] = []
    for s in searches:
        cur = conn.execute(
            "INSERT INTO person_searches "
            "(video_id, label, backend, positive_examples_json, negative_examples_json, "
            "config_json, threshold, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (s[0], s[1], s[2], "[]", "[]", "{}", s[3], s[4]),
        )
        search_ids.append(int(cur.lastrowid))

    # Add a couple of person_matches against the first stored search.
    conn.execute(
        "INSERT INTO person_matches (search_id, frame_id, confidence) "
        "VALUES (?, ?, ?)",
        (search_ids[0], frame_ids[0], 0.91),
    )
    conn.execute(
        "INSERT INTO person_matches (search_id, frame_id, confidence) "
        "VALUES (?, ?, ?)",
        (search_ids[0], frame_ids[1], 0.85),
    )
    conn.commit()
    conn.close()

    return {
        "video1": int(v1),
        "video2": int(v2),
        "search_alice_v1_face": search_ids[0],
    }


# Result types --------------------------------------------------------------


def test_result_types_are_frozen_dataclasses() -> None:
    """SubtitleHit / DetectionHit / FrameRow are dataclasses with the documented fields."""
    assert is_dataclass(SubtitleHit)
    assert is_dataclass(DetectionHit)
    assert is_dataclass(FrameRow)

    sub_fields = {f for f in SubtitleHit.__dataclass_fields__}
    assert sub_fields == {
        "id",
        "video_id",
        "start_ms",
        "end_ms",
        "text",
        "language",
        "source",
    }
    det_fields = {f for f in DetectionHit.__dataclass_fields__}
    assert det_fields == {
        "id",
        "video_id",
        "video_hash",
        "label",
        "model",
        "threshold",
        "created_at",
        "match_count",
    }
    frame_fields = {f for f in FrameRow.__dataclass_fields__}
    assert frame_fields == {
        "id",
        "video_id",
        "timestamp_ms",
        "path",
        "sampling_strategy",
    }

    # Frozen: assignment raises.
    hit = SubtitleHit(
        id=1,
        video_id=1,
        start_ms=0,
        end_ms=1,
        text="x",
        language=None,
        source="sidecar",
    )
    with pytest.raises(Exception):  # noqa: B017 - dataclass.FrozenInstanceError
        hit.text = "y"  # type: ignore[misc]


# subtitles() ---------------------------------------------------------------


def test_subtitles_matches_known_row(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    ids = _seed(db_path)

    s = Search(db_path=db_path)
    hits = s.subtitles("kubernetes")

    assert len(hits) == 1
    assert isinstance(hits[0], SubtitleHit)
    assert hits[0].text == "Welcome to kubernetes networking"
    assert hits[0].video_id == ids["video1"]
    assert hits[0].source == "sidecar"


def test_subtitles_filters_by_video_id(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    ids = _seed(db_path)

    s = Search(db_path=db_path)
    # 'pod' stems via porter to match the v1 cue but not the v2 cue.
    matched_all = s.subtitles("pod")
    assert len(matched_all) == 1
    assert matched_all[0].video_id == ids["video1"]

    # Filter to the second video, which has no 'pod' cue.
    matched_v2 = s.subtitles("pod", video_id=ids["video2"])
    assert matched_v2 == []

    # Filter to the first video, which does.
    matched_v1 = s.subtitles("pod", video_id=ids["video1"])
    assert len(matched_v1) == 1
    assert matched_v1[0].video_id == ids["video1"]


def test_subtitles_returns_empty_on_no_match(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    _seed(db_path)

    s = Search(db_path=db_path)
    assert s.subtitles("xyzzy") == []


def test_subtitles_respects_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    _seed(db_path)

    s = Search(db_path=db_path)
    # 'pod' matches a single cue but limit=1 confirms the LIMIT path is honoured.
    assert len(s.subtitles("pod", limit=1)) == 1
    assert s.subtitles("pod", limit=0) == []


# detections() --------------------------------------------------------------


def test_detections_filters_by_model(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    _seed(db_path)

    s = Search(db_path=db_path)
    face = s.detections(model="face_embed")
    assert len(face) == 3
    assert all(isinstance(d, DetectionHit) for d in face)
    assert all(d.model == "face_embed" for d in face)

    vision = s.detections(model="ollama_vision")
    assert len(vision) == 1
    assert vision[0].model == "ollama_vision"


def test_detections_filters_by_label(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    _seed(db_path)

    s = Search(db_path=db_path)
    alices = s.detections(label="alice")
    assert len(alices) == 3
    assert all(d.label == "alice" for d in alices)

    bobs = s.detections(label="bob")
    assert len(bobs) == 1
    assert bobs[0].label == "bob"


def test_detections_filters_by_video_id(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    ids = _seed(db_path)

    s = Search(db_path=db_path)
    on_v1 = s.detections(video_id=ids["video1"])
    assert len(on_v1) == 3
    assert all(d.video_id == ids["video1"] for d in on_v1)
    assert all(d.video_hash == "blake2b:aaa" for d in on_v1)

    on_v2 = s.detections(video_id=ids["video2"])
    assert len(on_v2) == 1
    assert on_v2[0].video_hash == "blake2b:bbb"


def test_detections_match_count_reflects_person_matches(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    ids = _seed(db_path)

    s = Search(db_path=db_path)
    target = next(
        d for d in s.detections() if d.id == ids["search_alice_v1_face"]
    )
    assert target.match_count == 2

    # Other detections in the fixture have no matches recorded.
    others = [d for d in s.detections() if d.id != target.id]
    assert all(d.match_count == 0 for d in others)


# frames_at() ---------------------------------------------------------------


def test_frames_at_returns_all_frames_for_video(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    ids = _seed(db_path)

    s = Search(db_path=db_path)
    frames = s.frames_at(video_id=ids["video1"])
    assert len(frames) == 4
    assert all(isinstance(f, FrameRow) for f in frames)
    assert [f.timestamp_ms for f in frames] == [0, 5000, 10000, 20000]
    assert all(f.video_id == ids["video1"] for f in frames)


def test_frames_at_with_window(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    ids = _seed(db_path)

    s = Search(db_path=db_path)
    # Inclusive bounds: [5000, 10000] returns the two frames at 5000 and 10000.
    in_window = s.frames_at(
        video_id=ids["video1"], ts_start_ms=5000, ts_end_ms=10000
    )
    assert [f.timestamp_ms for f in in_window] == [5000, 10000]

    # Start-only.
    from_5s = s.frames_at(video_id=ids["video1"], ts_start_ms=5000)
    assert [f.timestamp_ms for f in from_5s] == [5000, 10000, 20000]

    # End-only.
    upto_5s = s.frames_at(video_id=ids["video1"], ts_end_ms=5000)
    assert [f.timestamp_ms for f in upto_5s] == [0, 5000]


def test_frames_at_respects_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "idx.sqlite"
    ids = _seed(db_path)

    s = Search(db_path=db_path)
    assert len(s.frames_at(video_id=ids["video1"], limit=2)) == 2
    assert s.frames_at(video_id=ids["video1"], limit=0) == []


# Constructor / db_path behaviour ------------------------------------------


def test_constructor_default_uses_settings_db_path(tmp_path: Path) -> None:
    config.reset_settings()
    settings = config.get_settings()
    s = Search()
    assert s.db_path == settings.db_path


def test_explicit_db_path_does_not_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Constructing Search with db_path does not affect subsequent ``Search()``.

    Search must never mutate the global db_path override.
    """
    config.reset_settings()
    settings_db_path = config.get_settings().db_path

    explicit_db = tmp_path / "explicit.sqlite"
    _seed(explicit_db)

    s_explicit = Search(db_path=explicit_db)
    assert s_explicit.db_path == explicit_db.resolve()
    # Functional check: queries route to the explicit DB.
    assert len(s_explicit.detections()) == 4

    # A bare Search() now still resolves to the settings-default path.
    s_default = Search()
    assert s_default.db_path == settings_db_path
    # And the process-level override remains untouched.
    assert config.get_db_override() is None


# Issue 2: InvalidQueryError --------------------------------------------------


def test_subtitles_malformed_query_raises_invalid_query_error(tmp_path: Path) -> None:
    """An unterminated FTS string surfaces as InvalidQueryError, not sqlite3.OperationalError."""
    import sqlite3

    from reelgrep.search import InvalidQueryError, SearchError

    db_path = tmp_path / "idx.sqlite"
    _seed(db_path)

    s = Search(db_path=db_path)
    with pytest.raises(InvalidQueryError) as exc_info:
        s.subtitles('"unterminated')

    # InvalidQueryError is a SearchError.
    assert isinstance(exc_info.value, SearchError)
    # The original sqlite3.OperationalError is preserved as the cause.
    assert isinstance(exc_info.value.__cause__, sqlite3.OperationalError)


# Issue 3: explicit db_path must exist ---------------------------------------


def test_explicit_db_path_must_exist(tmp_path: Path) -> None:
    """Constructing Search with a typo'd db_path raises FileNotFoundError."""
    missing = tmp_path / "does-not-exist.sqlite"
    assert not missing.exists()
    with pytest.raises(FileNotFoundError):
        Search(db_path=missing)


def test_default_db_path_auto_creates(tmp_path: Path) -> None:
    """``Search()`` with no db_path arg still works on a fresh settings-resolved path."""
    # The autouse fixture sets REELGREP_HOME to a fresh tmp_path, so the
    # settings-default db file should not yet exist.
    settings = config.get_settings()
    assert not settings.db_path.exists()

    # Construction must succeed without raising.
    s = Search()
    # Issuing a query against a non-existent default path must transparently
    # bootstrap the schema (CLI bootstrap behaviour).
    assert s.detections() == []
    assert settings.db_path.exists()


# Issue 5: limit=None means unbounded ----------------------------------------


def test_frames_at_limit_none_returns_all_rows(tmp_path: Path) -> None:
    """``limit=None`` returns every frame, even past the default ``limit=200``."""
    db_path = tmp_path / "idx.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(db_path)
    db.migrate(conn)
    conn.execute(
        "INSERT INTO videos (file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("blake2b:ccc", "/v/big.mp4", 600000, "2026-05-17T12:00:00", "{}"),
    )
    video_id = int(
        conn.execute(
            "SELECT id FROM videos WHERE file_hash='blake2b:ccc'"
        ).fetchone()[0]
    )
    # Seed 250 frames -- comfortably above the default ``limit=200``.
    total = 250
    for i in range(total):
        conn.execute(
            "INSERT INTO frames (video_id, timestamp_ms, path, sampling_strategy) "
            "VALUES (?, ?, ?, ?)",
            (video_id, i * 1000, f"/cache/f{i}.jpg", "every_n"),
        )
    conn.commit()
    conn.close()

    s = Search(db_path=db_path)
    # Default limit caps at 200.
    assert len(s.frames_at(video_id=video_id)) == 200
    # ``limit=None`` returns all 250.
    all_frames = s.frames_at(video_id=video_id, limit=None)
    assert len(all_frames) == total
    assert [f.timestamp_ms for f in all_frames] == [i * 1000 for i in range(total)]


def test_detections_limit_none_returns_all_rows(tmp_path: Path) -> None:
    """``limit=None`` returns every recorded detection search."""
    db_path = tmp_path / "idx.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = db.connect(db_path)
    db.migrate(conn)
    conn.execute(
        "INSERT INTO videos (file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("blake2b:ddd", "/v/many.mp4", 600000, "2026-05-17T13:00:00", "{}"),
    )
    video_id = int(
        conn.execute(
            "SELECT id FROM videos WHERE file_hash='blake2b:ddd'"
        ).fetchone()[0]
    )
    # Seed 75 person_searches -- comfortably above the default ``limit=50``.
    total = 75
    for i in range(total):
        conn.execute(
            "INSERT INTO person_searches "
            "(video_id, label, backend, positive_examples_json, "
            "negative_examples_json, config_json, threshold, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                video_id,
                f"name{i}",
                "face_embed",
                "[]",
                "[]",
                "{}",
                0.3,
                f"2026-05-17T14:{i:02d}:00",
            ),
        )
    conn.commit()
    conn.close()

    s = Search(db_path=db_path)
    # Default limit caps at 50.
    assert len(s.detections()) == 50
    # ``limit=None`` returns all 75.
    assert len(s.detections(limit=None)) == total
