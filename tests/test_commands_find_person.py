"""Tests for the ``reelgrep find-person`` command."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from reelgrep.commands.find_person import find_person
from reelgrep.config import get_settings, reset_settings
from reelgrep.frames import Frame
from reelgrep.models import Match, ModelError
from reelgrep.probe import VideoMetadata


@pytest.fixture(autouse=True)
def _reelgrep_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    home = tmp_path / "rg-home"
    monkeypatch.setenv("REELGREP_HOME", str(home))
    monkeypatch.delenv("REELGREP_DB", raising=False)
    monkeypatch.delenv("REELGREP_CACHE", raising=False)
    reset_settings()
    yield home
    reset_settings()


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    v = tmp_path / "lecture.mp4"
    v.write_bytes(b"\x00fake-video-bytes")
    return v


@pytest.fixture
def positive_image(tmp_path: Path) -> Path:
    p = tmp_path / "speaker.jpg"
    p.write_bytes(b"\xff\xd8\xff")
    return p


def _make_meta() -> VideoMetadata:
    return VideoMetadata(
        path="filled-in",
        format_name="mp4",
        duration_ms=120000,
        size_bytes=1000,
        width=1920,
        height=1080,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        raw={"format": {"format_name": "mp4"}, "streams": []},
    )


def _open_db() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    return conn


class _FakeModelFactory:
    """Build configurable fake person models for monkeypatching."""

    def __init__(
        self,
        frames: list[Frame],
        *,
        return_matches: bool = True,
        raise_error: bool = False,
    ) -> None:
        self.frames = frames
        self.return_matches = return_matches
        self.raise_error = raise_error
        self.find_calls: list[dict[str, Any]] = []
        self.constructed_with: list[dict[str, Any]] = []

    def __call__(self, name: str, **kwargs: Any) -> Any:
        outer = self
        outer.constructed_with.append(kwargs)

        class FakeModel:
            name = "fake"

            def __init__(self, **kw: Any) -> None:
                self.kw = kw

            def config_dict(self) -> dict[str, Any]:
                return {"model_pack": "buffalo_l"}

            def find(
                self,
                frames_: list[Frame],
                positives: list[Path],
                negatives: list[Path],
                *,
                threshold: float,
                top_k: int | None = None,
            ) -> list[Match]:
                outer.find_calls.append(
                    {
                        "frames": list(frames_),
                        "positives": list(positives),
                        "negatives": list(negatives),
                        "threshold": threshold,
                        "top_k": top_k,
                    }
                )
                if outer.raise_error:
                    raise ModelError("backend down")
                if not outer.return_matches:
                    return []
                results = [
                    Match(
                        frame=outer.frames[1],
                        confidence=0.85,
                        bbox=(10, 20, 30, 40),
                        reasoning="cosine 0.87 to positive, 0.32 to nearest negative",
                    ),
                    Match(
                        frame=outer.frames[3],
                        confidence=0.62,
                        bbox=None,
                        reasoning="weaker but accepted",
                    ),
                ]
                if top_k is not None:
                    results = results[:top_k]
                return results

        return FakeModel(**kwargs)


@pytest.fixture
def patched_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    meta = _make_meta()
    frames: list[Frame] = []
    for i in range(5):
        fp = tmp_path / f"f{i}.jpg"
        fp.write_bytes(b"\xff\xd8\xff")
        frames.append(
            Frame(
                timestamp_ms=i * 5000,
                path=str(fp.resolve()),
                sampling_strategy="every_n",
            )
        )

    sample_calls: list[dict[str, Any]] = []

    def fake_sample_every(*args: Any, **kwargs: Any) -> list[Frame]:
        sample_calls.append({"args": args, "kwargs": kwargs})
        return frames

    monkeypatch.setattr(
        "reelgrep.commands.find_person.probe",
        lambda p: meta.model_copy(update={"path": str(p)}),
    )
    monkeypatch.setattr(
        "reelgrep.commands.find_person.sample_every", fake_sample_every
    )
    factory = _FakeModelFactory(frames)
    monkeypatch.setattr(
        "reelgrep.commands.find_person.get_person_model", factory
    )
    return {
        "meta": meta,
        "frames": frames,
        "factory": factory,
        "sample_calls": sample_calls,
    }


def test_top_k_zero_rejected(
    fake_video: Path, positive_image: Path, patched_pipeline: dict[str, Any],
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--top-k", "0",
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "--top-k" in result.output or "top-k" in result.output


def test_every_zero_rejected(
    fake_video: Path, positive_image: Path, patched_pipeline: dict[str, Any],
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--every", "0",
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "--every" in result.output or "every" in result.output


def test_missing_positive_rejected_by_click(
    fake_video: Path, patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "positive" in result.output.lower()


def test_directory_positives_are_expanded(
    fake_video: Path, patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in range(3):
        (refs / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(refs),
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    factory = patched_pipeline["factory"]
    assert factory.find_calls, "model.find was not called"
    assert len(factory.find_calls[0]["positives"]) == 3


def test_directory_with_no_images_rejected(
    fake_video: Path, patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "readme.txt").write_text("nope")
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(refs),
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "no positive images" in result.output.lower()


def test_happy_path(
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "matches:" in result.output
    assert "2" in result.output
    assert "00:00:05.000" in result.output
    assert "00:00:15.000" in result.output

    jpgs = sorted(out.glob("*.jpg"))
    assert len(jpgs) == 2
    names = {p.name for p in jpgs}
    assert "speaker_00:00:05.000.jpg" in names
    assert "speaker_00:00:15.000.jpg" in names

    manifest_path = out / "find-person.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["operation"] == "find-person"
    assert manifest["parameters"]["backend"] == "face_embed"
    assert manifest["parameters"]["threshold"] == 0.30
    assert len(manifest["results"]) == 2


def test_db_rows_populated_after_happy_path(
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output

    conn = _open_db()
    try:
        (search_count,) = conn.execute(
            "SELECT COUNT(*) FROM person_searches"
        ).fetchone()
        (match_count,) = conn.execute(
            "SELECT COUNT(*) FROM person_matches"
        ).fetchone()
        search_row = conn.execute(
            "SELECT id, video_id FROM person_searches"
        ).fetchone()
        match_rows = conn.execute(
            "SELECT search_id FROM person_matches"
        ).fetchall()
    finally:
        conn.close()

    assert search_count == 1
    assert match_count == 2
    assert all(r["search_id"] == search_row["id"] for r in match_rows)


def test_no_export_frames_skips_jpgs(
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--out", str(out),
            "--no-export-frames",
        ],
    )
    assert result.exit_code == 0, result.output
    jpgs = list(out.glob("*.jpg"))
    assert jpgs == []
    manifest_path = out / "find-person.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["results"]) == 2


def test_high_threshold_does_not_re_filter(
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    # The command itself does NOT re-filter; the backend is responsible.
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--threshold", "0.95",
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    # Fake model returns 2 matches regardless of threshold; the command
    # passes them through.
    assert "matches:   2" in result.output


def test_top_k_one_forwarded_to_model(
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--top-k", "1",
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    factory = patched_pipeline["factory"]
    assert factory.find_calls[0]["top_k"] == 1


def test_auto_ingest_creates_video_row(
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output

    conn = _open_db()
    try:
        (vcount,) = conn.execute("SELECT COUNT(*) FROM videos").fetchone()
        video_row = conn.execute("SELECT id FROM videos").fetchone()
        search_row = conn.execute("SELECT video_id FROM person_searches").fetchone()
    finally:
        conn.close()

    assert vcount == 1
    assert search_row["video_id"] == video_row["id"]


def test_reuse_skips_sample_every_when_frames_exist(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    # Pre-seed videos row + 5 frames at known timestamps for the hash.
    from reelgrep.db import connect, migrate
    from reelgrep.hashing import file_hash

    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    digest = file_hash(fake_video.resolve())
    pre_frames: list[Path] = []
    for i in range(5):
        fp = tmp_path / f"pre_f{i}.jpg"
        fp.write_bytes(b"\xff\xd8\xff")
        pre_frames.append(fp)

    conn = connect(settings.db_path)
    try:
        migrate(conn)
        with conn:
            cur = conn.execute(
                """
                INSERT INTO videos (
                    file_hash, path, duration_ms, width, height, fps,
                    container, video_codec, audio_codec, size_bytes,
                    ingested_at, probe_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    digest,
                    str(fake_video.resolve()),
                    120000, 1920, 1080, 30.0,
                    "mp4", "h264", "aac", 1000,
                    "2026-01-01T00:00:00+00:00",
                    json.dumps({"format": {}, "streams": []}),
                ),
            )
            video_id = cur.lastrowid
            pre_ids: list[int] = []
            for i, fp in enumerate(pre_frames):
                frame_cur = conn.execute(
                    """
                    INSERT INTO frames (
                        video_id, timestamp_ms, path, sampling_strategy, width, height
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (video_id, i * 5000, str(fp), "every_n", None, None),
                )
                pre_ids.append(int(frame_cur.lastrowid))
    finally:
        conn.close()

    sample_calls = patched_pipeline["sample_calls"]
    sample_calls.clear()

    # Have the FakeModel return matches that reference the pre-existing frames
    # by path (we need to re-wire it now that pre_frames exist).
    pre_frame_objs = [
        Frame(timestamp_ms=i * 5000, path=str(fp.resolve()), sampling_strategy="every_n")
        for i, fp in enumerate(pre_frames)
    ]

    class ReuseFactory:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def __call__(self, name: str, **kwargs: Any) -> Any:
            outer = self

            class M:
                name = "fake"

                def __init__(self, **kw: Any) -> None:
                    pass

                def config_dict(self) -> dict[str, Any]:
                    return {}

                def find(
                    self,
                    frames_: list[Frame],
                    positives: list[Path],
                    negatives: list[Path],
                    *,
                    threshold: float,
                    top_k: int | None = None,
                ) -> list[Match]:
                    outer.calls.append({"frames": frames_, "top_k": top_k})
                    return [
                        Match(
                            frame=frames_[1],
                            confidence=0.9,
                            bbox=None,
                            reasoning="reused frame match",
                        ),
                    ]

            return M(**kwargs)

    reuse_factory = ReuseFactory()
    monkeypatch.setattr(
        "reelgrep.commands.find_person.get_person_model", reuse_factory
    )

    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    # sample_every should NOT have been called.
    assert sample_calls == []
    # person_matches should reference the pre-existing frame_ids.
    conn = _open_db()
    try:
        match_rows = conn.execute(
            "SELECT frame_id FROM person_matches"
        ).fetchall()
    finally:
        conn.close()
    assert len(match_rows) == 1
    assert match_rows[0]["frame_id"] in pre_ids
    # The matched frame_objs were passed to the model.
    assert len(reuse_factory.calls[0]["frames"]) == 5
    assert reuse_factory.calls[0]["frames"][0].path == pre_frame_objs[0].path


def test_zero_matches_writes_empty_manifest(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    factory = _FakeModelFactory(patched_pipeline["frames"], return_matches=False)
    monkeypatch.setattr(
        "reelgrep.commands.find_person.get_person_model", factory
    )
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no matches above threshold" in result.output
    manifest_path = out / "find-person.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["results"] == []
    # out_dir created but empty (no jpgs).
    assert out.exists()
    assert list(out.glob("*.jpg")) == []
    # No export_artifacts rows.
    conn = _open_db()
    try:
        (artifact_count,) = conn.execute(
            "SELECT COUNT(*) FROM export_artifacts"
        ).fetchone()
    finally:
        conn.close()
    assert artifact_count == 0


def test_model_error_exits_with_code_two(
    monkeypatch: pytest.MonkeyPatch,
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    factory = _FakeModelFactory(patched_pipeline["frames"], raise_error=True)
    monkeypatch.setattr(
        "reelgrep.commands.find_person.get_person_model", factory
    )
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2
    assert "error:" in result.stderr
    assert "backend down" in result.stderr


def test_default_threshold_face_embed(
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--backend", "face_embed",
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    factory = patched_pipeline["factory"]
    assert factory.find_calls[0]["threshold"] == 0.30


def test_default_threshold_ollama_vision(
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--backend", "ollama_vision",
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    factory = patched_pipeline["factory"]
    assert factory.find_calls[0]["threshold"] == 0.65


def test_bbox_json_storage(
    fake_video: Path, positive_image: Path,
    patched_pipeline: dict[str, Any], tmp_path: Path,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        find_person,
        [
            str(fake_video),
            "--label", "speaker",
            "--positive", str(positive_image),
            "--out", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    conn = _open_db()
    try:
        row = conn.execute(
            "SELECT bbox_json FROM person_matches WHERE confidence > 0.8"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["bbox_json"] == "[10, 20, 30, 40]"
