"""Tests for reelgrep.commands.contact_sheet."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner
from PIL import Image

from reelgrep import config, manifest
from reelgrep.commands import contact_sheet as contact_sheet_module
from reelgrep.commands.contact_sheet import contact_sheet
from reelgrep.db import connect, migrate
from reelgrep.frames import Frame
from reelgrep.hashing import file_hash
from reelgrep.probe import VideoMetadata


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("REELGREP_HOME", str(tmp_path / "home"))
    config.reset_settings()
    yield
    config.reset_settings()


def _make_png(path: Path, color: tuple[int, int, int] = (200, 100, 50)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1, 1), color=color)
    img.save(path, format="PNG")
    return path


def _make_video_stub(tmp_path: Path, name: str = "video.mp4") -> Path:
    """Create a tiny file standing in for a video so click's exists=True passes."""
    p = tmp_path / name
    p.write_bytes(b"fake-video-bytes")
    return p


def _make_metadata(duration_ms: int = 60000) -> VideoMetadata:
    return VideoMetadata(
        path="/tmp/unused.mp4",
        format_name="mov,mp4",
        duration_ms=duration_ms,
        size_bytes=1024,
        width=1920,
        height=1080,
        fps=30.0,
        video_codec="h264",
        audio_codec="aac",
        raw={},
    )


def _make_frames(tmp_path: Path, count: int, step_ms: int = 5000) -> list[Frame]:
    frames: list[Frame] = []
    cache_root = tmp_path / "stub_frames"
    cache_root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        p = cache_root / f"frame_{i:03d}.png"
        _make_png(p, color=(i * 8 % 255, 50, 200))
        frames.append(
            Frame(
                timestamp_ms=i * step_ms,
                path=str(p),
                sampling_strategy="every_n",
            )
        )
    return frames


def _stub_build_sheet(recorded: dict[str, object]):
    """Return a fake build_sheet that records call kwargs and touches out_path."""

    def fake(
        *,
        frame_paths,
        out_path,
        cols,
        rows,
        thumb_width,
        timestamps_ms,
        **extra,
    ):
        recorded["frame_paths"] = list(frame_paths)
        recorded["out_path"] = Path(out_path)
        recorded["cols"] = cols
        recorded["rows"] = rows
        recorded["thumb_width"] = thumb_width
        recorded["timestamps_ms"] = list(timestamps_ms) if timestamps_ms is not None else None
        recorded["extra"] = extra
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color=(0, 0, 0)).save(out_path)
        return out_path

    return fake


def _insert_video(db_path: Path, digest: str, duration_ms: int = 60000) -> int:
    conn = connect(db_path)
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO videos (file_hash, path, ingested_at, probe_json, duration_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (digest, "/tmp/unused.mp4", "2026-01-01T00:00:00Z", "{}", duration_ms),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM videos WHERE file_hash = ?", (digest,)
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _insert_frames(db_path: Path, video_id: int, frame_paths: list[Path]) -> None:
    conn = connect(db_path)
    try:
        migrate(conn)
        for i, p in enumerate(frame_paths):
            conn.execute(
                "INSERT INTO frames "
                "(video_id, timestamp_ms, path, sampling_strategy) "
                "VALUES (?, ?, ?, ?)",
                (video_id, i * 5000, str(p), "every_n"),
            )
        conn.commit()
    finally:
        conn.close()


# -------------------- validation --------------------


def test_cols_zero_raises(tmp_path: Path) -> None:
    video = _make_video_stub(tmp_path)
    runner = CliRunner()
    out = tmp_path / "sheet.jpg"
    result = runner.invoke(
        contact_sheet,
        [str(video), "--out", str(out), "--cols", "0"],
    )
    assert result.exit_code != 0
    assert "--cols" in result.output or "cols" in result.output.lower()


def test_thumb_width_too_small_raises(tmp_path: Path) -> None:
    video = _make_video_stub(tmp_path)
    runner = CliRunner()
    out = tmp_path / "sheet.jpg"
    result = runner.invoke(
        contact_sheet,
        [str(video), "--out", str(out), "--thumb-width", "16"],
    )
    assert result.exit_code != 0
    assert "thumb-width" in result.output.lower() or "thumb_width" in result.output.lower()


def test_rows_zero_raises(tmp_path: Path) -> None:
    video = _make_video_stub(tmp_path)
    runner = CliRunner()
    out = tmp_path / "sheet.jpg"
    result = runner.invoke(
        contact_sheet,
        [str(video), "--out", str(out), "--rows", "0"],
    )
    assert result.exit_code != 0
    assert "--rows" in result.output or "rows" in result.output.lower()


# -------------------- fresh sampling --------------------


def test_fresh_sampling_path_invokes_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_stub(tmp_path)
    sampled = _make_frames(tmp_path, 12)
    monkeypatch.setattr(contact_sheet_module, "probe", lambda *_a, **_k: _make_metadata(60000))
    monkeypatch.setattr(contact_sheet_module, "sample_every", lambda *_a, **_k: sampled)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(contact_sheet_module, "build_sheet", _stub_build_sheet(recorded))

    out = tmp_path / "sheet.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [str(video), "--out", str(out), "--cols", "4", "--no-db"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert recorded["cols"] == 4
    assert recorded["rows"] == 3
    assert len(recorded["frame_paths"]) == 12  # type: ignore[arg-type]


def test_rows_and_cols_truncate_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_stub(tmp_path)
    sampled = _make_frames(tmp_path, 12)
    monkeypatch.setattr(contact_sheet_module, "probe", lambda *_a, **_k: _make_metadata(60000))
    monkeypatch.setattr(contact_sheet_module, "sample_every", lambda *_a, **_k: sampled)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(contact_sheet_module, "build_sheet", _stub_build_sheet(recorded))

    out = tmp_path / "sheet.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [
            str(video),
            "--out",
            str(out),
            "--rows",
            "2",
            "--cols",
            "4",
            "--no-db",
        ],
    )
    assert result.exit_code == 0, result.output
    assert recorded["cols"] == 4
    assert recorded["rows"] == 2
    assert len(recorded["frame_paths"]) == 8  # type: ignore[arg-type]


# -------------------- cached path --------------------


def test_use_cached_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_stub(tmp_path)
    digest = file_hash(video)

    # Prepare cached frames on disk so build_sheet stub doesn't actually need them.
    frame_paths = []
    for i in range(6):
        p = tmp_path / "cached" / f"cf_{i}.png"
        _make_png(p, color=(50, i * 30, 200))
        frame_paths.append(p)

    home = tmp_path / "home"
    db_path = home / "index.sqlite"
    video_id = _insert_video(db_path, digest)
    _insert_frames(db_path, video_id, frame_paths)

    sample_calls = {"n": 0}

    def _explode_sample(*_a, **_k):
        sample_calls["n"] += 1
        raise AssertionError("sample_every should not be called when cached frames exist")

    monkeypatch.setattr(contact_sheet_module, "sample_every", _explode_sample)
    monkeypatch.setattr(
        contact_sheet_module,
        "probe",
        lambda *_a, **_k: pytest.fail("probe should not run when cached frames present"),
    )
    recorded: dict[str, object] = {}
    monkeypatch.setattr(contact_sheet_module, "build_sheet", _stub_build_sheet(recorded))

    out = tmp_path / "x.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [
            str(video),
            "--use-cached",
            "--out",
            str(out),
            "--cols",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert sample_calls["n"] == 0
    assert recorded["cols"] == 3
    assert recorded["rows"] == 2
    assert len(recorded["frame_paths"]) == 6  # type: ignore[arg-type]


def test_use_cached_no_rows_falls_back_to_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_stub(tmp_path)
    digest = file_hash(video)
    # Video present, but no frame rows.
    home = tmp_path / "home"
    db_path = home / "index.sqlite"
    _insert_video(db_path, digest)

    sampled = _make_frames(tmp_path, 6)
    sample_calls = {"n": 0}

    def _record_sample(*_a, **_k):
        sample_calls["n"] += 1
        return sampled

    monkeypatch.setattr(contact_sheet_module, "probe", lambda *_a, **_k: _make_metadata(60000))
    monkeypatch.setattr(contact_sheet_module, "sample_every", _record_sample)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(contact_sheet_module, "build_sheet", _stub_build_sheet(recorded))

    out = tmp_path / "y.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [str(video), "--use-cached", "--out", str(out), "--cols", "3"],
    )
    assert result.exit_code == 0, result.output
    assert sample_calls["n"] == 1
    assert len(recorded["frame_paths"]) == 6  # type: ignore[arg-type]


def test_use_cached_video_not_indexed_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_stub(tmp_path)
    # DB exists but no row for this video.
    home = tmp_path / "home"
    db_path = home / "index.sqlite"
    conn = connect(db_path)
    try:
        migrate(conn)
    finally:
        conn.close()

    sampled = _make_frames(tmp_path, 4)
    sample_calls = {"n": 0}

    def _record_sample(*_a, **_k):
        sample_calls["n"] += 1
        return sampled

    monkeypatch.setattr(contact_sheet_module, "probe", lambda *_a, **_k: _make_metadata(60000))
    monkeypatch.setattr(contact_sheet_module, "sample_every", _record_sample)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(contact_sheet_module, "build_sheet", _stub_build_sheet(recorded))

    out = tmp_path / "z.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [str(video), "--use-cached", "--out", str(out), "--cols", "2", "--no-db"],
    )
    assert result.exit_code == 0, result.output
    assert sample_calls["n"] == 1
    assert len(recorded["frame_paths"]) == 4  # type: ignore[arg-type]


# -------------------- manifest --------------------


def test_manifest_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = _make_video_stub(tmp_path)
    sampled = _make_frames(tmp_path, 6)
    monkeypatch.setattr(contact_sheet_module, "probe", lambda *_a, **_k: _make_metadata(60000))
    monkeypatch.setattr(contact_sheet_module, "sample_every", lambda *_a, **_k: sampled)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(contact_sheet_module, "build_sheet", _stub_build_sheet(recorded))

    out = tmp_path / "sheet.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [
            str(video),
            "--out",
            str(out),
            "--cols",
            "3",
            "--rows",
            "2",
            "--no-db",
        ],
    )
    assert result.exit_code == 0, result.output
    m = manifest.read(manifest.sidecar_path(out))
    assert m.operation == "contact-sheet"
    assert m.parameters["cols"] == 3
    assert m.parameters["rows"] == 2
    assert m.parameters["use_cached"] is False
    assert m.parameters["source_frame_count"] == 6
    assert m.results[0]["tile_count"] == 6
    assert m.results[0]["output_path"] == str(out.resolve())


# -------------------- db side effects --------------------


def test_no_db_skips_export_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_stub(tmp_path)
    digest = file_hash(video)
    home = tmp_path / "home"
    db_path = home / "index.sqlite"
    _insert_video(db_path, digest)

    sampled = _make_frames(tmp_path, 6)
    monkeypatch.setattr(contact_sheet_module, "probe", lambda *_a, **_k: _make_metadata(60000))
    monkeypatch.setattr(contact_sheet_module, "sample_every", lambda *_a, **_k: sampled)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(contact_sheet_module, "build_sheet", _stub_build_sheet(recorded))

    out = tmp_path / "sheet.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [str(video), "--out", str(out), "--cols", "3", "--no-db"],
    )
    assert result.exit_code == 0, result.output
    conn = connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM export_artifacts").fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_export_artifacts_inserted_when_video_indexed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_stub(tmp_path)
    digest = file_hash(video)
    home = tmp_path / "home"
    db_path = home / "index.sqlite"
    video_id = _insert_video(db_path, digest)

    sampled = _make_frames(tmp_path, 6)
    monkeypatch.setattr(contact_sheet_module, "probe", lambda *_a, **_k: _make_metadata(60000))
    monkeypatch.setattr(contact_sheet_module, "sample_every", lambda *_a, **_k: sampled)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(contact_sheet_module, "build_sheet", _stub_build_sheet(recorded))

    out = tmp_path / "sheet.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [str(video), "--out", str(out), "--cols", "3"],
    )
    assert result.exit_code == 0, result.output
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT video_id, kind, path, manifest_path FROM export_artifacts"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    row = rows[0]
    assert row["video_id"] == video_id
    assert row["kind"] == "contact_sheet"
    assert row["path"] == str(out.resolve())
    assert row["manifest_path"] == str(manifest.sidecar_path(out))


def test_video_not_indexed_warns_and_skips_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video_stub(tmp_path)
    home = tmp_path / "home"
    db_path = home / "index.sqlite"
    # init db without inserting video
    conn = connect(db_path)
    try:
        migrate(conn)
    finally:
        conn.close()

    sampled = _make_frames(tmp_path, 6)
    monkeypatch.setattr(contact_sheet_module, "probe", lambda *_a, **_k: _make_metadata(60000))
    monkeypatch.setattr(contact_sheet_module, "sample_every", lambda *_a, **_k: sampled)
    recorded: dict[str, object] = {}
    monkeypatch.setattr(contact_sheet_module, "build_sheet", _stub_build_sheet(recorded))

    out = tmp_path / "sheet.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [str(video), "--out", str(out), "--cols", "3"],
    )
    assert result.exit_code == 0, result.output
    assert "warning" in result.output.lower() or "not indexed" in result.output.lower()
    conn = connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM export_artifacts").fetchone()[0]
    finally:
        conn.close()
    assert count == 0


# -------------------- integration --------------------


@pytest.mark.integration
def test_real_ffmpeg_lavfi(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    video = tmp_path / "clip.mp4"
    # 10s clip, sample every 4s => timestamps 0/4/8 (all in bounds).
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=10:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    out = tmp_path / "sheet.jpg"
    runner = CliRunner()
    result = runner.invoke(
        contact_sheet,
        [
            str(video),
            "--out",
            str(out),
            "--cols",
            "4",
            "--every",
            "4",
            "--no-db",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.stat().st_size > 0
    with Image.open(out) as img:
        assert img.size[0] > 0
        assert img.size[1] > 0
