"""CLI tests for the `reelgrep faces` command group."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from reelgrep import config
from reelgrep.cli import main as cli
from reelgrep.db import connect, migrate
from reelgrep.faces import cluster_faces


def _seed_clustered(tmp_path: Path) -> Path:
    db = tmp_path / "idx.sqlite"
    config.set_db_override(db)
    conn = connect(db)
    migrate(conn)
    conn.execute(
        "INSERT INTO videos(file_hash, path, duration_ms, ingested_at, probe_json) "
        "VALUES (?,?,?,?,?)",
        ("blake2b:" + "a" * 64, "/v.mp4", 100000, "2026-05-18T00:00:00Z", "{}"),
    )
    vid = conn.execute("SELECT id FROM videos").fetchone()[0]
    rng = np.random.RandomState(0)
    c1 = rng.randn(512).astype(np.float32)
    c1 /= np.linalg.norm(c1)
    c2 = rng.randn(512).astype(np.float32)
    c2 /= np.linalg.norm(c2)
    i = 0
    for c in (c1, c2):
        for _ in range(6):
            e = c + 0.02 * rng.randn(512).astype(np.float32)
            e /= np.linalg.norm(e)
            fp = tmp_path / f"f{i}.jpg"
            fp.write_bytes(b"x")
            i += 1
            cur = conn.execute(
                "INSERT INTO frames(video_id, timestamp_ms, path, sampling_strategy) "
                "VALUES (?,?,?,?)",
                (vid, i * 1000, str(fp), "every_n"),
            )
            conn.execute(
                "INSERT INTO face_detections(frame_id, bbox_x, bbox_y, bbox_w, bbox_h, "
                "embedding, embedding_model, detected_at) VALUES (?,?,?,?,?,?,?,?)",
                (cur.lastrowid, 0, 0, 80, 100, e.tobytes(),
                 "insightface_buffalo_l", "2026-05-18T00:00:00Z"),
            )
    conn.commit()
    conn.close()
    cluster_faces(min_cluster_size=5, db_path=db)
    return db


@pytest.fixture(autouse=True)
def _reset_config():
    yield
    config.reset_settings()


def _parse_cluster_ids(list_output: str) -> list[int]:
    ids = []
    for line in list_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            token = stripped.split()[0].lstrip("#")
            if token.isdigit():
                ids.append(int(token))
    return ids


def test_faces_list(tmp_path):
    _seed_clustered(tmp_path)
    result = CliRunner().invoke(cli, ["faces", "list"])
    assert result.exit_code == 0, result.output
    ids = _parse_cluster_ids(result.output)
    assert len(ids) >= 2


def test_faces_label_then_find(tmp_path):
    _seed_clustered(tmp_path)
    runner = CliRunner()
    list_out = runner.invoke(cli, ["faces", "list"]).output
    ids = _parse_cluster_ids(list_out)
    assert ids, "expected at least one cluster"
    assert runner.invoke(cli, ["faces", "label", str(ids[0]), "Speaker A"]).exit_code == 0
    find_result = runner.invoke(cli, ["faces", "find", "Speaker A"])
    assert find_result.exit_code == 0, find_result.output
    assert "/v.mp4" in find_result.output


def test_faces_label_collision_exits_nonzero(tmp_path):
    _seed_clustered(tmp_path)
    runner = CliRunner()
    list_out = runner.invoke(cli, ["faces", "list"]).output
    ids = _parse_cluster_ids(list_out)
    assert len(ids) >= 2
    assert runner.invoke(cli, ["faces", "label", str(ids[0]), "X"]).exit_code == 0
    second = runner.invoke(cli, ["faces", "label", str(ids[1]), "X"])
    assert second.exit_code != 0


def test_faces_cluster_recompute(tmp_path):
    _seed_clustered(tmp_path)
    result = CliRunner().invoke(cli, ["faces", "cluster", "--min-size", "5"])
    assert result.exit_code == 0, result.output
    assert "clusters: 2" in result.output


def test_faces_show(tmp_path):
    _seed_clustered(tmp_path)
    runner = CliRunner()
    ids = _parse_cluster_ids(runner.invoke(cli, ["faces", "list"]).output)
    result = runner.invoke(cli, ["faces", "show", str(ids[0])])
    assert result.exit_code == 0, result.output
    assert "/v.mp4" in result.output


def test_faces_purge_video(tmp_path):
    _seed_clustered(tmp_path)
    result = CliRunner().invoke(cli, ["faces", "purge", "/v.mp4", "--yes"])
    assert result.exit_code == 0, result.output
    assert "deleted" in result.output.lower()


def test_faces_purge_all(tmp_path):
    _seed_clustered(tmp_path)
    # --all is the sentinel literal target the purge subcommand accepts.
    result = CliRunner().invoke(cli, ["faces", "purge", "--all", "--yes"])
    assert result.exit_code == 0, result.output
    assert "deleted all" in result.output.lower() or "all" in result.output.lower()
