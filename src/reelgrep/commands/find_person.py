"""``reelgrep find-person`` command: locate frames containing a person."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import click

from reelgrep import timecode
from reelgrep.config import ensure_dirs, get_settings
from reelgrep.db import connect, migrate
from reelgrep.frames import Frame, sample_every
from reelgrep.hashing import file_hash
from reelgrep.manifest import Manifest, Source, write
from reelgrep.models import ModelError, get_person_model
from reelgrep.probe import probe

__all__ = ["find_person"]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_DEFAULT_THRESHOLDS = {"face_embed": 0.30, "ollama_vision": 0.65}


def _hash_slice(digest: str) -> str:
    """Return the deterministic cache subdirectory name for a file hash."""
    return digest[8:24] if digest.startswith("blake2b:") else digest[:16]


def _expand_images(paths: tuple[Path, ...]) -> list[Path]:
    """Expand a tuple of file or directory paths into a flat list of image files."""
    out: list[Path] = []
    for entry in paths:
        if entry.is_dir():
            for sub in sorted(entry.rglob("*")):
                if sub.is_file() and sub.suffix.lower() in _IMAGE_EXTS:
                    out.append(sub.resolve())
        elif entry.is_file():
            out.append(entry.resolve())
    return out


@click.command("find-person")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--label", required=True, help="Name to assign to this person search.")
@click.option(
    "--positive",
    "positives",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help="Path to a positive example image. Can be a directory; all images inside are used. "
    "Repeat to add more.",
)
@click.option(
    "--negative",
    "negatives",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="Path to a known false-positive image. Can be a directory. Repeat to add more.",
)
@click.option(
    "--backend",
    type=click.Choice(["face_embed", "ollama_vision"]),
    default="face_embed",
    show_default=True,
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Match acceptance threshold. Default: 0.30 for face_embed, 0.65 for ollama_vision.",
)
@click.option(
    "--top-k",
    "top_k",
    type=int,
    default=25,
    show_default=True,
    help="Maximum matches returned (and exported).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--every",
    "interval_seconds",
    type=float,
    default=2.0,
    show_default=True,
    help="Frame-sampling interval used when the video has not been ingested with finer sampling.",
)
@click.option(
    "--no-export-frames",
    is_flag=True,
    default=False,
    help="Skip writing per-match JPEGs; only update DB + manifest.",
)
def find_person(
    video_path: Path,
    label: str,
    positives: tuple[Path, ...],
    negatives: tuple[Path, ...],
    backend: str,
    threshold: float | None,
    top_k: int,
    out_dir: Path,
    interval_seconds: float,
    no_export_frames: bool,
) -> None:
    """Locate frames containing a person, using positive and negative reference images."""
    if top_k < 1:
        raise click.BadParameter("must be >= 1", param_hint="--top-k")
    if interval_seconds <= 0:
        raise click.BadParameter("must be > 0", param_hint="--every")
    if threshold is None:
        threshold = _DEFAULT_THRESHOLDS[backend]

    positive_files = _expand_images(positives)
    if not positive_files:
        raise click.BadParameter(
            "no positive images found (looked for .jpg/.jpeg/.png/.webp)",
            param_hint="--positive",
        )
    negative_files = _expand_images(negatives)

    resolved_video = video_path.resolve()
    resolved_out = out_dir.resolve()
    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    ensure_dirs(settings)
    digest = file_hash(resolved_video)

    conn = connect(settings.db_path)
    try:
        migrate(conn)

        row = conn.execute(
            "SELECT id, duration_ms FROM videos WHERE file_hash = ?",
            (digest,),
        ).fetchone()

        if row is None:
            meta = probe(resolved_video)
            ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
            probe_json = json.dumps(meta.raw)
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO videos (
                        file_hash, path, duration_ms, width, height, fps,
                        container, video_codec, audio_codec, size_bytes,
                        ingested_at, probe_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        digest,
                        str(resolved_video),
                        meta.duration_ms,
                        meta.width,
                        meta.height,
                        meta.fps,
                        meta.format_name,
                        meta.video_codec,
                        meta.audio_codec,
                        meta.size_bytes,
                        ingested_at,
                        probe_json,
                    ),
                )
                video_id = cursor.lastrowid
            duration_ms = meta.duration_ms
        else:
            video_id = row["id"]
            duration_ms = row["duration_ms"]

        frame_rows = conn.execute(
            "SELECT id, timestamp_ms, path, sampling_strategy "
            "FROM frames WHERE video_id = ? ORDER BY timestamp_ms",
            (video_id,),
        ).fetchall()

        frames: list[Frame]
        frame_ids: list[int]
        if frame_rows:
            frames = [
                Frame(
                    timestamp_ms=int(r["timestamp_ms"]),
                    path=str(r["path"]),
                    sampling_strategy=r["sampling_strategy"],
                )
                for r in frame_rows
            ]
            frame_ids = [int(r["id"]) for r in frame_rows]
        else:
            frames_cache = settings.cache_dir / "frames" / _hash_slice(digest)
            frames_cache.mkdir(parents=True, exist_ok=True)
            sampled = sample_every(
                resolved_video, frames_cache, interval_seconds=interval_seconds
            )
            frame_ids = []
            with conn:
                for frame in sampled:
                    cur = conn.execute(
                        """
                        INSERT INTO frames (
                            video_id, timestamp_ms, path, sampling_strategy,
                            width, height
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            video_id,
                            frame.timestamp_ms,
                            frame.path,
                            frame.sampling_strategy,
                            frame.width,
                            frame.height,
                        ),
                    )
                    frame_ids.append(int(cur.lastrowid))
            frames = sampled

        model = get_person_model(backend)
        cfg = model.config_dict()

        try:
            matches = model.find(
                frames,
                positive_files,
                negative_files,
                threshold=threshold,
                top_k=top_k,
            )
        except ModelError as exc:
            click.echo(f"error: {exc}", err=True)
            raise click.exceptions.Exit(2) from exc

        # Map frame path -> frame_id for DB linkage.
        path_to_frame_id: dict[str, int] = {
            f.path: fid for f, fid in zip(frames, frame_ids, strict=True)
        }

        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        with conn:
            search_cursor = conn.execute(
                """
                INSERT INTO person_searches (
                    video_id, label, backend,
                    positive_examples_json, negative_examples_json,
                    config_json, threshold, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    label,
                    backend,
                    json.dumps([str(p) for p in positive_files]),
                    json.dumps([str(p) for p in negative_files]),
                    json.dumps(cfg),
                    threshold,
                    created_at,
                ),
            )
            search_id = int(search_cursor.lastrowid)

            for match in matches:
                fid = path_to_frame_id.get(match.frame.path)
                if fid is None:
                    fid_row = conn.execute(
                        "SELECT id FROM frames WHERE video_id = ? AND timestamp_ms = ?",
                        (video_id, match.frame.timestamp_ms),
                    ).fetchone()
                    if fid_row is None:
                        continue
                    fid = int(fid_row["id"])
                bbox_json = json.dumps(list(match.bbox)) if match.bbox else None
                conn.execute(
                    """
                    INSERT INTO person_matches (
                        search_id, frame_id, confidence, bbox_json, reasoning
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (search_id, fid, match.confidence, bbox_json, match.reasoning),
                )

        # Export frames + build result records.
        results: list[dict] = []
        exported_paths: list[Path] = []
        if matches and not no_export_frames:
            resolved_out.mkdir(parents=True, exist_ok=True)
        elif not matches:
            # Still create out_dir so the manifest can land predictably.
            resolved_out.mkdir(parents=True, exist_ok=True)

        manifest_path = resolved_out / "find-person.manifest.json"

        for match in matches:
            exported_path: Path | None = None
            if not no_export_frames:
                target = resolved_out / (
                    f"{label}_{timecode.format(match.frame.timestamp_ms)}.jpg"
                )
                try:
                    shutil.copyfile(match.frame.path, target)
                    exported_path = target
                    exported_paths.append(target)
                except OSError as exc:
                    click.echo(
                        f"warning: could not export frame {match.frame.path}: {exc}",
                        err=True,
                    )
            results.append(
                {
                    "timestamp_ms": match.frame.timestamp_ms,
                    "frame_path": match.frame.path,
                    "exported_path": str(exported_path) if exported_path else None,
                    "confidence": match.confidence,
                    "bbox": list(match.bbox) if match.bbox else None,
                    "reasoning": match.reasoning,
                }
            )

        manifest = Manifest(
            operation="find-person",
            source=Source(
                path=str(resolved_video),
                file_hash=digest,
                duration_ms=duration_ms,
            ),
            parameters={
                "label": label,
                "backend": backend,
                "threshold": threshold,
                "top_k": top_k,
                "interval_seconds": interval_seconds,
                "positive_examples": [str(p) for p in positive_files],
                "negative_examples": [str(p) for p in negative_files],
                "model_config": cfg,
            },
            results=results,
        )
        write(manifest_path, manifest)

        if exported_paths:
            with conn:
                for exported in exported_paths:
                    conn.execute(
                        """
                        INSERT INTO export_artifacts (
                            video_id, kind, path, manifest_path, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            video_id,
                            "screenshot",
                            str(exported),
                            str(manifest_path),
                            created_at,
                        ),
                    )
    finally:
        conn.close()

    click.echo(f"label:     {label}")
    click.echo(f"backend:   {backend}")
    click.echo(f"threshold: {threshold}")

    if not matches:
        click.echo("no matches above threshold")
        click.echo(f"manifest:  {manifest_path}")
        return

    click.echo(
        f"matches:   {len(matches)} / {top_k} (showing top {len(matches)})"
    )
    click.echo("")
    for match in matches:
        ts = timecode.format(match.frame.timestamp_ms)
        click.echo(
            f"   {ts}  conf {match.confidence:.2f}  {match.reasoning}"
        )
    click.echo("")
    click.echo(f"manifest:  {manifest_path}")
    if no_export_frames:
        click.echo("exports:   (skipped, --no-export-frames)")
    else:
        click.echo(
            f"exports:   {resolved_out} ({len(exported_paths)} files)"
        )
