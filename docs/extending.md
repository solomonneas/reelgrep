# Extending and embedding reelgrep

This document is for downstream Python packages that want to depend on
reelgrep as a library: drive ingest, search, and transcription against
reelgrep's index, or plug in custom content sources and frame-level
person models.

CLI usage is covered in the main [README](../README.md).

## Overview

reelgrep is a local video-indexing toolkit. It exposes a small
library surface for programmatic use:

- An indexing entry point (`ingest_video`) that probes a file, samples
  frames, extracts subtitles, and writes everything to a SQLite index.
- A read-only query API (`Search`) over that index, with FTS5-backed
  subtitle search and structured rows for frames and stored person
  searches.
- A transcription entry point (`transcribe_video`) that runs Whisper
  against a video and persists the cues alongside any existing
  subtitle sources.
- Two extension registries: one for *content sources* (`BaseBackend`,
  `register_backend`) that resolve a URI to a local file path, and one
  for *person-finding models* (`BasePersonModel`, `register_person_model`)
  that score frames against reference images.

reelgrep does not pull frames or audio over the network on its own.
A consumer is expected to supply video files on the local filesystem,
or to register a backend that downloads remote content into a local
cache before reelgrep touches it.

This document is aimed at downstream packages that:

- Want to index, search, or transcribe videos under their own control
  flow (a worker queue, a web service, a notebook).
- Want to add a custom content source (S3, an HTTP archive, an
  internal asset manager) so that a single URI scheme can be passed
  through reelgrep's commands and library entry points.
- Want to plug in a custom frame-scoring model.

## Installing reelgrep as a dependency

From PyPI:

```bash
pip install reelgrep
```

Equivalent `pyproject.toml` snippet:

```toml
[project]
dependencies = ["reelgrep>=0.4"]
```

Optional extras unlock features that pull in heavier dependencies:

| Extra | Adds | Used for |
|---|---|---|
| `whisper` | `faster-whisper` | Local Whisper transcription via `transcribe_video`. |
| `face` | `insightface`, `onnxruntime`, `numpy` | The bundled `face_embed` person model. |
| `vision` | `httpx` | The bundled `ollama_vision` person model (talks to a local Ollama server). |
| `align` | `pypdf`, `rapidfuzz` | Prose-transcript alignment (CLI `reelgrep align`). |
| `web` | `starlette`, `uvicorn` | The local browser UI (`reelgrep serve`). |

Install multiple extras at once:

```bash
pip install "reelgrep[whisper,face,vision,align]"
```

System dependency: reelgrep shells out to `ffmpeg` and `ffprobe`, so
both must be on `PATH`. `ffprobe` ships in the same package as
`ffmpeg` on every mainstream distribution (`apt install ffmpeg`,
`brew install ffmpeg`, etc.). See <https://ffmpeg.org/> for source
builds.

## Quickstart: ingest, search, transcribe

All three entry points live on the top-level `reelgrep` package.
Submodule paths (`reelgrep.index`, `reelgrep.search`, etc.) work but
are not part of the stability contract: import from `reelgrep`
directly.

### Ingest a video

```python
from reelgrep import ingest_video, IngestResult

result: IngestResult = ingest_video(
    "/path/to/video.mp4",
    db_path="/tmp/reelgrep-demo.sqlite",
    interval_seconds=5.0,
)
print(result.video_id, result.frame_count, result.subtitle_cue_count)
for warning in result.warnings:
    print(f"[{warning.stage}] {warning.message}")
```

`ingest_video` is idempotent for a given file hash unless you pass
`force=True`. Re-running on the same video returns
`already_ingested=True` and writes nothing new.

### Search the index

```python
from reelgrep import Search, SubtitleHit

search = Search(db_path="/tmp/reelgrep-demo.sqlite")
hits: list[SubtitleHit] = search.subtitles("hello", limit=10)
for hit in hits:
    print(hit.start_ms, hit.text)
```

When `db_path` is provided explicitly, the file must already exist;
otherwise `FileNotFoundError` is raised. This is deliberate: passing
a typo would otherwise silently create an empty database and return
empty results forever. Omit `db_path` to use the configured default
(honours `REELGREP_DB` / `REELGREP_HOME`).

### Transcribe a video

```python
from reelgrep import transcribe_video, TranscribeResult

result: TranscribeResult = transcribe_video(
    "/path/to/lecture.mp4",
    model="tiny",
    db_path="/tmp/reelgrep-demo.sqlite",
)
print(result.cue_count, result.language_detected, result.already_transcribed)
```

Requires the `[whisper]` extra. The first call downloads the
faster-whisper model weights into the user's Hugging Face cache.
Subsequent calls on the same video short-circuit and return
`already_transcribed=True` unless you pass `force=True`.

## Custom content sources: writing a `BaseBackend`

A backend resolves a backend-specific URI to an absolute local file
path. reelgrep's indexer then probes that path with ffprobe like any
other file. This is the right extension point when your videos live
somewhere reelgrep cannot reach directly: an object store, an
internal HTTP archive, a build system's artifact store.

The contract is small. From `reelgrep.backends`:

```python
class BaseBackend(ABC):
    name: str = ""  # set by @register_backend

    @abstractmethod
    def resolve(self, uri: str) -> Path:
        """Resolve a backend-specific URI/identifier to an absolute local file path."""
```

A single method, sync, returning a `pathlib.Path` that already
exists on disk. If the backend's URI scheme refers to remote content,
`resolve` is responsible for downloading (or otherwise materialising)
that content into a local cache before returning the path.

If resolution fails, raise `reelgrep.backends.BackendError`. reelgrep
catches it and surfaces it as an ingest error.

### Worked example: an S3 backend

This example resolves `s3://bucket/key` URIs by downloading the
object into a per-user cache directory under reelgrep's settings
home, then returning the cached path. boto3 is illustrative only;
the same shape works with any S3-compatible client.

```python
"""mypkg/backends/s3.py"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from reelgrep import BaseBackend, register_backend
from reelgrep.backends import BackendError


@register_backend("s3")
class S3Backend(BaseBackend):
    """Resolves s3://bucket/key URIs by downloading into a local cache."""

    def __init__(
        self,
        *,
        bucket: str | None = None,
        region: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._default_bucket = bucket or os.environ.get("REELGREP_S3_BUCKET")
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        if cache_dir is None:
            cache_dir = Path.home() / ".local" / "share" / "reelgrep" / "cache" / "s3"
        self._cache_dir = Path(cache_dir).expanduser().resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise BackendError(f"S3Backend got non-s3 URI: {uri!r}")
        bucket = parsed.netloc or self._default_bucket
        key = parsed.path.lstrip("/")
        if not bucket:
            raise BackendError("no bucket: pass s3://bucket/key or set REELGREP_S3_BUCKET")
        if not key:
            raise BackendError(f"empty object key in {uri!r}")

        local = self._cache_dir / bucket / key
        if local.exists() and local.is_file():
            return local

        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendError("S3Backend requires boto3: pip install boto3") from exc

        client = boto3.client("s3", region_name=self._region)
        tmp = local.with_suffix(local.suffix + ".part")
        try:
            client.download_file(bucket, key, str(tmp))
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise BackendError(f"s3 download failed for {uri!r}: {exc}") from exc
        tmp.replace(local)
        return local
```

Register the module at startup (or rely on Python's normal import
mechanism by importing it from your package's `__init__.py`). Once
registered, the backend is selectable by name through the same
entry points the CLI uses:

```python
from reelgrep import ingest_video
import mypkg.backends.s3  # noqa: F401 - import for side-effect registration

result = ingest_video(
    "s3://my-bucket/path/to/video.mp4",
    backend="s3",
    db_path="/tmp/reelgrep-demo.sqlite",
)
```

Things to keep in mind:

- `resolve` MUST return a path that exists and is a regular file by
  the time it returns. reelgrep does not retry or wait.
- Backends are instantiated per call by `get_backend(name, **kwargs)`.
  Keep `__init__` cheap (no eager network calls). Defer real work
  until `resolve`.
- For long-lived caches, prefer content-addressed keys or a stable
  bucket/key layout so re-ingest doesn't redownload. The S3 example
  above uses the literal bucket/key path under the cache root.
- Raise `BackendError` for predictable failures (auth, missing
  object, malformed URI). Let unexpected exceptions propagate so
  bugs surface.

If you'd rather model an HTTP archive (`https://archive.example.com/clips/abc.mp4`)
the shape is identical: parse the URL, derive a cache path, download
once, return the local path.

## Custom person models: writing a `BasePersonModel`

The person-model registry is the second extension point. A model
takes a list of frames plus positive and negative reference images
and returns a sorted list of matches.

The contract from `reelgrep.models`:

```python
class BasePersonModel(ABC):
    name: str = ""  # set by @register_person_model

    @abstractmethod
    def find(
        self,
        frames: list[Frame],
        positive_examples: list[Path],
        negative_examples: list[Path],
        *,
        threshold: float,
        top_k: int | None = None,
    ) -> list[Match]:
        """Score frames against examples; return matches sorted by confidence desc."""

    def config_dict(self) -> dict[str, Any]:
        """Return the backend's effective configuration for manifest provenance."""
        return {}
```

`Frame` and `Match` currently live in submodules (`reelgrep.frames`
and `reelgrep.models` respectively) and are not re-exported from the
top-level package. A custom model must import them from their
submodules:

```python
from reelgrep.frames import Frame  # noqa: F401 - re-exported via reelgrep.models in future
from reelgrep.models import Match
```

Treat these submodule imports as the only stable place to get those
types until they are promoted to the public surface in a future
release.

### Worked example: a deliberately silly placeholder

The example below is intentionally a toy: it ignores the reference
images entirely and returns "matches" based on the average
brightness of each frame's filename hash. It exists to show the
plumbing, not to be useful. Do not ship this.

```python
"""mypkg/models/random_color.py"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from reelgrep import BasePersonModel, register_person_model
from reelgrep.frames import Frame
from reelgrep.models import Match


@register_person_model("random_color")
class RandomColorModel(BasePersonModel):
    """Toy model. Returns fake matches based on the frame path's hash."""

    def __init__(self, *, seed: str = "demo") -> None:
        self._seed = seed

    def config_dict(self) -> dict[str, Any]:
        return {"seed": self._seed}

    def find(
        self,
        frames: list[Frame],
        positive_examples: list[Path],
        negative_examples: list[Path],
        *,
        threshold: float,
        top_k: int | None = None,
    ) -> list[Match]:
        matches: list[Match] = []
        for frame in frames:
            digest = hashlib.sha256(
                (self._seed + frame.path).encode("utf-8")
            ).digest()
            confidence = digest[0] / 255.0
            if confidence < threshold:
                continue
            matches.append(
                Match(
                    frame=frame,
                    confidence=confidence,
                    bbox=None,
                    reasoning=f"toy: byte0={digest[0]}",
                )
            )
        matches.sort(key=lambda m: m.confidence, reverse=True)
        if top_k is not None:
            matches = matches[:top_k]
        return matches
```

The `face_embed` and `ollama_vision` models bundled with reelgrep
both implement this interface; their source under
`src/reelgrep/models/` is the canonical reference for what a real
model looks like.

## Schema and the `tags` table

The index schema lives at `src/reelgrep/schema.sql`. Most tables
(`videos`, `subtitles`, `frames`, `scenes`, `person_searches`,
`person_matches`, `export_artifacts`) are populated by reelgrep
itself and have a fixed shape.

The `tags` table is the explicit extension surface for downstream
consumers that need to attach arbitrary key-value metadata to a
video or a frame:

```sql
CREATE TABLE tags (
  id INTEGER PRIMARY KEY,
  video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
  frame_id INTEGER REFERENCES frames(id) ON DELETE CASCADE,
  key TEXT NOT NULL,
  value TEXT
);
```

Either `video_id` or `frame_id` should be set (or both, for a
frame-scoped tag whose parent video you want to denormalize). A
consumer that wants to map reelgrep rows back to its own primary
key can do so directly:

```python
import sqlite3

conn = sqlite3.connect("/tmp/reelgrep-demo.sqlite")
conn.execute(
    "INSERT INTO tags (video_id, key, value) VALUES (?, ?, ?)",
    (123, "external_id", "my-cdn-id-456"),
)
conn.commit()
```

Retrieve later via a plain `SELECT`:

```python
row = conn.execute(
    "SELECT value FROM tags WHERE video_id = ? AND key = ?",
    (123, "external_id"),
).fetchone()
external_id = row[0] if row else None
```

The `tags` table is a fallback. Prefer the backend and person-model
registries when your extension fits those shapes; reach for `tags`
only when it doesn't.

## Faces and clustering

reelgrep ships a face detection + clustering pipeline that programmatic
consumers can drive directly. The relevant entry points are
``extract_faces`` (detect + embed across a video), ``cluster_faces``
(group the detection pool), and ``Faces`` (the read-and-update API
over the resulting tables).

```python
from reelgrep import extract_faces, cluster_faces, Faces

extract_faces("/path/to/video.mp4")            # detect + embed
cluster_faces(min_cluster_size=5)               # cluster the whole pool

faces = Faces()
for c in faces.list_clusters(limit=10):
    print(c.id, c.label, c.size)

# Auto-extend: label a cluster once, query it everywhere.
faces.label_cluster(cluster_id=1, label="Speaker A")
for det in faces.find_by_label("Speaker A"):
    print(det.video_path, det.timestamp_ms, det.bbox)
```

``cluster_faces`` is destructive on the cluster tables (it wipes
``face_clusters`` and ``face_cluster_members``) but never on
``face_detections``. Labels are preserved across re-clusters when a
new cluster's centroid is within ``label_carry_threshold`` (default
``0.4`` cosine distance) of the previous labeled cluster's centroid;
labels that fall outside that band are reported in the
``ClusterReport.labels_orphaned`` list and the user can re-attach
them via ``Faces.label_cluster``.

### Embedding storage format

Each row in ``face_detections`` stores a 512-dim float32 numpy array
as a little-endian raw byte blob in the ``embedding`` column. To read
embeddings directly without going through reelgrep:

```python
import numpy as np
import sqlite3

conn = sqlite3.connect("~/.local/share/reelgrep/index.sqlite")
for row in conn.execute("SELECT embedding FROM face_detections LIMIT 5"):
    emb = np.frombuffer(row[0], dtype=np.float32)
    assert emb.shape == (512,)
```

The ``embedding_model`` column documents which insightface model pack
produced the embedding (``insightface_buffalo_l`` by default). Future
versions may persist multiple model outputs side by side.

### Privacy

Face detection is opt-in and runs only when the user invokes
``extract_faces`` (or the corresponding CLI) explicitly. The CLI
ships ``reelgrep faces purge <video> --yes`` for per-video deletes
and ``reelgrep faces purge --all --yes`` to wipe the face tables
entirely. Downstream packages embedding reelgrep should expose
equivalent controls in their own UIs.

## What reelgrep does NOT do (and what you may need to add yourself)

reelgrep is intentionally narrow. The following are out of scope at
the current version; if your application needs them, plan to layer
them on top:

- Visual or semantic embedding search across frames. There is no
  bundled CLIP, no FAISS, no vector store. Frames are indexed only
  by timestamp and path.
- Audio-feature search. Audio is used only as input to Whisper
  transcription; there is no audio-fingerprint or similarity index.
- Frame-level metadata beyond what registered models attach to
  `person_matches`. Object detection, OCR, scene labels, and similar
  per-frame structured data are not collected by default.
- Custom rerankers or score blending. `Search.subtitles` returns raw
  FTS5 matches in cue order; if you need BM25-weighted or hybrid
  ranking, do it in your own layer.
- Cross-modal search ("find scenes that look like this image").
  reelgrep has no image-to-image or text-to-image retrieval.

These are not promises that they will arrive. Build them on top of
the existing surface when you need them.

## Versioning and stability

reelgrep is currently at `0.5.x`. The library surface is settling
but is not yet API-stable: minor versions in the `0.y` series may
introduce breaking changes. Once a `1.0` release ships, reelgrep
will follow [semver](https://semver.org/).

For forward-compatibility within the `0.5` line, pin with
`reelgrep~=0.5.0`:

```toml
[project]
dependencies = ["reelgrep~=0.5.0"]
```

The definitive list of public names is `reelgrep.__all__`. Anything
not in that list, including submodule paths, may move or be renamed
without a major version bump. The two exceptions today are
`reelgrep.frames.Frame` and `reelgrep.models.Match`, which custom
person models must import directly until they are promoted to the
top-level package; this will land in a future release.
