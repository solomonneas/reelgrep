# Changelog

## 0.5.0 (unreleased)

Cross-video face clustering, behind the `[face]` extra (adds `hdbscan`).

- `reelgrep extract-faces <video>` and `--all` detect and embed faces in sampled frames.
- `reelgrep faces list|show|label|find|cluster|purge` browse, name, and query clusters. Labels carry over when you re-cluster.
- `reelgrep ingest --detect-faces` runs detection as part of ingest.
- Library API: `extract_faces`, `cluster_faces`, `Faces`.
- Browser UI: Faces view with a cluster grid, member frames, and a label editor.
- Schema v4 adds `face_detections`, `face_clusters`, and `face_cluster_members`. Existing indexes upgrade the next time reelgrep opens them.
- Library surface re-exported from the top-level `reelgrep` package: `Search` (with a typed error hierarchy), `transcribe_video`, `register_backend`, `register_person_model`. See [docs/extending.md](docs/extending.md).

## 0.4.0 (2026-05-17)

- `reelgrep align` maps an official prose transcript (`.txt`, `.md`, `.pdf`) onto Whisper cue timestamps, behind the `[align]` extra.
- `reelgrep.index.ingest_video` split out of the CLI as a library entry point.
- Schema v3 allows `source='aligned'` cues.

## 0.3.0 (2026-05-17)

- `reelgrep serve`: local browser UI on a Starlette JSON API with a vanilla HTML/CSS/JS frontend. Search across every video's subtitles at once.

## 0.2.0 (2026-05-17)

- `reelgrep transcribe` and `reelgrep ingest --transcribe`: local Whisper transcription via the `[whisper]` extra.

## 0.1.0 (2026-05-17)

- First release: `ingest`, `info`, `ls`, `search-subtitles` (SQLite FTS5), `export-clip`, `make-gif`, `contact-sheet`, `find-person` with `face_embed` and `ollama_vision` backends, and the Jellyfin resolver.
