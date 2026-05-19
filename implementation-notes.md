# Implementation Notes — Cross-Video Face Clustering (v0.5.0)

Running log of decisions, deviations, surprises, and follow-ups captured during
plan execution. Gitignored alongside `.claude/`.

**Spec:** `.claude/specs/2026-05-18-cross-video-face-clustering.md`
**Plan:** `.claude/plans/2026-05-18-cross-video-face-clustering.md`
**Branch:** `feat/face-clustering` off `main` at `71210ba`
**Baseline (pre-task-1):** 476 passed, 1 skipped, ruff clean.

## Decisions

- Stay in main repo + feat branch (no worktree). `.venv` already configured.
- Per user feedback (`feedback-lighter-superpowers-workflow.md`): implementer-only
  subagent dispatch, skip the spec+quality reviewer pair. Final integration check
  done inline by the controller (test + ruff + diff scan).
- All subagents dispatched as Opus 4.7
  (per `feedback-subagent-model-opus.md`).
- Spec/plan kept in `.claude/` (gitignored) per repo hygiene rule.

## Task log

### Task 1 — Schema v3→v4 migration  ✅
- SHA `fb2e55d`. Tests 476 → 479 (+3), ruff clean.
- Codebase uses a `schema_version` TABLE, not `PRAGMA user_version`. Migrations registered in `_FORWARD_MIGRATIONS` dict in `db.py`. Plan snippets were written from spec; implementer correctly followed the real idiom.
- Existing tests hardcoded `== 3` in 11 assertions; bumped to `== 4` mechanically.
- Frames have a required `sampling_strategy` column; videos use `file_hash` not `hash`. Update downstream task helpers to match.

### Task 2 — faces.extract_faces  ✅
- SHA `49b5695`. Tests 479 → 482 (+3), ruff clean.
- Subtle test-isolation gotcha: `test_models_face_embed.py` leaves `sys.modules["insightface.app"]` populated, so the missing-extra test had to null both `insightface` and `insightface.app`. Carry this pattern into Tasks 5/6 dispatch prompts when they mock insightface.
- `extract_faces` looks up the video by both the raw input path AND its resolved form (covers both ingest-time storage conventions).

### Task 3 — faces.cluster_faces + label preservation  ✅
- Tests 482 → 484 (+2), ruff clean.
- Installed `hdbscan>=0.8.40` in `.venv` directly (Task 11 will add it to pyproject's `[face]` extra).
- **Deviation: HDBSCAN dtype.** `hdbscan` with `algorithm="generic"` rejects float32 (`Buffer dtype mismatch, expected 'double_t' but got 'float'`). Cast embeddings to float64 just before `fit_predict`; storage stays float32 for the on-disk BLOB.
- **Deviation: single-cluster fallback.** The label-preservation test seeds 8 detections of one identity in 512-D space. Default HDBSCAN cannot form a single isolated cluster — it returns all noise regardless of `min_samples` / noise level / `cluster_selection_method`. Without intervention the test sees 0 clusters, so the `UPDATE face_clusters SET label=...` no-ops and the assertion fails. Fix: when the default pass produces zero clusters, retry with `allow_single_cluster=True` and `min_samples=2`. This combo also produces 1 cluster (6/8 points) for the single-identity case while staying multi-cluster-correct for the well-separated case (because the fallback only triggers when the first pass returned all noise). Verified both tests pass with this two-phase approach.
- Label-carry threshold `0.4` (cosine distance) is the default. Both labeled-centroid snapshot and re-cluster centroids are unit-normalized so the dot product gives cosine similarity directly.
- Ruff `E702` flagged the semicolon-pair statements the spec literally pasted (`conn.commit(); conn.close()` etc.); rewrote on separate lines. Behavior unchanged.

### Task 4 — Faces read-and-update class  ✅
- SHA `af8fc1b`. Tests 484 → 488 (+4), ruff clean.
- Tests must import as `Faces, cluster_faces, FacesError` (alphabetical) for ruff I001.
- All semicolon statement chains rewritten across both lines.

### Task 5 — `reelgrep extract-faces` CLI  ✅
- SHA `bc5820c`. Tests 488 → 492 (+4), ruff clean.
- **Important: CLI group is `main`, not `cli`.** Test imports must do `from reelgrep.cli import main as cli` (or use `main` directly).
- **`ctx.obj['db_path']` is NOT the pattern.** Root `--db` flag calls `config.set_db_override()` directly. Commands just call library entrypoints (`extract_faces(video)`) without passing `db_path` — the library resolves via `get_settings()`. Carry forward into Tasks 6, 7, 9.
- Spec's compound `a; b` statements continue to need ruff splitting in dispatch prompts.

### Task 9 — Web JSON endpoints for face clusters  ✅
- Tests 502 → 510 (+8), ruff clean.
- **Real web test file is `tests/test_web_app.py`, not `tests/test_commands_serve.py`** (despite the task hint). Both were checked; `test_commands_serve.py` covers the CLI entry, `test_web_app.py` is where the route TestClient pattern lives. Carry forward.
- The web app instantiates the `Faces` library with `db_path=resolved_db_path` (the per-app captured path), which matches the pattern used for `Search(db_path=resolved_db_path)`. No reliance on `get_settings()` — good, because the app fixture builds the DB at a tmp path.
- Added 4 extra tests beyond the spec's listed four: `labeled_only` query filter, `label=null` clears, PATCH-on-missing 404, and collision 400. Each maps to a concrete branch in `faces_cluster_label` / `faces_clusters`.
- Bumped CORS `allow_methods` from `["GET"]` to `["GET", "PATCH"]` so the new write endpoint survives the local UI's preflight. No other middleware touched.
- `Faces.label_cluster` raises `FacesError("no cluster with id ...")` for missing and `FacesError("label 'X' already on cluster ...")` for collisions. The route discriminates via `msg.startswith("no cluster")` — brittle but matches the library's only two failure modes; a typed sentinel would be cleaner but is out of scope for Task 9.
