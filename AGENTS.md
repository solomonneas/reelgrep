# Repository Guidance

reelgrep is a local video search and media analysis toolkit. Python 3.11+, hatchling
packaging, CLI entry point `reelgrep = reelgrep.cli:main` (click group).

## Definition of Done
A code change is done only when BOTH pass, run from the repo root (`.venv/` works
without activation):
- `.venv/bin/ruff check src tests`
- `.venv/bin/pytest`

Report actual results. If anything fails, report the failure verbatim and do not
claim success. Never report done without running both.

## Layout
- Source under `src/reelgrep/`. One file per subcommand in `src/reelgrep/commands/`.
  Backends (local, jellyfin) in `backends/`, person-model plugins (face_embed,
  ollama_vision) in `models/`, browser UI in `web/`.
- Index is SQLite with FTS5. Schema: `src/reelgrep/schema.sql` plus forward migrations
  (`migration_v*_to_v*.sql`) registered in `_FORWARD_MIGRATIONS` in `db.py`. Schema
  version lives in the `schema_version` table, not `PRAGMA user_version`.
- Heavy deps are gated behind extras: `face` (insightface, onnxruntime, hdbscan), `vision`
  (httpx for Ollama), `whisper` (faster-whisper), `web` (starlette, uvicorn), `align` (pypdf, rapidfuzz).

## Rules (trigger -> rule -> do instead)
- Touching core modules -> core install must work with zero extras -> import extra
  deps lazily inside the gated module, never at the top of core modules.
- Changing public exports (`ingest_video`, `Search`, `transcribe_video`,
  `register_backend`, `register_person_model`) -> `docs/extending.md` and
  `tests/test_public_api.py` must match -> update both in the same change.
- Changing the schema -> do not edit released schema in place -> add a new
  `migration_v*_to_v*.sql` and register it in `_FORWARD_MIGRATIONS`.
- Test needs real `ffmpeg`/`ffprobe` -> mark it `integration`; requires binaries on PATH.
- Iterating on one area -> use `.venv/bin/pytest tests/test_<area>.py -q` -> still run
  the full Definition of Done before reporting done.
- Lint config (pyproject): rules E, F, I, UP, B; line length 100. Fix code to satisfy
  the rules; do not add `noqa` to silence real findings.

## Prohibitions
- A test fails -> never weaken assertions, add skips, or delete tests to get green ->
  fix the code or report the failure verbatim and stop.
- Personal media files are the test subject domain -> never commit media files
  (`*.mp4|mkv|webm`, including `tests/fixtures/`) or analysis outputs with personal
  content (face crops, embeddings, transcripts, index DBs) -> gitignored paths and
  throwaway DBs only.
- Whisper or face model weights are missing -> never trigger heavy model downloads in
  sandboxed or metered environments -> ask the user before downloading.
- Jellyfin (`JELLYFIN_URL`, `JELLYFIN_API_KEY`) and Ollama vision talk to live
  services -> never hit a real server from tests -> mock the HTTP layer.
- `reelgrep faces purge` and cluster wipes are destructive -> never run them against
  the real index (`~/.local/share/reelgrep/index.sqlite`) -> use a throwaway DB via
  `--db` or `REELGREP_DB`.
- A pre-push hook exists -> never push with `--no-verify` -> only push when the user
  explicitly asks, and let the hook run.
- Everything is local by design -> never add code paths that send frames, clips, or
  index data over the network -> require explicit user opt-in first.
- `memory/`, `.brigade/`, and `.claude/` are local-only and gitignored -> never force-add them.
- Blocked (missing tool, permission, ambiguous requirement) -> do not work around it
  silently -> report the exact blocker and stop.

## Gotchas
- Env overrides: `REELGREP_HOME`, `REELGREP_DB`, `REELGREP_CACHE`, `REELGREP_FFMPEG`, `REELGREP_FFPROBE`.
- HDBSCAN `algorithm="generic"` rejects float32; cast to float64 before `fit_predict`,
  on-disk BLOB storage stays float32.
- Two-phase clustering fallback: if the default HDBSCAN pass returns all noise, retry with
  `allow_single_cluster=True`, `min_samples=2`. Do not remove; single-identity libraries depend on it.
- `test_models_face_embed.py` leaves `sys.modules["insightface.app"]` populated; tests
  simulating the missing `face` extra must null both `insightface` and `.app`.
- Videos use a `file_hash` column, not `hash`; frames require `sampling_strategy`.
- Face extraction looks up videos by raw input path and resolved path; both storage
  conventions exist in real indexes.
- Re-running ingest on the same file is a no-op unless `--force` is passed.

## Memory Handoff
At the end of any substantial task, write a handoff note to `.claude/memory-handoffs/`
using that directory's `TEMPLATE.md`. Record durable discoveries, gotchas, and
decisions. Do not wait to be reminded.
