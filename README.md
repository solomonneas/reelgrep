# reelgrep

> Local video search and media analysis. Find people, outfits, objects, scenes, spoken phrases, and useful clips inside your own video library.

Current release: v0.4.0. It includes transcript alignment for PDF, TXT, and MD sources, the local web UI, transcription, and the TypeScript MCP wrapper.

reelgrep indexes video files on disk: it runs `ffprobe` for metadata, samples frames at a configurable interval, builds contact sheets, extracts embedded and sidecar subtitles into a SQLite FTS5 table you can grep, and exports clips, screenshots, and animated WebP loops with a JSON manifest sidecar for each output. Person and object detection is pluggable, with a face-embedding backend and an Ollama vision-LLM backend; both accept confirmed positive and negative reference images so lookalikes do not slip through. Everything runs on your machine - frames, clips, manifests, and the index all stay on disk under your home directory by default, and nothing leaves the box without you configuring it to.

## Why

- Lectures and conference talks: jump to the moment the speaker discussed X.
- Personal and family videos: find clips of a specific person across years of footage.
- TV episodes and movies: build contact sheets, cut highlight reels, export GIFs.
- Training and onboarding videos: extract slides, search transcripts, build searchable archives.
- All processing is local. Frames, clips, manifests, and the index stay on disk under your home directory.

## Install

```bash
# CLI only (no model deps; subtitle search + ingest + export work):
pipx install reelgrep

# With local face recognition (insightface + onnxruntime, ~300MB on first model load):
pipx install "reelgrep[face]"

# With Ollama vision-LLM backend (httpx; assumes ollama serve is reachable):
pipx install "reelgrep[vision]"

# With local Whisper transcription (faster-whisper + ctranslate2, ~75MB-2.9GB depending on model):
pipx install "reelgrep[whisper]"

# With the local browser UI (starlette + uvicorn):
pipx install "reelgrep[web]"

# With prose-transcript alignment (pypdf + rapidfuzz, ~1MB):
pipx install "reelgrep[align]"

# Everything:
pipx install "reelgrep[face,vision,whisper,web,align]"
```

System dependency: `ffmpeg` and `ffprobe` must be on PATH. On Ubuntu:

```bash
sudo apt install ffmpeg
```

On macOS:

```bash
brew install ffmpeg
```

## Quickstart

### Ingest a video

```bash
reelgrep ingest /tmp/reelgrep-demo/demo.mp4
```

Synthetic example output:

```text
ingested: /tmp/reelgrep-demo/demo.mp4
hash:     blake2b:7343ecbd3429d7c3839e8e444bf8be180c84c1d01d528559d09c7f92d61ae75c
duration: 00:00:12.000
subtitle tracks: 1 (cues: 3)
frames sampled: 3
db:       /tmp/reelgrep-demo/home/index.sqlite
```

Ingest probes the file, samples one frame every five seconds by default (`--every 5`), pulls any embedded subtitle streams plus matching `.srt`/`.vtt` sidecars, and writes everything into the local index. Re-running on the same file is a no-op unless you pass `--force`.

### Search what was said

```bash
reelgrep search-subtitles /tmp/reelgrep-demo/demo.mp4 "kubernetes"
```

Synthetic example output:

```text
00:00:04.000  Today we are covering Kubernetes networking.
1 matches
```

Requires either embedded subtitles, a sidecar `.srt`/`.vtt` next to the video, or Whisper-transcribed cues (see next section). The `[whisper]` extra adds local transcription so screen recordings, lectures, and other un-captioned videos become searchable.

### Transcribe a video without subtitles

```bash
reelgrep transcribe /tmp/reelgrep-demo/lecture.mp4 --model tiny
```

Larger models (`small`, `medium`, `large-v3`, `large-v3-turbo`) trade speed for accuracy. After transcribing, `reelgrep search-subtitles` works against the new cues immediately.

The cues are stored alongside any embedded or sidecar subtitles with `source='whisper'`, so the index treats them uniformly. Re-running transcribe on the same video is a no-op unless you pass `--force`. Pass `--no-db` to print the cues as JSON to stdout instead of writing to the index.

You can also fold transcription into ingest itself: `reelgrep ingest ~/lecture.mp4 --transcribe` runs Whisper only when the normal embedded/sidecar pass finds nothing.

### Align an official transcript onto Whisper timestamps

If the institution ships a clean prose transcript next to the video (Canvas / Kaltura courses, conference talk hosts that post the speaker's text afterwards), Whisper's transcription is the wrong source of truth - the official transcript is cleaner and uses correct terminology. `reelgrep align` maps the official text onto the Whisper-derived timestamps so you keep accurate timing and the canonical wording.

```bash
reelgrep align /tmp/reelgrep-demo/demo.mp4 \
  --transcript /tmp/reelgrep-demo/demo_transcript.txt \
  --language en \
  --out /tmp/reelgrep-demo/aligned.srt
```

Synthetic example output:

```text
aligned:        /tmp/reelgrep-demo/demo.mp4
transcript:     /tmp/reelgrep-demo/demo_transcript.txt
language:       en
cues:           3 (matched 18/18 transcript words, coverage 100.0%)
avg similarity: 1.00
srt:            /tmp/reelgrep-demo/aligned.srt
```

Aligned cues preserve the transcript's terminology, punctuation, and capitalization while keeping the Whisper timestamps.

Accepts `.txt`, `.md`, `.pdf` transcripts. Auto-runs `whisper:tiny` if no cues exist for the video yet, so the typical flow is one-shot. Cues land in the `subtitles` table with `source='aligned'` so they coexist with `whisper`, `embedded`, and `sidecar` sources. The optional `--out file.srt` writes a standard SRT file you can hand to a video player.

Cues whose similarity to the transcript falls below `--min-similarity` (default 0.55) keep their original Whisper text rather than being fabricated - if the transcript doesn't actually match the audio for a stretch (Q&A inserted, slide change, etc.), the engine refuses to invent alignment.

### Browse the whole library in a local web UI

```bash
reelgrep serve
```

Opens `http://127.0.0.1:8765/` in your default browser. The UI surfaces every ingested video in a sidebar, every cue (embedded, sidecar, or Whisper) in a searchable Subtitles tab per video, every sampled frame in a paginated grid with a lightbox, every person-search result with thumbnails grouped by confidence, and every export artifact with its manifest sidecar link.

The search bar in the header runs an FTS5 query across every video in the index, then groups the hits by video. Clicking a hit opens that video's Subtitles tab with the term highlighted.

The server binds to loopback only by default (`--host 127.0.0.1`), reads exclusively from the local SQLite index, and serves frame and export files via an allow-list (paths must already be referenced in the index - it is not a general filesystem proxy). Pass `--no-open-browser` to skip the auto-launch, `--port` to change the port, and `--reload` for frontend development.

### Find a specific person

```bash
reelgrep find-person /tmp/reelgrep-demo/demo.mp4 \
  --label speaker_a \
  --positive /tmp/reelgrep-demo/refs/speaker_a/headshot1.jpg \
  --positive /tmp/reelgrep-demo/refs/speaker_a/headshot2.jpg \
  --negative /tmp/reelgrep-demo/refs/false_positives/lookalike.jpg \
  --out /tmp/reelgrep-demo/speaker_a_matches
```

#### Why negatives matter

Face matching from cast or speaker headshots alone is unreliable - lookalikes, twins, family members, and similar-looking people in similar settings will all score high. reelgrep treats matching as a precision-over-recall job: positive examples anchor the search, and negative examples (a known false positive caught in a prior run, a sibling's photo, a stock image of the same demographic) push lookalikes below the acceptance threshold. The default backend uses cosine distance to the positive centroid minus the nearest negative cosine; the more negatives you provide, the fewer false positives you get back.

### Cut a sub-clip

```bash
reelgrep export-clip ~/Videos/some-talk.mp4 --start 0:10:00 --end 0:10:30 --out highlight.mp4
```

Stream-copies by default (fast, no re-encode). Pass `--reencode` if the source codec or container is awkward for downstream tools. Writes `highlight.mp4` plus `highlight.mp4.manifest.json` next to it.

### Build a contact sheet

```bash
reelgrep contact-sheet ~/Videos/some-talk.mp4 --out sheet.jpg --cols 6 --every 30
```

Samples a frame every 30 seconds, lays them out in a 6-column grid, writes `sheet.jpg` plus a manifest. Pass `--use-cached` to reuse frames already sampled during ingest instead of re-sampling.

### Render a webp loop

```bash
reelgrep make-gif ~/Videos/some-talk.mp4 --start 0:10:00 --duration 5 --out highlight.webp
```

The output is animated WebP, not GIF - smaller files, better quality, supported in modern browsers and chat clients. Defaults: 12 fps, 480px wide. Tune with `--fps` and `--width`.

## Backends

reelgrep separates "where is the video file?" from "what do I want to do with it?" via a small backend layer:

- **local** (default): pass a file path, it is used directly.
- **jellyfin**: resolve a Jellyfin item name or 32-hex `ItemId` to its local file path via the Jellyfin HTTP API, then pipe into other commands. Configured via `JELLYFIN_URL` and `JELLYFIN_API_KEY` (same names as the `jellyfin-mcp` project).

Example:

```bash
export JELLYFIN_URL=http://media.example.com:8096
export JELLYFIN_API_KEY=<key>
reelgrep jellyfin resolve "Talk: Container Networking" | xargs -I {} reelgrep ingest {}
```

## Person and visual search models

Two backends ship in v0.1.0, both pluggable, both opt-in via extras:

- **face_embed** (default, `[face]` extra): insightface ArcFace 512-dim embeddings. Fast on CPU, deterministic, well-suited to face matching with the positive/negative anchor pattern. Default acceptance threshold `0.30` (cosine margin over nearest negative).
- **ollama_vision** (`[vision]` extra): per-frame chat against a local Ollama vision model (default `qwen2-vl:7b`, configurable via `OLLAMA_VISION_MODEL`). Slower per frame, but handles "find frames where the speaker is wearing a red jacket" or "find shots of the building exterior" - kinds of queries that pure face embeddings cannot answer. Default acceptance threshold `0.65`.

Switch engines with `--backend ollama_vision` on the `find-person` command. Both engines accept the same `--positive` / `--negative` / `--threshold` / `--top-k` flags.

## Using reelgrep as a library

Beyond the CLI, reelgrep ships a small Python library surface for
downstream packages that want to drive ingest, search, and
transcription programmatically, or to plug in a custom content
source (S3, an internal HTTP archive, etc.) or a custom person
model.

```bash
pip install reelgrep
```

See [docs/extending.md](docs/extending.md) for the consumer-facing
guide: installation, quickstart, `BaseBackend` and `BasePersonModel`
worked examples, the `tags` table extension surface, and the
current stability contract.

## Storage and privacy

- The index database lives at `~/.local/share/reelgrep/index.sqlite` by default. Override with `REELGREP_HOME` or `REELGREP_DB`.
- Sampled frames cache to `~/.local/share/reelgrep/cache/frames/<hash>/` and subtitles to `~/.local/share/reelgrep/cache/subtitles/<hash>/`.
- Every export (clip, gif, screenshot, contact sheet) writes a JSON manifest sidecar next to it with the parameters and source hash so outputs are reproducible.
- No telemetry. No background network calls. The Ollama backend talks to the Ollama URL you configure (default `http://127.0.0.1:11434`). The Jellyfin adapter talks only to the URL you set. Everything else stays local.
- You are responsible for confirming you have the rights to analyze and store frames, clips, and derived data from the videos you process.

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `REELGREP_HOME` | `~/.local/share/reelgrep` | Root directory for the index and cache. |
| `REELGREP_DB` | `<home>/index.sqlite` | Override the SQLite index path independently of `REELGREP_HOME`. |
| `REELGREP_CACHE` | `<home>/cache` | Override the frame / subtitle cache directory. |
| `REELGREP_FFMPEG` | `ffmpeg` | Path or name of the `ffmpeg` binary to invoke. |
| `REELGREP_FFPROBE` | `ffprobe` | Path or name of the `ffprobe` binary to invoke. |
| `JELLYFIN_URL` | (unset) | Base URL of a Jellyfin server for the `jellyfin` backend. |
| `JELLYFIN_API_KEY` | (unset) | API key for the Jellyfin server. |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Base URL of the Ollama server for the `ollama_vision` backend. |
| `OLLAMA_VISION_MODEL` | `qwen2-vl:7b` | Model id Ollama should serve for vision requests. |

## Commands

| Command | What it does |
|---|---|
| `reelgrep ingest <video>` | Probe, extract subtitles, sample frames, persist to the index. See [Ingest a video](#ingest-a-video). |
| `reelgrep info <hash-or-name>` | Print metadata + counts for an indexed video. |
| `reelgrep ls` | List indexed videos (most-recent first). |
| `reelgrep export-clip <video> --start --end --out` | Cut a sub-clip, stream-copy by default. See [Cut a sub-clip](#cut-a-sub-clip). |
| `reelgrep make-gif <video> --start --duration --out` | Render an animated WebP loop. See [Render a webp loop](#render-a-webp-loop). |
| `reelgrep contact-sheet <video> --out` | Build a grid of thumbnails. See [Build a contact sheet](#build-a-contact-sheet). |
| `reelgrep search-subtitles <video> <query>` | FTS5 search over indexed subtitle cues. See [Search what was said](#search-what-was-said). |
| `reelgrep transcribe <video> --model` | Whisper-transcribe and index cues for an un-captioned video. See [Transcribe a video without subtitles](#transcribe-a-video-without-subtitles). |
| `reelgrep align <video> --transcript <file>` | Map a clean prose transcript onto Whisper timestamps. See [Align an official transcript onto Whisper timestamps](#align-an-official-transcript-onto-whisper-timestamps). |
| `reelgrep find-person <video> --label --positive --out` | Locate frames containing a person. See [Find a specific person](#find-a-specific-person). |
| `reelgrep serve [--port 8765]` | Open the local browser UI for the whole index. See [Browse the whole library](#browse-the-whole-library-in-a-local-web-ui). |
| `reelgrep jellyfin resolve <query>` | Resolve a Jellyfin item to its local file path for piping. |
| `reelgrep --db PATH <subcommand>` | One-shot override for the index database path. |

## Development

```bash
git clone https://github.com/solomonneas/reelgrep
cd reelgrep
git config core.hooksPath .githooks
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,face,vision,whisper,web,align]"
.venv/bin/pytest
.venv/bin/ruff check .
```

Tests marked `integration` shell out to the real `ffmpeg`, `ffprobe`, and `insightface` stacks. To skip them in a quick local loop:

```bash
.venv/bin/pytest -m "not integration"
```

## Roadmap

Queued for later releases:

- Writing thumbnails and chapters back to Jellyfin.
- Cross-video person clustering ("find all distinct faces in this whole library").
- True wav2vec2 word-level alignment for cases where cue-level timing isn't tight enough.

Shipped in v0.4.0: prose-transcript alignment (`reelgrep align`) via the `[align]` extra, plus a public Python library API (`reelgrep.index.ingest_video`) for embedders. See [Align an official transcript onto Whisper timestamps](#align-an-official-transcript-onto-whisper-timestamps). The MCP wrapper for agentic use also shipped as its own repo at [solomonneas/reelgrep-mcp](https://github.com/solomonneas/reelgrep-mcp) (`npm install -g reelgrep-mcp`).

Shipped in v0.3.0: local browser UI (`reelgrep serve`) backed by a Starlette JSON API + vanilla HTML/CSS/JS frontend, with cross-library subtitle search as the headline feature. See [Browse the whole library in a local web UI](#browse-the-whole-library-in-a-local-web-ui).

Shipped in v0.2.0: local Whisper transcription via the `[whisper]` extra and the new `reelgrep transcribe` command. See [Transcribe a video without subtitles](#transcribe-a-video-without-subtitles).

## License

MIT. See [LICENSE](LICENSE).
