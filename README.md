# reelgrep

> Local video search and media analysis. Find people, outfits, objects, scenes, spoken phrases, and useful clips inside your own video library.

Status: pre-alpha, under active development.

reelgrep is a command-line tool for indexing and searching video files on disk. Point it at a folder of lectures, conference talks, movies, TV episodes, or training videos and it builds a local index of metadata, sampled frames, contact sheets, and subtitle text. From there you can grep transcripts for a spoken phrase, jump to the matching timecode, export a clip, or plug in optional models for person and object detection.

Everything runs locally. No cloud upload, no account, no telemetry. Your library never leaves the machine.

## Install

```bash
pipx install reelgrep
```

reelgrep shells out to `ffmpeg` and `ffprobe` for media work, so both must be installed system-wide and available on `PATH`. On Debian/Ubuntu:

```bash
sudo apt install ffmpeg
```

## Quickstart

TODO: index a folder, run a subtitle search, export a clip. Coming with the first usable release.

## License

MIT. See [LICENSE](LICENSE).
