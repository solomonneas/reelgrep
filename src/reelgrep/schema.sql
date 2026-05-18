CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
INSERT INTO schema_version VALUES (3);

CREATE TABLE videos (
  id INTEGER PRIMARY KEY,
  file_hash TEXT UNIQUE NOT NULL,
  path TEXT NOT NULL,
  duration_ms INTEGER,
  width INTEGER,
  height INTEGER,
  fps REAL,
  container TEXT,
  video_codec TEXT,
  audio_codec TEXT,
  size_bytes INTEGER,
  ingested_at TEXT NOT NULL,
  probe_json TEXT NOT NULL
);
CREATE INDEX idx_videos_path ON videos(path);

CREATE TABLE subtitles (
  id INTEGER PRIMARY KEY,
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  language TEXT,
  source TEXT NOT NULL CHECK(source IN ('embedded','sidecar','whisper','aligned')),
  stream_index INTEGER,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  text TEXT NOT NULL
);
CREATE INDEX idx_subtitles_video_ts ON subtitles(video_id, start_ms);

CREATE VIRTUAL TABLE subtitles_fts USING fts5(
  text, content='subtitles', content_rowid='id', tokenize='porter unicode61'
);

CREATE TABLE frames (
  id INTEGER PRIMARY KEY,
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  timestamp_ms INTEGER NOT NULL,
  path TEXT NOT NULL,
  sampling_strategy TEXT NOT NULL CHECK(sampling_strategy IN ('every_n','scene','manual')),
  width INTEGER,
  height INTEGER
);
CREATE INDEX idx_frames_video_ts ON frames(video_id, timestamp_ms);

CREATE TABLE scenes (
  id INTEGER PRIMARY KEY,
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  score REAL
);

CREATE TABLE person_searches (
  id INTEGER PRIMARY KEY,
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  backend TEXT NOT NULL,
  positive_examples_json TEXT NOT NULL,
  negative_examples_json TEXT NOT NULL,
  config_json TEXT NOT NULL,
  threshold REAL NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE person_matches (
  id INTEGER PRIMARY KEY,
  search_id INTEGER NOT NULL REFERENCES person_searches(id) ON DELETE CASCADE,
  frame_id INTEGER NOT NULL REFERENCES frames(id),
  confidence REAL NOT NULL,
  bbox_json TEXT,
  reasoning TEXT
);

CREATE TABLE export_artifacts (
  id INTEGER PRIMARY KEY,
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK(kind IN ('screenshot','clip','gif','contact_sheet')),
  path TEXT NOT NULL,
  start_ms INTEGER,
  end_ms INTEGER,
  manifest_path TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE tags (
  id INTEGER PRIMARY KEY,
  video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
  frame_id INTEGER REFERENCES frames(id) ON DELETE CASCADE,
  key TEXT NOT NULL,
  value TEXT
);
