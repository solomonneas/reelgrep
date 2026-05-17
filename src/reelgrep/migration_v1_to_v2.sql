PRAGMA foreign_keys = OFF;

DROP TABLE subtitles_fts;

ALTER TABLE subtitles RENAME TO subtitles_old;

CREATE TABLE subtitles (
  id INTEGER PRIMARY KEY,
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  language TEXT,
  source TEXT NOT NULL CHECK(source IN ('embedded','sidecar','whisper')),
  stream_index INTEGER,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  text TEXT NOT NULL
);

INSERT INTO subtitles (id, video_id, language, source, stream_index, start_ms, end_ms, text)
SELECT id, video_id, language, source, stream_index, start_ms, end_ms, text FROM subtitles_old;

DROP TABLE subtitles_old;

CREATE INDEX idx_subtitles_video_ts ON subtitles(video_id, start_ms);

CREATE VIRTUAL TABLE subtitles_fts USING fts5(
  text, content='subtitles', content_rowid='id', tokenize='porter unicode61'
);

INSERT INTO subtitles_fts(rowid, text) SELECT id, text FROM subtitles;

UPDATE schema_version SET version = 2;

PRAGMA foreign_keys = ON;
