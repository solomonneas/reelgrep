PRAGMA foreign_keys = OFF;

CREATE TABLE face_detections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
  bbox_x INTEGER NOT NULL,
  bbox_y INTEGER NOT NULL,
  bbox_w INTEGER NOT NULL,
  bbox_h INTEGER NOT NULL,
  embedding BLOB NOT NULL,
  embedding_model TEXT NOT NULL,
  detected_at TEXT NOT NULL
);
CREATE INDEX idx_face_detections_frame ON face_detections(frame_id);

CREATE TABLE face_clusters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  label TEXT,
  rep_detection_id INTEGER REFERENCES face_detections(id) ON DELETE SET NULL,
  size INTEGER NOT NULL DEFAULT 0,
  computed_at TEXT NOT NULL
);

CREATE TABLE face_cluster_members (
  cluster_id INTEGER NOT NULL REFERENCES face_clusters(id) ON DELETE CASCADE,
  detection_id INTEGER NOT NULL REFERENCES face_detections(id) ON DELETE CASCADE,
  distance REAL NOT NULL,
  PRIMARY KEY (cluster_id, detection_id)
);
CREATE INDEX idx_face_cluster_members_det ON face_cluster_members(detection_id);

UPDATE schema_version SET version = 4;

PRAGMA foreign_keys = ON;
