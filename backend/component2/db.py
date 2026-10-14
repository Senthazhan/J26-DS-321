import json
import sqlite3

from component2.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    activity_id TEXT PRIMARY KEY,
    team_id     TEXT NOT NULL,
    source      TEXT NOT NULL,
    member_id   TEXT,
    ts          TEXT NOT NULL,
    text        TEXT NOT NULL,
    meta        TEXT
);

CREATE TABLE IF NOT EXISTS stage_labels (
    activity_id TEXT NOT NULL,
    labeller    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    confidence  REAL,
    PRIMARY KEY (activity_id, labeller)
);
"""


def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def save_activities(conn, rows):
    before = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
    for r in rows:
        conn.execute(
            "INSERT OR IGNORE INTO activities VALUES (?, ?, ?, ?, ?, ?, ?)",
            (r["activity_id"], r["team_id"], r["source"], r["member_id"],
             r["ts"], r["text"], json.dumps(r["meta"])),
        )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
    return after - before

def save_labels(conn, labeller, labels):
    """labels: list of (activity_id, stage, confidence). Re-running replaces old labels."""
    conn.executemany(
        "INSERT OR REPLACE INTO stage_labels VALUES (?, ?, ?, ?)",
        [(aid, labeller, stage, conf) for aid, stage, conf in labels],
    )
    conn.commit()