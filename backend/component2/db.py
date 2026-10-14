"""
Database layer. Works with BOTH:
  - PostgreSQL  if DATABASE_URL is set in backend/.env   (the proposal's database)
  - SQLite      otherwise (a single file, handy for quick tests)

Every other file only uses connect(), save_activities(), save_labels(),
save_drift() and conn.execute(...), so they never need to know which one is used.
"""
import json
import sqlite3

from component2.config import DATABASE_URL, DB_PATH

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

CREATE TABLE IF NOT EXISTS drift_results (
    team_id     TEXT NOT NULL,
    window_end  TEXT NOT NULL,
    dimension   TEXT NOT NULL,
    planned     TEXT,
    seen_top    TEXT,
    drift_score REAL,
    severity    TEXT NOT NULL,
    explanation TEXT,
    PRIMARY KEY (team_id, window_end, dimension)
);

CREATE INDEX IF NOT EXISTS ix_activities_team_ts ON activities (team_id, ts);
"""


class Connection:
    """Gives SQLite and PostgreSQL the same methods, so no other file needs to care."""

    def __init__(self, url=DATABASE_URL):
        self.is_postgres = url.startswith(("postgresql://", "postgres://"))
        if self.is_postgres:
            import psycopg                      # pip install "psycopg[binary]"
            self.raw = psycopg.connect(url)
        else:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.raw = sqlite3.connect(DB_PATH)

    def _sql(self, sql):
        # SQLite writes placeholders as ?, PostgreSQL (psycopg) as %s
        return sql.replace("?", "%s") if self.is_postgres else sql

    def execute(self, sql, params=()):
        cur = self.raw.cursor()
        cur.execute(self._sql(sql), params)
        return cur

    def executemany(self, sql, rows):
        cur = self.raw.cursor()
        cur.executemany(self._sql(sql), rows)
        return cur

    def commit(self):
        self.raw.commit()


def connect():
    conn = Connection()
    for statement in SCHEMA.split(";"):
        if statement.strip():
            conn.execute(statement)
    conn.commit()
    return conn


def count_rows(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def save_activities(conn, rows):
    """Adds new activities; ones already saved are skipped. Returns how many were new."""
    before = count_rows(conn, "activities")
    conn.executemany(
        "INSERT INTO activities VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (activity_id) DO NOTHING",
        [(r["activity_id"], r["team_id"], r["source"], r["member_id"],
          r["ts"], r["text"], json.dumps(r["meta"])) for r in rows],
    )
    conn.commit()
    return count_rows(conn, "activities") - before


def save_labels(conn, labeller, labels):
    """labels: list of (activity_id, stage, confidence). Re-running replaces old labels."""
    conn.executemany(
        "INSERT INTO stage_labels VALUES (?, ?, ?, ?) "
        "ON CONFLICT (activity_id, labeller) DO UPDATE "
        "SET stage = excluded.stage, confidence = excluded.confidence",
        [(aid, labeller, str(stage), None if conf is None else float(conf)) for aid, stage, conf in labels],
    )
    conn.commit()


def save_drift(conn, results):
    """results: list of dicts from detect_drift. Re-running replaces old results."""
    conn.executemany(
        "INSERT INTO drift_results VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (team_id, window_end, dimension) DO UPDATE "
        "SET planned = excluded.planned, seen_top = excluded.seen_top, "
        "drift_score = excluded.drift_score, severity = excluded.severity, "
        "explanation = excluded.explanation",
        [(r["team_id"], r["window_end"], r["dimension"], r["planned"], r["seen_top"],
          r["drift_score"], r["severity"], r["explanation"]) for r in results],
    )
    conn.commit()