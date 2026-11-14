"""AI-reviewer agreement and human-adjudication store (queue/QUEUE-02's
supervision-quality phase). Separate table from log_store.py's `logs` deliberately -- that
table fits one recommendation + one correction per row; this audit needs
two independent reviewer decisions plus an adjudication per sample, a
different shape entirely.

Every reviewer decision is stored permanently and never overwritten by
adjudication -- Phase 1's explicit rule, so a later question ("what did
each reviewer actually say before adjudication changed the label") stays
answerable.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).with_name("review_audit.db")
DB_PATH = Path(os.environ.get("ARMOR_REVIEW_DB_PATH", _DEFAULT_DB_PATH))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT UNIQUE,
    log_id INTEGER,
    content TEXT,
    direction TEXT,
    existing_label TEXT,
    label_source_at_audit TEXT,
    reviewer_a_decision TEXT,
    reviewer_a_id TEXT,
    reviewer_a_reason_code TEXT,
    reviewer_b_decision TEXT,
    reviewer_b_id TEXT,
    reviewer_b_reason_code TEXT,
    agreement INTEGER,
    adjudicated_label TEXT,
    adjudicated_at TEXT,
    rubric_version TEXT,
    policy_subtype TEXT,
    created_at TEXT
)
"""


def init_db(path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(_SCHEMA)
        # Expand an existing audit database compatibly. Older rows retain
        # their decisions with NULL reviewer IDs; new AI reviews can record
        # the actual reviewer identity without relabeling legacy records.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(review_audit)")}
        for column in ("reviewer_a_id", "reviewer_b_id"):
            if column not in columns:
                conn.execute(f"ALTER TABLE review_audit ADD COLUMN {column} TEXT")
        conn.commit()
    finally:
        conn.close()


def add_sample(
    path: Path,
    *,
    sample_id: str,
    content: str,
    existing_label: str,
    label_source_at_audit: str,
    rubric_version: str,
    log_id: int | None = None,
    direction: str | None = None,
    reviewer_a_id: str | None = None,
    reviewer_b_id: str | None = None,
) -> int:
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO review_audit "
            "(sample_id, log_id, content, direction, existing_label, label_source_at_audit, rubric_version, "
            "reviewer_a_id, reviewer_b_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sample_id, log_id, content, direction, existing_label,
                label_source_at_audit, rubric_version,
                reviewer_a_id, reviewer_b_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        if cursor.rowcount == 0:
            prior = conn.execute(
                "SELECT id, content, rubric_version FROM review_audit WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
            if prior is None or prior[1] != content or prior[2] != rubric_version:
                raise ValueError(f"sample_id {sample_id!r} already exists with different audit data")
            sample_id_value = prior[0]
        else:
            sample_id_value = cursor.lastrowid
        conn.commit()
        return sample_id_value
    finally:
        conn.close()


def record_reviewer_decision(
    path: Path, sample_id: str, reviewer: str, decision: str,
    reason_code: str | None = None, reviewer_id: str | None = None,
) -> None:
    """reviewer: 'a' or 'b'. Setting both independently, then computing
    agreement only once both are present -- never inferred from a partial
    state."""
    if reviewer not in ("a", "b"):
        raise ValueError(f"reviewer must be 'a' or 'b', got {reviewer!r}")
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            f"SELECT reviewer_{reviewer}_decision, reviewer_{reviewer}_id FROM review_audit WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
        # Reviewer output is an audit record, not a mutable correction field.
        # Allow identical retries and adding a previously missing ID, but
        # reject attempts to replace a stored decision or reviewer identity.
        if row and row[0] not in (None, decision):
            conn.rollback()
            raise ValueError(f"reviewer {reviewer!r} decision is already recorded for {sample_id!r}")
        if row and reviewer_id and row[1] not in (None, reviewer_id):
            conn.rollback()
            raise ValueError(f"reviewer {reviewer!r} identity is already recorded for {sample_id!r}")
        conn.execute(
            f"UPDATE review_audit SET reviewer_{reviewer}_decision = COALESCE(reviewer_{reviewer}_decision, ?), "
            f"reviewer_{reviewer}_reason_code = COALESCE(reviewer_{reviewer}_reason_code, ?), "
            f"reviewer_{reviewer}_id = COALESCE(reviewer_{reviewer}_id, ?) WHERE sample_id = ?",
            (decision, reason_code, reviewer_id, sample_id),
        )
        row = conn.execute(
            "SELECT reviewer_a_decision, reviewer_b_decision FROM review_audit WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
        if row and row[0] is not None and row[1] is not None:
            conn.execute(
                "UPDATE review_audit SET agreement = ? WHERE sample_id = ?",
                (1 if row[0] == row[1] else 0, sample_id),
            )
        conn.commit()
    finally:
        conn.close()


def record_adjudication(path: Path, sample_id: str, adjudicated_label: str) -> None:
    """Never overwrites reviewer_a_decision/reviewer_b_decision -- adds the
    adjudicated_label alongside them, per Phase 1's explicit rule."""
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE review_audit SET adjudicated_label = ?, adjudicated_at = ? WHERE sample_id = ?",
            (adjudicated_label, datetime.now(timezone.utc).isoformat(), sample_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_disagreements(path: Path) -> list[dict]:
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM review_audit WHERE agreement = 0 AND adjudicated_label IS NULL"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def all_samples(path: Path = DB_PATH) -> list[dict]:
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM review_audit").fetchall()]
    finally:
        conn.close()
