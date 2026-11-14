"""Request log store for the admin GUI's debug log page.

ponytail: stdlib sqlite3, one connection per call — no pooling, no
persistent connection lifecycle. This is low-throughput debug logging
(one row per inspection decision), not a hot path that needs one.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).with_name("logs.db")
DB_PATH = Path(os.environ.get("ARMOR_LOG_DB_PATH", _DEFAULT_DB_PATH))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    source TEXT,
    provider TEXT,
    direction TEXT,
    content TEXT,
    decision TEXT,
    risk_score REAL,
    route TEXT,
    findings TEXT,
    latency_ms REAL,
    reason TEXT,
    bandit_action TEXT,
    bandit_context TEXT,
    analyst_correction TEXT,
    reward REAL,
    reviewed_at TEXT,
    decision_correction TEXT,
    decision_label_source TEXT,
    decision_reviewed_at TEXT,
    bandit_correction TEXT,
    bandit_label_source TEXT,
    bandit_reward REAL,
    bandit_reward_applied INTEGER,
    bandit_reviewed_at TEXT,
    policy_action TEXT,
    policy_confidence REAL,
    review_reason TEXT,
    label_source TEXT,
    new_policy_action TEXT,
    intervention_probability REAL,
    block_probability REAL,
    conformal_set TEXT,
    ood_score REAL,
    nearest_reference_class TEXT,
    nearest_reference_similarity REAL,
    inference_trace TEXT
)
"""

# ponytail: SQLite has no "ADD COLUMN IF NOT EXISTS" — this guard migrates an
# existing logs.db in place (fresh installs already get these columns from
# _SCHEMA above, which CREATE TABLE IF NOT EXISTS skips on an existing table).
_BANDIT_COLUMNS = {
    "bandit_action": "TEXT",
    "bandit_context": "TEXT",
    "analyst_correction": "TEXT",
    "reward": "REAL",
    "reviewed_at": "TEXT",
    # Explicit feedback ownership; legacy analyst_correction remains readable.
    "decision_correction": "TEXT",
    "decision_label_source": "TEXT",
    "decision_reviewed_at": "TEXT",
    "bandit_correction": "TEXT",
    "bandit_label_source": "TEXT",
    "bandit_reward": "REAL",
    "bandit_reward_applied": "INTEGER",
    "bandit_reviewed_at": "TEXT",
    # Phase 1 policy head (queue/QUEUE-02) -- kept in the same table as the
    # bandit's columns rather than a new one: one review queue, not two.
    "policy_action": "TEXT",
    "policy_confidence": "REAL",
    "review_reason": "TEXT",
    "label_source": "TEXT",
    # Hierarchical policy head A/B/C comparison (queue/QUEUE-02) -- logs
    # system B/C's recommendation alongside the flat head's (policy_action,
    # now "legacy" for this comparison) and the real production decision,
    # for live side-by-side shadow-mode comparison.
    "new_policy_action": "TEXT",
    "intervention_probability": "REAL",
    "block_probability": "REAL",
    "conformal_set": "TEXT",
    "ood_score": "REAL",
    "nearest_reference_class": "TEXT",
    "nearest_reference_similarity": "REAL",
    # Stage timings and bounded-inference metadata; nullable and additive for
    # compatibility with existing logs and older Armor processes.
    "inference_trace": "TEXT",
}


def init_db(path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(_SCHEMA)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(logs)")}
        for column, sql_type in _BANDIT_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE logs ADD COLUMN {column} {sql_type}")
        conn.commit()
    finally:
        conn.close()


def log_entry(
    path: Path,
    *,
    source: str,
    decision: str,
    provider: str | None = None,
    direction: str | None = None,
    content: str = "",
    risk_score: float = 0.0,
    route: str | None = None,
    findings: list | None = None,
    latency_ms: float = 0.0,
    reason: str | None = None,
    bandit_action: str | None = None,
    bandit_context: str | None = None,
    policy_action: str | None = None,
    policy_confidence: float | None = None,
    review_reason: str | None = None,
    new_policy_action: str | None = None,
    intervention_probability: float | None = None,
    block_probability: float | None = None,
    conformal_set: str | None = None,
    ood_score: float | None = None,
    nearest_reference_class: str | None = None,
    nearest_reference_similarity: float | None = None,
    inference_trace: dict | None = None,
) -> int:
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute(
            "INSERT INTO logs "
            "(timestamp, source, provider, direction, content, decision, risk_score, route, findings, latency_ms, reason, bandit_action, bandit_context, policy_action, policy_confidence, review_reason, "
            "new_policy_action, intervention_probability, block_probability, conformal_set, ood_score, nearest_reference_class, nearest_reference_similarity, inference_trace) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                source,
                provider,
                direction,
                content,
                decision,
                risk_score,
                route,
                json.dumps(findings or []),
                latency_ms,
                reason,
                bandit_action,
                bandit_context,
                policy_action,
                policy_confidence,
                review_reason,
                new_policy_action,
                intervention_probability,
                block_probability,
                conformal_set,
                ood_score,
                nearest_reference_class,
                nearest_reference_similarity,
                json.dumps(inference_trace) if inference_trace is not None else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def record_correction(
    path: Path, entry_id: int, correction: str, reward: float, reviewed_at: str, label_source: str = "human"
) -> None:
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE logs SET analyst_correction = ?, reward = ?, reviewed_at = ?, label_source = ? WHERE id = ?",
            (correction, reward, reviewed_at, label_source, entry_id),
        )
        conn.commit()
    finally:
        conn.close()


def record_feedback(
    path: Path,
    entry_id: int,
    correction: str,
    target: str,
    label_source: str,
    reviewed_at: str,
    bandit_reward: float | None = None,
) -> dict:
    """Store a correction against the production decision, bandit, or both.

    Existing analyst_correction columns are maintained as a legacy mirror for
    bandit-targeted feedback. New training/evaluation code uses the explicit
    target columns instead.
    """
    if target not in {"decision", "bandit", "both"}:
        raise ValueError("target must be decision, bandit, or both")
    if correction not in {"allow", "flag", "block", "review"}:
        raise ValueError("invalid correction action")
    if label_source not in {
        "human", "pseudo_model", "reviewer_ai_a", "reviewer_ai_b",
        "reviewer_ai_consensus", "reviewer_consensus",
    }:
        raise ValueError("invalid label source")

    init_db(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT decision, bandit_action, bandit_context, decision_correction, bandit_correction, "
            "decision_label_source, bandit_label_source, bandit_reward_applied "
            "FROM logs WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"no log entry with id {entry_id}")
        (decision, bandit_action, bandit_context, decision_correction, bandit_correction,
         decision_source, bandit_source, bandit_reward_applied) = row
        trusted_sources = {"human", "reviewer_consensus"}
        trusted = label_source in trusted_sources
        if target in {"decision", "both"}:
            if decision is None:
                raise ValueError("log entry has no production decision")
            if decision_correction is not None and not (
                decision_correction == correction
                and ((trusted and decision_source not in trusted_sources)
                     or (target == "both" and trusted and not bandit_reward_applied))
            ):
                raise FileExistsError("production decision already has feedback")
        if target in {"bandit", "both"}:
            if not bandit_action or not bandit_context:
                raise ValueError("log entry has no bandit recommendation context")
            if bandit_correction is not None and not (
                bandit_correction == correction
                and trusted
                and (bandit_source not in trusted_sources or not bandit_reward_applied)
            ):
                raise FileExistsError("bandit recommendation already has feedback")

        if target == "decision":
            conn.execute(
                "UPDATE logs SET decision_correction=?, decision_label_source=?, decision_reviewed_at=? WHERE id=?",
                (correction, label_source, reviewed_at, entry_id),
            )
        elif target == "bandit":
            conn.execute(
                "UPDATE logs SET bandit_correction=?, bandit_label_source=?, bandit_reward=?, "
                "bandit_reward_applied=0, bandit_reviewed_at=?, analyst_correction=COALESCE(analyst_correction,?), "
                "label_source=COALESCE(label_source,?), reviewed_at=COALESCE(reviewed_at,?) WHERE id=?",
                (correction, label_source, bandit_reward, reviewed_at, correction, label_source, reviewed_at, entry_id),
            )
        else:
            conn.execute(
                "UPDATE logs SET decision_correction=?, decision_label_source=?, decision_reviewed_at=?, "
                "bandit_correction=?, bandit_label_source=?, bandit_reward=?, bandit_reward_applied=0, "
                "bandit_reviewed_at=?, analyst_correction=COALESCE(analyst_correction,?), "
                "label_source=COALESCE(label_source,?), reviewed_at=COALESCE(reviewed_at,?) WHERE id=?",
                (correction, label_source, reviewed_at, correction, label_source, bandit_reward,
                 reviewed_at, correction, label_source, reviewed_at, entry_id),
            )
        conn.commit()
        return {"decision": decision, "bandit_action": bandit_action, "bandit_context": bandit_context}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def mark_bandit_reward_applied(path: Path, entry_id: int, reward: float) -> None:
    init_db(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE logs SET bandit_reward=?, bandit_reward_applied=1 WHERE id=? AND bandit_correction IS NOT NULL",
            (reward, entry_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_pending_reviews(path: Path, limit: int = 200) -> list[dict]:
    """Unresolved production review decisions, excluding shadow-head suggestions
    and duplicate proxy summary rows. The queue uses the existing logs table."""
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM logs WHERE source = 'inspect' AND decision = 'review' "
            "AND decision_correction IS NULL ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["findings"] = json.loads(result["findings"] or "[]")
            result["inference_trace"] = (
                json.loads(result["inference_trace"]) if result.get("inference_trace") else None
            )
            results.append(result)
        return results
    finally:
        conn.close()


def list_entries(path: Path, limit: int = 200) -> list[dict]:
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, timestamp, source, provider, direction, decision, route, latency_ms, inference_trace "
            "FROM logs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = [dict(row) for row in rows]
        for result in results:
            result["inference_trace"] = json.loads(result["inference_trace"]) if result["inference_trace"] else None
        return results
    finally:
        conn.close()


def get_entry(path: Path, entry_id: int) -> dict | None:
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM logs WHERE id = ?", (entry_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["findings"] = json.loads(result["findings"] or "[]")
        if result.get("inference_trace"):
            result["inference_trace"] = json.loads(result["inference_trace"])
        return result
    finally:
        conn.close()
