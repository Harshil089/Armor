import sqlite3

from armor.observability import log_store


def test_log_entry_round_trip(tmp_path):
    path = tmp_path / "logs.db"

    entry_id = log_store.log_entry(
        path,
        source="proxy",
        provider="groq",
        direction="prompt",
        content="hello world",
        decision="allow",
        risk_score=0.1,
        route="fast",
        findings=[{"label": "sensitive_data_exposure", "score": 0.1}],
        latency_ms=12.5,
        reason=None,
        inference_trace={"tokenization_ms": 1.25, "layer22_ms": 11.0},
    )

    entries = log_store.list_entries(path)
    assert len(entries) == 1
    assert entries[0]["id"] == entry_id
    assert entries[0]["provider"] == "groq"
    assert "content" not in entries[0]  # list is summary-only

    full = log_store.get_entry(path, entry_id)
    assert full["content"] == "hello world"
    assert full["findings"] == [{"label": "sensitive_data_exposure", "score": 0.1}]
    assert full["risk_score"] == 0.1
    assert full["inference_trace"] == {"tokenization_ms": 1.25, "layer22_ms": 11.0}
    assert entries[0]["inference_trace"] == {"tokenization_ms": 1.25, "layer22_ms": 11.0}


def test_list_entries_most_recent_first(tmp_path):
    path = tmp_path / "logs.db"
    log_store.log_entry(path, source="inspect", decision="allow")
    second_id = log_store.log_entry(path, source="inspect", decision="block")

    entries = log_store.list_entries(path)
    assert entries[0]["id"] == second_id


def test_get_entry_missing_returns_none(tmp_path):
    path = tmp_path / "logs.db"
    log_store.init_db(path)
    assert log_store.get_entry(path, 999) is None


def test_feedback_schema_migration_preserves_legacy_corrections(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE logs (id INTEGER PRIMARY KEY, analyst_correction TEXT, reward REAL, label_source TEXT)"
    )
    conn.execute("INSERT INTO logs VALUES (1, 'allow', -1.0, 'pseudo_model')")
    conn.commit()
    conn.close()

    log_store.init_db(path)
    log_store.init_db(path)  # idempotent retry
    conn = sqlite3.connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(logs)")}
    legacy = conn.execute(
        "SELECT analyst_correction, reward, label_source FROM logs WHERE id=1"
    ).fetchone()
    conn.close()
    assert {"decision_correction", "bandit_correction", "bandit_reward_applied"} <= columns
    assert legacy == ("allow", -1.0, "pseudo_model")


def test_inference_trace_migration_is_additive_and_idempotent(tmp_path):
    path = tmp_path / "old_logs.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, decision TEXT, content TEXT)")
    conn.execute("INSERT INTO logs VALUES (1, 'allow', 'preserved')")
    conn.commit()
    conn.close()

    log_store.init_db(path)
    log_store.init_db(path)
    conn = sqlite3.connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(logs)")}
    original = conn.execute("SELECT decision, content FROM logs WHERE id=1").fetchone()
    conn.close()
    assert "inference_trace" in columns
    assert original == ("allow", "preserved")
